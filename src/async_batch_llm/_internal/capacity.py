"""Concurrency-capacity admission and diagnostics shared by execution surfaces."""

from __future__ import annotations

import asyncio
import logging
import random
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core import StartupRampConfig
    from .guardrails import AbortController

from .guardrails import await_with_guardrails

_CAPACITY_DOCS_URL = (
    "https://geoff-davis.github.io/async-batch-llm/production-checklist/"
    "#2-connection-pool-max_connections-vs-max_workers"
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Admission:
    """Metadata for one acquired provider-capacity slot."""

    wait_seconds: float
    capacity: int | None
    startup_ramp_wait_seconds: float = 0.0


def strategy_max_concurrency(strategy: Any) -> int | None:
    """Return a strategy's advertised concurrency capacity, when valid."""
    try:
        capacity = strategy.max_concurrency
    except Exception:
        return None
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        return None
    return int(capacity)


class _CapacityGate:
    """Condition-based gate whose allowed concurrency can rise over time."""

    def __init__(self, capacity: int, ramp: StartupRampConfig | None) -> None:
        self.capacity = capacity
        self._ramp = ramp
        self._active = 0
        self._started = time.perf_counter()
        self._condition = asyncio.Condition()

    def _current_limit(self, now: float) -> tuple[int, float | None]:
        if self._ramp is None:
            return self.capacity, None
        elapsed = max(0.0, now - self._started)
        completed_intervals = int(elapsed / self._ramp.ramp_interval_seconds)
        limit = min(
            self.capacity,
            self._ramp.initial_concurrency + completed_intervals * self._ramp.concurrency_step,
        )
        if limit >= self.capacity:
            return limit, None
        next_boundary = (completed_intervals + 1) * self._ramp.ramp_interval_seconds
        return limit, max(0.0, next_boundary - elapsed)

    async def acquire(
        self,
        *,
        deadline: float | None = None,
        abort_controller: AbortController | None = None,
        item_id: str | None = None,
    ) -> tuple[float, float]:
        started = time.perf_counter()
        ramp_wait = 0.0
        if self._ramp is not None and self._ramp.jitter_seconds > 0:
            limit, _ = self._current_limit(started)
            if limit < self.capacity:
                jitter = random.uniform(0.0, self._ramp.jitter_seconds)
                await await_with_guardrails(
                    asyncio.sleep(jitter),
                    item_deadline=deadline,
                    item_id=item_id,
                    abort_controller=abort_controller,
                )
                ramp_wait += jitter

        async with self._condition:
            while True:
                now = time.perf_counter()
                limit, until_next_ramp = self._current_limit(now)
                if self._active < limit:
                    self._active += 1
                    return max(0.0, now - started), ramp_wait

                segment_started = time.perf_counter()
                if until_next_ramp is None:
                    await await_with_guardrails(
                        self._condition.wait(),
                        item_deadline=deadline,
                        item_id=item_id,
                        abort_controller=abort_controller,
                    )
                else:
                    try:
                        await await_with_guardrails(
                            self._condition.wait(),
                            item_deadline=deadline,
                            item_id=item_id,
                            abort_controller=abort_controller,
                            operation_timeout=max(until_next_ramp, 0.001),
                        )
                    except (TimeoutError, asyncio.TimeoutError):
                        pass
                if limit < self.capacity:
                    ramp_wait += max(0.0, time.perf_counter() - segment_started)

    async def release(self) -> None:
        async with self._condition:
            self._active -= 1
            self._condition.notify_all()


class CapacityLimiter:
    """Per-host capacity/ramp gates keyed by strategy/model scope."""

    def __init__(
        self,
        configured_capacity: int | None = None,
        *,
        max_workers: int = 1,
        startup_ramp: StartupRampConfig | None = None,
    ) -> None:
        self._configured_capacity = configured_capacity
        self._max_workers = max_workers
        self._startup_ramp = startup_ramp
        self._entries: dict[int, tuple[object, int, _CapacityGate]] = {}

    def effective_capacity(self, strategy: Any) -> int | None:
        advertised = strategy_max_concurrency(strategy)
        capacities = [
            capacity for capacity in (advertised, self._configured_capacity) if capacity is not None
        ]
        if self._startup_ramp is not None:
            capacities.append(self._startup_ramp.max_concurrency or self._max_workers)
        return min(capacities) if capacities else None

    @staticmethod
    def _scope(strategy: Any) -> object:
        try:
            scope = strategy.concurrency_scope
        except Exception:
            scope = strategy
        return scope if scope is not None else strategy

    @asynccontextmanager
    async def admit(
        self,
        strategy: Any,
        *,
        deadline: float | None = None,
        abort_controller: AbortController | None = None,
        item_id: str | None = None,
    ) -> AsyncIterator[Admission]:
        """Acquire capacity for one execute attempt and report queueing time."""
        capacity = self.effective_capacity(strategy)
        if capacity is None:
            yield Admission(wait_seconds=0.0, capacity=None)
            return

        scope = self._scope(strategy)
        scope_id = id(scope)
        entry = self._entries.get(scope_id)
        if entry is None or entry[0] is not scope:
            gate = _CapacityGate(capacity, self._startup_ramp)
            self._entries[scope_id] = (scope, capacity, gate)
        else:
            _, registered_capacity, gate = entry
            if registered_capacity != capacity:
                logger.warning(
                    "Concurrency capacity changed from %s to %s for %s; using the original limit.",
                    registered_capacity,
                    capacity,
                    type(strategy).__name__,
                )
                capacity = registered_capacity

        wait_seconds, ramp_wait_seconds = await gate.acquire(
            deadline=deadline,
            abort_controller=abort_controller,
            item_id=item_id,
        )
        if wait_seconds >= 0.001:
            logger.debug(
                "[ADMISSION] %s waited %.3fs for provider capacity=%s",
                type(strategy).__name__,
                wait_seconds,
                capacity,
            )
        try:
            yield Admission(
                wait_seconds=wait_seconds,
                capacity=capacity,
                startup_ramp_wait_seconds=ramp_wait_seconds,
            )
        finally:
            await gate.release()


def warn_if_worker_capacity_exceeded(
    *,
    strategy: Any,
    max_workers: int,
    surface: str,
    stacklevel: int = 2,
) -> None:
    """Warn when framework concurrency exceeds a known provider-client capacity."""
    capacity = strategy_max_concurrency(strategy)
    if capacity is None or max_workers <= capacity:
        return

    warnings.warn(
        f"{surface} max_workers={max_workers} exceeds "
        f"{type(strategy).__name__}.max_concurrency={capacity}. Excess attempts will wait "
        "in ABL admission before attempt_timeout starts. Set max_workers <= "
        f"{capacity}, or rebuild the model with max_connections >= {max_workers}. "
        f"See {_CAPACITY_DOCS_URL}",
        UserWarning,
        stacklevel=stacklevel,
    )
