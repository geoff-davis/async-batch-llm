"""Concurrency-capacity admission and diagnostics shared by execution surfaces."""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

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


def strategy_max_concurrency(strategy: Any) -> int | None:
    """Return a strategy's advertised concurrency capacity, when valid."""
    try:
        capacity = strategy.max_concurrency
    except Exception:
        return None
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        return None
    return int(capacity)


class CapacityLimiter:
    """Per-host provider-capacity semaphores keyed by strategy/model scope."""

    def __init__(self, configured_capacity: int | None = None) -> None:
        self._configured_capacity = configured_capacity
        self._entries: dict[int, tuple[object, int, asyncio.Semaphore]] = {}

    def effective_capacity(self, strategy: Any) -> int | None:
        advertised = strategy_max_concurrency(strategy)
        if advertised is None:
            return self._configured_capacity
        if self._configured_capacity is None:
            return advertised
        return min(advertised, self._configured_capacity)

    @staticmethod
    def _scope(strategy: Any) -> object:
        try:
            scope = strategy.concurrency_scope
        except Exception:
            scope = strategy
        return scope if scope is not None else strategy

    @asynccontextmanager
    async def admit(self, strategy: Any) -> AsyncIterator[Admission]:
        """Acquire capacity for one execute attempt and report queueing time."""
        capacity = self.effective_capacity(strategy)
        if capacity is None:
            yield Admission(wait_seconds=0.0, capacity=None)
            return

        scope = self._scope(strategy)
        scope_id = id(scope)
        entry = self._entries.get(scope_id)
        if entry is None or entry[0] is not scope:
            semaphore = asyncio.Semaphore(capacity)
            self._entries[scope_id] = (scope, capacity, semaphore)
        else:
            _, registered_capacity, semaphore = entry
            if registered_capacity != capacity:
                logger.warning(
                    "Concurrency capacity changed from %s to %s for %s; using the original limit.",
                    registered_capacity,
                    capacity,
                    type(strategy).__name__,
                )
                capacity = registered_capacity

        started = time.perf_counter()
        await semaphore.acquire()
        wait_seconds = time.perf_counter() - started
        if wait_seconds >= 0.001:
            logger.debug(
                "[ADMISSION] %s waited %.3fs for provider capacity=%s",
                type(strategy).__name__,
                wait_seconds,
                capacity,
            )
        try:
            yield Admission(wait_seconds=wait_seconds, capacity=capacity)
        finally:
            semaphore.release()


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
        "in ABL admission before timeout_per_item starts. Set max_workers <= "
        f"{capacity}, or rebuild the model with max_connections >= {max_workers}. "
        f"See {_CAPACITY_DOCS_URL}",
        UserWarning,
        stacklevel=stacklevel,
    )
