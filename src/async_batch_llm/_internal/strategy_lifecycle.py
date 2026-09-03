"""Strategy prepare/cleanup lifecycle shared by processors, hosts and gateways.

A single strategy instance may be shared across many work items (e.g. a
GeminiCachedModel that owns a context cache). This helper ensures:

- ``prepare()`` runs at most once per unique instance, even under
  concurrent workers (double-checked locking).
- Each prepared instance's ``cleanup()`` is a separate ordered cleanup step,
  checkpointed on success so a later close never repeats it, and retried by
  a later close if it failed or was interrupted.
- Once closing has started, new preparation is rejected.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from functools import partial
from typing import Any, Generic, Protocol, runtime_checkable

from ..base import TOutput
from ..llm_strategies import LLMCallStrategy
from .cleanup import CleanupStep, run_cleanup_steps

logger = logging.getLogger(__name__)


@runtime_checkable
class _HasCleanup(Protocol):
    async def cleanup(self) -> None: ...


def _has_cleanup(strategy: object) -> bool:
    return isinstance(strategy, _HasCleanup) or (
        hasattr(strategy, "cleanup") and callable(strategy.cleanup)
    )


class StrategyLifecycle(Generic[TOutput]):
    """Tracks prepared strategy instances and builds their cleanup steps.

    Holds weak references so that short-lived strategies (e.g. those
    created per request) don't keep Python objects alive longer than the
    caller intended, and so no per-strategy bookkeeping outlives them.
    """

    def __init__(self) -> None:
        self._prepared: weakref.WeakSet[LLMCallStrategy[Any]] = weakref.WeakSet()
        self._cleaned: weakref.WeakSet[LLMCallStrategy[Any]] = weakref.WeakSet()
        self._lock = asyncio.Lock()
        self._closing = False

    @property
    def closing(self) -> bool:
        return self._closing

    def mark_closing(self) -> None:
        """Reject new preparation from now on. One-way."""
        self._closing = True

    async def ensure_prepared(self, strategy: LLMCallStrategy[TOutput]) -> None:
        """Call ``strategy.prepare()`` if it hasn't been prepared yet.

        Safe under concurrent access via double-checked locking. The lock is
        never held across cleanup, so a slow ``cleanup()`` cannot block this.
        """
        if strategy in self._prepared:
            return
        if self._closing:
            raise RuntimeError("Strategy lifecycle is closing; new strategies cannot be prepared")

        async with self._lock:
            if strategy in self._prepared:
                return
            if self._closing:
                raise RuntimeError(
                    "Strategy lifecycle is closing; new strategies cannot be prepared"
                )
            strategy_id = id(strategy)
            logger.debug(f"Preparing strategy {strategy.__class__.__name__} (id={strategy_id})")
            await strategy.prepare()
            self._prepared.add(strategy)
            logger.debug(
                f"Strategy {strategy.__class__.__name__} prepared successfully (id={strategy_id})"
            )

    def cleanup_steps(self) -> list[CleanupStep]:
        """One ordered step per prepared strategy whose cleanup has not succeeded."""
        steps: list[CleanupStep] = []
        for strategy in list(self._prepared):
            if strategy in self._cleaned or not _has_cleanup(strategy):
                continue
            steps.append(
                CleanupStep(
                    name=f"strategy {strategy.__class__.__name__}",
                    run=partial(self._cleanup_one, strategy),
                    preserves_completed_result=True,
                )
            )
        return steps

    async def _cleanup_one(self, strategy: LLMCallStrategy[Any]) -> None:
        logger.debug(f"Cleaning up strategy {strategy.__class__.__name__} (id={id(strategy)})")
        await strategy.cleanup()
        self._cleaned.add(strategy)

    async def cleanup_all(self) -> None:
        """Close every prepared strategy not yet closed; raise the first ordinary failure.

        Siblings are always attempted. A failed or interrupted strategy stays
        retryable by a later call; a successful one is never repeated.
        """
        self._closing = True
        report = await run_cleanup_steps(
            self.cleanup_steps(), name="strategy lifecycle", logger=logger
        )
        report.raise_first()

    # Inspection helpers used by the processor and tests.

    @property
    def cleanup_complete(self) -> bool:
        """True once closing began and every prepared strategy has been closed."""
        return self._closing and all(
            strategy in self._cleaned or not _has_cleanup(strategy)
            for strategy in list(self._prepared)
        )

    def is_prepared(self, strategy: LLMCallStrategy[Any]) -> bool:
        return strategy in self._prepared
