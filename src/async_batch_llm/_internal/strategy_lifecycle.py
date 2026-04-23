"""Strategy prepare/cleanup lifecycle for ParallelBatchProcessor.

A single strategy instance may be shared across many work items (e.g. a
GeminiCachedModel that owns a context cache). This helper ensures:

- ``prepare()`` runs at most once per unique instance, even under
  concurrent workers (double-checked locking).
- ``cleanup()`` runs at most once across all prepared instances, and a
  failure in one instance's cleanup does not skip the others.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any, Generic, Protocol, runtime_checkable

from ..base import TOutput
from ..llm_strategies import LLMCallStrategy

logger = logging.getLogger(__name__)


@runtime_checkable
class _HasCleanup(Protocol):
    async def cleanup(self) -> None: ...


class StrategyLifecycle(Generic[TOutput]):
    """Tracks prepared strategy instances and coordinates cleanup.

    Holds weak references so that short-lived strategies (e.g. those
    created per request) don't keep Python objects alive longer than the
    caller intended.
    """

    def __init__(self) -> None:
        self._prepared: weakref.WeakSet[LLMCallStrategy[Any]] = weakref.WeakSet()
        self._lock = asyncio.Lock()
        self._cleaned_up = False

    async def ensure_prepared(self, strategy: LLMCallStrategy[TOutput]) -> None:
        """Call ``strategy.prepare()`` if it hasn't been prepared yet.

        Safe under concurrent access via double-checked locking.
        """
        if strategy in self._prepared:
            return

        async with self._lock:
            if strategy in self._prepared:
                return

            strategy_id = id(strategy)
            logger.debug(
                f"Preparing strategy {strategy.__class__.__name__} (id={strategy_id})"
            )
            await strategy.prepare()
            self._prepared.add(strategy)
            # A new prepare resets the cleanup flag so explicit lower-level
            # lifecycle reuse re-enters the cleanup path.
            self._cleaned_up = False
            logger.debug(
                f"Strategy {strategy.__class__.__name__} prepared successfully "
                f"(id={strategy_id})"
            )

    async def cleanup_all(self) -> None:
        """Cleanup every prepared strategy, exactly once.

        Iterates over a snapshot so that strategies removing themselves
        during cleanup don't cause skipped entries. A failure in one
        strategy's cleanup is logged as a warning and does not skip the
        others — the batch's results should not be invalidated by a
        flaky teardown.
        """
        if self._cleaned_up:
            return

        for strategy in list(self._prepared):
            if isinstance(strategy, _HasCleanup) or (
                hasattr(strategy, "cleanup") and callable(strategy.cleanup)
            ):
                try:
                    logger.debug(
                        f"Cleaning up strategy {strategy.__class__.__name__} "
                        f"(id={id(strategy)})"
                    )
                    await strategy.cleanup()
                except Exception as e:
                    # Keep `except Exception` (not BaseException) so
                    # KeyboardInterrupt / SystemExit still propagate.
                    logger.warning(
                        f"[WARN]Strategy cleanup failed for "
                        f"{strategy.__class__.__name__}: {e}",
                        exc_info=True,
                    )

        self._cleaned_up = True

    # Inspection helpers used by the processor and tests.

    def is_prepared(self, strategy: LLMCallStrategy[Any]) -> bool:
        return strategy in self._prepared

    def reset_cleanup_flag(self) -> None:
        """Re-arm cleanup for tests or custom owners that reuse this lifecycle helper."""
        self._cleaned_up = False
