"""Event and middleware dispatch for ParallelBatchProcessor.

Pulls the observer-emit and middleware-chain logic out of the processor
god class. Behavior is preserved 1:1 with the previous inline versions —
including the exact log-message prefixes, which several tests grep for.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Generic

from ..base import LLMWorkItem, TContext, TInput, TOutput, WorkItemResult
from ..observers import ProcessingEvent, ProcessorObserver

if TYPE_CHECKING:
    from ..middleware import Middleware

logger = logging.getLogger(__name__)

# Observer events should complete quickly; slow observers shouldn't block workers.
OBSERVER_CALLBACK_TIMEOUT = 5.0


class EventDispatcher(Generic[TInput, TOutput, TContext]):
    """Dispatches observer events and runs the middleware chain.

    Stateless with respect to the batch; holds only references to the
    registered observers/middlewares. Safe to share between worker tasks.
    """

    def __init__(
        self,
        observers: list[ProcessorObserver],
        middlewares: list[Middleware[TInput, TOutput, TContext]],
    ):
        self.observers = observers
        self.middlewares = middlewares

    # ── Observer events ──────────────────────────────────────────

    async def emit(self, event: ProcessingEvent, data: dict | None = None) -> None:
        """Notify every observer of `event`. Individual observer failures
        are logged but don't abort the batch."""
        if not self.observers:
            return

        event_data = data or {}
        for observer in self.observers:
            try:
                await asyncio.wait_for(
                    observer.on_event(event, event_data),
                    timeout=OBSERVER_CALLBACK_TIMEOUT,
                )
            except asyncio.CancelledError:
                raise
            except (TimeoutError, asyncio.TimeoutError):  # distinct classes on Python 3.10
                logger.warning(
                    f"[WARN]Observer callback timed out after {OBSERVER_CALLBACK_TIMEOUT}s "
                    f"for event {event.name}"
                )
            except Exception as e:
                logger.warning(f"[WARN]Observer error: {e}")

    # ── Middleware chain ──────────────────────────────────────────

    async def run_before(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMWorkItem[TInput, TOutput, TContext] | None:
        """Run `before_process` on each middleware in order. A middleware
        returning `None` skips the item entirely."""
        current_item = work_item
        for middleware in self.middlewares:
            try:
                result = await middleware.before_process(current_item)
                if result is None:
                    return None
                current_item = result
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"[WARN]Middleware before_process error for {work_item.item_id}: {e}"
                )
        return current_item

    async def run_after(
        self, result: WorkItemResult[TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        """Run `after_process` in reverse order (onion-style wrapping)."""
        current_result = result
        for middleware in reversed(self.middlewares):
            try:
                current_result = await middleware.after_process(current_result)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[WARN]Middleware after_process error for {result.item_id}: {e}")
        return current_result

    async def run_on_error(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], error: Exception
    ) -> WorkItemResult[TOutput, TContext] | None:
        """Give each middleware a chance to recover from `error`. Returns the
        first non-None result (middleware handled it), or None if none did."""
        for middleware in self.middlewares:
            try:
                result = await middleware.on_error(work_item, error)
                if result is not None:
                    return result
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[WARN]Middleware on_error error for {work_item.item_id}: {e}")
        return None
