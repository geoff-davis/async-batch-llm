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
from ..strategies import MiddlewareContractError
from .logical_item import PreparedLogicalItem

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
        """Notify every observer of `event`, in registration order.

        Individual observer failures are logged but don't abort the batch.
        Each observer receives its own shallow copy of the event data, so
        one observer mutating the dict can't corrupt what the next sees.

        Trusted built-in observers (``_abl_fast_observer``) are awaited
        directly; only third-party observers pay the per-event
        ``asyncio.wait_for`` task/timer overhead that guards against a
        slow/hanging ``on_event``. Callers should avoid building the ``data``
        payload at all when there are no observers (check ``self.observers``).
        """
        if not self.observers:
            return

        event_data = data or {}
        for observer in self.observers:
            try:
                if observer._abl_fast_observer:
                    await observer.on_event(event, dict(event_data))
                else:
                    await asyncio.wait_for(
                        observer.on_event(event, dict(event_data)),
                        timeout=OBSERVER_CALLBACK_TIMEOUT,
                    )
            except asyncio.CancelledError:
                raise
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning(
                    f"[WARN]Observer callback timed out after {OBSERVER_CALLBACK_TIMEOUT}s "
                    f"for event {event.name}"
                )
            except Exception as e:
                logger.warning(f"[WARN]Observer error: {e}")

    # ── Middleware chain ──────────────────────────────────────────

    async def run_before(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        *,
        prepared: PreparedLogicalItem[TInput, TOutput, TContext] | None = None,
    ) -> LLMWorkItem[TInput, TOutput, TContext] | None:
        """Run `before_process` on each middleware in order. A middleware
        returning `None` skips the item entirely."""
        current_item = work_item
        accepted_id = work_item.item_id
        submission_index = work_item.submission_index
        for middleware in self.middlewares:
            try:
                result = await middleware.before_process(current_item)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[WARN]Middleware before_process error for {accepted_id}: {e}")
                result = current_item
            # Validation is outside fail-open callback handling: an invalid
            # returned request cannot change accepted identity or reach a provider.
            candidate = current_item if result is None else result
            if not isinstance(candidate, LLMWorkItem):
                raise MiddlewareContractError("before_process must return LLMWorkItem or None")
            if candidate.item_id != accepted_id:
                candidate.item_id = accepted_id
                candidate.submission_index = submission_index
                raise MiddlewareContractError("before_process cannot change the accepted item_id")
            try:
                candidate.__post_init__()
            except (TypeError, ValueError) as exc:
                raise MiddlewareContractError(f"Invalid before_process work item: {exc}") from exc
            candidate.submission_index = submission_index
            candidate._artifact_key = None
            current_item = candidate
            if prepared is not None:
                prepared.effective_item = current_item
            if result is None:
                return None
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
