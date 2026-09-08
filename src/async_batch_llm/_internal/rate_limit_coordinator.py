"""Rate-limit coordination across concurrent workers.

When multiple workers hit an LLM rate limit at the same time, one of them
must take ownership of the cooldown and the others must pause. This helper
owns that coordination via:

- An ``asyncio.Event`` that gates all workers ("paused" when cleared).
- A generation counter that prevents a late-reporting worker from
  starting a redundant cooldown cycle.
- Slow-start ramp state (how many items since resume, consecutive
  cooldowns) that drives the :class:`RateLimitStrategy` backoff.

Extracted from ``parallel.py`` in v0.7.0 to make the state machine
testable in isolation and keep the processor focused on orchestration.
Behavior is preserved 1:1, including log message prefixes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..observers import ProcessingEvent
from ..strategies import RateLimitStrategy
from .cleanup import owned_task_failure, wait_detached
from .event_dispatcher import EventDispatcher

logger = logging.getLogger(__name__)

# Truncation for error strings included in COOLDOWN_ENDED payloads.
# Kept in sync with parallel.ERROR_MESSAGE_MAX_LENGTH.
_ERROR_MESSAGE_MAX_LENGTH = 200


class RateLimitCoordinator:
    """Owns rate-limit pause/resume state for a :class:`ParallelBatchProcessor`."""

    def __init__(
        self,
        rate_limit_strategy: RateLimitStrategy,
        # Parameterized explicitly: with PEP 696 defaults, a bare
        # `EventDispatcher` annotation would resolve to [str, Any, None] and
        # reject the processor's concrete dispatcher.
        events: EventDispatcher[Any, Any, Any],
        quota_scope_id: int | None = None,
    ) -> None:
        self._strategy = rate_limit_strategy
        self._events = events
        self._quota_scope_id = quota_scope_id

        self._rate_limit_event = asyncio.Event()
        self._rate_limit_event.set()  # Start un-paused.
        self._in_cooldown = False
        # Increments on every new cooldown cycle; used by workers to see
        # whether the cooldown they observed has already been handled.
        self._cooldown_generation = 0
        self._cooldown_complete_generation = 0
        # Per-generation event so late workers can wait for the exact cycle.
        self._current_generation_event: asyncio.Event = asyncio.Event()
        self._current_generation_event.set()

        # Slow-start ramp state.
        self._items_since_resume = 0
        self._slow_start_active = False
        self._consecutive_rate_limits = 0

        # The cooldown sleep runs in a coordinator-OWNED task (issue #88):
        # cancelling the caller that reported the rate limit (gateway
        # submit_timeout, item deadline) must cancel only that caller, never
        # finish the shared pause early. shutdown() cancels this task.
        self._cooldown_task: asyncio.Task[None] | None = None
        # Owned-cooldown bookkeeping for shutdown(): the shielded finalization
        # started by the task's cancellation handler, the context needed to
        # finalize from shutdown() itself when that handler never ran, and
        # whether shutdown() already sent its one cancellation.
        self._cooldown_finalization: asyncio.Future[None] | None = None
        self._cooldown_context: tuple[float, str | None] | None = None
        self._cooldown_cancel_sent = False
        self._cooldown_failure_surfaced = False

        self._lock = asyncio.Lock()

    # ── Worker-side hooks ────────────────────────────────────────

    async def wait_if_paused(self) -> None:
        """Block until the processor is not in cooldown."""
        await self._rate_limit_event.wait()

    async def apply_slow_start(self) -> float:
        """Return the slow-start delay to apply before the next item, or 0.

        Also advances the slow-start counter and ends the slow-start window
        when the ramp finishes.
        """
        # Unlocked fast-path: slow-start is inactive the vast majority of the
        # time (only between a rate limit and the end of its ramp), so this
        # check spares every worker an asyncio.Lock acquire/release per item.
        # The race is benign: `_slow_start_active` is only ever set True under
        # the lock by handle_rate_limit(); a worker that reads a stale False
        # the instant it flips just skips the delay for one item (it'll see it
        # on the next), and a stale True falls through to the locked check
        # below which re-reads the flag authoritatively.
        if not self._slow_start_active:
            return 0.0

        async with self._lock:
            if not self._slow_start_active:
                return 0.0
            should_delay, delay = self._strategy.should_apply_slow_start(self._items_since_resume)
            if should_delay:
                self._items_since_resume += 1
                return float(delay)
            # Ramp finished — reset counters until the next rate limit.
            self._slow_start_active = False
            self._items_since_resume = 0
            return 0.0

    @property
    def current_generation(self) -> int:
        """The current cooldown generation counter. Snapshot for workers
        to pass back into :meth:`handle_rate_limit`."""
        return self._cooldown_generation

    async def on_item_success(self) -> None:
        """Reset the consecutive-rate-limit counter after a successful call."""
        async with self._lock:
            self._consecutive_rate_limits = 0

    # ── Cooldown coordination ────────────────────────────────────

    async def handle_rate_limit(
        self,
        worker_id: int,
        observed_generation: int | None = None,
        suggested_wait: float | None = None,
        strategy_type: str | None = None,
    ) -> None:
        """Coordinate a cooldown among workers.

        Exactly one worker becomes the coordinator for each cycle; the rest
        wait on the current generation's event.

        Args:
            worker_id: The worker reporting the rate limit.
            observed_generation: The cooldown generation this worker observed
                before reporting, for the atomic check-and-set.
            suggested_wait: A server-suggested minimum wait (e.g. parsed from a
                ``Retry-After`` header by the error classifier). When provided,
                it acts as a *floor* on the strategy-computed cooldown — the
                backoff strategy can wait longer, but never shorter than the
                server asked. Only the coordinating worker's value is applied.
            strategy_type: Safe class name for scoped diagnostics. Arbitrary
                quota-scope values are never formatted or logged.
        """
        if observed_generation is None:
            observed_generation = self._cooldown_generation

        async with self._lock:
            current_generation = self._cooldown_generation
            generation_event = self._current_generation_event
            if self._in_cooldown or observed_generation < current_generation:
                logger.debug(
                    f"Worker {worker_id} waiting for cooldown gen {current_generation} "
                    f"(obs={observed_generation})"
                )
                generation = current_generation
            else:
                self._in_cooldown = True
                self._cooldown_generation += 1
                generation = self._cooldown_generation
                self._slow_start_active = True
                self._consecutive_rate_limits += 1
                self._rate_limit_event.clear()
                self._current_generation_event = asyncio.Event()
                generation_event = self._current_generation_event
                # The cooldown runs in a coordinator-owned task, NOT in this
                # caller's task (issue #88): a gateway submit timeout or item
                # deadline cancelling the reporting caller must cancel only
                # that caller — a caller cancellation used to success-finalize
                # the shared cooldown and wake every waiter early. Only
                # shutdown() cancels the owned task (and then waking waiters
                # is the intended teardown behavior).
                self._cooldown_context = (time.time(), strategy_type)
                self._cooldown_finalization = None
                self._cooldown_cancel_sent = False
                self._cooldown_failure_surfaced = False
                self._cooldown_task = asyncio.create_task(
                    self._run_cooldown(
                        worker_id,
                        generation,
                        self._consecutive_rate_limits,
                        suggested_wait,
                        strategy_type,
                    )
                )

        # Coordinator and waiters alike wait for the generation to complete;
        # each caller's cancellation affects only itself.
        await generation_event.wait()
        logger.debug(f"Worker {worker_id} resumed after cooldown gen {generation}")

    async def _run_cooldown(
        self,
        worker_id: int,
        generation: int,
        consecutive: int,
        suggested_wait: float | None,
        strategy_type: str | None,
    ) -> None:
        """Owned cooldown cycle: compute the wait, sleep, finalize."""
        pause_started_at = time.time()
        cooldown_error: Exception | None = None

        try:
            try:
                cooldown = await self._strategy.on_rate_limit(worker_id, consecutive)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                cooldown_error = exc
                cooldown = 0.0
                logger.warning(
                    "[WARN]Rate limit strategy failed to determine cooldown: %s. "
                    "Resuming workers immediately.",
                    exc,
                )

            # Respect a server-suggested wait (e.g. Retry-After) as a floor:
            # the backoff strategy may ask for longer, but we never undershoot
            # the server's request. Only applied when the strategy itself
            # didn't error.
            if cooldown_error is None and suggested_wait is not None and suggested_wait > cooldown:
                logger.info(
                    "[RATE-LIMIT]Raising cooldown from %.1fs to server-suggested %.1fs.",
                    cooldown,
                    suggested_wait,
                )
                cooldown = suggested_wait

            payload: dict[str, Any] = {
                "worker_id": worker_id,
                "duration": cooldown,
                "consecutive": consecutive,
            }
            if self._quota_scope_id is not None:
                payload["quota_scope_id"] = self._quota_scope_id
            if strategy_type is not None:
                payload["strategy_type"] = strategy_type
            await self._events.emit(ProcessingEvent.COOLDOWN_STARTED, payload)

            scope_log = f" scope={self._quota_scope_id}" if self._quota_scope_id is not None else ""
            strategy_log = f" strategy={strategy_type}" if strategy_type is not None else ""

            if cooldown_error is not None:
                logger.warning(
                    "[RATE-LIMIT]Rate limit detected by worker %s (gen %d%s%s). "
                    "Skipping cooldown due to prior error.",
                    worker_id,
                    generation,
                    scope_log,
                    strategy_log,
                )
            elif cooldown > 0:
                logger.warning(
                    "[RATE-LIMIT]Rate limit detected by worker %s (gen %d%s%s). "
                    "Pausing all workers for %.1fs...",
                    worker_id,
                    generation,
                    scope_log,
                    strategy_log,
                    cooldown,
                )
            else:
                # A strategy can legitimately return 0.0 (no cooldown wanted);
                # don't mislabel that as an error.
                logger.warning(
                    "[RATE-LIMIT]Rate limit detected by worker %s (gen %d%s%s). "
                    "Strategy requested no cooldown; resuming immediately.",
                    worker_id,
                    generation,
                    scope_log,
                    strategy_log,
                )

            try:
                if cooldown > 0:
                    await asyncio.sleep(cooldown)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[WARN]Cooldown sleep interrupted for worker %s: %s. Resuming immediately.",
                    worker_id,
                    exc,
                )
                cooldown_error = cooldown_error or exc

            await self._finalize_cooldown(pause_started_at, cooldown_error, strategy_type)
        except asyncio.CancelledError:
            if not self._cooldown_cancel_sent:
                # Not shutdown's cancellation: an observer raised
                # CancelledError while an event was being delivered, or a
                # third party cancelled this task. Nothing may finalize the
                # generation on shutdown's behalf here (after the state
                # transition that would deliver COOLDOWN_ENDED a second
                # time); shutdown() classifies this as an interruption and
                # the explicit close after it finalizes.
                raise
            # shutdown() cancelled this task: wake the waiters so host
            # teardown never hangs on the pause, then propagate. The
            # finalization is shielded and its handle is kept so shutdown()
            # can wait for *it*, not merely for this wrapper task.
            finalization = asyncio.ensure_future(
                self._finalize_cooldown(pause_started_at, None, strategy_type)
            )
            self._cooldown_finalization = finalization
            await asyncio.shield(finalization)
            raise

    async def shutdown(self) -> None:
        """Cancel an in-flight cooldown and finalize its generation.

        Safe to call repeatedly, and a cancelled or failed call is retryable:
        the owned task handle is kept until the generation is demonstrably
        finalized (``_in_cooldown`` is false) and the task is cancelled at most
        once. A failure of the owned task or of its finalization — including a
        cancellation this coordinator did not send, reported as
        :class:`CleanupInterruptedError` — is raised by the first call that
        observes it, without a second finalization attempt in that call; the
        next explicit call finalizes the generation from here. The generation
        is also finalized from here when the task's own cancellation handler
        never ran (cancelled before its first instruction). Nothing about a
        reported failure is retained once it has been raised.
        """
        task = self._cooldown_task
        if task is None:
            return
        if not task.done() and not self._cooldown_cancel_sent:
            self._cooldown_cancel_sent = True
            task.cancel()
        # Detached: the caller's own cancellation propagates instead of being
        # mistaken for the owned task's, and never cancels the owned work.
        await wait_detached(task)
        finalization = self._cooldown_finalization
        if finalization is not None:
            await wait_detached(finalization)

        if not self._cooldown_failure_surfaced:
            failure = owned_task_failure(
                task, name="cooldown", cancel_sent=self._cooldown_cancel_sent
            )
            if failure is None and finalization is not None:
                # The finalization is never cancelled by this coordinator.
                failure = owned_task_failure(
                    finalization, name="cooldown finalization", cancel_sent=False
                )
            if failure is not None:
                # Report once, without retrying in this call (clause 4). The
                # settled handles stay so the next call resumes here; only
                # the flag is kept, never the exception.
                self._cooldown_failure_surfaced = True
                self._cooldown_finalization = None
                raise failure

        if self._in_cooldown:
            # Nothing owned is still working on the generation: the task's
            # handler never ran (cancelled before its first instruction), or
            # a failure was reported by an earlier call and this later call
            # retries. A failure here keeps the handle and paused state.
            started_at, strategy_type = self._cooldown_context or (time.time(), None)
            await self._finalize_cooldown(started_at, None, strategy_type)

        # Generation demonstrably finalized: checkpoint.
        if self._cooldown_task is task:
            self._cooldown_task = None
            self._cooldown_finalization = None
            self._cooldown_context = None
            self._cooldown_failure_surfaced = False

    async def _finalize_cooldown(
        self,
        start_time: float,
        error: Exception | None,
        strategy_type: str | None = None,
    ) -> None:
        """Resume workers and emit COOLDOWN_ENDED."""
        actual_duration = max(0.0, time.time() - start_time)

        async with self._lock:
            self._items_since_resume = 0
            self._in_cooldown = False
            self._cooldown_complete_generation = self._cooldown_generation
            self._rate_limit_event.set()
            self._current_generation_event.set()

        payload: dict[str, float | str | int] = {"duration": actual_duration}
        if self._quota_scope_id is not None:
            payload["quota_scope_id"] = self._quota_scope_id
        if strategy_type is not None:
            payload["strategy_type"] = strategy_type
        if error is not None:
            payload["error"] = str(error)[:_ERROR_MESSAGE_MAX_LENGTH]

        await self._events.emit(ProcessingEvent.COOLDOWN_ENDED, payload)

        if error is not None:
            logger.warning(
                "[WARN]Cooldown ended early due to error: %s. Workers resumed immediately.",
                error,
            )
        else:
            logger.info("[OK]Cooldown complete. Resuming with slow-start...")
