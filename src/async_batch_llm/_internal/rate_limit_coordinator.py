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
        # Any-parameterized explicitly: with PEP 696 defaults, a bare
        # `EventDispatcher` would resolve to [str, Any, None] and reject
        # dispatchers from differently-parameterized processors.
        events: EventDispatcher[Any, Any, Any],
    ) -> None:
        self._strategy = rate_limit_strategy
        self._events = events

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
        """
        if observed_generation is None:
            observed_generation = self._cooldown_generation

        async with self._lock:
            current_generation = self._cooldown_generation
            generation_event = self._current_generation_event
            consecutive = 0  # set on coordinator path below
            if self._in_cooldown or observed_generation < current_generation:
                logger.debug(
                    f"Worker {worker_id} waiting for cooldown gen {current_generation} "
                    f"(obs={observed_generation})"
                )
                should_wait = True
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
                consecutive = self._consecutive_rate_limits
                should_wait = False

        if should_wait:
            await generation_event.wait()
            logger.debug(f"Worker {worker_id} resumed after cooldown gen {generation}")
            return

        # We're the coordinator — run the cooldown sleep.
        pause_started_at = time.time()
        cooldown_error: Exception | None = None

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

        # Respect a server-suggested wait (e.g. Retry-After) as a floor: the
        # backoff strategy may ask for longer, but we never undershoot the
        # server's request. Only applied when the strategy itself didn't error.
        if cooldown_error is None and suggested_wait is not None and suggested_wait > cooldown:
            logger.info(
                "[RATE-LIMIT]Raising cooldown from %.1fs to server-suggested %.1fs.",
                cooldown,
                suggested_wait,
            )
            cooldown = suggested_wait

        await self._events.emit(
            ProcessingEvent.COOLDOWN_STARTED,
            {
                "worker_id": worker_id,
                "duration": cooldown,
                "consecutive": consecutive,
            },
        )

        if cooldown_error is None and cooldown > 0:
            logger.warning(
                "[RATE-LIMIT]Rate limit detected by worker %s (gen %d). "
                "Pausing all workers for %.1fs...",
                worker_id,
                generation,
                cooldown,
            )
        else:
            logger.warning(
                "[RATE-LIMIT]Rate limit detected by worker %s (gen %d). "
                "Skipping cooldown due to prior error.",
                worker_id,
                generation,
            )

        try:
            if cooldown > 0:
                await asyncio.sleep(cooldown)
        except asyncio.CancelledError:
            # Shield so workers still resume even when cancelled.
            await asyncio.shield(self._finalize_cooldown(pause_started_at, None))
            raise
        except Exception as exc:
            logger.warning(
                "[WARN]Cooldown sleep interrupted for worker %s: %s. Resuming immediately.",
                worker_id,
                exc,
            )
            cooldown_error = cooldown_error or exc
            await self._finalize_cooldown(pause_started_at, cooldown_error)
            return

        await self._finalize_cooldown(pause_started_at, cooldown_error)

    async def _finalize_cooldown(self, start_time: float, error: Exception | None) -> None:
        """Resume workers and emit COOLDOWN_ENDED."""
        actual_duration = max(0.0, time.time() - start_time)

        async with self._lock:
            self._items_since_resume = 0
            self._in_cooldown = False
            self._cooldown_complete_generation = self._cooldown_generation
            self._rate_limit_event.set()
            self._current_generation_event.set()

        payload: dict[str, float | str] = {"duration": actual_duration}
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
