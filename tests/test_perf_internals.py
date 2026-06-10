"""Tests for the worker-loop micro-optimizations (perf items #2 and #3)."""

import pytest

from async_batch_llm import MetricsObserver
from async_batch_llm._internal.event_dispatcher import EventDispatcher
from async_batch_llm._internal.rate_limit_coordinator import RateLimitCoordinator
from async_batch_llm.observers.base import BaseObserver, ProcessingEvent
from async_batch_llm.strategies.rate_limit import ExponentialBackoffStrategy

# ── Item 2: observer fast-path ──────────────────────────────────────────────


def test_metrics_observer_marked_fast_user_observer_not():
    assert MetricsObserver()._abl_fast_observer is True
    # The default for any user-defined observer is False (keeps the wait_for
    # guard so a slow on_event can't block workers).
    assert BaseObserver()._abl_fast_observer is False


@pytest.mark.asyncio
async def test_emit_delivers_to_both_fast_and_slow_observers():
    received: list[tuple[str, ProcessingEvent]] = []

    class FastObserver(BaseObserver):
        _abl_fast_observer = True

        async def on_event(self, event, data):
            received.append(("fast", event))

    class SlowObserver(BaseObserver):
        # default _abl_fast_observer = False -> goes through wait_for
        async def on_event(self, event, data):
            received.append(("slow", event))

    dispatcher = EventDispatcher(observers=[FastObserver(), SlowObserver()], middlewares=[])
    await dispatcher.emit(ProcessingEvent.ITEM_COMPLETED, {"item_id": "x"})

    assert ("fast", ProcessingEvent.ITEM_COMPLETED) in received
    assert ("slow", ProcessingEvent.ITEM_COMPLETED) in received


@pytest.mark.asyncio
async def test_emit_noop_without_observers():
    dispatcher = EventDispatcher(observers=[], middlewares=[])
    # Must return without error and without needing a payload.
    await dispatcher.emit(ProcessingEvent.ITEM_STARTED, None)


# ── Item 3: slow-start unlocked fast-path ───────────────────────────────────


@pytest.mark.asyncio
async def test_apply_slow_start_returns_zero_when_inactive():
    coordinator = RateLimitCoordinator(
        rate_limit_strategy=ExponentialBackoffStrategy(),
        events=EventDispatcher(observers=[], middlewares=[]),
    )
    assert coordinator._slow_start_active is False
    # Fast-path: returns 0.0 without touching the lock or the strategy.
    assert await coordinator.apply_slow_start() == 0.0
    # Still inactive afterwards (no state mutated).
    assert coordinator._slow_start_active is False
