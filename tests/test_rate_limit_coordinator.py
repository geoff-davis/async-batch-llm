"""Unit tests for RateLimitCoordinator cooldown duration logic.

Exercises the coordinator's state machine in isolation (no processor), in
particular the ``suggested_wait`` floor added in v0.10.0: a server-suggested
wait (e.g. a parsed ``Retry-After``) raises the cooldown but never lowers it.
"""

from __future__ import annotations

import asyncio

import pytest

from async_batch_llm._internal.event_dispatcher import EventDispatcher
from async_batch_llm._internal.rate_limit_coordinator import RateLimitCoordinator
from async_batch_llm.strategies.rate_limit import FixedDelayStrategy


def _make_coordinator(cooldown: float) -> RateLimitCoordinator:
    events: EventDispatcher = EventDispatcher(observers=[], middlewares=[])
    return RateLimitCoordinator(FixedDelayStrategy(cooldown=cooldown), events)


async def _captured_cooldown(coord: RateLimitCoordinator, monkeypatch, **kwargs) -> float:
    """Run one cooldown cycle and return the duration passed to asyncio.sleep."""
    slept: list[float] = []

    real_sleep = asyncio.sleep

    async def fake_sleep(delay, *a, **kw):
        slept.append(delay)
        await real_sleep(0)  # yield without actually waiting

    monkeypatch.setattr(
        "async_batch_llm._internal.rate_limit_coordinator.asyncio.sleep", fake_sleep
    )
    await coord.handle_rate_limit(worker_id=0, observed_generation=0, **kwargs)
    return slept[0] if slept else 0.0


@pytest.mark.asyncio
async def test_suggested_wait_raises_cooldown(monkeypatch):
    coord = _make_coordinator(cooldown=5.0)
    duration = await _captured_cooldown(coord, monkeypatch, suggested_wait=30.0)
    assert duration == 30.0


@pytest.mark.asyncio
async def test_suggested_wait_below_cooldown_is_ignored(monkeypatch):
    coord = _make_coordinator(cooldown=20.0)
    duration = await _captured_cooldown(coord, monkeypatch, suggested_wait=5.0)
    assert duration == 20.0


@pytest.mark.asyncio
async def test_no_suggested_wait_uses_strategy_cooldown(monkeypatch):
    coord = _make_coordinator(cooldown=8.0)
    duration = await _captured_cooldown(coord, monkeypatch)
    assert duration == 8.0


# ── Issue #88: caller cancellation must not finish the shared cooldown ──


@pytest.mark.asyncio
async def test_cancelled_reporter_does_not_finish_shared_cooldown():
    """A gateway submit timeout cancels the caller that reported the 429;
    the shared cooldown must survive and other waiters stay paused."""
    coord = _make_coordinator(cooldown=0.5)

    reporter = asyncio.create_task(coord.handle_rate_limit(worker_id=0, observed_generation=0))
    await asyncio.sleep(0.05)  # cooldown underway
    assert coord._in_cooldown

    waiter = asyncio.create_task(coord.handle_rate_limit(worker_id=1, observed_generation=0))
    await asyncio.sleep(0.02)

    # Simulate the submit timeout: cancel only the reporting caller.
    reporter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reporter

    # The shared pause survives the caller's cancellation.
    await asyncio.sleep(0.1)
    assert coord._in_cooldown
    assert not coord._rate_limit_event.is_set()
    assert not waiter.done()

    # The waiter resumes only once the real cooldown expires.
    await asyncio.wait_for(waiter, timeout=1.0)
    assert not coord._in_cooldown
    assert coord._rate_limit_event.is_set()
    await coord.shutdown()


@pytest.mark.asyncio
async def test_two_waiters_survive_reporter_cancellation():
    """Regression per the issue: two waiters + a cancelled reporter."""
    coord = _make_coordinator(cooldown=0.4)
    started = asyncio.get_running_loop().time()

    reporter = asyncio.create_task(coord.handle_rate_limit(worker_id=0, observed_generation=0))
    await asyncio.sleep(0.02)
    waiters = [
        asyncio.create_task(coord.handle_rate_limit(worker_id=i, observed_generation=0))
        for i in (1, 2)
    ]
    await asyncio.sleep(0.02)
    reporter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reporter

    await asyncio.wait_for(asyncio.gather(*waiters), timeout=1.0)
    elapsed = asyncio.get_running_loop().time() - started
    # Both waiters waited out the REAL cooldown, not the 0.04s to cancellation.
    assert elapsed >= 0.35
    await coord.shutdown()


@pytest.mark.asyncio
async def test_shutdown_stops_cooldown_and_wakes_waiters():
    """Host teardown stops the owned task without leaking it; waiters wake."""
    coord = _make_coordinator(cooldown=30.0)

    reporter = asyncio.create_task(coord.handle_rate_limit(worker_id=0, observed_generation=0))
    await asyncio.sleep(0.05)
    task = next(iter(coord._owned_cooldowns), None)
    assert task is not None and not task.done()

    await asyncio.wait_for(coord.shutdown(), timeout=1.0)
    assert task.done()
    assert next(iter(coord._owned_cooldowns), None) is None
    # Waiters (including the reporter) are woken by teardown finalization.
    await asyncio.wait_for(reporter, timeout=1.0)
    assert coord._rate_limit_event.is_set()
    # Idempotent.
    await coord.shutdown()


@pytest.mark.asyncio
async def test_shutdown_without_cooldown_is_noop():
    coord = _make_coordinator(cooldown=1.0)
    await coord.shutdown()
    assert next(iter(coord._owned_cooldowns), None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suggestion,expected", [(86400, 0.1), (float("inf"), 0.01), (float("nan"), 0.01)]
)
async def test_prov5_coordinator_caps_untrusted_suggestion(monkeypatch, suggestion, expected):
    events = EventDispatcher(observers=[], middlewares=[])
    coord = RateLimitCoordinator(
        FixedDelayStrategy(cooldown=0.01), events, max_cooldown_seconds=0.1
    )
    assert await _captured_cooldown(coord, monkeypatch, suggested_wait=suggestion) == expected
    await coord.shutdown()


@pytest.mark.asyncio
async def test_adm5_failed_strategy_uses_fallback_and_suggested_wait(monkeypatch):
    class Broken(FixedDelayStrategy):
        async def on_rate_limit(self, worker_id, consecutive_limit_count):
            raise RuntimeError("broken policy")

    coord = RateLimitCoordinator(Broken(), EventDispatcher([], []))
    coord._fallback_cooldown_seconds = 5
    assert await _captured_cooldown(coord, monkeypatch, suggested_wait=8) == 8
    await coord.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -float("inf"), -5.0, 0.0, 0.25])
@pytest.mark.parametrize("suggested_wait", [None, 0.0, 8.0])
async def test_sd1_invalid_cooldown_return_uses_fallback(
    duration, suggested_wait, monkeypatch, caplog
):
    import math

    coord = RateLimitCoordinator(
        FixedDelayStrategy(cooldown=duration),
        EventDispatcher([], []),
        fallback_cooldown_seconds=5,
    )
    invalid = not math.isfinite(duration) or duration < 0
    expected = max(5 if invalid else duration, suggested_wait or 0)
    try:
        actual = await _captured_cooldown(coord, monkeypatch, suggested_wait=suggested_wait)
        assert actual == expected
        assert ("Using the configured fallback cooldown" in caplog.text) is invalid
        assert not coord.is_paused
    finally:
        await coord.shutdown()
    assert not coord._owned_cooldowns
