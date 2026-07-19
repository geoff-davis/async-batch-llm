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
async def test_shutdown_cancels_cooldown_and_wakes_waiters():
    """Host teardown cancels the owned task without leaking it; waiters wake."""
    coord = _make_coordinator(cooldown=30.0)

    reporter = asyncio.create_task(coord.handle_rate_limit(worker_id=0, observed_generation=0))
    await asyncio.sleep(0.05)
    task = coord._cooldown_task
    assert task is not None and not task.done()

    await asyncio.wait_for(coord.shutdown(), timeout=1.0)
    assert task.done()
    assert coord._cooldown_task is None
    # Waiters (including the reporter) are woken by teardown finalization.
    await asyncio.wait_for(reporter, timeout=1.0)
    assert coord._rate_limit_event.is_set()
    # Idempotent.
    await coord.shutdown()


@pytest.mark.asyncio
async def test_shutdown_without_cooldown_is_noop():
    coord = _make_coordinator(cooldown=1.0)
    await coord.shutdown()
    assert coord._cooldown_task is None
