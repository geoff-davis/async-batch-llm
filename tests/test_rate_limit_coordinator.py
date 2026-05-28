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
