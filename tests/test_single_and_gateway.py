"""Tests for the queue-less single-call helper and gateway."""

import asyncio

import pytest
from pydantic import BaseModel

from async_batch_llm import (
    LLMCallError,
    LLMGateway,
    ProcessorConfig,
    PydanticAIStrategy,
    call,
    try_call,
)
from async_batch_llm.core import RateLimitConfig, RetryConfig
from async_batch_llm.testing import MockAgent


class Out(BaseModel):
    text: str


def _agent(**kwargs):
    return MockAgent(response_factory=lambda p: Out(text=f"ok:{p}"), latency=0.01, **kwargs)


def _strategy(**kwargs):
    return PydanticAIStrategy(agent=_agent(**kwargs))


# ── single call ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_success_returns_output():
    out = await call(_strategy(), "hello")
    assert isinstance(out, Out)
    assert out.text == "ok:hello"


@pytest.mark.asyncio
async def test_try_call_reports_tokens():
    result = await try_call(_strategy(), "hello")
    assert result.success
    assert result.token_usage["total_tokens"] == 30


@pytest.mark.asyncio
async def test_call_retries_through_rate_limit():
    # Rate-limited on the first call, succeeds on retry — exercises the
    # coordinator's cooldown/slow-start path with no worker pool.
    cfg = ProcessorConfig(
        max_workers=1,
        rate_limit=RateLimitConfig(cooldown_seconds=0.05),
        retry=RetryConfig(max_attempts=3, max_rate_limit_retries=5),
    )
    out = await call(_strategy(rate_limit_on_call=1), "hi", config=cfg)
    assert out.text == "ok:hi"


@pytest.mark.asyncio
async def test_call_failure_raises_and_try_call_does_not():
    cfg = ProcessorConfig(max_workers=1, retry=RetryConfig(max_attempts=2))
    strat = _strategy(failure_rate=1.0)
    with pytest.raises((LLMCallError, Exception)):
        await call(strat, "boom", config=cfg)

    result = await try_call(_strategy(failure_rate=1.0), "boom", config=cfg)
    assert not result.success
    assert result.error


# ── gateway ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gateway_concurrent_submits():
    async with LLMGateway(_strategy(), config=ProcessorConfig(max_workers=4)) as gw:
        outs = await asyncio.gather(*(gw.submit(f"p{i}") for i in range(12)))
    assert [o.text for o in outs] == [f"ok:p{i}" for i in range(12)]


@pytest.mark.asyncio
async def test_gateway_shared_cooldown_recovers():
    # A 429 on one request pauses via the shared coordinator; all still succeed.
    cfg = ProcessorConfig(
        max_workers=4,
        rate_limit=RateLimitConfig(cooldown_seconds=0.05),
        retry=RetryConfig(max_attempts=3, max_rate_limit_retries=5),
    )
    async with LLMGateway(_strategy(rate_limit_on_call=2), config=cfg) as gw:
        outs = await asyncio.gather(*(gw.submit(f"p{i}") for i in range(6)))
    assert all(o.text.startswith("ok:") for o in outs)


@pytest.mark.asyncio
async def test_gateway_try_submit_reports_failure():
    cfg = ProcessorConfig(max_workers=2, retry=RetryConfig(max_attempts=2))
    async with LLMGateway(_strategy(failure_rate=1.0), config=cfg) as gw:
        result = await gw.try_submit("boom")
    assert not result.success


@pytest.mark.asyncio
async def test_gateway_closed_rejects():
    gw = LLMGateway(_strategy(), config=ProcessorConfig(max_workers=2))
    await gw.aclose()
    with pytest.raises(RuntimeError):
        await gw.submit("x")


@pytest.mark.asyncio
async def test_gateway_cancelled_submit_frees_slot():
    # max_workers=1: a cancelled in-flight submit must release its slot so the
    # next submit can proceed (queue-less cancellation is free).
    async with LLMGateway(_strategy(), config=ProcessorConfig(max_workers=1)) as gw:
        slow = asyncio.create_task(gw.submit("slow"))
        await asyncio.sleep(0)
        slow.cancel()
        with pytest.raises(asyncio.CancelledError):
            await slow
        out = await asyncio.wait_for(gw.submit("after"), timeout=2.0)
    assert out.text == "ok:after"
