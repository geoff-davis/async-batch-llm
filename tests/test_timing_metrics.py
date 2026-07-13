"""Structured per-attempt timing and processor percentile summaries."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from async_batch_llm import (
    LLMCallStrategy,
    LLMResponse,
    LLMWorkItem,
    ModelStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    RateLimitConfig,
    RetryConfig,
    TokenUsage,
    call_result,
)
from async_batch_llm._internal.executor_host import ExecutorHost


class _TimingStrategy(LLMCallStrategy[str]):
    def __init__(
        self,
        *,
        delay: float = 0.01,
        fail_first: bool = False,
        rate_limit_first: bool = False,
    ) -> None:
        self.delay = delay
        self.fail_first = fail_first
        self.rate_limit_first = rate_limit_first
        self.calls = 0

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: Any = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail_first and self.calls == 1:
            raise ConnectionError("retry me")
        if self.rate_limit_first and self.calls == 1:
            raise RuntimeError("rate limit exceeded")
        return prompt, {"total_tokens": 0}, None


class _TimedModel:
    max_concurrency = 1

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        await asyncio.sleep(0.02)
        return LLMResponse(text=prompt, input_tokens=1, output_tokens=1, total_tokens=2)


@pytest.mark.asyncio
async def test_retry_timing_records_attempts_execution_and_backoff() -> None:
    strategy = _TimingStrategy(delay=0.01, fail_first=True)
    config = ProcessorConfig(
        retry=RetryConfig(
            max_attempts=2,
            initial_wait=0.01,
            max_wait=0.01,
            jitter=False,
        )
    )
    result = await call_result(strategy, "x", config=config)

    assert result.success
    assert len(result.timing.attempts) == 2
    first, second = result.timing.attempts
    assert first.try_number == 1 and first.attempt == 1
    assert first.error_type == "ConnectionError"
    assert first.error_category == "connection_error"
    assert first.retry_backoff_seconds >= 0.005
    assert not first.success
    assert second.try_number == 2 and second.attempt == 2
    assert second.success
    assert first.execution_seconds > 0
    assert second.execution_seconds > 0
    assert result.timing.total_seconds >= 0.025


@pytest.mark.asyncio
async def test_model_strategy_records_provider_duration() -> None:
    strategy = ModelStrategy(_TimedModel())
    result = await call_result(strategy, "provider")

    assert result.success
    attempt = result.timing.attempts[0]
    assert attempt.provider_seconds is not None
    assert attempt.provider_seconds >= 0.015
    assert attempt.execution_seconds >= attempt.provider_seconds
    assert result.timing.provider_seconds >= 0.015


@pytest.mark.asyncio
async def test_framework_timeout_has_execution_timeout_category() -> None:
    strategy = _TimingStrategy(delay=0.05)
    config = ProcessorConfig(
        timeout_per_item=0.01,
        retry=RetryConfig(max_attempts=1),
    )
    result = await call_result(strategy, "timeout", config=config)

    assert not result.success
    assert result.timing.timeout_category == "framework_execution_timeout"
    assert len(result.timing.attempts) == 1
    assert result.timing.attempts[0].timeout_category == "framework_execution_timeout"
    assert result.timing.attempts[0].execution_seconds >= 0.008


@pytest.mark.asyncio
async def test_permanent_failure_records_classifier_category() -> None:
    class _PermanentFailure(_TimingStrategy):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: Any = None
        ) -> tuple[str, TokenUsage, None]:
            raise ValueError("bad configuration")

    result = await call_result(_PermanentFailure(), "x")

    assert not result.success
    assert result.timing.attempts[0].error_type == "ValueError"
    assert result.timing.attempts[0].error_category == "logic_error"


@pytest.mark.asyncio
async def test_rate_limit_timing_separates_cooldown() -> None:
    strategy = _TimingStrategy(delay=0.005, rate_limit_first=True)
    config = ProcessorConfig(
        retry=RetryConfig(max_attempts=1, max_rate_limit_retries=2),
        rate_limit=RateLimitConfig(
            cooldown_seconds=0.01,
            slow_start_items=0,
            slow_start_initial_delay=0.0,
            slow_start_final_delay=0.0,
            backoff_multiplier=1.0,
        ),
    )
    result = await call_result(strategy, "rate", config=config)

    assert result.success
    assert len(result.timing.attempts) == 2
    first, second = result.timing.attempts
    assert first.attempt == second.attempt == 1
    assert first.error_category == "rate_limit"
    assert first.cooldown_wait_seconds >= 0.008
    assert first.retry_backoff_seconds == 0


@pytest.mark.asyncio
async def test_initial_capacity_waits_are_attributed_to_first_successful_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _TimingStrategy(delay=0)
    host: ExecutorHost[str, str, None] = ExecutorHost(
        ProcessorConfig(max_workers=1),
        strategy=strategy,
    )

    async def wait_if_paused() -> None:
        await asyncio.sleep(0.01)

    async def apply_slow_start() -> float:
        return 0.01

    monkeypatch.setattr(host._rate_limit_coord, "wait_if_paused", wait_if_paused)
    monkeypatch.setattr(host._rate_limit_coord, "apply_slow_start", apply_slow_start)
    try:
        result = await host.executor.execute(
            LLMWorkItem(item_id="timed", strategy=strategy, prompt="x")
        )
    finally:
        await host.aclose()

    attempt = result.timing.attempts[0]
    assert result.success
    assert attempt.cooldown_wait_seconds >= 0.008
    assert attempt.startup_ramp_wait_seconds >= 0.008


@pytest.mark.asyncio
async def test_processor_stats_include_timing_percentiles() -> None:
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(max_workers=3, max_provider_concurrency=1)
    )
    strategy = _TimingStrategy(delay=0.01)
    for i in range(3):
        await processor.add_work(LLMWorkItem(item_id=str(i), strategy=strategy, prompt=str(i)))
    await processor.process_all()
    stats = await processor.get_stats()
    await processor.cleanup()

    assert stats["admission_wait_p50_seconds"] >= 0
    assert stats["admission_wait_p95_seconds"] >= stats["admission_wait_p50_seconds"]
    assert stats["admission_wait_p99_seconds"] >= stats["admission_wait_p95_seconds"]
    assert stats["execution_p50_seconds"] > 0
    assert stats["execution_p95_seconds"] >= stats["execution_p50_seconds"]
    assert stats["execution_p99_seconds"] >= stats["execution_p95_seconds"]
