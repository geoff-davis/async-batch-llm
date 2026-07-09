"""Startup concurrency ramp behavior and timeout boundaries."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from async_batch_llm import (
    LLMCallStrategy,
    LLMGateway,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryConfig,
    StartupRampConfig,
    TokenUsage,
)


class _RampStrategy(LLMCallStrategy[str]):
    def __init__(self, delays: dict[str, float], capacity: int | None = None) -> None:
        self.delays = delays
        self.capacity = capacity
        self.active = 0
        self.peak = 0
        self.started_at: list[float] = []
        self.epoch = time.perf_counter()

    @property
    def max_concurrency(self) -> int | None:
        return self.capacity

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: Any = None
    ) -> tuple[str, TokenUsage, None]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started_at.append(time.perf_counter() - self.epoch)
        try:
            await asyncio.sleep(self.delays[prompt])
            return prompt, {"total_tokens": 0}, None
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_startup_ramp_increases_allowed_concurrency_over_time() -> None:
    strategy = _RampStrategy({str(i): 0.09 for i in range(5)})
    config = ProcessorConfig(
        max_workers=5,
        timeout_per_item=1.0,
        startup_ramp=StartupRampConfig(
            initial_concurrency=1,
            concurrency_step=1,
            ramp_interval_seconds=0.04,
            max_concurrency=3,
        ),
    )
    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(5):
            await processor.add_work(LLMWorkItem(item_id=str(i), strategy=strategy, prompt=str(i)))
        result = await processor.process_all()

    assert result.succeeded == 5
    assert strategy.peak == 3
    starts = sorted(strategy.started_at)
    assert starts[1] >= 0.025
    assert starts[2] >= 0.06
    assert any(item.timing.startup_ramp_wait_seconds > 0 for item in result.results)


@pytest.mark.asyncio
async def test_ramp_wait_is_outside_execution_timeout_in_gateway() -> None:
    strategy = _RampStrategy({"first": 0.12, "second": 0.08})
    config = ProcessorConfig(
        max_workers=2,
        timeout_per_item=0.10,
        retry=RetryConfig(max_attempts=1),
        startup_ramp=StartupRampConfig(
            initial_concurrency=1,
            concurrency_step=1,
            ramp_interval_seconds=0.20,
            max_concurrency=2,
        ),
    )
    gateway = LLMGateway(strategy, config=config)
    first = asyncio.create_task(gateway.submit_result("first"))
    await asyncio.sleep(0.01)
    second = asyncio.create_task(gateway.submit_result("second"))
    first_result, second_result = await asyncio.gather(first, second)
    await gateway.aclose()

    assert not first_result.success  # execution itself exceeds 0.10s
    assert second_result.success
    assert second_result.timing.startup_ramp_wait_seconds >= 0.07
    assert second_result.timing.attempts[0].execution_seconds < 0.10


@pytest.mark.asyncio
async def test_ramp_composes_with_lower_advertised_capacity() -> None:
    strategy = _RampStrategy({str(i): 0.05 for i in range(4)}, capacity=2)
    config = ProcessorConfig(
        max_workers=4,
        timeout_per_item=1.0,
        startup_ramp=StartupRampConfig(
            initial_concurrency=1,
            concurrency_step=4,
            ramp_interval_seconds=0.01,
            max_concurrency=4,
        ),
    )
    processor = ParallelBatchProcessor[str, str, None](config=config)
    with pytest.warns(UserWarning):
        for i in range(4):
            await processor.add_work(LLMWorkItem(item_id=str(i), strategy=strategy, prompt=str(i)))
    result = await processor.process_all()
    await processor.cleanup()

    assert result.succeeded == 4
    assert strategy.peak == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_concurrency": 0}, "initial_concurrency"),
        ({"concurrency_step": 0}, "concurrency_step"),
        ({"ramp_interval_seconds": 0}, "ramp_interval_seconds"),
        ({"max_concurrency": 0}, "max_concurrency"),
        (
            {"initial_concurrency": 3, "max_concurrency": 2},
            "initial_concurrency must be <= max_concurrency",
        ),
        ({"jitter_seconds": -1}, "jitter_seconds"),
    ],
)
def test_startup_ramp_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StartupRampConfig(**kwargs)
