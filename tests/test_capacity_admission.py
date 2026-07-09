"""Provider-capacity admission outside per-attempt execution timeouts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from async_batch_llm import (
    LLMCallStrategy,
    LLMGateway,
    LLMWorkItem,
    MetricsObserver,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryConfig,
    TokenUsage,
    call_result,
)


@dataclass
class _ConcurrencyState:
    active: int = 0
    peak: int = 0
    calls: int = 0


class _TrackingStrategy(LLMCallStrategy[str]):
    def __init__(
        self,
        *,
        capacity: int | None,
        delays: dict[str, float] | None = None,
        state: _ConcurrencyState | None = None,
        scope: object | None = None,
        fail_first: bool = False,
    ) -> None:
        self._capacity = capacity
        self._delays = delays or {}
        self._state = state or _ConcurrencyState()
        self._scope = scope or self
        self._fail_first = fail_first
        self._failed_prompts: set[str] = set()
        self.started = asyncio.Event()

    @property
    def max_concurrency(self) -> int | None:
        return self._capacity

    @property
    def concurrency_scope(self) -> object:
        return self._scope

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: Any = None
    ) -> tuple[str, TokenUsage, None]:
        self._state.active += 1
        self._state.peak = max(self._state.peak, self._state.active)
        self._state.calls += 1
        self.started.set()
        try:
            await asyncio.sleep(self._delays.get(prompt, 0.02))
            if self._fail_first and prompt not in self._failed_prompts:
                self._failed_prompts.add(prompt)
                raise ConnectionError("retry me")
            return prompt, {"total_tokens": 0}, None
        finally:
            self._state.active -= 1


@pytest.mark.asyncio
async def test_advertised_capacity_limits_batch_execute_concurrency() -> None:
    strategy = _TrackingStrategy(capacity=2)
    metrics = MetricsObserver()
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(max_workers=8, timeout_per_item=1.0),
        observers=[metrics],
    )

    with pytest.warns(UserWarning, match="Excess attempts will wait in ABL admission"):
        for i in range(8):
            await processor.add_work(LLMWorkItem(item_id=str(i), strategy=strategy, prompt=str(i)))
    result = await processor.process_all()
    stats = await processor.get_stats()
    observed = await metrics.get_metrics()
    await processor.cleanup()

    assert result.succeeded == 8
    assert strategy._state.peak == 2
    assert sum(item.admission_wait_seconds for item in result.results) > 0
    assert stats["total_admission_wait_seconds"] > 0
    assert stats["max_admission_wait_seconds"] > 0
    assert observed["admission_wait_count"] == 8
    assert observed["admission_wait_seconds_sum"] > 0
    assert observed["avg_admission_wait_seconds"] > 0


@pytest.mark.asyncio
async def test_explicit_capacity_limits_strategy_with_unknown_capacity() -> None:
    strategy = _TrackingStrategy(capacity=None)
    config = ProcessorConfig(
        max_workers=6,
        max_provider_concurrency=2,
        timeout_per_item=1.0,
    )
    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(6):
            await processor.add_work(LLMWorkItem(item_id=str(i), strategy=strategy, prompt=str(i)))
        result = await processor.process_all()

    assert result.succeeded == 6
    assert strategy._state.peak == 2


@pytest.mark.asyncio
async def test_admission_wait_does_not_consume_execution_timeout() -> None:
    strategy = _TrackingStrategy(
        capacity=1,
        delays={"first": 0.15, "second": 0.10},
    )
    config = ProcessorConfig(
        max_workers=2,
        timeout_per_item=0.20,
        retry=RetryConfig(max_attempts=1),
    )

    with pytest.warns(UserWarning, match="max_concurrency=1"):
        gateway = LLMGateway(strategy, config=config)
    first = asyncio.create_task(gateway.submit_result("first"))
    await strategy.started.wait()
    second = asyncio.create_task(gateway.submit_result("second"))
    first_result, second_result = await asyncio.gather(first, second)
    await gateway.aclose()

    assert first_result.success
    assert second_result.success
    assert second_result.admission_wait_seconds >= 0.10


@pytest.mark.asyncio
async def test_retries_release_and_reacquire_capacity() -> None:
    strategy = _TrackingStrategy(capacity=1, fail_first=True)
    config = ProcessorConfig(
        max_workers=2,
        timeout_per_item=1.0,
        retry=RetryConfig(
            max_attempts=2,
            initial_wait=0.001,
            max_wait=0.001,
            jitter=False,
        ),
    )
    processor = ParallelBatchProcessor[str, str, None](config=config)
    with pytest.warns(UserWarning):
        for item_id in ("a", "b"):
            await processor.add_work(
                LLMWorkItem(item_id=item_id, strategy=strategy, prompt=item_id)
            )
    result = await processor.process_all()
    await processor.cleanup()

    assert result.succeeded == 2
    assert strategy._state.calls == 4
    assert strategy._state.peak == 1
    assert all(item.admission_wait_seconds >= 0 for item in result.results)


@pytest.mark.asyncio
async def test_strategies_sharing_scope_share_capacity() -> None:
    shared_state = _ConcurrencyState()
    shared_scope = object()
    first = _TrackingStrategy(capacity=1, state=shared_state, scope=shared_scope)
    second = _TrackingStrategy(capacity=1, state=shared_state, scope=shared_scope)
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(max_workers=4, timeout_per_item=1.0)
    )
    with pytest.warns(UserWarning) as caught:
        for i in range(4):
            strategy = first if i % 2 == 0 else second
            await processor.add_work(LLMWorkItem(item_id=str(i), strategy=strategy, prompt=str(i)))
    result = await processor.process_all()
    await processor.cleanup()

    assert len(caught) == 2  # one diagnostic per distinct strategy
    assert result.succeeded == 4
    assert shared_state.peak == 1


@pytest.mark.asyncio
async def test_single_call_uses_shared_executor_admission_path() -> None:
    strategy = _TrackingStrategy(capacity=1)
    result = await call_result(
        strategy,
        "single",
        config=ProcessorConfig(max_provider_concurrency=1),
    )

    assert result.success
    assert result.admission_wait_seconds >= 0


def test_explicit_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_provider_concurrency must be >= 1"):
        ProcessorConfig(max_provider_concurrency=0)


def test_new_capacity_field_preserves_existing_positional_config_order() -> None:
    config = ProcessorConfig(7, 42.0)

    assert config.max_workers == 7
    assert config.timeout_per_item == 42.0
    assert config.max_provider_concurrency is None
