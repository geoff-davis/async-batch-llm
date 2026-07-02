"""Tests for process_iter() streaming and request_stop() early cancellation."""

import asyncio

import pytest
from pydantic import BaseModel

from async_batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,
    RetryState,
)
from async_batch_llm.base import TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.testing import MockAgent


class StreamOutput(BaseModel):
    text: str


def make_item(i: int, latency: float = 0.01) -> LLMWorkItem:
    agent = MockAgent(response_factory=lambda p: StreamOutput(text=p), latency=latency)
    return LLMWorkItem(
        item_id=f"item_{i}",
        strategy=PydanticAIStrategy(agent=agent),
        prompt=f"prompt {i}",
    )


class SlowEcho(LLMCallStrategy[str]):
    """Echo strategy with configurable latency and call tracking."""

    def __init__(self, latency: float = 0.05) -> None:
        self.latency = latency
        self.executed_ids: list[str] = []

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        await asyncio.sleep(self.latency)
        self.executed_ids.append(prompt)
        return prompt, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


@pytest.mark.asyncio
async def test_process_iter_yields_every_result():
    config = ProcessorConfig(max_workers=2, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, StreamOutput, None](config=config)

    for i in range(5):
        await processor.add_work(make_item(i))

    seen: list[str] = []
    async for result in processor.process_iter():
        assert result.success
        seen.append(result.item_id)

    assert sorted(seen) == [f"item_{i}" for i in range(5)]
    stats = await processor.get_stats()
    assert stats["succeeded"] == 5


@pytest.mark.asyncio
async def test_process_iter_streams_incrementally():
    """Results arrive while later items are still being processed — the
    consumer must not have to wait for the whole batch."""
    strategy = SlowEcho(latency=0.05)
    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, str, None](config=config)

    for i in range(3):
        await processor.add_work(LLMWorkItem(item_id=f"s{i}", strategy=strategy, prompt=f"s{i}"))

    executed_when_first_yield = None
    count = 0
    async for _result in processor.process_iter():
        count += 1
        if executed_when_first_yield is None:
            executed_when_first_yield = len(strategy.executed_ids)

    assert count == 3
    # With 1 worker and 50ms latency per item, the first yield must happen
    # before the third item has executed.
    assert executed_when_first_yield is not None
    assert executed_when_first_yield < 3


@pytest.mark.asyncio
async def test_process_iter_early_break_stops_batch():
    """Breaking out of the loop cancels remaining work instead of leaking
    running workers."""
    strategy = SlowEcho(latency=0.05)
    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, str, None](config=config)

    for i in range(10):
        await processor.add_work(LLMWorkItem(item_id=f"b{i}", strategy=strategy, prompt=f"b{i}"))

    import contextlib

    async with contextlib.aclosing(processor.process_iter()) as results:
        async for _result in results:
            break  # Consumer bails after the first result.

    # Workers must be finished/cancelled — nothing left running.
    assert all(worker.done() for worker in processor._workers)
    # Far fewer than 10 items should have executed.
    assert len(strategy.executed_ids) < 10


@pytest.mark.asyncio
async def test_process_iter_propagates_worker_crash():
    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, StreamOutput, None](config=config)

    async def crashing_worker(worker_id: int) -> None:
        raise RuntimeError("worker exploded")

    processor._worker = crashing_worker  # type: ignore[method-assign]
    await processor.add_work(make_item(0))

    with pytest.raises(RuntimeError, match="Worker crashed"):
        async for _result in processor.process_iter():
            pass


@pytest.mark.asyncio
async def test_request_stop_discards_queued_items_but_finishes_in_flight():
    """After request_stop(), queued items are recorded as cancelled failures
    (accounting stays consistent) and are never executed."""
    strategy = SlowEcho(latency=0.02)
    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, str, None](config=config)

    for i in range(6):
        await processor.add_work(LLMWorkItem(item_id=f"q{i}", strategy=strategy, prompt=f"q{i}"))

    stopped_after: list[str] = []
    async for result in processor.process_iter():
        if not stopped_after:
            processor.request_stop()
        stopped_after.append(result.item_id)

    # Every queued item is accounted for: the first (and possibly a second
    # already in flight) succeeded, the rest are cancelled failures.
    assert len(stopped_after) == 6
    result_batch = processor._results
    succeeded = [r for r in result_batch if r.success]
    cancelled = [r for r in result_batch if not r.success]
    assert len(succeeded) >= 1
    assert len(succeeded) + len(cancelled) == 6
    for r in cancelled:
        assert "request_stop" in (r.error or "")
    # Cancelled items were never executed by the strategy.
    executed = set(strategy.executed_ids)
    for r in cancelled:
        assert r.item_id not in executed

    stats = await processor.get_stats()
    assert stats["processed"] == 6
    assert stats["succeeded"] + stats["failed"] == 6


@pytest.mark.asyncio
async def test_request_stop_works_with_process_all():
    """request_stop() from a post-processor also works with process_all()."""
    strategy = SlowEcho(latency=0.02)
    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)

    processor: ParallelBatchProcessor[str, str, None] | None = None

    async def stop_after_first(result) -> None:
        assert processor is not None
        processor.request_stop()

    processor = ParallelBatchProcessor[str, str, None](
        config=config, post_processor=stop_after_first
    )

    for i in range(5):
        await processor.add_work(LLMWorkItem(item_id=f"p{i}", strategy=strategy, prompt=f"p{i}"))

    result = await processor.process_all()

    assert result.total_items == 5
    assert result.succeeded >= 1
    assert result.succeeded + result.failed == 5
    assert any("request_stop" in (r.error or "") for r in result.results if not r.success)
