"""Regression tests for worker-death robustness in the processing loop.

Covers three related fixes:

- ``task_done()`` is guaranteed via try/finally in the worker loop, so an
  unexpected exception in the per-item pipeline can no longer strand
  ``queue.join()``.
- ``process_all()`` watches worker tasks and raises if one crashes, instead
  of hanging forever on a queue nobody is draining.
- Post-processor timeouts are caught with ``(TimeoutError, asyncio.TimeoutError)``
  so they can't kill a worker on Python 3.10, where the two are distinct classes.
"""

import asyncio

import pytest
from pydantic import BaseModel

from async_batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,
)
from async_batch_llm.testing import MockAgent


class EchoOutput(BaseModel):
    text: str


def make_item(i: int) -> LLMWorkItem:
    agent = MockAgent(response_factory=lambda prompt: EchoOutput(text=prompt), latency=0.0)
    return LLMWorkItem(
        item_id=f"item_{i}",
        strategy=PydanticAIStrategy(agent=agent),
        prompt=f"prompt {i}",
    )


@pytest.mark.asyncio
async def test_crashed_worker_raises_instead_of_hanging():
    """A worker that dies before draining the queue must fail the batch fast."""
    config = ProcessorConfig(max_workers=2, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, EchoOutput, None](config=config)

    async def crashing_worker(worker_id: int) -> None:
        raise RuntimeError("worker exploded")

    processor._worker = crashing_worker  # type: ignore[method-assign]

    await processor.add_work(make_item(0))

    with pytest.raises(RuntimeError, match="Worker crashed"):
        # Without the watchdog this would hang on queue.join() forever;
        # wait_for is a belt-and-suspenders guard for the regression case.
        await asyncio.wait_for(processor.process_all(), timeout=5.0)


@pytest.mark.asyncio
async def test_crashed_worker_with_full_bounded_queue_does_not_hang():
    """Sentinel injection must not block on a bounded queue after a crash."""
    config = ProcessorConfig(max_workers=2, timeout_per_item=5.0, max_queue_size=2)
    processor = ParallelBatchProcessor[str, EchoOutput, None](config=config)

    async def crashing_worker(worker_id: int) -> None:
        raise RuntimeError("worker exploded")

    processor._worker = crashing_worker  # type: ignore[method-assign]

    for i in range(2):
        await processor.add_work(make_item(i))

    with pytest.raises(RuntimeError, match="Worker crashed"):
        await asyncio.wait_for(processor.process_all(), timeout=5.0)


@pytest.mark.asyncio
async def test_unexpected_pipeline_error_surfaces_instead_of_hanging():
    """An exception escaping the per-item pipeline (after the item's failed
    result would normally be recorded) must surface, not deadlock the batch."""
    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, EchoOutput, None](config=config)

    async def broken_post_processor(result) -> None:
        raise RuntimeError("pipeline bug outside the strategy call")

    processor._run_post_processor = broken_post_processor  # type: ignore[method-assign]

    for i in range(2):
        await processor.add_work(make_item(i))

    with pytest.raises(RuntimeError, match="Worker crashed"):
        await asyncio.wait_for(processor.process_all(), timeout=5.0)


@pytest.mark.asyncio
async def test_slow_post_processor_times_out_without_killing_batch(caplog):
    """A post-processor exceeding config.post_processor_timeout is logged and
    skipped; the batch still completes successfully."""
    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0, post_processor_timeout=0.05)

    async def slow_post_processor(result) -> None:
        await asyncio.sleep(1.0)

    processor = ParallelBatchProcessor[str, EchoOutput, None](
        config=config, post_processor=slow_post_processor
    )

    for i in range(2):
        await processor.add_work(make_item(i))

    with caplog.at_level("ERROR"):
        result = await asyncio.wait_for(processor.process_all(), timeout=5.0)

    assert result.total_items == 2
    assert result.succeeded == 2
    assert any("Post-processor exceeded timeout" in r.message for r in caplog.records)
