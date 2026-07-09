"""Tests for ProcessorConfig.concurrent_post_processing (perf item #4).

When enabled, post-processors run as tracked background tasks (bounded by a
semaphore of size max_workers) instead of inline. The guarantees:

- ordering is RELAXED (post-processors may finish out of item order / overlap);
- process_all() still awaits every outstanding post-processor before returning;
- cleanup() cancels any still-pending post-processor tasks.
"""

import asyncio
import logging
import threading
import time

import pytest

from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from async_batch_llm.base import RetryState, TokenUsage, WorkItemResult
from async_batch_llm.llm_strategies import LLMCallStrategy


class _InstantStrategy(LLMCallStrategy[str]):
    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        return prompt, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None


@pytest.mark.asyncio
async def test_concurrent_post_processing_relaxes_ordering():
    """A later-submitted fast post-processor overtakes an earlier slow one."""
    finish_order: list[str] = []
    lock = asyncio.Lock()

    async def post(result: WorkItemResult) -> None:
        if result.item_id == "slow":
            await asyncio.sleep(0.1)
        async with lock:
            finish_order.append(result.item_id)

    config = ProcessorConfig(max_workers=4, concurrent_post_processing=True)
    strategy = _InstantStrategy()

    async with ParallelBatchProcessor[str, str, None](
        config=config, post_processor=post
    ) as processor:
        await processor.add_work(LLMWorkItem(item_id="slow", strategy=strategy, prompt="a"))
        await processor.add_work(LLMWorkItem(item_id="fast", strategy=strategy, prompt="b"))
        await processor.process_all()

    # Both ran, and "fast" finished first despite being submitted second —
    # proof that completion order is no longer item-submission order.
    assert set(finish_order) == {"slow", "fast"}
    assert finish_order[0] == "fast"


@pytest.mark.asyncio
async def test_concurrent_post_processing_completes_before_return():
    """process_all() must not return until every background post-processor is done."""
    done: list[str] = []
    lock = asyncio.Lock()

    async def post(result: WorkItemResult) -> None:
        # Each post-processor outlives its worker's item handling, so this only
        # passes if process_all() explicitly awaits the background tasks.
        await asyncio.sleep(0.05)
        async with lock:
            done.append(result.item_id)

    config = ProcessorConfig(max_workers=4, concurrent_post_processing=True)
    strategy = _InstantStrategy()

    async with ParallelBatchProcessor[str, str, None](
        config=config, post_processor=post
    ) as processor:
        for i in range(10):
            await processor.add_work(LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt="x"))
        result = await processor.process_all()

    assert result.total_items == 10
    # All 10 post-processors completed before process_all() returned.
    assert len(done) == 10
    assert set(done) == {f"i{i}" for i in range(10)}


@pytest.mark.asyncio
async def test_concurrent_post_processing_bounded_by_max_workers():
    """At most max_workers post-processors run at once (semaphore bound)."""
    concurrent = 0
    peak = 0
    lock = asyncio.Lock()

    async def post(result: WorkItemResult) -> None:
        nonlocal concurrent, peak
        async with lock:
            concurrent += 1
            peak = max(peak, concurrent)
        await asyncio.sleep(0.02)
        async with lock:
            concurrent -= 1

    config = ProcessorConfig(max_workers=3, concurrent_post_processing=True)
    strategy = _InstantStrategy()

    async with ParallelBatchProcessor[str, str, None](
        config=config, post_processor=post
    ) as processor:
        for i in range(20):
            await processor.add_work(LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt="x"))
        await processor.process_all()

    assert peak <= 3, f"post-processor concurrency {peak} exceeded max_workers=3"


@pytest.mark.asyncio
async def test_default_mode_runs_post_processors_inline():
    """Default (inline) mode schedules no background post-processor tasks."""
    seen: list[str] = []

    async def post(result: WorkItemResult) -> None:
        seen.append(result.item_id)

    config = ProcessorConfig(max_workers=2)  # concurrent_post_processing defaults False
    strategy = _InstantStrategy()

    async with ParallelBatchProcessor[str, str, None](
        config=config, post_processor=post
    ) as processor:
        await processor.add_work(LLMWorkItem(item_id="a", strategy=strategy, prompt="a"))
        await processor.process_all()
        assert processor._post_processor_tasks == set()

    assert seen == ["a"]


@pytest.mark.asyncio
async def test_cleanup_cancels_outstanding_post_processors():
    """cleanup() cancels background post-processor tasks that are still running."""
    config = ProcessorConfig(max_workers=2, concurrent_post_processing=True)
    started = asyncio.Event()
    cancelled: list[str] = []

    async def post(result: WorkItemResult) -> None:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(result.item_id)
            raise

    processor = ParallelBatchProcessor[str, str, None](config=config, post_processor=post)
    # Stand in for process_all()'s per-run setup, then spawn a long task directly.
    processor._post_processor_semaphore = asyncio.Semaphore(2)
    processor._spawn_post_processor(WorkItemResult(item_id="x", success=True), timeout=30.0)

    await started.wait()
    assert len(processor._post_processor_tasks) == 1

    await processor.cleanup()

    assert processor._post_processor_tasks == set()
    assert cancelled == ["x"]


@pytest.mark.asyncio
async def test_sync_post_processor_does_not_block_event_loop():
    """A synchronous callback runs in a thread so other async work can advance."""
    release = threading.Event()

    def post(result: WorkItemResult) -> None:
        release.wait(timeout=0.5)

    async def release_from_event_loop() -> None:
        await asyncio.sleep(0.02)
        release.set()

    config = ProcessorConfig(max_workers=1, post_processor_timeout=0.3)
    strategy = _InstantStrategy()
    releaser = asyncio.create_task(release_from_event_loop())
    started = time.monotonic()
    async with ParallelBatchProcessor[str, str, None](
        config=config, post_processor=post
    ) as processor:
        await processor.add_work(LLMWorkItem(item_id="x", strategy=strategy, prompt="x"))
        await processor.process_all()
    elapsed = time.monotonic() - started
    await releaser

    assert elapsed < 0.2


@pytest.mark.asyncio
async def test_sync_post_processor_respects_wait_timeout(caplog):
    """The processor stops waiting when a synchronous callback exceeds its budget."""

    def post(result: WorkItemResult) -> None:
        time.sleep(0.2)

    config = ProcessorConfig(max_workers=1, post_processor_timeout=0.02)
    strategy = _InstantStrategy()
    caplog.set_level(logging.ERROR)
    started = time.monotonic()
    async with ParallelBatchProcessor[str, str, None](
        config=config, post_processor=post
    ) as processor:
        await processor.add_work(LLMWorkItem(item_id="x", strategy=strategy, prompt="x"))
        result = await processor.process_all()
    elapsed = time.monotonic() - started

    assert result.succeeded == 1
    assert elapsed < 0.15
    assert "Post-processor execution timed out" in caplog.text
