"""Tests for first-class streaming mode and the high-level streaming API."""

import asyncio
import contextlib

import pytest

from async_batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    process_prompts,
    process_stream,
)
from async_batch_llm.base import RetryState, TokenUsage, WorkItemResult
from async_batch_llm.llm_strategies import LLMCallStrategy

_TOK: TokenUsage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class _Strategy(LLMCallStrategy[str]):
    """Echo the prompt upper-cased, with optional per-prompt latency."""

    def __init__(self, latency: float = 0.0, delays: dict[str, float] | None = None) -> None:
        self.latency = latency
        self.delays = delays or {}

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        delay = self.delays.get(prompt, self.latency)
        if delay:
            await asyncio.sleep(delay)
        return prompt.upper(), _TOK, None


def _item(item_id: str, strategy: LLMCallStrategy, prompt: str = "p") -> LLMWorkItem:
    return LLMWorkItem(item_id=item_id, strategy=strategy, prompt=prompt)


# ── High-level process_prompts / process_stream ─────────────────────────────


@pytest.mark.asyncio
async def test_process_prompts_collects_results():
    result = await process_prompts(_Strategy(), ["a", "b"], config=ProcessorConfig(max_workers=2))
    assert result.total_items == 2
    assert result.succeeded == 2
    by_id = result.by_id()
    assert by_id["item_0"].output == "A"
    assert by_id["item_1"].output == "B"


@pytest.mark.asyncio
async def test_process_prompts_tuple_ids():
    result = await process_prompts(
        _Strategy(), [("x", "hi"), ("y", "yo")], config=ProcessorConfig(max_workers=2)
    )
    assert set(result.by_id()) == {"x", "y"}


@pytest.mark.asyncio
async def test_process_stream_yields_in_completion_order():
    # "slow" submitted first but sleeps; "fast" should stream out first.
    strategy = _Strategy(delays={"slow": 0.1})
    order = []
    async for result in process_stream(
        strategy, [("a", "slow"), ("b", "fast")], config=ProcessorConfig(max_workers=2)
    ):
        order.append(result.item_id)
    assert order[0] == "b"
    assert set(order) == {"a", "b"}


@pytest.mark.asyncio
async def test_process_stream_accepts_async_iterable():
    async def gen():
        for p in ["a", "b", "c"]:
            yield p

    seen = []
    async for r in process_stream(_Strategy(), gen(), config=ProcessorConfig(max_workers=2)):
        seen.append(r.item_id)
    assert set(seen) == {"item_0", "item_1", "item_2"}


@pytest.mark.asyncio
async def test_process_stream_forwards_post_processor():
    seen: list[str] = []

    async def post(result: WorkItemResult) -> None:
        seen.append(result.item_id)

    out = [
        r.item_id
        async for r in process_stream(
            _Strategy(), ["a", "b"], config=ProcessorConfig(max_workers=2), post_processor=post
        )
    ]
    assert set(out) == {"item_0", "item_1"}
    assert set(seen) == {"item_0", "item_1"}


# ── Backpressure (the headline reason streaming exists) ─────────────────────


@pytest.mark.asyncio
async def test_bounded_queue_applies_backpressure_in_streaming_mode():
    config = ProcessorConfig(max_workers=2, max_queue_size=2)
    processor = ParallelBatchProcessor[None, str, None](config=config)
    strategy = _Strategy(latency=0.01)
    max_qsize = 0

    async def feed():
        nonlocal max_qsize
        for i in range(20):
            await processor.add_work(_item(f"i{i}", strategy))
            max_qsize = max(max_qsize, processor._queue.qsize())
        await processor.finish()

    async with processor:
        processor.start()
        producer = asyncio.create_task(feed())
        results = [r async for r in processor.results()]
        await producer

    # All 20 arrive despite the tiny bounded queue, and the queue never grows
    # past its bound — add_work() blocked (backpressure) instead of raising.
    assert len(results) == 20
    assert max_qsize <= 2


# ── results() termination + worker-crash surfacing ──────────────────────────


@pytest.mark.asyncio
async def test_results_terminates_on_end_of_stream():
    processor = ParallelBatchProcessor[None, str, None](config=ProcessorConfig(max_workers=2))
    strategy = _Strategy()
    async with processor:
        processor.start()
        for i in range(5):
            await processor.add_work(_item(f"i{i}", strategy))
        await processor.finish()
        results = [r async for r in processor.results()]
    assert len(results) == 5


@pytest.mark.asyncio
async def test_worker_crash_is_reraised_by_results():
    class _Boom(BaseException):
        """Escapes the worker's `except Exception` so the task truly dies."""

    processor = ParallelBatchProcessor[None, str, None](config=ProcessorConfig(max_workers=2))

    async def _crash(*args, **kwargs):
        raise _Boom("worker exploded")

    # Monkeypatch the per-item entry point to crash the worker outright.
    processor._process_item_with_retries = _crash  # type: ignore[method-assign]

    with pytest.raises(_Boom, match="worker exploded"):
        async with processor:
            processor.start()
            await processor.add_work(_item("x", _Strategy()))
            await processor.finish()
            async for _ in processor.results():
                pass


# ── Producer errors + consumer early-exit cleanup ───────────────────────────


@pytest.mark.asyncio
async def test_producer_exception_propagates_to_consumer():
    async def bad_prompts():
        yield "ok"
        raise ValueError("producer boom")

    with pytest.raises(ValueError, match="producer boom"):
        async for _ in process_stream(
            _Strategy(), bad_prompts(), config=ProcessorConfig(max_workers=2)
        ):
            pass


@pytest.mark.asyncio
async def test_consumer_early_exit_cancels_producer_no_leaks():
    async def many():
        i = 0
        while True:  # unbounded producer — only safe because the consumer breaks
            yield f"p{i}"
            i += 1

    before = {t for t in asyncio.all_tasks() if not t.done()}

    config = ProcessorConfig(max_workers=2, max_queue_size=4)
    stream = process_stream(_Strategy(latency=0.001), many(), config=config)
    async with contextlib.aclosing(stream):
        async for _result in stream:
            break  # bail out after the first result

    await asyncio.sleep(0.05)  # let cancellations settle
    after = {t for t in asyncio.all_tasks() if not t.done()}
    leaked = after - before
    assert not leaked, f"leaked {len(leaked)} task(s): {leaked}"


# ── Mode guards + batch mode unchanged ──────────────────────────────────────


@pytest.mark.asyncio
async def test_results_without_start_raises():
    processor = ParallelBatchProcessor[None, str, None](config=ProcessorConfig())
    with pytest.raises(RuntimeError, match="streaming mode"):
        async for _ in processor.results():
            pass


@pytest.mark.asyncio
async def test_finish_without_start_raises():
    processor = ParallelBatchProcessor[None, str, None](config=ProcessorConfig())
    with pytest.raises(RuntimeError, match="streaming mode"):
        await processor.finish()


@pytest.mark.asyncio
async def test_add_work_after_finish_raises():
    processor = ParallelBatchProcessor[None, str, None](config=ProcessorConfig(max_workers=1))
    async with processor:
        processor.start()
        await processor.finish()
        with pytest.raises(RuntimeError, match="after finish"):
            await processor.add_work(_item("late", _Strategy()))


@pytest.mark.asyncio
async def test_batch_mode_bounded_queue_full_points_to_streaming():
    config = ProcessorConfig(max_workers=1, max_queue_size=2)
    processor = ParallelBatchProcessor[None, str, None](config=config)
    strategy = _Strategy()
    await processor.add_work(_item("1", strategy))
    await processor.add_work(_item("2", strategy))
    with pytest.raises(ValueError, match="streaming mode"):
        await processor.add_work(_item("3", strategy))


@pytest.mark.asyncio
async def test_process_all_unchanged_and_ignores_stream():
    """Batch mode still returns a BatchResult and never enters streaming mode."""
    config = ProcessorConfig(max_workers=2)
    processor = ParallelBatchProcessor[None, str, None](config=config)
    strategy = _Strategy()
    async with processor:
        for i in range(4):
            await processor.add_work(_item(f"i{i}", strategy))
        result = await processor.process_all()
    assert result.total_items == 4
    assert result.succeeded == 4
    assert processor._streaming is False
    assert processor._result_stream is None
