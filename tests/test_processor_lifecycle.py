"""Public one-shot lifecycle and bounded producer ownership."""

import asyncio

import pytest

from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.middleware import BaseMiddleware


class Echo(LLMCallStrategy):
    async def execute(self, prompt, attempt, timeout, state=None):
        return prompt, {"total_tokens": 1}


def item(name="x"):
    return LLMWorkItem(item_id=name, prompt=name, strategy=Echo())


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["start", "process_all"])
@pytest.mark.parametrize("mode", ["batch", "streaming", "closed"])
async def test_processor_rejects_mode_reentry(mode, operation):
    p = ParallelBatchProcessor(config=ProcessorConfig(max_workers=1))
    try:
        if mode == "batch":
            await p.process_all()
        elif mode == "streaming":
            p.start()
            if operation == "start":
                await p.finish()
        else:
            await p.shutdown()
        with pytest.raises(RuntimeError, match="new processor"):
            if operation == "start":
                p.start()
            else:
                await p.process_all()
    finally:
        await p.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.parametrize("mode", ["new", "streaming"])
async def test_closed_processor_rejects_work_before_configuration(mode, middleware):
    p = ParallelBatchProcessor(middlewares=[BaseMiddleware()] if middleware else [])
    if mode == "streaming":
        p.start()
    await p.shutdown()
    work = item()
    with pytest.raises(RuntimeError) as caught:
        await p.add_work(work)
    assert type(caught.value).__name__ == "BatchAdmissionClosedError"
    assert work.submission_index is None
    assert (await p.get_stats())["total"] == 0
    assert p._queue.empty()


@pytest.mark.asyncio
async def test_finish_after_shutdown_does_not_spawn_finalizer():
    p = ParallelBatchProcessor()
    p.start()
    await p.shutdown()
    with pytest.raises(RuntimeError, match="new processor"):
        await p.finish()
    assert p._finalize_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["shutdown", "finish"])
async def test_bounded_producer_wakes_at_admission_boundary(boundary):
    entered = asyncio.Event()
    release = asyncio.Event()

    class Blocked(Echo):
        async def execute(self, *args, **kwargs):
            entered.set()
            await release.wait()
            return await super().execute(*args, **kwargs)

    p = ParallelBatchProcessor(config=ProcessorConfig(max_workers=1, max_queue_size=2))
    strategy = Blocked()
    p.start()
    await p.add_work(LLMWorkItem(item_id="running", prompt="x", strategy=strategy))
    await entered.wait()
    await p.add_work(item("queued1"))
    await p.add_work(item("queued2"))
    blocked = item("blocked")
    producer = asyncio.create_task(p.add_work(blocked))
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        assert not producer.done()
        await getattr(p, boundary)()
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(asyncio.shield(producer), 1)
        assert type(caught.value).__name__ == "BatchAdmissionClosedError"
        assert blocked.submission_index is None
        assert (await p.get_stats())["total"] == 3
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        release.set()
        await p.shutdown()


@pytest.mark.asyncio
async def test_cancelled_bounded_producer_has_no_orphan_acceptance():
    p = ParallelBatchProcessor(config=ProcessorConfig(max_workers=1, max_queue_size=2))
    # Block worker setup, keeping queue capacity deterministic.
    blocker = asyncio.Event()

    async def worker(_):
        await blocker.wait()

    p._worker = worker
    p.start()
    await p.add_work(item("one"))
    await p.add_work(item("two"))
    work = item("cancelled")
    task = asyncio.create_task(p.add_work(work))
    for _ in range(10):
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await p.shutdown()
    assert work.submission_index is None
    assert not p._queue.qsize()


@pytest.mark.asyncio
async def test_repeated_finish_after_normal_finalization_is_noop():
    p = ParallelBatchProcessor()
    p.start()
    await p.finish()
    assert [r async for r in p.results()] == []
    finalizer = p._finalize_task
    await p.finish()
    assert p._finalize_task is finalizer
    await p.shutdown()


@pytest.mark.asyncio
async def test_raising_bounded_consumer_leaves_no_producer_or_owned_tasks():
    before = set(asyncio.all_tasks())
    producer = None
    try:
        with pytest.raises(ValueError, match="consumer failed"):
            async with ParallelBatchProcessor(
                config=ProcessorConfig(max_workers=1, max_queue_size=2, max_result_queue_size=1),
                middlewares=[BaseMiddleware()],
            ) as p:
                p.start()

                async def produce():
                    try:
                        for i in range(1000):
                            await p.add_work(item(str(i)))
                    finally:
                        await p.finish()

                producer = asyncio.create_task(produce())
                async for _ in p.results():
                    raise ValueError("consumer failed")
        assert producer is not None
        # Wait for the producer's public rejection, without cancelling it.
        done, _ = await asyncio.wait({producer}, timeout=1)
        assert producer in done
        assert isinstance(producer.exception(), RuntimeError)
        assert p._queue.empty()
        await asyncio.sleep(0)
        assert not (set(asyncio.all_tasks()) - before)
    finally:
        if producer is not None:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_cleanup_stays_closed_to_admission_then_retries():
    class FailingCleanup(Echo):
        calls = 0

        async def cleanup(self):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("cleanup failed")

    strategy = FailingCleanup()
    p = ParallelBatchProcessor()
    await p.add_work(LLMWorkItem(item_id="x", prompt="x", strategy=strategy))
    await p.process_all()
    with pytest.raises(ValueError, match="cleanup failed"):
        await p.shutdown()
    with pytest.raises(RuntimeError):
        await p.add_work(item("later"))
    with pytest.raises(RuntimeError):
        p.start()
    await p.shutdown()
    assert strategy.calls == 2
