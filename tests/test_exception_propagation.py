"""Regression tests for WorkItemResult.exception propagation (issue #101).

Carried from the pre-0.13 backlog: verify that the originating exception of
a failed item reaches the caller on every execution surface. The single-call
and gateway surfaces were already covered (test_single_and_gateway.py);
these tests pin down the batch and streaming paths plus the serialization
boundary.
"""

from async_batch_llm import (
    FrameworkTimeoutError,
    LLMCallStrategy,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    WorkItemResult,
    process_prompts,
)


class BoomError(RuntimeError):
    pass


class AlwaysFailStrategy(LLMCallStrategy[str]):
    """Fails every attempt with a distinctive exception."""

    async def execute(self, prompt, attempt, timeout, state=None):
        raise BoomError(f"boom on attempt {attempt}")


class SucceedStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        tokens = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return "ok", tokens, None


class SlowStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        import asyncio

        await asyncio.sleep(30)
        return "never", {}, None


class TestBatchPath:
    async def test_failed_item_carries_originating_exception(self, fast_retry):
        config = ProcessorConfig(max_workers=2, retry=fast_retry)
        async with ParallelBatchProcessor[None, str, None](config=config) as processor:
            await processor.add_work(
                LLMWorkItem(item_id="fail", strategy=AlwaysFailStrategy(), prompt="p")
            )
            batch = await processor.process_all()

        result = batch.results[0]
        assert not result.success
        assert isinstance(result.exception, BoomError)
        # Retries exhausted: the stored exception is from the LAST attempt.
        assert "attempt 3" in str(result.exception)

    async def test_traceback_is_detached_on_batch_path(self, fast_retry):
        config = ProcessorConfig(max_workers=1, retry=fast_retry)
        async with ParallelBatchProcessor[None, str, None](config=config) as processor:
            await processor.add_work(
                LLMWorkItem(item_id="fail", strategy=AlwaysFailStrategy(), prompt="p")
            )
            batch = await processor.process_all()

        assert batch.results[0].exception is not None
        assert batch.results[0].exception.__traceback__ is None

    async def test_successful_item_has_no_exception(self):
        batch = await process_prompts(SucceedStrategy(), ["a"])
        assert batch.results[0].success
        assert batch.results[0].exception is None

    async def test_framework_timeout_surfaces_as_exception(self, fast_retry):
        config = ProcessorConfig(max_workers=1, attempt_timeout=0.05, retry=fast_retry)
        batch = await process_prompts(SlowStrategy(), ["a"], config=config)
        result = batch.results[0]
        assert not result.success
        assert isinstance(result.exception, FrameworkTimeoutError)


class TestStreamingPath:
    async def test_process_prompts_preserves_exception(self, fast_retry):
        config = ProcessorConfig(retry=fast_retry)
        batch = await process_prompts(AlwaysFailStrategy(), ["a"], config=config)
        result = batch.results[0]
        assert isinstance(result.exception, BoomError)

    async def test_exception_reaches_post_processor(self, fast_retry):
        seen: list[Exception | None] = []

        async def post(result: WorkItemResult) -> None:
            seen.append(result.exception)

        config = ProcessorConfig(retry=fast_retry)
        await process_prompts(AlwaysFailStrategy(), ["a"], config=config, post_processor=post)
        assert len(seen) == 1
        assert isinstance(seen[0], BoomError)


class TestSerializationBoundary:
    async def test_exception_serializes_as_descriptor_and_restores_as_none(self, fast_retry):
        config = ProcessorConfig(retry=fast_retry)
        batch = await process_prompts(AlwaysFailStrategy(), ["a"], config=config)

        data = batch.to_dict()
        descriptor = data["results"][0]["exception"]
        assert descriptor["class_name"] == "BoomError"
        assert "boom" in descriptor["message"]

        # By design, deserialization never instantiates a class named by
        # untrusted JSON — the live exception does not round-trip.
        from async_batch_llm import BatchResult

        restored = BatchResult.from_dict(data)
        assert restored.results[0].exception is None
        assert restored.results[0].error is not None
