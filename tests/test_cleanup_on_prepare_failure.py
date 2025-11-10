"""Tests for cleanup() being called even when prepare() fails."""

import pytest

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, RetryState
from batch_llm.base import TokenUsage
from batch_llm.llm_strategies import LLMCallStrategy


@pytest.mark.asyncio
async def test_cleanup_called_even_when_prepare_fails():
    """Verify cleanup() is called even when prepare() fails."""
    cleanup_called = False
    prepare_called = False

    class FailingPrepareStrategy(LLMCallStrategy[str]):
        """Strategy that fails during prepare()."""

        async def prepare(self):
            nonlocal prepare_called
            prepare_called = True
            raise RuntimeError("Simulated prepare() failure")

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            # Should never be called since prepare fails
            raise AssertionError("execute() should not be called when prepare() fails")

        async def cleanup(self):
            nonlocal cleanup_called
            cleanup_called = True

    strategy = FailingPrepareStrategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="test_item", strategy=strategy, prompt="Test")
        )

        result = await processor.process_all()

    # Verify behavior
    assert prepare_called, "prepare() should have been called"
    assert cleanup_called, "cleanup() should be called even when prepare() fails"
    assert result.total_items == 1
    assert result.failed == 1
    assert result.succeeded == 0
    assert "Simulated prepare() failure" in result.results[0].error


@pytest.mark.asyncio
async def test_cleanup_called_once_per_item_even_with_prepare_failure():
    """Verify cleanup() is called once per item, not once per strategy, even when prepare fails."""
    cleanup_count = 0
    prepare_count = 0

    class FailingPrepareWithCountStrategy(LLMCallStrategy[str]):
        """Strategy that fails during prepare() and counts calls."""

        async def prepare(self):
            nonlocal prepare_count
            prepare_count += 1
            raise RuntimeError("Prepare always fails")

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            raise AssertionError("Should not reach execute")

        async def cleanup(self):
            nonlocal cleanup_count
            cleanup_count += 1

    # Share strategy across 5 items
    strategy = FailingPrepareWithCountStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

    # Verify counts
    assert result.total_items == 5
    assert result.failed == 5
    # Note: Current behavior is that prepare() is retried for each item when it fails
    # This is by design - a failed prepare might be transient (e.g., network issue creating cache)
    assert prepare_count == 5, (
        "prepare() is called once per item when it fails (retried for each item)"
    )
    assert cleanup_count == 5, (
        "cleanup() should be called once per item (5 times), even when prepare() fails"
    )


@pytest.mark.asyncio
async def test_cleanup_exception_doesnt_break_processing():
    """Verify that exceptions in cleanup() don't break the processing."""
    cleanup_called = False

    class CleanupFailsStrategy(LLMCallStrategy[str]):
        """Strategy where cleanup() raises an exception."""

        async def prepare(self):
            pass  # Succeeds

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return "Success", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        async def cleanup(self):
            nonlocal cleanup_called
            cleanup_called = True
            raise RuntimeError("Cleanup failed!")

    strategy = CleanupFailsStrategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="test_item", strategy=strategy, prompt="Test")
        )

        result = await processor.process_all()

    # Verify behavior - item should still succeed even though cleanup failed
    assert cleanup_called, "cleanup() should have been called"
    assert result.total_items == 1
    assert result.succeeded == 1, "Item should succeed even though cleanup() raised exception"
    assert result.failed == 0


@pytest.mark.asyncio
async def test_cleanup_called_when_prepare_fails_then_succeeds_for_other_items():
    """Verify cleanup behavior when first item's prepare fails but strategy is reused."""
    prepare_count = 0
    cleanup_count = 0
    prepare_fail_on_first = True

    class ConditionalPrepareStrategy(LLMCallStrategy[str]):
        """Strategy that fails prepare() only on first call."""

        async def prepare(self):
            nonlocal prepare_count, prepare_fail_on_first
            prepare_count += 1
            if prepare_fail_on_first:
                prepare_fail_on_first = False
                raise RuntimeError("First prepare fails")
            # Second prepare succeeds

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return f"Result: {prompt}", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        async def cleanup(self):
            nonlocal cleanup_count
            cleanup_count += 1

    # Note: Current behavior is that prepare() is retried for each item when it fails.
    # This means if prepare() fails once but succeeds later, subsequent items can succeed.
    strategy = ConditionalPrepareStrategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test 1")
        )
        await processor.add_work(
            LLMWorkItem(item_id="item_2", strategy=strategy, prompt="Test 2")
        )

        result = await processor.process_all()

    # The framework retries prepare() for each item when it fails
    # First item: prepare() fails → cleanup() called
    # Second item: prepare() succeeds → execute() called → cleanup() called
    assert prepare_count == 2, "prepare() attempted twice (once per item, since first failed)"
    assert result.total_items == 2
    assert result.failed == 1, "First item fails (prepare failed)"
    assert result.succeeded == 1, "Second item succeeds (prepare succeeded)"
    # cleanup() called once per item that was attempted
    assert cleanup_count == 2, "cleanup() should be called for both items"
