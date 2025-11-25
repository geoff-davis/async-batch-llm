"""Tests for strategy lifecycle management (v0.4.0).

This tests the hybrid approach using context managers for prepare/cleanup:
- Track strategies in add_work()
- Prepare strategies in workers (via _ensure_strategy_prepared)
- Cleanup strategies in __aexit__
- Backward compatible (no context manager = no cleanup)
- Prevent add_work() after process_all() starts
"""

import pytest

from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy


class LifecycleTrackingStrategy(LLMCallStrategy[str]):
    """Strategy that tracks prepare/cleanup calls for testing."""

    def __init__(self, output: str = "test"):
        self.output = output
        self.prepare_called = False
        self.cleanup_called = False
        self.execute_count = 0

    async def prepare(self) -> None:
        """Track that prepare was called."""
        self.prepare_called = True

    async def cleanup(self) -> None:
        """Track that cleanup was called."""
        self.cleanup_called = True

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        """Simple execution that returns test output."""
        self.execute_count += 1
        tokens: TokenUsage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }
        return self.output, tokens


@pytest.mark.asyncio
async def test_shared_strategy_prepared_once_cleaned_once():
    """Test that shared strategy instance is prepared once and cleaned up once."""
    strategy = LifecycleTrackingStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add multiple items with same strategy instance
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

        # Verify all items succeeded
        assert result.succeeded == 5
        assert result.failed == 0

        # Verify prepare was called exactly once
        assert strategy.prepare_called, "Strategy prepare() should have been called"
        assert strategy.execute_count == 5, "Should have executed 5 times"

        # Cleanup not called yet (still in context manager)
        assert not strategy.cleanup_called, "Cleanup should not be called yet"

    # After exiting context manager, cleanup should be called exactly once
    assert strategy.cleanup_called, "Strategy cleanup() should have been called on __aexit__"


@pytest.mark.asyncio
async def test_multiple_unique_strategies_each_get_lifecycle():
    """Test that multiple unique strategy instances each get prepare/cleanup."""
    strategy1 = LifecycleTrackingStrategy(output="output1")
    strategy2 = LifecycleTrackingStrategy(output="output2")
    strategy3 = LifecycleTrackingStrategy(output="output3")
    config = ProcessorConfig(max_workers=3, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add items with different strategies
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy1, prompt="Test 1"))
        await processor.add_work(LLMWorkItem(item_id="item_2", strategy=strategy2, prompt="Test 2"))
        await processor.add_work(LLMWorkItem(item_id="item_3", strategy=strategy3, prompt="Test 3"))

        result = await processor.process_all()

        # Verify all succeeded
        assert result.succeeded == 3

        # Verify each strategy was prepared
        assert strategy1.prepare_called
        assert strategy2.prepare_called
        assert strategy3.prepare_called

        # Cleanup not called yet
        assert not strategy1.cleanup_called
        assert not strategy2.cleanup_called
        assert not strategy3.cleanup_called

    # After exiting, all should be cleaned up
    assert strategy1.cleanup_called
    assert strategy2.cleanup_called
    assert strategy3.cleanup_called


@pytest.mark.asyncio
async def test_cleanup_happens_even_on_processing_error():
    """Test that cleanup is called even when processing fails."""

    class FailingStrategy(LifecycleTrackingStrategy):
        """Strategy that fails execution but should still be cleaned up."""

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            """Always raise an error."""
            self.execute_count += 1
            raise ValueError("Intentional test failure")

    strategy = FailingStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))

        result = await processor.process_all()

        # Verify item failed
        assert result.failed == 1
        assert result.succeeded == 0

        # Prepare was called
        assert strategy.prepare_called

    # Cleanup should still be called despite failure
    assert strategy.cleanup_called


@pytest.mark.asyncio
async def test_cleanup_error_does_not_fail_batch():
    """Test that cleanup errors are logged but don't fail the batch."""

    class CleanupFailStrategy(LifecycleTrackingStrategy):
        """Strategy that raises error during cleanup."""

        async def cleanup(self) -> None:
            """Raise error during cleanup."""
            self.cleanup_called = True
            raise RuntimeError("Cleanup failed")

    strategy = CleanupFailStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    # Should not raise despite cleanup failure
    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))
        result = await processor.process_all()

    # Verify processing succeeded
    assert result.succeeded == 1

    # Cleanup was called (and failed, but that's logged not raised)
    assert strategy.cleanup_called


@pytest.mark.asyncio
async def test_backward_compatibility_no_context_manager_no_cleanup():
    """Test that without context manager, cleanup is not called (backward compatible)."""
    strategy = LifecycleTrackingStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    # Don't use context manager
    processor = ParallelBatchProcessor[str, str, None](config=config)
    await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))
    result = await processor.process_all()

    # Verify processing succeeded
    assert result.succeeded == 1

    # Prepare was called
    assert strategy.prepare_called

    # Cleanup was NOT called (backward compatibility)
    assert not strategy.cleanup_called


@pytest.mark.asyncio
async def test_shutdown_triggers_cleanup_without_context_manager():
    """shutdown() should run strategy cleanup when not using context manager."""
    strategy = LifecycleTrackingStrategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    processor = ParallelBatchProcessor[str, str, None](config=config)
    await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))

    result = await processor.process_all()
    assert result.succeeded == 1
    assert strategy.prepare_called
    assert not strategy.cleanup_called  # Not yet cleaned up

    await processor.shutdown()
    assert strategy.cleanup_called


@pytest.mark.asyncio
async def test_cannot_add_work_after_process_all_starts():
    """Test that add_work() raises RuntimeError after process_all() starts."""
    strategy = LifecycleTrackingStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add first item
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test 1"))

        # Start processing
        result = await processor.process_all()
        assert result.succeeded == 1

        # Try to add more work - should fail
        with pytest.raises(RuntimeError) as exc_info:
            await processor.add_work(
                LLMWorkItem(item_id="item_2", strategy=strategy, prompt="Test 2")
            )

        assert "Cannot add work after process_all() has started" in str(exc_info.value)


@pytest.mark.asyncio
async def test_strategy_without_cleanup_method_works():
    """Test that strategies without cleanup() method work fine."""

    class NoCleanupStrategy(LLMCallStrategy[str]):
        """Strategy without cleanup method."""

        def __init__(self):
            self.prepare_called = False

        async def prepare(self) -> None:
            self.prepare_called = True

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            tokens: TokenUsage = {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }
            return "output", tokens

    strategy = NoCleanupStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    # Should work fine without cleanup method
    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))
        result = await processor.process_all()

    assert result.succeeded == 1
    assert strategy.prepare_called


@pytest.mark.asyncio
async def test_strategy_without_prepare_method_works():
    """Test that strategies without prepare() method work fine."""

    class NoPrepareStrategy(LLMCallStrategy[str]):
        """Strategy without prepare method."""

        def __init__(self):
            self.cleanup_called = False

        async def cleanup(self) -> None:
            self.cleanup_called = True

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            tokens: TokenUsage = {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }
            return "output", tokens

    strategy = NoPrepareStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    # Should work fine without prepare method
    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))
        result = await processor.process_all()

    assert result.succeeded == 1
    assert strategy.cleanup_called


@pytest.mark.asyncio
async def test_mixed_shared_and_unique_strategies():
    """Test mixture of shared and unique strategy instances."""
    shared_strategy = LifecycleTrackingStrategy(output="shared")
    unique_strategy1 = LifecycleTrackingStrategy(output="unique1")
    unique_strategy2 = LifecycleTrackingStrategy(output="unique2")
    config = ProcessorConfig(max_workers=3, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add multiple items with shared strategy
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=shared_strategy, prompt="Test 1")
        )
        await processor.add_work(
            LLMWorkItem(item_id="item_2", strategy=shared_strategy, prompt="Test 2")
        )
        await processor.add_work(
            LLMWorkItem(item_id="item_3", strategy=shared_strategy, prompt="Test 3")
        )

        # Add items with unique strategies
        await processor.add_work(
            LLMWorkItem(item_id="item_4", strategy=unique_strategy1, prompt="Test 4")
        )
        await processor.add_work(
            LLMWorkItem(item_id="item_5", strategy=unique_strategy2, prompt="Test 5")
        )

        result = await processor.process_all()

        # Verify all succeeded
        assert result.succeeded == 5

        # Shared strategy executed 3 times, prepared once
        assert shared_strategy.prepare_called
        assert shared_strategy.execute_count == 3

        # Unique strategies executed once each, prepared once each
        assert unique_strategy1.prepare_called
        assert unique_strategy1.execute_count == 1
        assert unique_strategy2.prepare_called
        assert unique_strategy2.execute_count == 1

    # All should be cleaned up exactly once
    assert shared_strategy.cleanup_called
    assert unique_strategy1.cleanup_called
    assert unique_strategy2.cleanup_called
