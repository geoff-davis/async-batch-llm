"""Tests for shared strategy lifecycle (v0.2.0 feature).

This module tests the critical cost-optimization feature where one strategy
instance is shared across multiple work items. The framework must ensure:
1. prepare() is called exactly once per unique strategy instance
2. execute() is called for each work item + retries
3. cleanup() is called once per work item (not per strategy)

This enables 70-90% cost savings with Gemini prompt caching.
"""

import asyncio
import gc

import pytest

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, RetryState
from batch_llm.base import TokenUsage
from batch_llm.llm_strategies import LLMCallStrategy


@pytest.mark.asyncio
async def test_shared_strategy_prepare_called_once():
    """Verify prepare() called exactly once for shared strategy across 100 items."""
    prepare_count = 0
    cleanup_count = 0
    execute_count = 0

    class CountingStrategy(LLMCallStrategy[str]):
        """Strategy that counts lifecycle method calls."""

        async def prepare(self):
            nonlocal prepare_count
            prepare_count += 1
            await asyncio.sleep(0.01)  # Simulate slow prepare (cache creation)

        async def cleanup(self):
            nonlocal cleanup_count
            cleanup_count += 1

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            nonlocal execute_count
            execute_count += 1
            return f"Result: {prompt}", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    # Create ONE strategy instance
    strategy = CountingStrategy()
    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 100 items all sharing the same strategy
        for i in range(100):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")  # SHARED
            )

        result = await processor.process_all()

    # Critical assertions for cost optimization
    assert result.total_items == 100
    assert result.succeeded == 100
    assert prepare_count == 1, (
        f"prepare() should be called exactly once for shared strategy, "
        f"but was called {prepare_count} times. "
        f"This breaks the cost optimization (70-90% savings)."
    )
    assert execute_count == 100, (
        f"execute() should be called once per item, got {execute_count}"
    )
    assert cleanup_count == 100, (
        f"cleanup() should be called once per item (not per strategy), " f"got {cleanup_count}"
    )


@pytest.mark.asyncio
async def test_shared_strategy_concurrent_prepare_no_race():
    """Verify no race condition when concurrent workers access shared strategy."""
    prepare_count = 0
    prepare_lock = asyncio.Lock()

    class ConcurrentStrategy(LLMCallStrategy[str]):
        """Strategy with explicit locking to detect races."""

        async def prepare(self):
            nonlocal prepare_count
            # Use lock to safely increment counter
            async with prepare_lock:
                prepare_count += 1
            # Long prepare to increase chance of race if framework is broken
            await asyncio.sleep(0.1)

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return "Result", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    strategy = ConcurrentStrategy()
    config = ProcessorConfig(max_workers=20, timeout_per_item=10.0)  # High concurrency

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 100 items to stress-test concurrent access
        for i in range(100):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

    assert result.succeeded == 100
    assert prepare_count == 1, (
        f"Race condition detected: prepare() called {prepare_count} times instead of once. "
        f"Multiple workers called prepare() concurrently on shared strategy."
    )


@pytest.mark.asyncio
async def test_different_strategies_prepare_separately():
    """Verify different strategy instances each get prepare() called."""
    prepare_counts = {}

    class TrackedStrategy(LLMCallStrategy[str]):
        """Strategy that tracks prepare() calls by instance name."""

        def __init__(self, name: str):
            self.name = name

        async def prepare(self):
            prepare_counts[self.name] = prepare_counts.get(self.name, 0) + 1

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return f"{self.name}: {prompt}", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    # Create THREE different strategy instances
    strategy_a = TrackedStrategy("A")
    strategy_b = TrackedStrategy("B")
    strategy_c = TrackedStrategy("C")

    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add items with different strategies (reuse each strategy multiple times)
        for i in range(10):
            await processor.add_work(LLMWorkItem(item_id=f"a_{i}", strategy=strategy_a, prompt="Test"))
        for i in range(10):
            await processor.add_work(LLMWorkItem(item_id=f"b_{i}", strategy=strategy_b, prompt="Test"))
        for i in range(10):
            await processor.add_work(LLMWorkItem(item_id=f"c_{i}", strategy=strategy_c, prompt="Test"))

        result = await processor.process_all()

    assert result.total_items == 30
    assert result.succeeded == 30

    # Each unique strategy instance should have prepare() called exactly once
    assert prepare_counts["A"] == 1, f"Strategy A should prepare once, got {prepare_counts['A']}"
    assert prepare_counts["B"] == 1, f"Strategy B should prepare once, got {prepare_counts['B']}"
    assert prepare_counts["C"] == 1, f"Strategy C should prepare once, got {prepare_counts['C']}"


@pytest.mark.asyncio
async def test_shared_strategy_cleanup_per_item():
    """Verify cleanup() called once per work item, not once per strategy."""
    cleanup_calls = []

    class CleanupTrackingStrategy(LLMCallStrategy[str]):
        """Strategy that tracks which items trigger cleanup()."""

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return f"Result: {prompt}", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        async def cleanup(self):
            # Record that cleanup was called (can't track item_id here, just count)
            cleanup_calls.append(True)

    strategy = CleanupTrackingStrategy()
    config = ProcessorConfig(max_workers=3, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 20 items sharing the same strategy
        for i in range(20):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

    assert result.succeeded == 20
    assert len(cleanup_calls) == 20, (
        f"cleanup() should be called 20 times (once per item), " f"but was called {len(cleanup_calls)} times"
    )


@pytest.mark.asyncio
async def test_mixed_shared_and_unique_strategies():
    """Test mixture of shared and per-item strategies."""
    prepare_counts = {"shared": 0, "unique": 0}
    unique_count = 0

    class SharedStrategy(LLMCallStrategy[str]):
        """Shared strategy used by multiple items."""

        async def prepare(self):
            prepare_counts["shared"] += 1

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return "Shared result", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    class UniqueStrategy(LLMCallStrategy[str]):
        """Unique strategy created per item."""

        def __init__(self, item_num: int):
            self.item_num = item_num

        async def prepare(self):
            nonlocal unique_count
            unique_count += 1

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return f"Unique {self.item_num}", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    shared_strategy = SharedStrategy()  # One instance
    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 10 items with shared strategy
        for i in range(10):
            await processor.add_work(
                LLMWorkItem(item_id=f"shared_{i}", strategy=shared_strategy, prompt="Test")
            )

        # Add 10 items with unique strategies (new instance per item)
        for i in range(10):
            unique_strat = UniqueStrategy(i)  # New instance
            await processor.add_work(
                LLMWorkItem(item_id=f"unique_{i}", strategy=unique_strat, prompt="Test")
            )

        result = await processor.process_all()

    assert result.total_items == 20
    assert result.succeeded == 20

    # Shared strategy: prepare() called once
    assert prepare_counts["shared"] == 1, (
        f"Shared strategy should prepare once, got {prepare_counts['shared']}"
    )

    # Unique strategies: prepare() called 10 times (once per instance)
    assert unique_count == 10, f"10 unique strategies should each prepare, got {unique_count}"


@pytest.mark.asyncio
async def test_shared_strategy_with_retries():
    """Verify shared strategy works correctly with retry logic."""
    prepare_count = 0
    execute_count = 0
    attempt_numbers = []

    class RetryingStrategy(LLMCallStrategy[str]):
        """Strategy that fails first attempt, succeeds on second."""

        async def prepare(self):
            nonlocal prepare_count
            prepare_count += 1

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            nonlocal execute_count
            execute_count += 1
            attempt_numbers.append(attempt)

            # Fail first attempt, succeed on second
            if attempt == 1:
                raise Exception("Simulated transient failure")

            return "Success on retry", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    strategy = RetryingStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 5 items that will all retry once
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

    assert result.succeeded == 5
    assert result.failed == 0

    # prepare() should still be called only once despite retries
    assert prepare_count == 1, (
        f"prepare() should be called once even with retries, " f"got {prepare_count}"
    )

    # execute() should be called 10 times (2 attempts × 5 items)
    assert execute_count == 10, f"execute() should be called 10 times (with retries), got {execute_count}"

    # Verify we saw both attempt 1 and attempt 2
    assert 1 in attempt_numbers, "Should have seen attempt 1"
    assert 2 in attempt_numbers, "Should have seen attempt 2"


@pytest.mark.asyncio
async def test_prepared_strategy_entries_released_after_gc():
    """Weak references should drop prepared strategies after garbage collection."""

    prepare_count = 0

    class TemporaryStrategy(LLMCallStrategy[str]):
        async def prepare(self):
            nonlocal prepare_count
            prepare_count += 1

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return "ok", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        strategy = TemporaryStrategy()
        await processor.add_work(
            LLMWorkItem(item_id="temp_strategy", strategy=strategy, prompt="Test")
        )
        result = await processor.process_all()

        assert result.succeeded == 1
        assert len(processor._prepared_strategies) == 1

        # Drop last strong reference and force GC; WeakSet should shrink automatically
        del strategy
        gc.collect()

        assert len(processor._prepared_strategies) == 0, (
            "Prepared strategy cache should release entries once strategies are garbage collected"
        )
        assert prepare_count == 1
