"""Tests for shared strategy instances across multiple work items."""

import asyncio

import pytest

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, RetryState
from batch_llm.base import TokenUsage
from batch_llm.llm_strategies import LLMCallStrategy


class CountingStrategy(LLMCallStrategy[str]):
    """Strategy that counts how many times prepare() is called."""

    def __init__(self):
        self.prepare_count = 0
        self.execute_count = 0
        self.prepare_lock = asyncio.Lock()

    async def prepare(self) -> None:
        """Track prepare() calls."""
        async with self.prepare_lock:
            self.prepare_count += 1
            # Simulate slow preparation (database connection, cache creation, etc.)
            await asyncio.sleep(0.1)

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        """Track execute() calls."""
        self.execute_count += 1
        return f"Response: {prompt}", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


@pytest.mark.asyncio
async def test_shared_strategy_prepare_called_once():
    """Test that prepare() is called only once for shared strategy instance."""
    strategy = CountingStrategy()
    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 20 work items all sharing the same strategy instance
        for i in range(20):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"item_{i}",
                    strategy=strategy,  # SHARED instance
                    prompt=f"Test prompt {i}",
                )
            )

        result = await processor.process_all()

    # All items should succeed
    assert result.total_items == 20
    assert result.succeeded == 20
    assert result.failed == 0

    # prepare() should be called exactly once despite 20 work items
    assert (
        strategy.prepare_count == 1
    ), f"Expected prepare() to be called once, but it was called {strategy.prepare_count} times"

    # execute() should be called 20 times (once per work item)
    assert strategy.execute_count == 20


@pytest.mark.asyncio
async def test_different_strategies_prepare_called_separately():
    """Test that different strategy instances each get prepare() called."""
    strategy1 = CountingStrategy()
    strategy2 = CountingStrategy()
    strategy3 = CountingStrategy()

    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add work items with different strategy instances
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy1, prompt="Test 1"))
        await processor.add_work(LLMWorkItem(item_id="item_2", strategy=strategy2, prompt="Test 2"))
        await processor.add_work(LLMWorkItem(item_id="item_3", strategy=strategy3, prompt="Test 3"))

        # Add more items reusing the strategies
        await processor.add_work(LLMWorkItem(item_id="item_4", strategy=strategy1, prompt="Test 4"))
        await processor.add_work(LLMWorkItem(item_id="item_5", strategy=strategy2, prompt="Test 5"))

        result = await processor.process_all()

    assert result.total_items == 5
    assert result.succeeded == 5

    # Each unique strategy instance should have prepare() called exactly once
    assert strategy1.prepare_count == 1
    assert strategy2.prepare_count == 1
    assert strategy3.prepare_count == 1

    # strategy1 and strategy2 used twice, strategy3 used once
    assert strategy1.execute_count == 2
    assert strategy2.execute_count == 2
    assert strategy3.execute_count == 1


@pytest.mark.asyncio
async def test_shared_strategy_concurrent_workers():
    """Test shared strategy with high concurrency (simulates production load)."""
    strategy = CountingStrategy()
    config = ProcessorConfig(max_workers=10, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 100 work items to stress-test concurrent prepare() protection
        for i in range(100):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"item_{i}",
                    strategy=strategy,
                    prompt=f"Test {i}",
                )
            )

        result = await processor.process_all()

    assert result.total_items == 100
    assert result.succeeded == 100

    # Critical assertion: prepare() called exactly once despite 10 concurrent workers
    assert strategy.prepare_count == 1, (
        f"With 10 concurrent workers and 100 items, prepare() should be called once, "
        f"but was called {strategy.prepare_count} times"
    )

    assert strategy.execute_count == 100


@pytest.mark.asyncio
async def test_prepare_failure_propagates():
    """Test that prepare() failure is properly propagated."""

    class FailingPrepareStrategy(LLMCallStrategy[str]):
        async def prepare(self) -> None:
            raise ValueError("Simulated prepare failure")

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return "Should not reach here", {"input_tokens": 0}

    strategy = FailingPrepareStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))

        result = await processor.process_all()

    # Item should fail due to prepare() failure
    assert result.total_items == 1
    assert result.failed == 1
    assert "prepare" in result.results[0].error.lower()


@pytest.mark.asyncio
async def test_mixed_shared_and_unique_strategies():
    """Test mixture of shared and per-item strategies."""

    class Strategy(LLMCallStrategy[str]):
        def __init__(self, name: str):
            self.name = name
            self.prepare_count = 0

        async def prepare(self) -> None:
            self.prepare_count += 1

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return f"{self.name}: {prompt}", {"input_tokens": 10}

    # Shared strategies
    shared_strategy_a = Strategy("SharedA")
    shared_strategy_b = Strategy("SharedB")

    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add items with shared strategies
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"shared_a_{i}",
                    strategy=shared_strategy_a,
                    prompt=f"Test {i}",
                )
            )

        for i in range(5):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"shared_b_{i}",
                    strategy=shared_strategy_b,
                    prompt=f"Test {i}",
                )
            )

        # Add items with unique strategies (new instance per item)
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"unique_{i}",
                    strategy=Strategy(f"Unique{i}"),  # New instance each time
                    prompt=f"Test {i}",
                )
            )

        result = await processor.process_all()

    assert result.total_items == 15
    assert result.succeeded == 15

    # Shared strategies should have prepare() called once each
    assert shared_strategy_a.prepare_count == 1
    assert shared_strategy_b.prepare_count == 1

    # Each unique strategy instance had prepare() called (but we can't easily verify
    # since they're not tracked - this test mainly verifies no crashes)


@pytest.mark.asyncio
async def test_strategy_prepare_idempotency():
    """Test that calling _ensure_strategy_prepared multiple times is safe."""

    class Strategy(LLMCallStrategy[str]):
        def __init__(self):
            self.prepare_count = 0

        async def prepare(self) -> None:
            self.prepare_count += 1

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return "Response", {"input_tokens": 10}

    strategy = Strategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    processor = ParallelBatchProcessor[str, str, None](config=config)

    # Manually call _ensure_strategy_prepared multiple times
    await processor._ensure_strategy_prepared(strategy)
    await processor._ensure_strategy_prepared(strategy)
    await processor._ensure_strategy_prepared(strategy)

    # Should only prepare once
    assert strategy.prepare_count == 1

    await processor.cleanup()
