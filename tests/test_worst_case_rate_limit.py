"""Tests for worst-case rate limit scenarios."""

import asyncio

import pytest

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, RetryState
from batch_llm.base import TokenUsage
from batch_llm.llm_strategies import LLMCallStrategy
from batch_llm.strategies.errors import DefaultErrorClassifier, ErrorInfo
from batch_llm.strategies.rate_limit import FixedDelayStrategy


class FastRateLimitStrategy(FixedDelayStrategy):
    """Rate limit strategy with short cooldown for testing."""

    def __init__(self):
        """Initialize with 1 second cooldown instead of default 300 seconds."""
        super().__init__(cooldown=1.0, delay_between_requests=0.1)


@pytest.mark.asyncio
async def test_all_workers_hit_rate_limit_simultaneously():
    """
    Test worst case: all workers hit rate limit on their first attempts.

    This tests the framework's ability to handle widespread rate limit hits
    where all workers encounter 429 errors. The framework should:
    1. Trigger cooldown appropriately
    2. All workers should pause during cooldown
    3. Resume processing after cooldown without deadlock
    4. Complete all items successfully
    """
    rate_limit_count = 0
    execute_count = 0

    num_workers = 10

    class MultipleRateLimitStrategy(LLMCallStrategy[str]):
        """Strategy that hits rate limit on first attempt for all workers."""

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            nonlocal execute_count, rate_limit_count
            execute_count += 1

            # On first attempt, all workers hit rate limit
            if attempt == 1:
                rate_limit_count += 1
                raise Exception("429 Resource Exhausted: Rate limit exceeded")

            # Second attempt succeeds
            return f"Result: {prompt}", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    config = ProcessorConfig(
        max_workers=num_workers,
        timeout_per_item=30.0,  # Longer timeout to allow cooldown
    )

    async with ParallelBatchProcessor[str, str, None](
        config=config, rate_limit_strategy=FastRateLimitStrategy()
    ) as processor:
        # Add work items equal to number of workers
        # This ensures all workers are active and hit rate limits
        for i in range(num_workers):
            strategy = MultipleRateLimitStrategy()  # Each item gets its own strategy
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

    # Verify results
    assert result.total_items == num_workers
    assert result.succeeded == num_workers, f"Expected all {num_workers} to succeed, got {result.succeeded}"
    assert result.failed == 0

    # Verify rate limit was hit by all workers
    assert rate_limit_count == num_workers, (
        f"Expected {num_workers} rate limit hits, got {rate_limit_count}"
    )

    # Verify retry logic worked (2 attempts per item: fail then succeed)
    assert execute_count == num_workers * 2, (
        f"Expected {num_workers * 2} total executions (2 per item), got {execute_count}"
    )

    # Check that rate limit cooldown was triggered
    stats = await processor.get_stats()
    assert stats["rate_limit_count"] >= 1, "Expected at least one rate limit cooldown"


@pytest.mark.asyncio
async def test_cascading_rate_limits_under_high_load():
    """
    Test cascading rate limits where hitting one limit causes another.

    Simulates a scenario where:
    1. Multiple workers hit rate limit
    2. After cooldown, resuming too quickly causes another rate limit
    3. Framework should handle multiple consecutive cooldowns
    """
    rate_limit_wave = 1  # Track which "wave" of rate limits we're in
    attempt_counts = {}  # Track attempts per item

    class CascadingRateLimitStrategy(LLMCallStrategy[str]):
        """Strategy that causes cascading rate limits."""

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            nonlocal rate_limit_wave

            # Track this attempt
            item_id = prompt.split()[-1]  # Extract item ID from prompt
            attempt_counts[item_id] = attempt_counts.get(item_id, 0) + 1

            # First 2 waves of attempts hit rate limit
            if attempt <= 2:
                # Simulate rate limit
                await asyncio.sleep(0.01)  # Small delay to simulate network
                raise Exception(f"429 Resource Exhausted (wave {rate_limit_wave})")

            # Third attempt succeeds
            return f"Result after {attempt} attempts: {prompt}", {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            }

    config = ProcessorConfig(
        max_workers=5,
        timeout_per_item=60.0,  # Long timeout for multiple cooldowns
    )

    async with ParallelBatchProcessor[str, str, None](
        config=config, rate_limit_strategy=FastRateLimitStrategy()
    ) as processor:
        # Add 15 items (3x workers) to create sustained load
        for i in range(15):
            strategy = CascadingRateLimitStrategy()
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test item_{i}")
            )

        result = await processor.process_all()

    # Verify all items eventually succeeded despite cascading rate limits
    assert result.total_items == 15
    assert result.succeeded == 15, f"Expected all 15 to succeed, got {result.succeeded}"
    assert result.failed == 0

    # Verify multiple cooldowns occurred
    stats = await processor.get_stats()
    assert stats["rate_limit_count"] >= 2, (
        f"Expected at least 2 rate limit cooldowns (cascading), got {stats['rate_limit_count']}"
    )

    # Verify items required multiple attempts
    for item_id, count in attempt_counts.items():
        assert count >= 3, f"Item {item_id} should need 3+ attempts, got {count}"


@pytest.mark.asyncio
async def test_rate_limit_with_mixed_success_and_failures():
    """
    Test rate limiting when some items succeed and others hit rate limits.

    This tests the framework's ability to handle partial failures where:
    - Some workers successfully process items
    - Other workers hit rate limits
    - The rate limit doesn't block successful workers unnecessarily
    """
    successful_items = 0
    rate_limited_items = 0

    class MixedRateLimitStrategy(LLMCallStrategy[str]):
        """Strategy where some items hit rate limit, others succeed."""

        def __init__(self, should_rate_limit: bool):
            self.should_rate_limit = should_rate_limit

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            nonlocal successful_items, rate_limited_items

            if self.should_rate_limit and attempt == 1:
                rate_limited_items += 1
                await asyncio.sleep(0.01)
                raise Exception("429 Resource Exhausted")

            successful_items += 1
            return f"Result: {prompt}", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    config = ProcessorConfig(
        max_workers=5,
        timeout_per_item=30.0,
    )

    async with ParallelBatchProcessor[str, str, None](
        config=config, rate_limit_strategy=FastRateLimitStrategy()
    ) as processor:
        # Add 20 items: 10 will rate limit, 10 will succeed immediately
        for i in range(20):
            should_rate_limit = (i % 2 == 0)  # Every other item rate limits
            strategy = MixedRateLimitStrategy(should_rate_limit=should_rate_limit)
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

    # All items should eventually succeed
    assert result.total_items == 20
    assert result.succeeded == 20
    assert result.failed == 0

    # Verify mixed outcomes
    assert rate_limited_items == 10, f"Expected 10 rate limited items, got {rate_limited_items}"
    # Successful includes both immediate successes (10) and post-retry successes (10) = 20 total
    # Each item succeeds exactly once (on whichever attempt succeeds)
    assert successful_items == 20, f"Expected 20 successful executions, got {successful_items}"
