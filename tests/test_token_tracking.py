"""Tests for token tracking, especially cached tokens (v0.2.0)."""

import pytest

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, RetryState
from batch_llm.base import TokenUsage
from batch_llm.llm_strategies import LLMCallStrategy


class CachedTokenStrategy(LLMCallStrategy[str]):
    """Strategy that returns token usage including cached tokens."""

    def __init__(
        self,
        input_tokens: int = 500,
        output_tokens: int = 100,
        cached_tokens: int = 450,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        """Return mock response with cached token info."""
        return "Response", {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cached_input_tokens": self.cached_tokens,
        }


@pytest.mark.asyncio
async def test_cached_token_aggregation():
    """Test that cached tokens are properly aggregated in BatchResult."""
    strategy = CachedTokenStrategy(
        input_tokens=500,
        output_tokens=100,
        cached_tokens=450,  # 90% cached
    )
    config = ProcessorConfig(max_workers=2)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(10):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt="Test")
            )

        result = await processor.process_all()

    # Check basic aggregation
    assert result.total_items == 10
    assert result.succeeded == 10
    assert result.total_input_tokens == 5000  # 500 * 10
    assert result.total_output_tokens == 1000  # 100 * 10
    assert result.total_cached_tokens == 4500  # 450 * 10


@pytest.mark.asyncio
async def test_cache_hit_rate_calculation():
    """Test cache_hit_rate() method."""
    strategy = CachedTokenStrategy(
        input_tokens=500,
        output_tokens=100,
        cached_tokens=450,  # 90% cached
    )
    config = ProcessorConfig(max_workers=2)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(10):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt="Test")
            )

        result = await processor.process_all()

    # Should be 90% cache hit rate (4500 / 5000 * 100)
    assert result.cache_hit_rate() == 90.0


@pytest.mark.asyncio
async def test_effective_input_tokens_calculation():
    """Test effective_input_tokens() method (cost after caching)."""
    strategy = CachedTokenStrategy(
        input_tokens=500,
        output_tokens=100,
        cached_tokens=450,  # 90% cached
    )
    config = ProcessorConfig(max_workers=2)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(10):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt="Test")
            )

        result = await processor.process_all()

    # Effective tokens = total_input - (cached * 0.9)
    # = 5000 - (4500 * 0.9)
    # = 5000 - 4050
    # = 950
    assert result.effective_input_tokens() == 950


@pytest.mark.asyncio
async def test_no_cached_tokens():
    """Test that missing cached_input_tokens doesn't break aggregation."""

    class NonCachedStrategy(LLMCallStrategy[str]):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return "Response", {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                # No cached_input_tokens field
            }

    strategy = NonCachedStrategy()
    config = ProcessorConfig(max_workers=1)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))

        result = await processor.process_all()

    assert result.total_cached_tokens == 0
    assert result.cache_hit_rate() == 0.0
    assert result.effective_input_tokens() == 100  # No caching discount


@pytest.mark.asyncio
async def test_mixed_cached_and_noncached():
    """Test mixture of cached and non-cached strategies."""
    cached_strategy = CachedTokenStrategy(input_tokens=500, output_tokens=100, cached_tokens=450)

    class NonCachedStrategy(LLMCallStrategy[str]):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return "Response", {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }

    non_cached_strategy = NonCachedStrategy()
    config = ProcessorConfig(max_workers=2)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 5 cached items
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(item_id=f"cached_{i}", strategy=cached_strategy, prompt="Test")
            )

        # Add 5 non-cached items
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"noncached_{i}",
                    strategy=non_cached_strategy,
                    prompt="Test",
                )
            )

        result = await processor.process_all()

    # Total input: (500 * 5) + (100 * 5) = 2500 + 500 = 3000
    assert result.total_input_tokens == 3000

    # Only cached items contribute to cached tokens: 450 * 5 = 2250
    assert result.total_cached_tokens == 2250

    # Cache hit rate: 2250 / 3000 * 100 = 75%
    assert result.cache_hit_rate() == 75.0


@pytest.mark.asyncio
async def test_cache_hit_rate_with_zero_input_tokens():
    """Test that cache_hit_rate() handles edge case of zero input tokens."""

    class ZeroTokenStrategy(LLMCallStrategy[str]):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return "Response", {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

    strategy = ZeroTokenStrategy()
    config = ProcessorConfig(max_workers=1)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))

        result = await processor.process_all()

    # Should handle division by zero gracefully
    assert result.cache_hit_rate() == 0.0
    assert result.effective_input_tokens() == 0


@pytest.mark.asyncio
async def test_100_percent_cache_hit_rate():
    """Test 100% cache hit rate scenario."""
    strategy = CachedTokenStrategy(
        input_tokens=500,
        output_tokens=100,
        cached_tokens=500,  # 100% cached
    )
    config = ProcessorConfig(max_workers=2)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(10):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt="Test")
            )

        result = await processor.process_all()

    assert result.cache_hit_rate() == 100.0

    # Effective tokens = 5000 - (5000 * 0.9) = 500
    assert result.effective_input_tokens() == 500


@pytest.mark.asyncio
async def test_partial_cache_hit_rate():
    """Test various partial cache hit rates."""
    test_cases = [
        (1000, 500, 50.0),  # 50% cached
        (1000, 250, 25.0),  # 25% cached
        (1000, 750, 75.0),  # 75% cached
        (1000, 100, 10.0),  # 10% cached
    ]

    for input_tokens, cached_tokens, expected_rate in test_cases:
        strategy = CachedTokenStrategy(
            input_tokens=input_tokens,
            output_tokens=100,
            cached_tokens=cached_tokens,
        )
        config = ProcessorConfig(max_workers=1)

        async with ParallelBatchProcessor[str, str, None](config=config) as processor:
            await processor.add_work(
                LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test")
            )

            result = await processor.process_all()

        assert (
            result.cache_hit_rate() == expected_rate
        ), f"Expected {expected_rate}% hit rate for {cached_tokens}/{input_tokens} cached tokens"
