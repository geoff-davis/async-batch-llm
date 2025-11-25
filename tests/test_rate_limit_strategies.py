"""Tests for rate limit strategies."""

import pytest

from async_batch_llm.strategies.rate_limit import ExponentialBackoffStrategy, FixedDelayStrategy


@pytest.mark.asyncio
async def test_exponential_backoff_strategy_consecutive_limits():
    """Test ExponentialBackoffStrategy with consecutive rate limits."""
    strategy = ExponentialBackoffStrategy(
        initial_cooldown=60.0, max_cooldown=600.0, backoff_multiplier=2.0
    )

    # First rate limit
    cooldown1 = await strategy.on_rate_limit(worker_id=1, consecutive_count=1)
    assert cooldown1 == 60.0  # initial_cooldown

    # Second consecutive rate limit
    cooldown2 = await strategy.on_rate_limit(worker_id=1, consecutive_count=2)
    assert cooldown2 == 120.0  # 60.0 * 2^1

    # Third consecutive rate limit
    cooldown3 = await strategy.on_rate_limit(worker_id=1, consecutive_count=3)
    assert cooldown3 == 240.0  # 60.0 * 2^2

    # Fourth consecutive rate limit
    cooldown4 = await strategy.on_rate_limit(worker_id=1, consecutive_count=4)
    assert cooldown4 == 480.0  # 60.0 * 2^3


@pytest.mark.asyncio
async def test_exponential_backoff_strategy_max_cooldown():
    """Test ExponentialBackoffStrategy respects max_cooldown."""
    strategy = ExponentialBackoffStrategy(
        initial_cooldown=60.0, max_cooldown=300.0, backoff_multiplier=2.0
    )

    # 10 consecutive rate limits should hit max
    for count in range(1, 11):
        cooldown = await strategy.on_rate_limit(worker_id=1, consecutive_count=count)
        assert cooldown <= 300.0  # Never exceeds max


@pytest.mark.asyncio
async def test_exponential_backoff_slow_start_early_items():
    """Test ExponentialBackoffStrategy slow start for early items."""
    strategy = ExponentialBackoffStrategy(
        slow_start_items=50,
        slow_start_initial_delay=2.0,
        slow_start_final_delay=0.1,
    )

    # First item after resume
    should_delay, delay = strategy.should_apply_slow_start(items_since_resume=0)
    assert should_delay is True
    assert delay == pytest.approx(2.0, abs=0.01)  # Full initial delay

    # Middle of slow start (item 25)
    should_delay, delay = strategy.should_apply_slow_start(items_since_resume=25)
    assert should_delay is True
    # Halfway through: 2.0 - (2.0 - 0.1) * 0.5 = 1.05
    assert delay == pytest.approx(1.05, abs=0.01)

    # Near end of slow start (item 45)
    should_delay, delay = strategy.should_apply_slow_start(items_since_resume=45)
    assert should_delay is True
    # 90% through: 2.0 - (2.0 - 0.1) * 0.9 = 0.29
    assert delay == pytest.approx(0.29, abs=0.01)


@pytest.mark.asyncio
async def test_exponential_backoff_slow_start_complete():
    """Test ExponentialBackoffStrategy after slow start completes."""
    strategy = ExponentialBackoffStrategy(
        slow_start_items=50,
        slow_start_initial_delay=2.0,
        slow_start_final_delay=0.1,
    )

    # After slow start completes
    should_delay, delay = strategy.should_apply_slow_start(items_since_resume=50)
    assert should_delay is False
    assert delay == 0.0

    should_delay, delay = strategy.should_apply_slow_start(items_since_resume=100)
    assert should_delay is False
    assert delay == 0.0


@pytest.mark.asyncio
async def test_fixed_delay_strategy_consistent_cooldown():
    """Test FixedDelayStrategy returns consistent cooldown."""
    strategy = FixedDelayStrategy(cooldown=300.0, delay_between_requests=1.0)

    # Should return same cooldown regardless of consecutive count
    cooldown1 = await strategy.on_rate_limit(worker_id=1, consecutive_count=1)
    assert cooldown1 == 300.0

    cooldown2 = await strategy.on_rate_limit(worker_id=1, consecutive_count=5)
    assert cooldown2 == 300.0

    cooldown3 = await strategy.on_rate_limit(worker_id=1, consecutive_count=10)
    assert cooldown3 == 300.0


@pytest.mark.asyncio
async def test_fixed_delay_strategy_always_applies_delay():
    """Test FixedDelayStrategy always applies delay between requests."""
    strategy = FixedDelayStrategy(cooldown=300.0, delay_between_requests=1.5)

    # Should always apply delay, regardless of items processed
    should_delay, delay = strategy.should_apply_slow_start(items_since_resume=0)
    assert should_delay is True
    assert delay == 1.5

    should_delay, delay = strategy.should_apply_slow_start(items_since_resume=50)
    assert should_delay is True
    assert delay == 1.5

    should_delay, delay = strategy.should_apply_slow_start(items_since_resume=1000)
    assert should_delay is True
    assert delay == 1.5


@pytest.mark.asyncio
async def test_exponential_backoff_custom_multiplier():
    """Test ExponentialBackoffStrategy with custom multiplier."""
    strategy = ExponentialBackoffStrategy(
        initial_cooldown=10.0, max_cooldown=1000.0, backoff_multiplier=3.0
    )

    # Test with 3x multiplier
    cooldown1 = await strategy.on_rate_limit(worker_id=1, consecutive_count=1)
    assert cooldown1 == 10.0  # 10 * 3^0

    cooldown2 = await strategy.on_rate_limit(worker_id=1, consecutive_count=2)
    assert cooldown2 == 30.0  # 10 * 3^1

    cooldown3 = await strategy.on_rate_limit(worker_id=1, consecutive_count=3)
    assert cooldown3 == 90.0  # 10 * 3^2


@pytest.mark.asyncio
async def test_exponential_backoff_slow_start_zero_items():
    """Test ExponentialBackoffStrategy with slow_start_items=0."""
    strategy = ExponentialBackoffStrategy(
        slow_start_items=0,
        slow_start_initial_delay=2.0,
        slow_start_final_delay=0.1,
    )

    # With slow_start_items=0, should never apply slow start
    should_delay, delay = strategy.should_apply_slow_start(items_since_resume=0)
    assert should_delay is False
    assert delay == 0.0
