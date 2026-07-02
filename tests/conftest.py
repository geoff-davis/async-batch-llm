"""Shared fixtures for the test suite.

The retry/rate-limit fixtures exist to keep tests fast: the library's
default ``RetryConfig(initial_wait=1.0)`` means any test that exercises a
retry against the defaults pays 1-2s of real sleep per retry. Tests that
don't specifically assert timing behavior should use these.
"""

import pytest

from async_batch_llm.core import RateLimitConfig, RetryConfig


@pytest.fixture
def fast_retry() -> RetryConfig:
    """Retry config with millisecond waits (deterministic: jitter off)."""
    return RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.05, jitter=False)


@pytest.fixture
def fast_rate_limit() -> RateLimitConfig:
    """Rate-limit config with a near-zero cooldown and no slow-start ramp."""
    return RateLimitConfig(
        cooldown_seconds=0.01,
        slow_start_items=0,
        slow_start_initial_delay=0.0,
        slow_start_final_delay=0.0,
        backoff_multiplier=1.0,
    )
