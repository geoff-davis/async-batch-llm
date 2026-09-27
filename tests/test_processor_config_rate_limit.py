"""ProcessorConfig proactive-rate validation regressions (issue #147)."""

import logging

import pytest

from async_batch_llm import ProcessorConfig
from async_batch_llm.core import RateLimitConfig, RetryConfig, StartupRampConfig


@pytest.mark.parametrize("rpm", [10.0, 59.9, 1.0, 0.25])
def test_low_positive_rpm_with_multiple_workers_is_valid_and_quiet(
    rpm: float, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="async_batch_llm.core.config"):
        config = ProcessorConfig(max_workers=2, max_requests_per_minute=rpm)

    assert config.max_requests_per_minute == rpm
    assert not caplog.records


_D4_FIELDS = [
    (RetryConfig, "max_attempts", True),
    (RetryConfig, "max_rate_limit_retries", True),
    (RetryConfig, "initial_wait", False),
    (RetryConfig, "max_wait", False),
    (RetryConfig, "exponential_base", False),
    (RateLimitConfig, "cooldown_seconds", False),
    (RateLimitConfig, "slow_start_items", True),
    (RateLimitConfig, "slow_start_initial_delay", False),
    (RateLimitConfig, "slow_start_final_delay", False),
    (RateLimitConfig, "backoff_multiplier", False),
    (RateLimitConfig, "max_cooldown_seconds", False),
    (StartupRampConfig, "initial_concurrency", True),
    (StartupRampConfig, "concurrency_step", True),
    (StartupRampConfig, "max_concurrency", True),
    (StartupRampConfig, "ramp_interval_seconds", False),
    (StartupRampConfig, "jitter_seconds", False),
    (ProcessorConfig, "max_workers", True),
    (ProcessorConfig, "concurrency", True),
    (ProcessorConfig, "max_provider_concurrency", True),
    (ProcessorConfig, "max_queue_size", True),
    (ProcessorConfig, "max_result_queue_size", True),
    (ProcessorConfig, "progress_interval", True),
    (ProcessorConfig, "max_tokens_per_minute", True),
    (ProcessorConfig, "max_requests_per_minute", False),
    (ProcessorConfig, "attempt_timeout", False),
    (ProcessorConfig, "post_processor_timeout", False),
    (ProcessorConfig, "progress_callback_timeout", False),
    (ProcessorConfig, "progress_refresh_interval_seconds", False),
]


@pytest.mark.parametrize("factory,name,integer", _D4_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 2.5, True])
def test_adm4_numeric_fields(factory, name, integer, value):
    kwargs = {name: value}
    if factory is RateLimitConfig:
        kwargs.setdefault("slow_start_initial_delay", 3.0)
    if value == 2.5 and not integer:
        factory(**kwargs)
    else:
        with pytest.raises(ValueError):
            factory(**kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True])
def test_adm3_burst_validation(value):
    with pytest.raises(ValueError):
        ProcessorConfig(quota_burst_seconds=value)


def test_adm4_processor_does_not_repeat_validation_warning(caplog):
    from async_batch_llm import ParallelBatchProcessor

    with caplog.at_level(logging.WARNING):
        config = ProcessorConfig(max_workers=5, max_queue_size=2)
        ParallelBatchProcessor(config=config)
    assert sum("less than max_workers" in r.message for r in caplog.records) == 1
