"""Configuration management for batch processor."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum attempts per item for ordinary retryable
            errors (network, timeout, validation).
        initial_wait: Base wait before the first retry (seconds).
        max_wait: Cap on the exponential-backoff wait (seconds).
        exponential_base: Backoff multiplier per failed attempt.
        jitter: Randomize waits to 50-100% to avoid thundering herds.
        count_rate_limits: When False (default), rate-limited attempts do
            NOT consume the ``max_attempts`` budget — the framework already
            paused and cooled down globally, so the item never got a clean
            attempt. Set True to restore pre-v0.10 accounting where a 429
            counts like any other failure.
        max_rate_limit_retries: Hard cap on budget-exempt rate-limited
            attempts per item (only used when ``count_rate_limits`` is
            False), so a persistently-throttled item can't retry forever.
    """

    max_attempts: int = 3
    initial_wait: float = 1.0
    max_wait: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    count_rate_limits: bool = False
    max_rate_limit_retries: int = 10

    def __post_init__(self) -> None:
        """Validate configuration on construction."""
        self.validate()

    def validate(self) -> None:
        """Validate retry configuration."""
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1 (got {self.max_attempts}). "
                f"Set retry.max_attempts to a positive integer."
            )
        if self.max_rate_limit_retries < 1:
            raise ValueError(
                f"max_rate_limit_retries must be >= 1 (got {self.max_rate_limit_retries}). "
                f"Set retry.max_rate_limit_retries to a positive integer, or set "
                f"retry.count_rate_limits=True to count rate limits against max_attempts."
            )
        if self.initial_wait <= 0:
            raise ValueError(
                f"initial_wait must be > 0 (got {self.initial_wait}). "
                f"Set retry.initial_wait to a positive number in seconds."
            )
        if self.max_wait < self.initial_wait:
            raise ValueError(
                f"max_wait must be >= initial_wait (got max_wait={self.max_wait}, initial_wait={self.initial_wait}). "
                f"Set retry.max_wait to be at least as large as retry.initial_wait."
            )
        if self.exponential_base < 1:
            raise ValueError(
                f"exponential_base must be >= 1 (got {self.exponential_base}). "
                f"Set retry.exponential_base to 1.0 or higher (typical values: 2.0-3.0)."
            )


@dataclass
class RateLimitConfig:
    """Configuration for rate limit handling."""

    cooldown_seconds: float = 300.0
    # Cap on the exponentially-backed-off cooldown (cooldown_seconds *
    # backoff_multiplier^n never exceeds this).
    max_cooldown_seconds: float = 600.0
    slow_start_items: int = 50
    slow_start_initial_delay: float = 2.0
    slow_start_final_delay: float = 0.1
    backoff_multiplier: float = 1.5  # Increase cooldown on repeated rate limits

    def __post_init__(self) -> None:
        """Validate configuration on construction."""
        self.validate()

    def validate(self) -> None:
        """Validate rate limit configuration."""
        if self.cooldown_seconds < 0:
            raise ValueError(
                f"cooldown_seconds must be >= 0 (got {self.cooldown_seconds}). "
                f"Set rate_limit.cooldown_seconds to a non-negative number."
            )
        if self.max_cooldown_seconds < self.cooldown_seconds:
            raise ValueError(
                f"max_cooldown_seconds must be >= cooldown_seconds "
                f"(got max_cooldown_seconds={self.max_cooldown_seconds}, "
                f"cooldown_seconds={self.cooldown_seconds})."
            )
        if self.slow_start_items < 0:
            raise ValueError(
                f"slow_start_items must be >= 0 (got {self.slow_start_items}). "
                f"Set rate_limit.slow_start_items to 0 to disable or a positive number."
            )
        if self.slow_start_final_delay < 0:
            raise ValueError(
                f"slow_start_final_delay must be >= 0 (got {self.slow_start_final_delay})."
            )
        if self.slow_start_initial_delay < self.slow_start_final_delay:
            raise ValueError(
                f"slow_start_initial_delay must be >= slow_start_final_delay "
                f"(got initial={self.slow_start_initial_delay}, final={self.slow_start_final_delay}). "
                f"The delay should decrease during slow start, not increase."
            )
        if self.backoff_multiplier < 1.0:
            raise ValueError(
                f"backoff_multiplier must be >= 1.0 (got {self.backoff_multiplier}). "
                f"Set rate_limit.backoff_multiplier to 1.0 (no increase) or higher (typical: 1.5-2.0)."
            )


@dataclass
class ProcessorConfig:
    """Complete configuration for batch processor."""

    max_workers: int = 5
    timeout_per_item: float = 120.0
    # Timeout for user-supplied post-processor functions (seconds).
    # Post-processors may do database/IO work; leave generous but bounded.
    post_processor_timeout: float = 90.0

    retry: RetryConfig = field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Proactive rate limiting (prevents hitting rate limits)
    max_requests_per_minute: float | None = None  # None = no proactive limit

    # Progress reporting
    progress_interval: int = 10  # Log every N items
    progress_callback_timeout: float | None = 5.0  # Timeout for progress callback (seconds)

    # Observability
    enable_detailed_logging: bool = False

    # Queue management
    max_queue_size: int = 0  # 0 = unlimited, >0 = max items in queue

    # Dry-run mode (for testing configuration without making API calls)
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Validate configuration on construction."""
        self.validate()

    def validate(self) -> None:
        """Validate complete configuration."""
        if self.max_workers < 1:
            raise ValueError(
                f"max_workers must be >= 1 (got {self.max_workers}). "
                f"Set config.max_workers to a positive integer (typical: 5-20)."
            )
        if self.timeout_per_item <= 0:
            raise ValueError(
                f"timeout_per_item must be > 0 (got {self.timeout_per_item}). "
                f"Set config.timeout_per_item to a positive number in seconds (typical: 60-300)."
            )
        if self.post_processor_timeout <= 0:
            raise ValueError(
                f"post_processor_timeout must be > 0 (got {self.post_processor_timeout}). "
                f"Set config.post_processor_timeout to a positive number in seconds (typical: 30-120)."
            )
        if self.progress_interval < 1:
            raise ValueError(
                f"progress_interval must be >= 1 (got {self.progress_interval}). "
                f"Set config.progress_interval to a positive integer."
            )
        if self.progress_callback_timeout is not None and self.progress_callback_timeout <= 0:
            raise ValueError(
                f"progress_callback_timeout must be > 0 (got {self.progress_callback_timeout}). "
                f"Set config.progress_callback_timeout to None to disable or a positive number of seconds."
            )
        if self.max_queue_size < 0:
            raise ValueError(
                f"max_queue_size must be >= 0 (got {self.max_queue_size}). "
                f"Set config.max_queue_size to 0 for unlimited, or a positive number to limit queue size."
            )
        if self.max_requests_per_minute is not None and self.max_requests_per_minute <= 0:
            raise ValueError(
                f"max_requests_per_minute must be > 0 or None (got {self.max_requests_per_minute}). "
                f"Set config.max_requests_per_minute to None to disable proactive rate limiting, "
                f"or a positive number (typical: 10-500 requests/minute)."
            )

        # Validate nested configs first
        self.retry.validate()
        self.rate_limit.validate()

        # Cross-field validations
        if self.max_queue_size > 0 and self.max_queue_size < self.max_workers:
            logger.warning(
                f"max_queue_size ({self.max_queue_size}) is less than max_workers ({self.max_workers}). "
                f"This may cause workers to starve waiting for work. "
                f"Consider setting max_queue_size >= max_workers or 0 for unlimited."
            )

        # Note: timeout_per_item is a PER-ATTEMPT limit enforced around each
        # strategy.execute() call; between-attempt retry waits happen outside
        # it, so no cross-validation between the two is meaningful. (Earlier
        # versions warned when timeout_per_item was smaller than cumulative
        # retry waits — that comparison was conceptually wrong and confusing.)

        # Validate proactive rate limit vs workers
        if self.max_requests_per_minute is not None:
            requests_per_second = self.max_requests_per_minute / 60.0
            if requests_per_second < self.max_workers:
                logger.warning(
                    f"max_requests_per_minute ({self.max_requests_per_minute}) is less than "
                    f"max_workers ({self.max_workers}). "
                    f"At {requests_per_second:.2f} requests/second with {self.max_workers} workers, "
                    f"workers may frequently wait for rate limit tokens. "
                    f"Consider reducing max_workers to {int(requests_per_second)} or increasing "
                    f"max_requests_per_minute."
                )
