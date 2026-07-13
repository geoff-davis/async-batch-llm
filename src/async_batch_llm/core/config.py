"""Configuration management for batch processor."""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    ``max_attempts`` bounds retries for *content/transport* failures (validation
    errors, timeouts, connection errors, generic 5xx). **Rate-limit errors are
    exempt from this budget**: a 429 / quota / coordinated-cooldown signal is a
    "wait and try again", not a failed attempt, so it does not consume an
    ``attempt`` — the same logical ``attempt`` number is retried after the
    cooldown. Rate-limit retries are instead bounded separately by
    ``max_rate_limit_retries`` (a backstop against an endpoint that throttles
    forever). This keeps model-escalation strategies (which key behavior off the
    attempt number) from escalating just because the endpoint was busy.
    """

    max_attempts: int = 3
    initial_wait: float = 1.0
    max_wait: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    # Max times a single item may be retried after a rate-limit/cooldown without
    # consuming its max_attempts budget. 0 = rate limits immediately count as
    # failures. Exceeding it fails the item with a clear "exceeded N rate-limit
    # retries" error.
    max_rate_limit_retries: int = 20

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
        if self.max_rate_limit_retries < 0:
            raise ValueError(
                f"max_rate_limit_retries must be >= 0 (got {self.max_rate_limit_retries}). "
                f"Set retry.max_rate_limit_retries to 0 (rate limits fail immediately) "
                f"or a positive integer."
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


_DEFAULT_MAX_COOLDOWN_SECONDS = 600.0


@dataclass
class RateLimitConfig:
    """Configuration for rate limit handling."""

    cooldown_seconds: float = 300.0
    slow_start_items: int = 50
    slow_start_initial_delay: float = 2.0
    slow_start_final_delay: float = 0.1
    backoff_multiplier: float = 1.5  # Increase cooldown on repeated rate limits
    # Cap on the exponentially-backed-off cooldown (cooldown_seconds *
    # backoff_multiplier^n never exceeds this).
    max_cooldown_seconds: float = _DEFAULT_MAX_COOLDOWN_SECONDS

    def __post_init__(self) -> None:
        """Validate configuration on construction."""
        # The cap applies to the *escalated* cooldown, so it can never sit
        # below the base cooldown — lift it to match rather than rejecting
        # the config. This must be an unconditional, idempotent lift (not an
        # error, and not gated on the cap equaling its default) because
        # dataclasses.replace() re-runs __post_init__ on the *resolved*
        # values of the source config: a previously lifted cap no longer
        # equals the default and is indistinguishable from an explicit one.
        # Warn when the lift overrides a value the user plausibly set.
        if self.max_cooldown_seconds < self.cooldown_seconds:
            if self.max_cooldown_seconds != _DEFAULT_MAX_COOLDOWN_SECONDS:
                logger.warning(
                    f"max_cooldown_seconds ({self.max_cooldown_seconds}) is below "
                    f"cooldown_seconds ({self.cooldown_seconds}); lifting the cap to "
                    f"match. Set max_cooldown_seconds >= cooldown_seconds to silence."
                )
            self.max_cooldown_seconds = self.cooldown_seconds
        self.validate()

    def validate(self) -> None:
        """Validate rate limit configuration."""
        if self.cooldown_seconds < 0:
            raise ValueError(
                f"cooldown_seconds must be >= 0 (got {self.cooldown_seconds}). "
                f"Set rate_limit.cooldown_seconds to a non-negative number."
            )
        if self.slow_start_items < 0:
            raise ValueError(
                f"slow_start_items must be >= 0 (got {self.slow_start_items}). "
                f"Set rate_limit.slow_start_items to 0 to disable or a positive number."
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
        # No max_cooldown_seconds >= cooldown_seconds check here: __post_init__
        # lifts the cap unconditionally, so the invariant holds by construction.
        if self.slow_start_final_delay < 0:
            raise ValueError(
                f"slow_start_final_delay must be >= 0 (got {self.slow_start_final_delay})."
            )


@dataclass
class StartupRampConfig:
    """Optional concurrency ramp applied when an execution host starts."""

    initial_concurrency: int = 1
    concurrency_step: int = 1
    ramp_interval_seconds: float = 1.0
    max_concurrency: int | None = None
    jitter_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.initial_concurrency < 1:
            raise ValueError("initial_concurrency must be >= 1")
        if self.concurrency_step < 1:
            raise ValueError("concurrency_step must be >= 1")
        if self.ramp_interval_seconds <= 0:
            raise ValueError("ramp_interval_seconds must be > 0")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("startup ramp max_concurrency must be >= 1 or None")
        if self.max_concurrency is not None and self.initial_concurrency > self.max_concurrency:
            raise ValueError("initial_concurrency must be <= max_concurrency")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds must be >= 0")


class AbortMode(str, Enum):
    """How a controlled batch abort treats provider calls already running."""

    DRAIN_ACTIVE = "drain_active"
    CANCEL_ACTIVE = "cancel_active"


@dataclass
class GuardrailConfig:
    """Optional end-to-end deadlines and terminal-category fail-fast policy."""

    total_timeout_per_item: float | None = None
    batch_timeout: float | None = None
    abort_on_error_categories: frozenset[str] = field(default_factory=frozenset)
    abort_mode: AbortMode = AbortMode.DRAIN_ACTIVE

    def __post_init__(self) -> None:
        self.abort_mode = AbortMode(self.abort_mode)
        self.validate()

    def validate(self) -> None:
        for name, value in (
            ("total_timeout_per_item", self.total_timeout_per_item),
            ("batch_timeout", self.batch_timeout),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and > 0 or None (got {value!r})")
        if any(
            not isinstance(category, str) or not category
            for category in self.abort_on_error_categories
        ):
            raise ValueError("abort_on_error_categories must contain only non-empty strings")


@dataclass
class ProcessorConfig:
    """Complete configuration for batch processor."""

    max_workers: int = 5
    # Timeout applied to EACH execute() attempt (seconds), enforced via
    # asyncio.wait_for() around the strategy call. This is per-attempt, NOT a
    # total budget across retries: with retry.max_attempts=3 a single item can
    # legitimately spend up to ~3 × timeout_per_item in execute() time (plus any
    # backoff waits between attempts). Size it for one slow call, not the whole
    # retry chain.
    timeout_per_item: float = 120.0
    # Timeout for user-supplied post-processor functions (seconds).
    # Post-processors may do database/IO work; leave generous but bounded.
    # This is the single source of truth — it is threaded into
    # _run_post_processor and governs the wait directly (no hidden inner cap).
    post_processor_timeout: float = 90.0
    # When True, post-processors run as tracked background tasks (bounded by a
    # semaphore of size max_workers) instead of inline, so a slow post-processor
    # doesn't cap worker throughput. Trade-off: post-processors no longer run in
    # item-completion order, and may overlap. process_all() still awaits every
    # outstanding post-processor before returning. Default False (inline, ordered).
    concurrent_post_processing: bool = False

    retry: RetryConfig = field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Proactive rate limiting (prevents hitting rate limits)
    max_requests_per_minute: float | None = None  # None = no proactive limit

    # Progress reporting
    progress_interval: int = 10  # Log every N items
    progress_callback_timeout: float | None = 5.0  # Timeout for progress callback (seconds)

    # Observability
    enable_detailed_logging: bool = False

    # Queue management.
    # 0 = unlimited (recommended). A positive bound must be >= the total number
    # of items you add_work() before process_all(): workers only start in
    # process_all() and add_work() is rejected once processing begins, so a
    # bounded queue cannot drain while you are still adding work. add_work()
    # raises ValueError (rather than blocking forever) if the bound is reached.
    max_queue_size: int = 0

    # Dry-run mode (for testing configuration without making API calls)
    dry_run: bool = False

    # Optional explicit provider/client capacity. Kept last to preserve the
    # positional ordering of every pre-existing dataclass field. Attempts above
    # this limit wait before strategy.execute() and timeout_per_item; when the
    # strategy advertises max_concurrency, the lower limit applies.
    max_provider_concurrency: int | None = None

    # Optional cold-start ramp. Kept after all prior fields for positional
    # compatibility; None preserves immediate full concurrency.
    startup_ramp: StartupRampConfig | None = None

    # End-to-end item/batch deadlines and opt-in fail-fast behavior. Appended
    # for positional compatibility; defaults preserve pre-guardrail semantics.
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)

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
        if self.max_provider_concurrency is not None and self.max_provider_concurrency < 1:
            raise ValueError(
                "max_provider_concurrency must be >= 1 or None "
                f"(got {self.max_provider_concurrency})."
            )
        if self.startup_ramp is not None:
            self.startup_ramp.validate()
        self.guardrails.validate()
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
