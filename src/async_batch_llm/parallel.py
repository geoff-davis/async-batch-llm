"""Parallel batch processor"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Generic, cast

if TYPE_CHECKING:
    from types import TracebackType

from ._internal.error_logging import log_retryable_error, log_validation_error
from ._internal.event_dispatcher import EventDispatcher
from ._internal.rate_limit_coordinator import RateLimitCoordinator
from ._internal.strategy_lifecycle import StrategyLifecycle
from .base import (
    BatchProcessor,
    LLMWorkItem,
    PostProcessorFunc,
    ProgressCallbackFunc,
    RetryState,
    TContext,
    TInput,
    TokenUsage,
    TOutput,
    WorkItemResult,
    _unpack_strategy_result,
)
from .core import ProcessorConfig
from .llm_strategies import LLMCallStrategy
from .middleware import Middleware
from .observers import ProcessingEvent, ProcessorObserver
from .strategies import (
    DefaultErrorClassifier,
    ErrorClassifier,
    ExponentialBackoffStrategy,
    FrameworkTimeoutError,
    RateLimitRetriesExceeded,
    RateLimitStrategy,
)
from .token_extractor import TokenExtractor

logger = logging.getLogger(__name__)

# Maximum length for error messages in logs / result payloads.
# Longer errors get truncated to keep logs readable.
ERROR_MESSAGE_MAX_LENGTH = 200
# Larger truncation used for detailed diagnostic logs (final attempt, validation traces).
ERROR_MESSAGE_DETAILED_LENGTH = 500

# Warn when the soft open-file limit leaves less than this many fds of headroom
# above max_workers (each in-flight request typically holds a socket).
_FD_LIMIT_HEADROOM = 128
# Docs section explaining the open-file-limit footgun and how to fix it.
_FD_LIMIT_DOCS_URL = (
    "https://geoff-davis.github.io/async-batch-llm/getting-started/"
    "#open-file-limits-and-high-concurrency"
)


def _warn_if_fd_limit_low(max_workers: int) -> None:
    """Warn if ``max_workers`` is close to the process's soft open-file limit.

    Each in-flight request typically holds a socket (a file descriptor), so a
    ``max_workers`` near the OS soft limit (``RLIMIT_NOFILE`` — 256 by default
    on macOS) risks ``OSError: [Errno 24] Too many open files`` once the
    connection pools, workers, and the app's own fds are all drawing from it.

    This only *warns* — raising the limit mutates process-global state, which is
    the operator's call, not the library's. No-op on non-Unix platforms (no
    ``RLIMIT_NOFILE``) or when the limit is unlimited / comfortably high.
    """
    try:
        import resource
    except ImportError:
        return  # non-Unix (e.g. Windows): no RLIMIT_NOFILE
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        return
    if soft == resource.RLIM_INFINITY or soft - max_workers >= _FD_LIMIT_HEADROOM:
        return

    import warnings

    warnings.warn(
        f"max_workers={max_workers} is close to this process's soft open-file "
        f"limit (RLIMIT_NOFILE={soft}); high-concurrency runs may fail with "
        f"'OSError: [Errno 24] Too many open files'. Raise the limit "
        f"(e.g. `ulimit -n {max(8192, max_workers * 4)}`, or resource.setrlimit "
        f"in-process) or lower max_workers. See {_FD_LIMIT_DOCS_URL}",
        UserWarning,
        stacklevel=3,
    )


class ParallelBatchProcessor(
    BatchProcessor[TInput, TOutput, TContext], Generic[TInput, TOutput, TContext]
):
    """
    Batch processor that executes items in parallel as individual agent calls.

    This refactored version uses:
    - Pluggable error classification (provider-agnostic)
    - Pluggable rate limit strategies
    - Middleware pipeline for extensibility
    - Observer pattern for monitoring
    - Configuration objects for easier setup
    """

    def __init__(
        self,
        max_workers: int | None = None,
        post_processor: PostProcessorFunc[TOutput, TContext] | None = None,
        timeout_per_item: float | None = None,
        rate_limit_cooldown: float | None = None,  # Deprecated, use config
        # New parameters
        config: ProcessorConfig | None = None,
        error_classifier: ErrorClassifier | None = None,
        rate_limit_strategy: RateLimitStrategy | None = None,
        middlewares: list[Middleware[TInput, TOutput, TContext]] | None = None,
        observers: list[ProcessorObserver] | None = None,
        progress_callback: "ProgressCallbackFunc | None" = None,
    ):
        """
        Initialize the parallel batch processor.

        Args:
            max_workers: Maximum concurrent workers (deprecated, use config)
            post_processor: Optional async function called after each successful item
            timeout_per_item: Timeout per item in seconds (deprecated, use config)
            rate_limit_cooldown: Cooldown duration (deprecated, use config)
            config: Processor configuration object (recommended)
            error_classifier: Strategy for classifying errors (default: DefaultErrorClassifier)
            rate_limit_strategy: Strategy for handling rate limits
            middlewares: List of middleware to apply
            observers: List of observers for events
            progress_callback: Optional callback(completed, total, current_item_id) for progress updates
        """
        import warnings

        # Emit deprecation warnings for legacy parameters
        if max_workers is not None:
            warnings.warn(
                "The 'max_workers' parameter is deprecated. "
                "Use ProcessorConfig(max_workers=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if timeout_per_item is not None:
            warnings.warn(
                "The 'timeout_per_item' parameter is deprecated. "
                "Use ProcessorConfig(timeout_per_item=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if rate_limit_cooldown is not None:
            warnings.warn(
                "The 'rate_limit_cooldown' parameter is deprecated. "
                "Use ProcessorConfig(rate_limit=RateLimitConfig(cooldown_seconds=...)) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Handle backward compatibility
        if config is None:
            from .core import RateLimitConfig

            config = ProcessorConfig(
                max_workers=max_workers or 5,
                timeout_per_item=timeout_per_item or 120.0,
                rate_limit=RateLimitConfig(cooldown_seconds=rate_limit_cooldown or 300.0),
            )
        else:
            # Override config with explicit legacy parameters if provided — but
            # build a NEW config via dataclasses.replace rather than mutating the
            # caller's object (and its nested RateLimitConfig). Callers may reuse
            # the same ProcessorConfig across processors and shouldn't see it
            # silently rewritten under them.
            import dataclasses

            overrides: dict = {}
            if max_workers is not None:
                overrides["max_workers"] = max_workers
            if timeout_per_item is not None:
                overrides["timeout_per_item"] = timeout_per_item
            if rate_limit_cooldown is not None:
                overrides["rate_limit"] = dataclasses.replace(
                    config.rate_limit, cooldown_seconds=rate_limit_cooldown
                )
            if overrides:
                config = dataclasses.replace(config, **overrides)

        config.validate()

        super().__init__(
            config.max_workers,
            post_processor,
            max_queue_size=config.max_queue_size,
            progress_callback=progress_callback,
            progress_callback_timeout=config.progress_callback_timeout,
        )
        self.config = config

        # Diagnostic: high max_workers can outrun the OS open-file limit.
        _warn_if_fd_limit_low(config.max_workers)

        # Set up strategies. When the caller didn't pass an explicit
        # error_classifier, it's auto-selected from the work items' strategies at
        # batch start (see _resolve_error_classifier); until then we hold a
        # DefaultErrorClassifier so the processor is always usable.
        self._user_supplied_classifier = error_classifier is not None
        self.error_classifier: ErrorClassifier = error_classifier or DefaultErrorClassifier()
        self._recommended_classifiers: list[ErrorClassifier] = []
        self._classifier_resolved = self._user_supplied_classifier
        self.rate_limit_strategy = rate_limit_strategy or ExponentialBackoffStrategy(
            initial_cooldown=config.rate_limit.cooldown_seconds,
            backoff_multiplier=config.rate_limit.backoff_multiplier,
            slow_start_items=config.rate_limit.slow_start_items,
            slow_start_initial_delay=config.rate_limit.slow_start_initial_delay,
            slow_start_final_delay=config.rate_limit.slow_start_final_delay,
        )

        # Set up middleware and observers
        self.middlewares = middlewares or []
        self.observers = observers or []

        # Event + middleware dispatch. Delegates observer emits and the
        # middleware chain (before/after/on_error) to a stateless helper.
        self._events: EventDispatcher[TInput, TOutput, TContext] = EventDispatcher(
            observers=self.observers, middlewares=self.middlewares
        )

        # Rate-limit coordination (extracted in v0.7.0).
        self._rate_limit_coord = RateLimitCoordinator(
            rate_limit_strategy=self.rate_limit_strategy,
            events=self._events,
        )
        # Back-compat aliases — existing private methods and some tests
        # reach into these attributes directly.
        self._rate_limit_event = self._rate_limit_coord._rate_limit_event
        self._current_generation_event = self._rate_limit_coord._current_generation_event
        self._rate_limit_lock = self._rate_limit_coord._lock

        # Thread-safety locks (_stats_lock / _results_lock) live on the base
        # class so both batch and streaming modes share them.

        # Strategy lifecycle management (v0.2.0, extracted in v0.7.0).
        # Tracks prepared strategies via a WeakSet so sharing one instance
        # across work items invokes prepare() exactly once.
        self._strategy_lifecycle: StrategyLifecycle[TOutput] = StrategyLifecycle()
        # Back-compat aliases used by existing private methods and tests.
        self._prepared_strategies = self._strategy_lifecycle._prepared
        self._strategy_lock = self._strategy_lifecycle._lock

        # Proactive rate limiting (prevents hitting rate limits)
        if config.max_requests_per_minute:
            from aiolimiter import AsyncLimiter

            # aiolimiter doesn't have explicit burst_size - it uses max_rate as burst capacity
            # To support burst_size, we'd need to use max_rate + burst_size
            # For now, we use max_rate directly (no additional burst)
            self._proactive_rate_limiter: AsyncLimiter | None = AsyncLimiter(
                max_rate=config.max_requests_per_minute,
                time_period=60,  # per minute
            )
        else:
            self._proactive_rate_limiter = None

        # Centralized token-usage extraction across all exception shapes.
        self._token_extractor = TokenExtractor()

    @property
    def _strategies_cleaned_up(self) -> bool:
        return self._strategy_lifecycle._cleaned_up

    @_strategies_cleaned_up.setter
    def _strategies_cleaned_up(self, value: bool) -> None:
        self._strategy_lifecycle._cleaned_up = value

    # Back-compat attribute accessors for tests and subclasses that read
    # the rate-limit coordinator's state directly.

    @property
    def _in_cooldown(self) -> bool:
        return self._rate_limit_coord._in_cooldown

    @property
    def _cooldown_generation(self) -> int:
        return self._rate_limit_coord._cooldown_generation

    @property
    def _cooldown_complete_generation(self) -> int:
        return self._rate_limit_coord._cooldown_complete_generation

    @property
    def _consecutive_rate_limits(self) -> int:
        return self._rate_limit_coord._consecutive_rate_limits

    @property
    def _slow_start_active(self) -> bool:
        return self._rate_limit_coord._slow_start_active

    @property
    def _items_since_resume(self) -> int:
        return self._rate_limit_coord._items_since_resume

    async def _cleanup_strategies(self) -> None:
        """Delegate to StrategyLifecycle (see _internal/strategy_lifecycle.py)."""
        await self._strategy_lifecycle.cleanup_all()

    async def _ensure_strategy_prepared(self, strategy: LLMCallStrategy[TOutput]) -> None:
        """Delegate to StrategyLifecycle.ensure_prepared."""
        await self._strategy_lifecycle.ensure_prepared(strategy)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> bool:
        """
        Context manager exit - ensures cleanup of strategies and resources.

        Calls cleanup() on all prepared strategies, then delegates to parent cleanup.

        Args:
            exc_type: Exception type (if any exception occurred)
            exc_val: Exception value (if any exception occurred)
            exc_tb: Exception traceback (if any exception occurred)

        Returns:
            False to indicate exceptions should not be suppressed
        """
        await self._cleanup_strategies()

        # Call parent cleanup to handle workers and queue
        await self.cleanup()
        return False  # Don't suppress exceptions

    async def get_stats(self) -> dict:
        """
        Get processor statistics (thread-safe).

        Returns:
            Dictionary containing processing statistics including:
            - processed: Number of items processed
            - succeeded: Number of successful items
            - failed: Number of failed items
            - rate_limit_count: Number of rate limit errors encountered
            - error_counts: Dictionary of error types and their counts
            - total: Total number of items queued
            - start_time: Timestamp when processing started
        """
        async with self._stats_lock:
            return self._stats.copy()

    async def add_work(self, work_item: LLMWorkItem[TInput, TOutput, TContext]) -> None:
        """Queue a work item, recording its strategy's classifier recommendation.

        Extends the base queueing with the bookkeeping needed to auto-select an
        error classifier at batch start when the caller didn't pass one. A
        no-op recommendation (custom strategies returning ``None``) is ignored.
        """
        await super().add_work(work_item)
        if self._user_supplied_classifier or self._classifier_resolved:
            return
        try:
            recommended = work_item.strategy.recommended_error_classifier()
        except Exception:
            # A buggy override must never break queueing — just skip the hint.
            recommended = None
        if recommended is not None:
            self._recommended_classifiers.append(recommended)
            # Streaming mode has no "batch start" barrier and workers are
            # already running, so resolve eagerly from the first recommendation
            # rather than waiting to collect every item's (we can't).
            if self._streaming:
                self._resolve_error_classifier()

    def _resolve_error_classifier(self) -> None:
        """Pick an error classifier from the queued strategies' recommendations.

        Runs once, at batch start, only when the caller didn't pass an explicit
        ``error_classifier``. If every recommending strategy agrees, that
        classifier is used; if they disagree (mixed providers), we keep the
        :class:`DefaultErrorClassifier` and warn. No recommendations → keep the
        default silently (debug log).
        """
        if self._classifier_resolved:
            return
        self._classifier_resolved = True

        recommendations = self._recommended_classifiers
        if not recommendations:
            logger.debug(
                "No work-item strategy recommended an error classifier; using %s.",
                type(self.error_classifier).__name__,
            )
            return

        distinct_types = {type(r) for r in recommendations}
        if len(distinct_types) == 1:
            self.error_classifier = recommendations[0]
            logger.debug(
                "Auto-selected %s from work-item strategies.",
                type(self.error_classifier).__name__,
            )
        else:
            names = ", ".join(sorted(t.__name__ for t in distinct_types))
            logger.warning(
                "[WARN]Work items recommend mixed error classifiers (%s); falling back "
                "to %s. Pass error_classifier=... to ParallelBatchProcessor to choose "
                "one explicitly.",
                names,
                type(self.error_classifier).__name__,
            )

    async def _on_batch_started(self) -> None:
        """Emit batch start event with initial stats snapshot."""
        # Resolve the auto-selected error classifier before any worker runs.
        self._resolve_error_classifier()

        async with self._stats_lock:
            stats_snapshot = self._stats.copy()

        await self._emit_event(
            ProcessingEvent.BATCH_STARTED,
            {
                "total": stats_snapshot["total"],
                "max_workers": self.max_workers,
                "start_time": stats_snapshot["start_time"],
            },
        )

    async def _on_batch_completed(self) -> None:
        """Emit batch completion event with final stats snapshot."""
        async with self._stats_lock:
            stats_snapshot = self._stats.copy()

        duration = 0.0
        if stats_snapshot.get("start_time"):
            duration = time.time() - float(stats_snapshot["start_time"])

        await self._emit_event(
            ProcessingEvent.BATCH_COMPLETED,
            {
                "processed": stats_snapshot["processed"],
                "succeeded": stats_snapshot["succeeded"],
                "failed": stats_snapshot["failed"],
                "total": stats_snapshot["total"],
                "total_tokens": stats_snapshot["total_tokens"],
                "cached_input_tokens": stats_snapshot.get("cached_input_tokens", 0),
                "duration": duration,
            },
        )

    async def _emit_event(self, event: ProcessingEvent, data: dict | None = None) -> None:
        """Delegate to EventDispatcher (see _internal/event_dispatcher.py)."""
        await self._events.emit(event, data)

    async def _run_middlewares_before(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMWorkItem[TInput, TOutput, TContext] | None:
        return await self._events.run_before(work_item)

    async def _run_middlewares_after(
        self, result: WorkItemResult[TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        return await self._events.run_after(result)

    async def _run_middlewares_on_error(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], error: Exception
    ) -> WorkItemResult[TOutput, TContext] | None:
        return await self._events.run_on_error(work_item, error)

    async def _worker(self, worker_id: int):
        """Worker coroutine that processes items from the queue."""
        # Per-worker / per-item logs are DEBUG with lazy %-formatting: at 100
        # workers and thousands of items these are pure hot-path overhead, and
        # the f-string would be built even when DEBUG is disabled. INFO is
        # reserved for batch start/end and the periodic progress line.
        logger.debug("[Worker %s] started and waiting for work", worker_id)
        if self._events.observers:
            await self._emit_event(ProcessingEvent.WORKER_STARTED, {"worker_id": worker_id})

        while True:
            try:
                work_item = await self._queue.get()
            except asyncio.CancelledError:
                logger.debug("[Worker %s] cancelled while waiting for work", worker_id)
                raise

            if work_item is None:  # Sentinel value
                self._queue.task_done()
                logger.debug("[Worker %s] finished (no more work)", worker_id)
                if self._events.observers:
                    await self._emit_event(ProcessingEvent.WORKER_STOPPED, {"worker_id": worker_id})
                return

            logger.debug("[Worker %s] Picked up %s from queue", worker_id, work_item.item_id)

            # Wait if we're in rate limit cooldown, then apply slow-start ramp.
            await self._rate_limit_coord.wait_if_paused()
            delay = await self._rate_limit_coord.apply_slow_start()
            if delay > 0:
                await asyncio.sleep(delay)

            # Process the item
            try:
                result = await self._process_item_with_retries(work_item, worker_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # All retries exhausted or unhandled exception
                # Create a failed result so the item is recorded

                # Extract token usage from exception if available
                failed_tokens = {}
                if hasattr(e, "__dict__") and "_failed_token_usage" in e.__dict__:
                    failed_tokens = e.__dict__["_failed_token_usage"]

                token_msg = ""
                if failed_tokens.get("total_tokens", 0) > 0:
                    token_msg = (
                        f" (consumed {failed_tokens['total_tokens']} tokens across all attempts)"
                    )

                logger.error(
                    f"[FAIL]Worker {worker_id} failed to process {work_item.item_id} after all retries: "
                    f"{type(e).__name__}: {str(e)[:ERROR_MESSAGE_MAX_LENGTH]}{token_msg}"
                )

                # Try middleware error handlers
                middleware_result = await self._run_middlewares_on_error(work_item, e)
                if middleware_result is not None:
                    result = middleware_result
                else:
                    result = WorkItemResult(
                        item_id=work_item.item_id,
                        success=False,
                        error=f"{type(e).__name__}: {str(e)[:ERROR_MESSAGE_MAX_LENGTH]}",
                        context=work_item.context,
                        token_usage=cast(TokenUsage, failed_tokens),
                    )

                # Emit ITEM_FAILED here too. Items that exhaust retries reach
                # this fallback (the exception propagates out of
                # _process_item_with_retries) rather than the non-retryable
                # branch in _handle_execution_error, so without this emit a
                # MetricsObserver would undercount failures vs BatchResult.
                if not result.success:
                    await self._emit_event(
                        ProcessingEvent.ITEM_FAILED,
                        {"item_id": work_item.item_id, "error_type": type(e).__name__},
                    )
                # Fall through to store result and call task_done()

            # Store the result. In streaming mode publish it to the result
            # stream (constant memory — we don't accumulate _results); in batch
            # mode accumulate for process_all()'s BatchResult.
            if self._streaming:
                assert self._result_stream is not None  # set by start()
                await self._result_stream.put(result)
            else:
                async with self._results_lock:
                    self._results.append(result)

            # Update stats (thread-safe). A single lock acquisition handles the
            # counters, the progress-callback decision, AND the periodic-log
            # snapshot — stats don't change between here and the log below, so
            # there's no reason to take _stats_lock twice per item.
            should_call_progress = False
            completed = 0
            total = 0
            current_item = ""
            should_log = False
            stats_snapshot: dict | None = None

            async with self._stats_lock:
                self._stats.processed += 1
                if result.success:
                    self._stats.succeeded += 1
                else:
                    self._stats.failed += 1
                    if result.error:
                        error_type = result.error.split(":")[0]
                        self._stats.error_counts[error_type] = (
                            self._stats.error_counts.get(error_type, 0) + 1
                        )

                # Track token usage in real-time
                if result.token_usage:
                    self._stats.total_input_tokens += result.token_usage.get("input_tokens", 0)
                    self._stats.total_output_tokens += result.token_usage.get("output_tokens", 0)
                    self._stats.total_tokens += result.token_usage.get("total_tokens", 0)
                    self._stats.cached_input_tokens += result.token_usage.get(
                        "cached_input_tokens", 0
                    )

                # Both the progress callback and the periodic progress log fire
                # on the same progress_interval boundary — decide both here.
                should_log = self._stats.processed % self.config.progress_interval == 0
                if should_log:
                    stats_snapshot = self._stats.copy()
                    if self.progress_callback:
                        should_call_progress = True
                        completed = self._stats.processed
                        total = self._stats.total
                        current_item = work_item.item_id

            # Invoke progress callback outside of lock
            if should_call_progress:
                await self._run_progress_callback(completed, total, current_item)

            # Run post-processor for both success AND failure
            # Note: Post-processors should check result.success and handle accordingly
            # Most post-processors return early for failures, but some may want to
            # save failed items (e.g., dedupe_authors saves failed clusters as singletons)
            # The timeout is enforced inside _run_post_processor from config — a
            # single source of truth rather than a hardcoded inner cap. When
            # concurrent_post_processing is enabled, run it as a tracked
            # background task so a slow post-processor doesn't cap throughput.
            if self.config.concurrent_post_processing:
                self._spawn_post_processor(result, self.config.post_processor_timeout)
            else:
                await self._run_post_processor(result, timeout=self.config.post_processor_timeout)

            self._queue.task_done()

            # Per-item completion log: DEBUG + lazy %-formatting (hot path).
            logger.debug(
                "[Worker %s] Completed %s (%s)",
                worker_id,
                work_item.item_id,
                "success" if result.success else "failed",
            )

            if should_log and stats_snapshot is not None:
                elapsed = time.time() - stats_snapshot["start_time"]
                calls_per_sec = stats_snapshot["processed"] / elapsed if elapsed > 0 else 0

                error_breakdown = ""
                if stats_snapshot["error_counts"]:
                    error_strs = [
                        f"{err}: {count}" for err, count in stats_snapshot["error_counts"].items()
                    ]
                    error_breakdown = f" | Errors: {', '.join(error_strs)}"

                # Token summary
                token_summary = ""
                if stats_snapshot["total_tokens"] > 0:
                    cached_info = ""
                    if stats_snapshot.get("cached_input_tokens", 0) > 0:
                        cached_info = f", {stats_snapshot['cached_input_tokens']:,} cached"
                    token_summary = (
                        f" | Tokens: {stats_snapshot['total_tokens']:,} "
                        f"({stats_snapshot['total_input_tokens']:,} in, "
                        f"{stats_snapshot['total_output_tokens']:,} out{cached_info})"
                    )

                logger.info(
                    f"[INFO]Progress: {stats_snapshot['processed']}/{stats_snapshot['total']} "
                    f"({stats_snapshot['processed'] / stats_snapshot['total'] * 100:.1f}%) | "
                    f"Succeeded: {stats_snapshot['succeeded']}, Failed: {stats_snapshot['failed']}"
                    f"{error_breakdown} | {calls_per_sec:.2f} calls/sec{token_summary}"
                )

    async def _handle_rate_limit(
        self,
        worker_id: int,
        observed_generation: int | None = None,
        suggested_wait: float | None = None,
    ) -> None:
        """Delegate to RateLimitCoordinator."""
        await self._rate_limit_coord.handle_rate_limit(
            worker_id, observed_generation, suggested_wait
        )

    async def _finalize_cooldown(self, start_time: float, error: Exception | None) -> None:
        """Delegate to RateLimitCoordinator._finalize_cooldown (kept for subclass overrides)."""
        await self._rate_limit_coord._finalize_cooldown(start_time, error)

    def _should_retry_error(self, exception: Exception) -> bool:
        """Determine if error should be retried using error classifier."""
        error_info = self.error_classifier.classify(exception)
        return error_info.is_retryable

    def _extract_token_usage(self, exception: Exception) -> dict[str, int]:
        """Extract token usage from a failed LLM call exception.

        Thin wrapper around :class:`TokenExtractor`; retained as a method
        for subclass override points. See ``token_extractor.py`` for the
        three extraction strategies.
        """
        return self._token_extractor.extract_from_exception(exception)

    async def _process_item_with_retries(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], worker_id: int
    ) -> WorkItemResult[TOutput, TContext]:
        """Wrapper that applies retry logic and strategy lifecycle."""
        # Track cumulative token usage across all failed attempts
        cumulative_failed_tokens = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }

        # Get the strategy
        strategy = self._get_strategy(work_item)

        # Create retry state for this work item (v0.3.0)
        # This state persists across all retry attempts for multi-stage strategies
        retry_state = RetryState()

        # Ensure strategy is prepared (framework ensures this is called only once per unique strategy instance)
        # (v0.4.0: cleanup now happens in __aexit__, not per-item)
        try:
            await self._ensure_strategy_prepared(strategy)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[FAIL]Strategy prepare() failed for {work_item.item_id}: {e}")
            raise

        # Two independent counters: `attempt` is the *logical* attempt number
        # (what execute()/on_error() see, and what model-escalation strategies
        # key off). `rate_limit_retries` bounds throttling retries separately.
        # Rate-limit errors retry the SAME logical attempt — they don't consume
        # the max_attempts budget — so a busy endpoint can't trigger escalation.
        attempt = 1
        rate_limit_retries = 0
        max_attempts = self.config.retry.max_attempts
        max_rate_limit_retries = self.config.retry.max_rate_limit_retries

        while True:
            try:
                result = await self._process_item(
                    work_item,
                    worker_id,
                    attempt_number=attempt,
                    strategy=strategy,
                    retry_state=retry_state,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Accumulate token usage across every attempt (including
                # rate-limit retries) so users see the true cost of a failure.
                attempt_tokens = self._extract_token_usage(e)
                self._token_extractor.accumulate(cumulative_failed_tokens, attempt_tokens)

                error_info = self.error_classifier.classify(e)
                error_snippet = str(e)[:ERROR_MESSAGE_MAX_LENGTH]
                error_type = type(e).__name__

                if not error_info.is_retryable:
                    # Surface an operator hint (e.g. a 402 insufficient-balance
                    # remediation) at WARNING so a misconfiguration doesn't read
                    # like a generic API/code bug; otherwise a quiet debug line.
                    if error_info.hint:
                        logger.warning(
                            f"[FAIL]Non-retryable error for {work_item.item_id}: {error_info.hint}"
                        )
                    else:
                        logger.debug(f"Error not retryable: {error_type}")
                    self._attach_failed_tokens(e, cumulative_failed_tokens)
                    raise

                if error_info.is_rate_limit:
                    # Rate limits do NOT consume the max_attempts budget — they're
                    # "wait and try again", not a failed attempt. The coordinated
                    # cooldown already ran inside _handle_rate_limit(), so we retry
                    # the SAME logical attempt immediately. A separate counter
                    # bounds this so a permanently-throttled endpoint can't hang.
                    rate_limit_retries += 1
                    if rate_limit_retries > max_rate_limit_retries:
                        token_summary = self._cumulative_token_summary(cumulative_failed_tokens)
                        logger.error(
                            f"[FAIL]EXCEEDED {max_rate_limit_retries} RATE-LIMIT RETRIES "
                            f"for {work_item.item_id}:\n"
                            f"  Last error type: {error_type}\n"
                            f"  Last error message: "
                            f"{str(e)[:ERROR_MESSAGE_DETAILED_LENGTH]}{token_summary}"
                        )
                        exhausted = RateLimitRetriesExceeded(
                            f"Exceeded {max_rate_limit_retries} rate-limit retries for "
                            f"{work_item.item_id} (last error: {error_type}: {error_snippet})",
                            item_id=work_item.item_id,
                            rate_limit_retries=rate_limit_retries,
                        )
                        exhausted.__dict__["_failed_token_usage"] = cumulative_failed_tokens
                        raise exhausted from e
                    logger.warning(
                        f"[WARN]Rate-limit retry {rate_limit_retries} for {work_item.item_id} "
                        f"(attempt {attempt}/{max_attempts} budget unchanged): "
                        f"{error_type} - {error_snippet}. "
                        f"Retrying immediately (cooldown already applied)..."
                    )
                    continue  # attempt unchanged; no extra backoff (cooldown done)

                # --- Non-rate-limit retryable error: consumes the budget. ---
                if attempt >= max_attempts:
                    token_summary = self._cumulative_token_summary(cumulative_failed_tokens)
                    logger.error(
                        f"[FAIL]ALL {max_attempts} ATTEMPTS EXHAUSTED "
                        f"for {work_item.item_id}:\n"
                        f"  Final error type: {error_type}\n"
                        f"  Final error message: "
                        f"{str(e)[:ERROR_MESSAGE_DETAILED_LENGTH]}{token_summary}"
                    )
                    self._attach_failed_tokens(e, cumulative_failed_tokens)
                    raise

                # Validation errors retry immediately — the strategy adjusts on
                # retry; other transient errors get exponential backoff keyed off
                # the logical attempt number. (PydanticAI wraps validation errors
                # in UnexpectedModelBehavior.)
                error_msg_for_check = str(e)
                is_validation_error = (
                    "validation" in error_type.lower()
                    or "parse" in error_type.lower()
                    or "unexpectedmodelbehavior" in error_type.lower()
                    or "result validation" in error_msg_for_check.lower()
                    or error_info.error_category == "validation_error"
                )

                if is_validation_error:
                    wait_time = 0.0
                else:
                    wait_time = min(
                        self.config.retry.initial_wait
                        * (self.config.retry.exponential_base ** (attempt - 1)),
                        self.config.retry.max_wait,
                    )
                    if self.config.retry.jitter:
                        import random

                        # Jitter to 50-100% of the computed wait to spread retries.
                        wait_time = wait_time * (0.5 + random.random() * 0.5)

                retry_desc = "immediately" if wait_time == 0 else f"in {wait_time:.1f}s"
                logger.warning(
                    f"[WARN]Attempt {attempt}/{max_attempts} failed for "
                    f"{work_item.item_id}: {error_type} - {error_snippet}. "
                    f"Retrying {retry_desc}..."
                )

                attempt += 1
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            else:
                # _process_item returned without raising (success, or a result
                # produced by middleware / non-retryable handling). Fold in the
                # tokens consumed by any earlier failed attempts so cost
                # reporting is aggregated across retries, not just the final
                # attempt (see README "aggregated across retries").
                self._merge_failed_tokens(result, cumulative_failed_tokens)
                return result

    @staticmethod
    def _attach_failed_tokens(exception: Exception, tokens: dict[str, int]) -> None:
        """Stamp cumulative failed-attempt tokens onto an exception for the worker
        to surface in the failed ``WorkItemResult``. No-op if the exception has
        no writable ``__dict__``."""
        if hasattr(exception, "__dict__"):
            exception.__dict__["_failed_token_usage"] = tokens

    @staticmethod
    def _cumulative_token_summary(tokens: dict[str, int]) -> str:
        """Format the cross-attempt token total for final-failure logs (or '')."""
        if tokens.get("total_tokens", 0) > 0:
            return f"\n  Total tokens consumed across all attempts: {tokens['total_tokens']}"
        return ""

    @staticmethod
    def _merge_failed_tokens(
        result: WorkItemResult[TOutput, TContext], failed_tokens: dict[str, int]
    ) -> None:
        """Add tokens consumed by prior failed attempts into ``result.token_usage``.

        Mutates ``result.token_usage`` in place. Keys that would stay zero and
        weren't already present are left out so a clean success keeps its tidy
        ``{input, output, total}`` shape.
        """
        existing = cast("dict[str, int]", result.token_usage)
        usage: dict[str, int] = {}
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
            combined = existing.get(key, 0) + failed_tokens.get(key, 0)
            if combined or key in existing:
                usage[key] = combined
        result.token_usage = cast(TokenUsage, usage)

    def _get_strategy(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMCallStrategy[TOutput]:
        """Get the LLM call strategy for this work item."""
        return work_item.strategy

    async def _process_item(  # type: ignore[override]  # ty:ignore[invalid-method-override]
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int = 1,
        strategy: LLMCallStrategy[TOutput] | None = None,
        retry_state: RetryState | None = None,
    ) -> WorkItemResult[TOutput, TContext]:
        """Process a single work item using the provided strategy."""
        start_time = time.time()

        # Store original item_id before middleware might return None
        original_item_id = work_item.item_id

        # Skip building the event payload entirely when nobody is listening.
        if self._events.observers:
            await self._emit_event(
                ProcessingEvent.ITEM_STARTED,
                {"item_id": original_item_id, "worker_id": worker_id},
            )

        try:
            # Run before middlewares
            processed_item = await self._run_middlewares_before(work_item)
            if processed_item is None:
                logger.debug("Skipping %s (filtered by middleware)", original_item_id)
                return WorkItemResult(
                    item_id=original_item_id,
                    success=False,
                    error="Skipped by middleware",
                    context=work_item.context,
                )
            work_item = processed_item

            # Execute the strategy
            if attempt_number > 1:
                logger.debug(
                    "[Worker %s] Retry attempt %s for %s",
                    worker_id,
                    attempt_number,
                    work_item.item_id,
                )
            logger.debug(
                "[STRATEGY] Starting strategy.execute() for %s (attempt %s, timeout=%ss)",
                work_item.item_id,
                attempt_number,
                self.config.timeout_per_item,
            )
            llm_start_time = time.time()

            # Ensure strategy is not None (it shouldn't be since we always pass it)
            if strategy is None:
                raise RuntimeError("Strategy is None in _process_item - this should not happen")

            try:
                # Proactive rate limiting: acquire token before making request
                if self._proactive_rate_limiter:
                    logger.debug(
                        "[RATE-LIMIT] Acquiring token for %s (attempt %s)",
                        work_item.item_id,
                        attempt_number,
                    )
                    await self._proactive_rate_limiter.acquire()
                    logger.debug(
                        "[RATE-LIMIT] Token acquired for %s (attempt %s)",
                        work_item.item_id,
                        attempt_number,
                    )

                # Dry-run mode: use strategy's dry_run method instead of making API call
                if self.config.dry_run:
                    logger.debug("[DRY-RUN] Skipping API call for %s", work_item.item_id)
                    # Delegate to strategy's dry_run method for mock output.
                    # dry_run is fixed at 2-tuple — no metadata for mock data.
                    output, token_usage = await strategy.dry_run(work_item.prompt)
                    response_metadata = None
                else:
                    # Call strategy.execute() with prompt, attempt number, timeout, and retry state (v0.3.0)
                    # Wrap in asyncio.wait_for to enforce timeout at framework level.
                    # _unpack_strategy_result accepts both legacy 2-tuple and the
                    # current 3-tuple (output, tokens, metadata) shape (v0.10.0).
                    raw_result = await asyncio.wait_for(
                        strategy.execute(
                            work_item.prompt,
                            attempt_number,
                            self.config.timeout_per_item,
                            retry_state,
                        ),
                        timeout=self.config.timeout_per_item,
                    )
                    output, token_usage, response_metadata = _unpack_strategy_result(raw_result)
            except (TimeoutError, asyncio.TimeoutError) as timeout_exc:
                elapsed = time.time() - llm_start_time
                logger.error(
                    f"⏱ FRAMEWORK TIMEOUT for {work_item.item_id} after {elapsed:.1f}s "
                    f"(limit: {self.config.timeout_per_item}s, attempt {attempt_number}). "
                    f"Consider increasing config.timeout_per_item if this error persists."
                )
                # Wrap in FrameworkTimeoutError to differentiate from API timeouts
                framework_timeout = FrameworkTimeoutError(
                    f"Framework timeout after {elapsed:.1f}s (limit: {self.config.timeout_per_item}s)",
                    item_id=work_item.item_id,
                    elapsed=elapsed,
                    timeout_limit=self.config.timeout_per_item,
                )
                # This except also catches a TimeoutError *subclass the strategy
                # raised itself* (not just asyncio.wait_for's own fresh timeout),
                # and that one can carry token usage. Copy it across so failed-
                # attempt tokens aren't lost — the token extractor only reads the
                # top exception's __dict__, not its __cause__'s.
                if (
                    hasattr(timeout_exc, "__dict__")
                    and "_failed_token_usage" in timeout_exc.__dict__
                ):
                    framework_timeout.__dict__["_failed_token_usage"] = timeout_exc.__dict__[
                        "_failed_token_usage"
                    ]
                raise framework_timeout from timeout_exc

            llm_duration = time.time() - llm_start_time
            logger.debug(
                "[STRATEGY] Completed strategy.execute() for %s in %.1fs",
                work_item.item_id,
                llm_duration,
            )

            # Log success after previous failures
            if attempt_number > 1:
                logger.debug(
                    "SUCCESS on attempt %s for %s (after %s failure(s), took %.1fs)",
                    attempt_number,
                    work_item.item_id,
                    attempt_number - 1,
                    llm_duration,
                )

            # Log first few results for debugging (lazy: the big banner string is
            # only built when DEBUG is actually enabled).
            if self._stats.succeeded < 3:
                logger.debug(
                    "\n%s\nRESULT for %s:\n%s\n%s\n%s",
                    "=" * 80,
                    work_item.item_id,
                    "=" * 80,
                    output,
                    "=" * 80,
                )

            # Create result
            work_result = WorkItemResult(
                item_id=work_item.item_id,
                success=True,
                output=output,
                context=work_item.context,
                token_usage=token_usage,
                metadata=response_metadata,
            )

            # Run after middlewares
            work_result = await self._run_middlewares_after(work_result)

            # Skip the duration calc + payload dict when nobody is observing.
            if self._events.observers:
                duration = time.time() - start_time
                await self._emit_event(
                    ProcessingEvent.ITEM_COMPLETED,
                    {
                        "item_id": work_item.item_id,
                        "duration": duration,
                        "tokens": token_usage.get("total_tokens", 0),
                    },
                )

            # Reset consecutive rate limit counter on success (thread-safe).
            await self._rate_limit_coord.on_item_success()

            return work_result

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Notify strategy about the error before handling it
            # This allows strategy to adjust behavior for next retry (v0.3.0: now includes retry_state)
            if strategy is not None:  # Type guard for mypy
                try:
                    await strategy.on_error(e, attempt_number, retry_state)
                except asyncio.CancelledError:
                    raise
                except Exception as callback_error:
                    # Log but don't fail if on_error callback has bugs
                    logger.warning(
                        f"Strategy.on_error callback failed for {work_item.item_id}: {callback_error}"
                    )

            # Delegate error handling to separate method
            return await self._handle_execution_error(e, work_item, worker_id, attempt_number)

    async def _handle_execution_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
    ) -> WorkItemResult[TOutput, TContext]:
        """
        Handle exceptions from LLM execution.

        This method classifies errors, extracts token usage, handles rate limits,
        and determines whether errors should be retried or treated as permanent failures.

        Args:
            exception: The exception that was raised during execution
            work_item: The work item being processed
            worker_id: ID of the worker processing this item
            attempt_number: Current attempt number (for logging)

        Returns:
            WorkItemResult for permanent failures

        Raises:
            RateLimitException: If rate limit detected (for re-queueing)
            Exception: If error is retryable (for retry logic to handle)
        """
        # Try to extract token usage from failed LLM calls using robust extraction
        # Even if validation fails, the LLM consumed tokens
        failed_token_usage = self._extract_token_usage(exception)
        if failed_token_usage and failed_token_usage.get("total_tokens", 0) > 0:
            logger.debug(
                f"Extracted token usage from failed attempt for {work_item.item_id}: "
                f"{failed_token_usage['total_tokens']} tokens"
            )

        # Classify the error
        error_info = self.error_classifier.classify(exception)

        # Check if it's a rate limit
        if error_info.is_rate_limit:
            # Update stats (thread-safe)
            async with self._stats_lock:
                self._stats.rate_limit_count += 1

            await self._emit_event(
                ProcessingEvent.RATE_LIMIT_HIT,
                {"item_id": work_item.item_id, "worker_id": worker_id},
            )

            # Handle rate limit (cooldown) - this will pause all workers.
            # Pass the classifier's suggested_wait (e.g. a parsed Retry-After)
            # as a floor on the cooldown duration.
            observed_generation = self._rate_limit_coord.current_generation
            await self._handle_rate_limit(worker_id, observed_generation, error_info.suggested_wait)

            # Re-raise the original exception to trigger retry logic
            # The retry loop will increment attempt and try again after cooldown
            raise

        # If error is retryable, re-raise to trigger retry in _process_item_with_retries
        # Note: Cache invalidation is automatic because retries use different temperatures,
        # which creates different cache keys and bypasses any cached bad responses
        if error_info.is_retryable:
            self._log_retryable_error(exception, work_item, attempt_number, failed_token_usage)
            raise

        # Try middleware error handlers
        middleware_result = await self._run_middlewares_on_error(work_item, exception)
        if middleware_result is not None:
            return middleware_result

        # Log non-retryable error with full details
        error_name = type(exception).__name__
        error_msg = str(exception)

        token_summary = ""
        if failed_token_usage:
            token_summary = f"\n  Tokens consumed: {failed_token_usage.get('total_tokens', 0)}"

        logger.error(
            f"[FAIL]PERMANENT FAILURE for {work_item.item_id}:\n"
            f"  Error type: {error_name}\n"
            f"  Error message: {error_msg[:ERROR_MESSAGE_DETAILED_LENGTH]}\n"
            f"  This error will NOT be retried (not retryable){token_summary}"
        )

        await self._emit_event(
            ProcessingEvent.ITEM_FAILED,
            {"item_id": work_item.item_id, "error_type": error_name},
        )

        return WorkItemResult(
            item_id=work_item.item_id,
            success=False,
            error=f"{error_name}: {error_msg[:ERROR_MESSAGE_DETAILED_LENGTH]}",
            context=work_item.context,
            token_usage=cast(TokenUsage, failed_token_usage),
        )

    def _log_retryable_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        attempt_number: int,
        failed_token_usage: dict[str, int],
    ) -> None:
        """Delegate to :func:`_internal.error_logging.log_retryable_error`."""
        log_retryable_error(exception, work_item.item_id, attempt_number, failed_token_usage)

    def _log_validation_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        attempt_number: int,
        token_msg: str,
    ) -> None:
        """Delegate to :func:`_internal.error_logging.log_validation_error`."""
        log_validation_error(exception, work_item.item_id, attempt_number, token_msg)

    async def shutdown(self):
        """Clean up resources: flush observers and cancel pending tasks."""
        await self._cleanup_strategies()
        await self.cleanup()
