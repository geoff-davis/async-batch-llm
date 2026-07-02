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
    RateLimitStrategy,
)
from .token_extractor import TokenExtractor

logger = logging.getLogger(__name__)

# Maximum length for error messages in logs / result payloads.
# Longer errors get truncated to keep logs readable.
ERROR_MESSAGE_MAX_LENGTH = 200
# Larger truncation used for detailed diagnostic logs (final attempt, validation traces).
ERROR_MESSAGE_DETAILED_LENGTH = 500


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

        # Handle backward compatibility. ProcessorConfig validates itself in
        # __post_init__, so no extra validate() call is needed here (it would
        # just re-emit the cross-field warnings a second time).
        if config is None:
            from .core import RateLimitConfig

            config = ProcessorConfig(
                max_workers=max_workers or 5,
                timeout_per_item=timeout_per_item or 120.0,
                rate_limit=RateLimitConfig(cooldown_seconds=rate_limit_cooldown or 300.0),
            )
        elif (
            max_workers is not None
            or timeout_per_item is not None
            or rate_limit_cooldown is not None
        ):
            # Never mutate the caller's config — it may be shared across
            # processors. Copy (including the nested rate_limit when it's
            # being overridden) before applying deprecated overrides.
            import dataclasses

            rate_limit = config.rate_limit
            if rate_limit_cooldown is not None:
                rate_limit = dataclasses.replace(rate_limit, cooldown_seconds=rate_limit_cooldown)
            config = dataclasses.replace(
                config,
                max_workers=max_workers if max_workers is not None else config.max_workers,
                timeout_per_item=timeout_per_item
                if timeout_per_item is not None
                else config.timeout_per_item,
                rate_limit=rate_limit,
            )

        super().__init__(
            config.max_workers,
            post_processor,
            max_queue_size=config.max_queue_size,
            progress_callback=progress_callback,
            progress_callback_timeout=config.progress_callback_timeout,
        )
        self.config = config

        # Set up strategies
        self.error_classifier = error_classifier or DefaultErrorClassifier()
        self.rate_limit_strategy = rate_limit_strategy or ExponentialBackoffStrategy(
            initial_cooldown=config.rate_limit.cooldown_seconds,
            max_cooldown=config.rate_limit.max_cooldown_seconds,
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
        # reach into these attributes directly. (_rate_limit_event is safe as
        # a plain alias because the coordinator mutates it in place;
        # _current_generation_event is REPLACED on every cooldown, so it's
        # exposed as a delegating property below instead.)
        self._rate_limit_event = self._rate_limit_coord._rate_limit_event
        self._rate_limit_lock = self._rate_limit_coord._lock

        # Thread safety locks (stats + results remain processor-owned).
        self._stats_lock = asyncio.Lock()
        self._results_lock = asyncio.Lock()

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
    def _current_generation_event(self) -> asyncio.Event:
        # Delegating property (not a plain alias set in __init__): the
        # coordinator replaces this event with a fresh one on every cooldown,
        # so a snapshot taken at construction would go permanently stale
        # after the first cooldown.
        return self._rate_limit_coord._current_generation_event

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
        try:
            await self._cleanup_strategies()
        finally:
            # Parent cleanup (cancel workers, drain queue, cancel progress
            # tasks) must run even if a strategy's cleanup raised or was
            # cancelled mid-await (e.g. Ctrl-C during __aexit__) — otherwise
            # workers stay alive and progress tasks leak.
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

    async def _on_batch_started(self) -> None:
        """Emit batch start event with initial stats snapshot."""
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
        logger.info(f"[OK]Worker {worker_id} started and waiting for work")
        await self._emit_event(ProcessingEvent.WORKER_STARTED, {"worker_id": worker_id})

        while True:
            try:
                work_item = await self._queue.get()
            except asyncio.CancelledError:
                logger.info(f"[WARN]Worker {worker_id} cancelled while waiting for work")
                raise

            if work_item is None:  # Sentinel value
                self._queue.task_done()
                logger.info(f"[OK]Worker {worker_id} finished (no more work)")
                await self._emit_event(ProcessingEvent.WORKER_STOPPED, {"worker_id": worker_id})
                return

            try:
                await self._handle_work_item(work_item, worker_id)
            finally:
                # Guarantee task_done() even if an unexpected error escapes:
                # a missed task_done() would leave queue.join() waiting forever.
                self._queue.task_done()

    async def _handle_work_item(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], worker_id: int
    ) -> None:
        """Process one dequeued item end-to-end: retries, result recording,
        stats, progress callback, and post-processing."""
        logger.info(f"[INFO][Worker {worker_id}] Picked up {work_item.item_id} from queue")

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

            # Emit ITEM_FAILED for the retries-exhausted path. (The
            # non-retryable path emits inside _handle_execution_error and
            # returns a result instead of raising, so there's no double
            # emit here.) Skip it if a middleware recovered the item.
            if not result.success:
                await self._emit_event(
                    ProcessingEvent.ITEM_FAILED,
                    {"item_id": work_item.item_id, "error_type": type(e).__name__},
                )
            # Fall through to store result

        # Store result (thread-safe)
        async with self._results_lock:
            self._results.append(result)

        # Update stats (thread-safe)
        should_call_progress = False
        completed = 0
        total = 0
        current_item = ""

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
                self._stats.cached_input_tokens += result.token_usage.get("cached_input_tokens", 0)

            # Check if we should call progress callback (based on progress_interval)
            if (
                self.progress_callback
                and self._stats.processed % self.config.progress_interval == 0
            ):
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
        try:
            await asyncio.wait_for(
                self._run_post_processor(result),
                timeout=self.config.post_processor_timeout,
            )
        except (TimeoutError, asyncio.TimeoutError):  # distinct classes on Python 3.10
            logger.error(f"⏱ Post-processor exceeded timeout for {work_item.item_id}")

        # Log completion
        status = "[OK]" if result.success else "[FAIL]"
        outcome = "success" if result.success else "failed"
        logger.info(f"{status} [Worker {worker_id}] Completed {work_item.item_id} ({outcome})")

        # Log progress (thread-safe read of stats)
        async with self._stats_lock:
            should_log = self._stats.processed % self.config.progress_interval == 0
            if should_log:
                stats_snapshot = self._stats.copy()

        if should_log:
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

        retry_cfg = self.config.retry
        attempt = 0  # Total attempts made (including rate-limited ones)
        budget_used = 0  # Failed attempts counted against retry.max_attempts
        rate_limited_attempts = 0  # Failed attempts exempted from the budget

        while True:
            # Re-check the cooldown gate before every attempt — another worker
            # may have started a cooldown while this item was between retries,
            # and firing a retry mid-cooldown burns quota.
            await self._rate_limit_coord.wait_if_paused()
            attempt += 1
            try:
                return await self._process_item(
                    work_item,
                    worker_id,
                    attempt_number=attempt,
                    strategy=strategy,
                    retry_state=retry_state,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Accumulate token usage across retry attempts so users see
                # the true cost of a failure, not just the final attempt.
                attempt_tokens = self._extract_token_usage(e)
                self._token_extractor.accumulate(cumulative_failed_tokens, attempt_tokens)

                if not self._should_retry_error(e):
                    logger.debug(f"Error not retryable: {type(e).__name__}")
                    # Attach token usage to exception so it can be included in failed result
                    if hasattr(e, "__dict__"):
                        e.__dict__["_failed_token_usage"] = cumulative_failed_tokens
                    raise

                # Classify error to determine budget accounting and delay
                error_info = self.error_classifier.classify(e)

                # Rate-limited attempts are exempt from the retry budget by
                # default: the framework already paused and cooled down, so the
                # item never got a clean attempt — an item that happened to be
                # in flight during several 429 bursts shouldn't permanently
                # fail. A separate cap (max_rate_limit_retries) keeps a
                # persistently-throttled item from looping forever.
                if error_info.is_rate_limit and not retry_cfg.count_rate_limits:
                    rate_limited_attempts += 1
                    exhausted = rate_limited_attempts >= retry_cfg.max_rate_limit_retries
                else:
                    budget_used += 1
                    exhausted = budget_used >= retry_cfg.max_attempts

                if exhausted:
                    error_msg = str(e)
                    token_summary = ""
                    if cumulative_failed_tokens["total_tokens"] > 0:
                        total = cumulative_failed_tokens["total_tokens"]
                        token_summary = f"\n  Total tokens consumed across all attempts: {total}"
                    logger.error(
                        f"[FAIL]ALL ATTEMPTS EXHAUSTED for {work_item.item_id} "
                        f"({attempt} attempts, {rate_limited_attempts} rate-limited):\n"
                        f"  Final error type: {type(e).__name__}\n"
                        f"  Final error message: {error_msg[:ERROR_MESSAGE_DETAILED_LENGTH]}{token_summary}"
                    )
                    # Attach token usage to exception so it can be included in failed result
                    if hasattr(e, "__dict__"):
                        e.__dict__["_failed_token_usage"] = cumulative_failed_tokens
                    raise

                # Only delay for network/timeout errors, not for validation errors
                # Validation errors are immediate - strategy can adjust its behavior on retry
                # PydanticAI wraps validation errors in UnexpectedModelBehavior
                error_msg_for_check = str(e)
                is_validation_error = (
                    "validation" in type(e).__name__.lower()
                    or "parse" in type(e).__name__.lower()
                    or "unexpectedmodelbehavior" in type(e).__name__.lower()
                    or "result validation" in error_msg_for_check.lower()
                    or error_info.error_category == "validation_error"
                )

                if is_validation_error or error_info.is_rate_limit:
                    # Validation errors retry immediately — the strategy adjusts on retry.
                    # Rate limits already waited inside _handle_rate_limit()'s coordinated
                    # cooldown; adding exponential backoff here would double the delay.
                    wait_time = 0.0
                else:
                    # Calculate wait time with exponential backoff for network/timeout
                    # errors. The exponent uses budget-counted failures so interleaved
                    # rate limits don't inflate the backoff.
                    wait_time = min(
                        retry_cfg.initial_wait * (retry_cfg.exponential_base ** (budget_used - 1)),
                        retry_cfg.max_wait,
                    )

                    # Apply jitter if enabled to prevent thundering herd
                    if retry_cfg.jitter:
                        import random

                        # Apply jitter: multiply by random factor between 0.5 and 1.0
                        # This reduces wait time by up to 50% to spread out retries
                        wait_time = wait_time * (0.5 + random.random() * 0.5)

                # Log retry attempt
                # Brief snippet for retry logs (shorter than the detailed final-failure log)
                error_snippet = str(e)[:ERROR_MESSAGE_MAX_LENGTH]
                error_type = type(e).__name__
                if error_info.is_rate_limit:
                    retry_desc = "immediately (cooldown already applied)"
                elif wait_time == 0:
                    retry_desc = "immediately"
                else:
                    retry_desc = f"in {wait_time:.1f}s"
                logger.warning(
                    f"[WARN]Attempt {budget_used}/{retry_cfg.max_attempts} "
                    f"(+{rate_limited_attempts} rate-limited) failed for "
                    f"{work_item.item_id}: {error_type} - {error_snippet}. "
                    f"Retrying {retry_desc}..."
                )

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

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

        # Snapshot the cooldown generation BEFORE issuing the request. If this
        # attempt's 429 surfaces after a cooldown that started later has
        # already completed, the stale snapshot lets the coordinator no-op
        # instead of starting a redundant second cooldown. (Snapshotting in
        # the error handler — microseconds before use — made the guard cover
        # only the lock-acquisition window.)
        observed_generation = self._rate_limit_coord.current_generation

        # Store original item_id before middleware might return None
        original_item_id = work_item.item_id

        await self._emit_event(
            ProcessingEvent.ITEM_STARTED,
            {"item_id": original_item_id, "worker_id": worker_id},
        )

        try:
            # Run before middlewares
            processed_item = await self._run_middlewares_before(work_item)
            if processed_item is None:
                logger.info(f"[INFO]Skipping {original_item_id} (filtered by middleware)")
                await self._emit_event(
                    ProcessingEvent.ITEM_FAILED,
                    {"item_id": original_item_id, "error_type": "middleware_skip"},
                )
                return WorkItemResult(
                    item_id=original_item_id,
                    success=False,
                    error="Skipped by middleware",
                    context=work_item.context,
                )
            work_item = processed_item

            # Execute the strategy
            if attempt_number > 1:
                logger.info(
                    f"[INFO][Worker {worker_id}] Retry attempt {attempt_number} for {work_item.item_id}"
                )
            logger.debug(
                f"[STRATEGY] Starting strategy.execute() for {work_item.item_id} "
                f"(attempt {attempt_number}, timeout={self.config.timeout_per_item}s)"
            )
            llm_start_time = time.time()

            # Ensure strategy is not None (it shouldn't be since we always pass it)
            if strategy is None:
                raise RuntimeError("Strategy is None in _process_item - this should not happen")

            try:
                # Proactive rate limiting: acquire token before making request
                if self._proactive_rate_limiter:
                    logger.debug(
                        f"[RATE-LIMIT] Acquiring token for {work_item.item_id} (attempt {attempt_number})"
                    )
                    await self._proactive_rate_limiter.acquire()
                    logger.debug(
                        f"[RATE-LIMIT] Token acquired for {work_item.item_id} (attempt {attempt_number})"
                    )

                # Dry-run mode: use strategy's dry_run method instead of making API call
                if self.config.dry_run:
                    logger.info(f"[DRY-RUN] Skipping API call for {work_item.item_id}")
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
                # Preserve token usage when the caught TimeoutError carries it.
                # This never applies to wait_for's own timeout (asyncio creates
                # that exception fresh), but this except clause also catches
                # TimeoutErrors raised BY the strategy — e.g. an API timeout
                # after the provider already billed the prompt.
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
                f"[STRATEGY] Completed strategy.execute() for {work_item.item_id} "
                f"in {llm_duration:.1f}s"
            )

            # Log success after previous failures
            if attempt_number > 1:
                failures = attempt_number - 1
                logger.info(
                    f"[OK]SUCCESS on attempt {attempt_number} for {work_item.item_id} "
                    f"(after {failures} failure(s), took {llm_duration:.1f}s)"
                )

            # Log first few results for debugging
            if self._stats.succeeded < 3:
                logger.info(
                    f"[INFO]\n{'=' * 80}\nRESULT for {work_item.item_id}:\n{'=' * 80}\n{output}\n{'=' * 80}"
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
            return await self._handle_execution_error(
                e, work_item, worker_id, attempt_number, observed_generation
            )

    async def _handle_execution_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
        observed_generation: int | None = None,
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
            observed_generation: Cooldown generation snapshotted before the
                request was issued; lets the coordinator drop stale reports

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
            # as a floor on the cooldown duration. observed_generation was
            # snapshotted before the request went out (see _process_item);
            # fall back to a late snapshot only if a subclass didn't pass one.
            if observed_generation is None:
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
            if not middleware_result.success:
                await self._emit_event(
                    ProcessingEvent.ITEM_FAILED,
                    {"item_id": work_item.item_id, "error_type": type(exception).__name__},
                )
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
