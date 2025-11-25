"""Parallel batch processor"""

import asyncio
import logging
import time
import weakref
from typing import Any, Generic, cast

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

logger = logging.getLogger(__name__)

# Timeout constants (seconds)
OBSERVER_CALLBACK_TIMEOUT = 5.0  # Observer events should complete quickly
POST_PROCESSOR_TIMEOUT = 90.0  # Post-processors may do database/IO work


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
            # Override config with explicit parameters if provided
            if max_workers is not None:
                config.max_workers = max_workers
            if timeout_per_item is not None:
                config.timeout_per_item = timeout_per_item
            if rate_limit_cooldown is not None:
                config.rate_limit.cooldown_seconds = rate_limit_cooldown

        config.validate()

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
            backoff_multiplier=config.rate_limit.backoff_multiplier,
            slow_start_items=config.rate_limit.slow_start_items,
            slow_start_initial_delay=config.rate_limit.slow_start_initial_delay,
            slow_start_final_delay=config.rate_limit.slow_start_final_delay,
        )

        # Set up middleware and observers
        self.middlewares = middlewares or []
        self.observers = observers or []

        # Rate limit coordination
        self._rate_limit_event = asyncio.Event()
        self._rate_limit_event.set()  # Start in "not paused" state
        self._in_cooldown = False
        self._cooldown_generation = 0  # Track which cooldown cycle we're in (fixes race condition)
        self._cooldown_complete_generation = 0
        self._current_generation_event: asyncio.Event = asyncio.Event()
        self._current_generation_event.set()
        self._items_since_resume = 0
        self._slow_start_active = False
        self._consecutive_rate_limits = 0

        # Thread safety locks
        self._rate_limit_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()
        self._results_lock = asyncio.Lock()

        # Strategy lifecycle management (v0.2.0)
        # Track which strategy instances have been prepared to avoid duplicate prepare() calls
        self._prepared_strategies: weakref.WeakSet[LLMCallStrategy[Any]] = weakref.WeakSet()
        self._strategy_lock = asyncio.Lock()  # Protect strategy initialization

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

        self._strategies_cleaned_up = False

    async def _cleanup_strategies(self) -> None:
        """Cleanup all prepared strategies exactly once."""
        if self._strategies_cleaned_up:
            return

        # Iterate over copy of set since cleanup may modify it
        for strategy in list(self._prepared_strategies):
            if hasattr(strategy, "cleanup") and callable(strategy.cleanup):
                try:
                    logger.debug(
                        f"Cleaning up strategy {strategy.__class__.__name__} (id={id(strategy)})"
                    )
                    await strategy.cleanup()
                except Exception as e:
                    # Log but don't fail the batch - cleanup failures shouldn't invalidate work
                    logger.warning(
                        f"[WARN]Strategy cleanup failed for {strategy.__class__.__name__}: {e}"
                    )

        self._strategies_cleaned_up = True

    async def _ensure_strategy_prepared(self, strategy: LLMCallStrategy[TOutput]) -> None:
        """
        Ensure strategy is prepared exactly once, even with concurrent calls.

        When the same strategy instance is shared across multiple work items
        (e.g., for caching cost optimization), this method ensures prepare()
        is called only once per unique strategy instance.

        Thread-safe via double-checked locking pattern.

        Args:
            strategy: The LLM call strategy to prepare
        """
        # Fast path: already prepared (no lock needed for read)
        if strategy in self._prepared_strategies:
            return

        # Slow path: acquire lock and prepare
        async with self._strategy_lock:
            # Double-check after acquiring lock (another worker may have prepared)
            if strategy in self._prepared_strategies:
                return

            strategy_id = id(strategy)
            logger.debug(f"Preparing strategy {strategy.__class__.__name__} (id={strategy_id})")
            await strategy.prepare()
            self._prepared_strategies.add(strategy)
            self._strategies_cleaned_up = False
            logger.debug(
                f"Strategy {strategy.__class__.__name__} prepared successfully (id={strategy_id})"
            )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
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
        """Emit event to all observers."""
        if not self.observers:
            return

        event_data = data or {}
        for observer in self.observers:
            try:
                await asyncio.wait_for(
                    observer.on_event(event, event_data),
                    timeout=OBSERVER_CALLBACK_TIMEOUT,
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                logger.warning(
                    f"[WARN]Observer callback timed out after {OBSERVER_CALLBACK_TIMEOUT}s for event {event.name}"
                )
            except Exception as e:
                logger.warning(f"[WARN]Observer error: {e}")

    async def _run_middlewares_before(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMWorkItem[TInput, TOutput, TContext] | None:
        """Run before_process on all middlewares."""
        current_item = work_item
        for middleware in self.middlewares:
            try:
                result = await middleware.before_process(current_item)
                if result is None:
                    return None  # Skip this item
                current_item = result
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"[WARN]Middleware before_process error for {work_item.item_id}: {e}"
                )
        return current_item

    async def _run_middlewares_after(
        self, result: WorkItemResult[TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        """Run after_process on all middlewares in reverse order (onion pattern)."""
        current_result = result
        # Run in reverse order to maintain onion-style middleware pattern
        for middleware in reversed(self.middlewares):
            try:
                current_result = await middleware.after_process(current_result)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[WARN]Middleware after_process error for {result.item_id}: {e}")
        return current_result

    async def _run_middlewares_on_error(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], error: Exception
    ) -> WorkItemResult[TOutput, TContext] | None:
        """Run on_error on all middlewares."""
        for middleware in self.middlewares:
            try:
                result = await middleware.on_error(work_item, error)
                if result is not None:
                    return result  # Middleware handled the error
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[WARN]Middleware on_error error for {work_item.item_id}: {e}")
        return None

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

            logger.info(f"[INFO][Worker {worker_id}] Picked up {work_item.item_id} from queue")

            # Wait if we're in rate limit cooldown
            await self._rate_limit_event.wait()

            # Slow start after rate limit recovery (thread-safe)
            should_delay = False
            delay = 0.0

            async with self._rate_limit_lock:
                if self._slow_start_active:
                    should_delay, delay = self.rate_limit_strategy.should_apply_slow_start(
                        self._items_since_resume
                    )
                    if should_delay:
                        self._items_since_resume += 1
                    else:
                        # Slow-start window finished; reset counters until next rate limit
                        self._slow_start_active = False
                        self._items_since_resume = 0

            if should_delay:
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
                    f"{type(e).__name__}: {str(e)[:200]}{token_msg}"
                )

                # Try middleware error handlers
                middleware_result = await self._run_middlewares_on_error(work_item, e)
                if middleware_result is not None:
                    result = middleware_result
                else:
                    result = WorkItemResult(
                        item_id=work_item.item_id,
                        success=False,
                        error=f"{type(e).__name__}: {str(e)[:200]}",
                        context=work_item.context,
                        token_usage=cast(TokenUsage, failed_tokens),
                    )
                # Fall through to store result and call task_done()

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
                    self._stats.cached_input_tokens += result.token_usage.get(
                        "cached_input_tokens", 0
                    )

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
                    self._run_post_processor(result), timeout=POST_PROCESSOR_TIMEOUT
                )
            except TimeoutError:
                logger.error(f"⏱ Post-processor exceeded timeout for {work_item.item_id}")

            self._queue.task_done()

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

    async def _handle_rate_limit(self, worker_id: int, observed_generation: int | None = None):
        """Handle rate limit by pausing all workers and coordinating cooldown."""
        # Use generation counter to track cooldown cycles and prevent race conditions.
        # Workers that detect a rate limit while another cooldown is active simply wait
        # for that generation's completion event instead of attempting to coordinate again.

        if observed_generation is None:
            observed_generation = self._cooldown_generation

        async with self._rate_limit_lock:
            current_generation = self._cooldown_generation
            generation_event = self._current_generation_event
            if self._in_cooldown or observed_generation < current_generation:
                # Another worker already coordinating this (or a newer) cooldown cycle
                logger.debug(
                    f"Worker {worker_id} waiting for cooldown gen {current_generation} (obs={observed_generation})"
                )
                should_wait = True
                generation = current_generation
            else:
                # We're the coordinator for this cooldown
                self._in_cooldown = True
                self._cooldown_generation += 1  # Increment generation for new cooldown cycle
                generation = self._cooldown_generation
                self._slow_start_active = True
                self._consecutive_rate_limits += 1
                self._rate_limit_event.clear()  # Pause all workers
                self._current_generation_event = asyncio.Event()
                generation_event = self._current_generation_event
                consecutive = self._consecutive_rate_limits
                should_wait = False

        if should_wait:
            await generation_event.wait()
            logger.debug(f"Worker {worker_id} resumed after cooldown gen {generation}")
            return

        # We're the coordinator - perform the cooldown

        pause_started_at = time.time()
        cooldown_error: Exception | None = None

        try:
            cooldown = await self.rate_limit_strategy.on_rate_limit(worker_id, consecutive)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            cooldown_error = exc
            cooldown = 0.0
            logger.warning(
                "[WARN]Rate limit strategy failed to determine cooldown: %s. "
                "Resuming workers immediately.",
                exc,
            )

        await self._emit_event(
            ProcessingEvent.COOLDOWN_STARTED,
            {
                "worker_id": worker_id,
                "duration": cooldown,
                "consecutive": consecutive,
            },
        )

        if cooldown_error is None and cooldown > 0:
            logger.warning(
                "[RATE-LIMIT]Rate limit detected by worker %s (gen %d). Pausing all workers for %.1fs...",
                worker_id,
                generation,
                cooldown,
            )
        else:
            logger.warning(
                "[RATE-LIMIT]Rate limit detected by worker %s (gen %d). Skipping cooldown due to prior error.",
                worker_id,
                generation,
            )

        try:
            if cooldown > 0:
                await asyncio.sleep(cooldown)
        except asyncio.CancelledError:
            await asyncio.shield(self._finalize_cooldown(pause_started_at, None))
            raise
        except Exception as exc:
            logger.warning(
                "[WARN]Cooldown sleep interrupted for worker %s: %s. Resuming immediately.",
                worker_id,
                exc,
            )
            cooldown_error = cooldown_error or exc
            await self._finalize_cooldown(pause_started_at, cooldown_error)
            return

        await self._finalize_cooldown(pause_started_at, cooldown_error)

    async def _finalize_cooldown(self, start_time: float, error: Exception | None) -> None:
        """Resume workers after cooldown and emit completion event."""
        actual_duration = max(0.0, time.time() - start_time)

        # Reset state and resume all workers atomically
        async with self._rate_limit_lock:
            self._items_since_resume = 0
            self._in_cooldown = False
            # Track completion so late reporters don't start a new cycle
            self._cooldown_complete_generation = self._cooldown_generation
            self._rate_limit_event.set()  # Resume all workers
            self._current_generation_event.set()

        payload: dict[str, float | str] = {"duration": actual_duration}
        if error is not None:
            payload["error"] = str(error)[:200]

        await self._emit_event(ProcessingEvent.COOLDOWN_ENDED, payload)

        if error is not None:
            logger.warning(
                "[WARN]Cooldown ended early due to error: %s. Workers resumed immediately.",
                error,
            )
        else:
            logger.info("[OK]Cooldown complete. Resuming with slow-start...")

    def _should_retry_error(self, exception: Exception) -> bool:
        """Determine if error should be retried using error classifier."""
        error_info = self.error_classifier.classify(exception)
        return error_info.is_retryable

    def _extract_token_usage(self, exception: Exception) -> dict[str, int]:
        """
        Extract token usage from a failed LLM call exception.

        Attempts multiple strategies to extract token usage from different provider
        exception structures. Returns empty dict if extraction fails.

        Args:
            exception: The exception from which to extract token usage

        Returns:
            Dictionary with input_tokens, output_tokens, total_tokens (or empty dict)
        """
        try:
            # Strategy 1: PydanticAI-style exception with result in __cause__
            if hasattr(exception, "__cause__") and exception.__cause__:
                cause = exception.__cause__
                if hasattr(cause, "result"):
                    result = cause.result
                    if hasattr(result, "usage") and callable(result.usage):
                        usage = result.usage()
                        if usage:
                            return {
                                "input_tokens": getattr(usage, "request_tokens", 0),
                                "output_tokens": getattr(usage, "response_tokens", 0),
                                "total_tokens": getattr(usage, "total_tokens", 0),
                            }

            # Strategy 2: Direct usage attribute on exception
            if hasattr(exception, "usage"):
                usage = exception.usage
                if callable(usage):
                    usage = usage()
                if usage:
                    return {
                        "input_tokens": getattr(
                            usage, "request_tokens", getattr(usage, "input_tokens", 0)
                        ),
                        "output_tokens": getattr(
                            usage, "response_tokens", getattr(usage, "output_tokens", 0)
                        ),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }

            # Strategy 3: Custom _failed_token_usage attribute (set by this framework)
            if hasattr(exception, "__dict__") and "_failed_token_usage" in exception.__dict__:
                failed_usage = exception.__dict__["_failed_token_usage"]
                if isinstance(failed_usage, dict):
                    return failed_usage

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Extraction failed - log for debugging and return empty dict
            logger.debug(
                f"Failed to extract token usage from {type(exception).__name__}: {e}. "
                "Returning 0 tokens. This is normal for non-LLM exceptions."
            )

        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }

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

        for attempt in range(1, self.config.retry.max_attempts + 1):
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
                # Try to extract token usage from this failed attempt using robust extraction
                attempt_tokens = self._extract_token_usage(e)
                if attempt_tokens:
                    cumulative_failed_tokens["input_tokens"] += attempt_tokens.get(
                        "input_tokens", 0
                    )
                    cumulative_failed_tokens["output_tokens"] += attempt_tokens.get(
                        "output_tokens", 0
                    )
                    cumulative_failed_tokens["total_tokens"] += attempt_tokens.get(
                        "total_tokens", 0
                    )
                    cumulative_failed_tokens["cached_input_tokens"] += attempt_tokens.get(
                        "cached_input_tokens", 0
                    )

                if not self._should_retry_error(e):
                    logger.debug(f"Error not retryable: {type(e).__name__}")
                    # Attach token usage to exception so it can be included in failed result
                    if hasattr(e, "__dict__"):
                        e.__dict__["_failed_token_usage"] = cumulative_failed_tokens
                    raise
                if attempt >= self.config.retry.max_attempts:
                    error_msg = str(e)
                    token_summary = ""
                    if cumulative_failed_tokens["total_tokens"] > 0:
                        total = cumulative_failed_tokens["total_tokens"]
                        token_summary = f"\n  Total tokens consumed across all attempts: {total}"
                    logger.error(
                        f"[FAIL]ALL {self.config.retry.max_attempts} ATTEMPTS EXHAUSTED "
                        f"for {work_item.item_id}:\n"
                        f"  Final error type: {type(e).__name__}\n"
                        f"  Final error message: {error_msg[:500]}{token_summary}"
                    )
                    # Attach token usage to exception so it can be included in failed result
                    if hasattr(e, "__dict__"):
                        e.__dict__["_failed_token_usage"] = cumulative_failed_tokens
                    raise

                # Classify error to determine if we should delay
                error_info = self.error_classifier.classify(e)

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

                if is_validation_error:
                    wait_time = 0.0  # No delay for validation - retry immediately
                else:
                    # Calculate wait time with exponential backoff for network/timeout errors
                    wait_time = min(
                        self.config.retry.initial_wait
                        * (self.config.retry.exponential_base ** (attempt - 1)),
                        self.config.retry.max_wait,
                    )

                    # Apply jitter if enabled to prevent thundering herd
                    if self.config.retry.jitter:
                        import random

                        # Apply jitter: multiply by random factor between 0.5 and 1.0
                        # This reduces wait time by up to 50% to spread out retries
                        wait_time = wait_time * (0.5 + random.random() * 0.5)

                # Log retry attempt
                error_snippet = str(e)[:150]
                error_type = type(e).__name__
                if is_validation_error:
                    logger.warning(
                        f"[WARN]Attempt {attempt}/{self.config.retry.max_attempts} failed for "
                        f"{work_item.item_id}: {error_type} - {error_snippet}. "
                        f"Retrying immediately..."
                    )
                else:
                    logger.warning(
                        f"[WARN]Attempt {attempt}/{self.config.retry.max_attempts} failed for "
                        f"{work_item.item_id}: {error_type} - {error_snippet}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

        # Unreachable - all paths raise or return
        raise RuntimeError(
            f"Unexpected: all retry attempts should have raised for {work_item.item_id}"
        )

    def _get_strategy(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMCallStrategy[TOutput]:
        """Get the LLM call strategy for this work item."""
        return work_item.strategy

    async def _process_item(  # type: ignore[override]
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

        await self._emit_event(
            ProcessingEvent.ITEM_STARTED,
            {"item_id": original_item_id, "worker_id": worker_id},
        )

        try:
            # Run before middlewares
            processed_item = await self._run_middlewares_before(work_item)
            if processed_item is None:
                logger.info(f"[INFO]Skipping {original_item_id} (filtered by middleware)")
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
                    # Delegate to strategy's dry_run method for mock output
                    output, token_usage = await strategy.dry_run(work_item.prompt)
                else:
                    # Call strategy.execute() with prompt, attempt number, timeout, and retry state (v0.3.0)
                    # Wrap in asyncio.wait_for to enforce timeout at framework level
                    output, token_usage = await asyncio.wait_for(
                        strategy.execute(
                            work_item.prompt,
                            attempt_number,
                            self.config.timeout_per_item,
                            retry_state,
                        ),
                        timeout=self.config.timeout_per_item,
                    )
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
                # Preserve token usage if the underlying exception had it (including cached tokens)
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

            # Reset consecutive rate limit counter on success (thread-safe)
            async with self._rate_limit_lock:
                self._consecutive_rate_limits = 0

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

            # Handle rate limit (cooldown) - this will pause all workers
            observed_generation = self._cooldown_generation
            await self._handle_rate_limit(worker_id, observed_generation)

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
            f"  Error message: {error_msg[:500]}\n"
            f"  This error will NOT be retried (not retryable){token_summary}"
        )

        await self._emit_event(
            ProcessingEvent.ITEM_FAILED,
            {"item_id": work_item.item_id, "error_type": error_name},
        )

        return WorkItemResult(
            item_id=work_item.item_id,
            success=False,
            error=f"{error_name}: {error_msg[:500]}",
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
        """
        Log detailed information about retryable errors.

        For validation errors, this extracts field-level details from Pydantic
        ValidationErrors. For other retryable errors, logs basic info.

        Args:
            exception: The exception that was raised
            work_item: The work item being processed
            attempt_number: Current attempt number
            failed_token_usage: Token usage extracted from the failed attempt
        """
        error_name = type(exception).__name__
        error_msg = str(exception)

        # Log token usage for failed attempt if we have it
        token_msg = ""
        if failed_token_usage:
            token_msg = f" ({failed_token_usage.get('total_tokens', 0)} tokens consumed)"

        # For validation errors, try to extract field-level details
        # PydanticAI wraps validation errors in UnexpectedModelBehavior
        is_validation_type = (
            "validation" in error_name.lower()
            or "unexpectedmodelbehavior" in error_name.lower()
            or "result validation" in error_msg.lower()
        )

        if is_validation_type:
            self._log_validation_error(exception, work_item, attempt_number, token_msg)
        else:
            logger.debug(
                f"Retryable {error_name} on attempt {attempt_number} for {work_item.item_id}: {error_msg[:300]}"
            )

    def _log_validation_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        attempt_number: int,
        token_msg: str,
    ) -> None:
        """
        Log detailed validation error information.

        Walks the exception chain to find Pydantic ValidationError and extracts
        field-level error details and raw LLM responses.

        Args:
            exception: The exception that was raised
            work_item: The work item being processed
            attempt_number: Current attempt number
            token_msg: Formatted token usage message
        """
        error_name = type(exception).__name__
        error_msg = str(exception)

        # Try to extract raw LLM response and underlying ValidationError
        raw_response = None
        underlying_validation_error = None

        try:
            # Walk the exception chain to find ValidationError and raw response
            exc_chain_current = exception
            depth = 0
            while exc_chain_current and depth < 10:
                # Try to extract raw response
                if hasattr(exc_chain_current, "response"):
                    raw_response = str(exc_chain_current.response)[:1000]
                if hasattr(exc_chain_current, "messages"):
                    try:
                        raw_response = str(exc_chain_current.messages)[:1000]
                    except Exception:
                        pass

                # Check if this is a ValidationError
                from pydantic import ValidationError

                if isinstance(exc_chain_current, ValidationError):
                    underlying_validation_error = exc_chain_current
                    break

                # Move to cause
                next_cause = getattr(exc_chain_current, "__cause__", None)
                if next_cause is None or not isinstance(next_cause, BaseException):
                    break
                exc_chain_current = next_cause
                depth += 1
        except Exception:
            pass

        # Try to parse Pydantic ValidationError for field details
        try:
            from pydantic import ValidationError

            if underlying_validation_error or isinstance(exception, ValidationError):
                validation_err = cast(ValidationError, underlying_validation_error or exception)
                error_details = []
                for err in validation_err.errors():
                    field_path = " -> ".join(str(loc) for loc in err["loc"])
                    error_details.append(
                        f"    Field: {field_path}\n"
                        f"      Type: {err['type']}\n"
                        f"      Message: {err['msg']}\n"
                        f"      Input: {str(err.get('input', 'N/A'))[:100]}"
                    )

                log_msg = (
                    f"[FAIL]Validation error on attempt {attempt_number} for {work_item.item_id}{token_msg}:\n"
                    f"  Error type: {error_name}\n"
                    f"  Field-level errors:\n" + "\n".join(error_details)
                )
                if raw_response:
                    log_msg += f"\n  Raw LLM response (first 1000 chars):\n{raw_response}"
                logger.error(log_msg)
            else:
                # Not a Pydantic ValidationError, log full error with more context
                log_msg = (
                    f"[FAIL]Validation error on attempt {attempt_number} for {work_item.item_id}{token_msg}:\n"
                    f"  Error type: {error_name}\n"
                    f"  Full error message: {error_msg}\n"
                    f"  Exception chain:"
                )
                # Show exception chain for debugging
                current: BaseException | None = exception
                depth = 0
                while current and depth < 5:
                    log_msg += f"\n    {depth}: {type(current).__name__}: {str(current)[:200]}"
                    next_cause = getattr(current, "__cause__", None)
                    if next_cause is None or not isinstance(next_cause, BaseException):
                        break
                    current = next_cause
                    depth += 1
                if raw_response:
                    log_msg += f"\n  Raw LLM response (first 1000 chars):\n{raw_response}"
                logger.error(log_msg)
        except Exception as parse_error:
            # Fallback if we can't parse the error
            log_msg = (
                f"[FAIL]Validation error on attempt {attempt_number} for {work_item.item_id}{token_msg}:\n"
                f"  Error type: {error_name}\n"
                f"  Full error: {error_msg}\n"
                f"  (Failed to parse error details: {parse_error})"
            )
            if raw_response:
                log_msg += f"\n  Raw LLM response (first 1000 chars):\n{raw_response}"
            logger.error(log_msg)

    async def shutdown(self):
        """Clean up resources: flush observers and cancel pending tasks."""
        await self._cleanup_strategies()
        await self.cleanup()
