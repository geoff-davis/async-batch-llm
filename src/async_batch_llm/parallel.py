"""Parallel batch processor"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Generic

if TYPE_CHECKING:
    from types import TracebackType

from ._internal.event_dispatcher import EventDispatcher
from ._internal.item_executor import ItemExecutor
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

        # Per-item execution engine (extracted so the single-call helper and the
        # gateway can share the exact same retry/rate-limit/token pipeline). The
        # processor is one host for the executor; it reads its deps live from
        # `self`, so it must be built here, before start(), and the worker keeps
        # calling the instance methods below (which delegate to it) so tests that
        # monkeypatch them still take effect.
        self._executor: ItemExecutor[TInput, TOutput, TContext] = ItemExecutor(self)

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
            await self._executor.wait_for_capacity()

            # Process the item
            try:
                result = await self._process_item_with_retries(work_item, worker_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # All retries exhausted or unhandled exception: build the failed
                # result (token extraction → middleware on_error → ITEM_FAILED
                # emit) via the shared executor so the batch worker and the
                # queue-less surfaces produce identical failures. Falls through to
                # store the result and call task_done().
                result = await self._executor.build_failure_result(work_item, e, worker_id)

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
        """Apply retry logic and strategy lifecycle (delegates to ItemExecutor).

        Kept as an instance method (not inlined to the executor call) because
        tests monkeypatch *this* method and call it directly on a non-started
        processor; the worker must keep routing through it.
        """
        return await self._executor._process_item_with_retries(work_item, worker_id)

    async def _process_item(  # type: ignore[override]  # ty:ignore[invalid-method-override]
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int = 1,
        strategy: LLMCallStrategy[TOutput] | None = None,
        retry_state: RetryState | None = None,
    ) -> WorkItemResult[TOutput, TContext]:
        """Process a single work item (delegates to ItemExecutor)."""
        return await self._executor._process_item(
            work_item, worker_id, attempt_number, strategy, retry_state
        )

    async def _handle_execution_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
    ) -> WorkItemResult[TOutput, TContext]:
        """Handle exceptions from LLM execution (delegates to ItemExecutor).

        Classifies errors, extracts token usage, handles rate limits, and either
        re-raises (retryable / rate-limited) or returns a permanent-failure
        result. See :class:`ItemExecutor` for the implementation.
        """
        return await self._executor._handle_execution_error(
            exception, work_item, worker_id, attempt_number
        )

    async def shutdown(self):
        """Clean up resources: flush observers and cancel pending tasks."""
        await self._cleanup_strategies()
        await self.cleanup()
