"""Base classes and interfaces for batch LLM processing."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar  # noqa: F401

# Conditional imports for type checking
if TYPE_CHECKING:
    from .llm_strategies import LLMCallStrategy

# Type variables for generic typing
TInput = TypeVar("TInput")  # Input data type
TOutput = TypeVar("TOutput")  # Agent output type
TContext = TypeVar("TContext")  # Optional context passed through

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class RetryState:
    """
    Mutable state that persists across retry attempts for a single work item.

    This allows strategies to maintain state between retry attempts, enabling
    multi-stage retry patterns where each attempt can build on previous attempts.

    Example use cases:
    - Partial recovery: Save successfully parsed fields from failed attempts
    - Progressive refinement: Track which parts of output need improvement
    - Smart retry prompts: Build targeted prompts based on what failed
    - Model escalation: Track validation errors vs network errors separately

    Usage:
        # In strategy's on_error():
        if state:
            state.set('last_error_type', type(exception).__name__)
            state.set('validation_failures', state.get('validation_failures', 0) + 1)

        # In strategy's execute():
        if state and state.get('validation_failures', 0) > 1:
            # Use smarter model or modified prompt
            pass

    Added in v0.3.0.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the state.

        Args:
            key: State key to retrieve
            default: Value to return if key doesn't exist

        Returns:
            Value associated with key, or default if not found
        """
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the state.

        Args:
            key: State key to set
            value: Value to store (can be any JSON-serializable type)
        """
        self.data[key] = value

    def delete(self, key: str, raise_if_missing: bool = False) -> None:
        """
        Delete a value from the state.

        Args:
            key: State key to delete
            raise_if_missing: If True, raise KeyError if key doesn't exist.
                             If False, silently ignore missing keys (default).

        Raises:
            KeyError: If key doesn't exist and raise_if_missing=True
        """
        if raise_if_missing:
            del self.data[key]
        else:
            self.data.pop(key, None)

    def clear(self) -> None:
        """Clear all state data."""
        self.data.clear()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in state."""
        return key in self.data

    def __repr__(self) -> str:
        """String representation showing all state data."""
        return f"RetryState({self.data!r})"

# Timeout constants (seconds)
WORKER_CANCELLATION_TIMEOUT = 2.0  # Time to wait for workers to cancel gracefully
WORKER_SHUTDOWN_TIMEOUT = 30.0  # Time to wait for workers to finish after queue is done
POST_PROCESSOR_EXECUTION_TIMEOUT = 75.0  # Maximum time for post-processor to execute
PROGRESS_TASK_CANCELLATION_TIMEOUT = 2.0  # Time to wait for progress callbacks to cancel


class TokenUsage(TypedDict, total=False):
    """
    Token usage statistics from LLM API calls.

    All fields are optional to accommodate different provider APIs.
    Different providers may return different subsets of these fields.

    Fields:
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/completion
        total_tokens: Total tokens used (input + output)
        cached_input_tokens: Number of input tokens served from cache (Gemini)
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int


@dataclass
class LLMWorkItem(Generic[TInput, TOutput, TContext]):
    """
    Represents a single work item to be processed by an LLM strategy.

    Attributes:
        item_id: Unique identifier for this work item
        strategy: LLM call strategy that encapsulates how to make the LLM call
        prompt: The prompt/input to pass to the LLM
        context: Optional context data passed through to results/post-processor
    """

    item_id: str
    strategy: "LLMCallStrategy[TOutput]"
    prompt: str = ""
    context: TContext | None = None

    def __post_init__(self):
        """Validate work item fields."""
        if not self.item_id or not isinstance(self.item_id, str):
            raise ValueError(
                f"item_id must be a non-empty string (got {type(self.item_id).__name__}: {repr(self.item_id)}). "
                f"Provide a unique string identifier for this work item."
            )
        if not self.item_id.strip():
            raise ValueError(
                f"item_id cannot be whitespace only (got {repr(self.item_id)}). "
                f"Provide a non-whitespace string identifier."
            )


@dataclass
class WorkItemResult(Generic[TOutput, TContext]):
    """
    Result of processing a single work item.

    Attributes:
        item_id: ID of the work item
        success: Whether processing succeeded
        output: Agent output if successful, None if failed
        error: Error message if failed, None if successful
        context: Context data from the work item
        token_usage: Token usage stats (input_tokens, output_tokens, total_tokens)
        gemini_safety_ratings: Gemini API safety ratings if available
    """

    item_id: str
    success: bool
    output: TOutput | None = None
    error: str | None = None
    context: TContext | None = None
    token_usage: TokenUsage = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    gemini_safety_ratings: dict[str, str] | None = None


@dataclass
class BatchResult(Generic[TOutput, TContext]):
    """
    Result of processing a batch of work items.

    Attributes:
        results: List of individual work item results
        total_items: Total number of items in the batch
        succeeded: Number of successful items
        failed: Number of failed items
        total_input_tokens: Sum of input tokens across all items
        total_output_tokens: Sum of output tokens across all items
        total_cached_tokens: Sum of cached input tokens across all items (v0.2.0)
    """

    results: list[WorkItemResult[TOutput, TContext]]
    total_items: int = 0
    succeeded: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0  # v0.2.0: Track cached tokens separately

    def __post_init__(self):
        """Calculate summary statistics from results."""
        self.total_items = len(self.results)
        self.succeeded = sum(1 for r in self.results if r.success)
        self.failed = sum(1 for r in self.results if not r.success)
        self.total_input_tokens = sum(r.token_usage.get("input_tokens", 0) for r in self.results)
        self.total_output_tokens = sum(r.token_usage.get("output_tokens", 0) for r in self.results)
        # v0.2.0: Aggregate cached tokens
        self.total_cached_tokens = sum(
            r.token_usage.get("cached_input_tokens", 0) for r in self.results
        )

    def cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate as percentage of input tokens that were cached.

        Returns:
            Percentage (0.0 to 100.0) of input tokens served from cache
        """
        if self.total_input_tokens == 0:
            return 0.0
        return (self.total_cached_tokens / self.total_input_tokens) * 100.0

    def effective_input_tokens(self) -> int:
        """
        Calculate effective input tokens (actual cost after caching).

        Gemini charges 10% of the normal price for cached tokens.

        Returns:
            Effective number of input tokens billed
        """
        # Cached tokens cost 10% of normal, so discount is 90%
        discount = int(self.total_cached_tokens * 0.9)
        return self.total_input_tokens - discount


# Type alias for post-processor function
PostProcessorFunc = Callable[[WorkItemResult[TOutput, TContext]], Awaitable[None] | None]

# Type alias for progress callback function (completed, total, current_item_id)
ProgressCallbackFunc = Callable[[int, int, str], Awaitable[None] | None]


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""

    total: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    start_time: float | None = None
    error_counts: dict[str, int] = field(default_factory=dict)
    rate_limit_count: int = 0
    # Token usage tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0  # Gemini context caching (same as total_cached_tokens)
    total_cached_tokens: int = 0  # v0.2.0: Alias for cached_input_tokens (preferred name)

    def copy(self) -> dict[str, Any]:
        """Return a dictionary copy of the stats for backwards compatibility."""
        return {
            "total": self.total,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "start_time": self.start_time,
            "error_counts": self.error_counts.copy(),
            "rate_limit_count": self.rate_limit_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "total_cached_tokens": self.total_cached_tokens,
        }


class BatchProcessor(ABC, Generic[TInput, TOutput, TContext]):
    """
    Abstract base class for batch LLM processing strategies.

    Subclasses implement different strategies for processing batches:
    - ParallelBatchProcessor: Process items in parallel as individual requests
    - BatchAPIProcessor: Use Google's true batch API (future)
    """

    def __init__(
        self,
        max_workers: int = 5,
        post_processor: PostProcessorFunc[TOutput, TContext] | None = None,
        max_queue_size: int = 0,
        progress_callback: ProgressCallbackFunc | None = None,
        progress_callback_timeout: float | None = None,
    ):
        """
        Initialize the batch processor.

        Args:
            max_workers: Maximum number of concurrent workers
            post_processor: Optional async function called after each successful item
            max_queue_size: Maximum queue size (0 = unlimited)
            progress_callback: Optional callback(completed, total, current_item_id) for progress updates
            progress_callback_timeout: Maximum seconds to wait for progress callback (None = no limit)
        """
        self.max_workers = max_workers
        self.post_processor = post_processor
        self.max_queue_size = max_queue_size
        self.progress_callback = progress_callback
        self.progress_callback_timeout = progress_callback_timeout
        self._progress_callback_is_async = False
        if progress_callback is not None:
            self._progress_callback_is_async = asyncio.iscoroutinefunction(progress_callback) or (
                callable(progress_callback)
                and asyncio.iscoroutinefunction(progress_callback.__call__)  # type: ignore[operator]
            )
        self._queue: asyncio.Queue[LLMWorkItem[TInput, TOutput, TContext] | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._results: list[WorkItemResult[TOutput, TContext]] = []
        self._stats = ProcessingStats()
        self._workers: list[asyncio.Task] = []
        self._is_processing = False
        self._progress_tasks: set[asyncio.Task[Any]] = set()

        self._processing_started = False  # Prevent add_work() after process_all() starts

    async def __aenter__(self):
        """Context manager entry - returns self for use in async with."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup of resources."""
        await self.cleanup()
        return False  # Don't suppress exceptions

    async def cleanup(self):
        """
        Clean up resources: cancel pending workers and clear queue.

        This method should be called when you're done with the processor,
        or use the processor as an async context manager.
        """
        # Cancel any running workers
        if self._workers:
            logger.debug(f"Cleaning up {len(self._workers)} workers")
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()

            # Wait briefly for cancellations
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=WORKER_CANCELLATION_TIMEOUT,
                )
            except TimeoutError:
                logger.warning("Some workers did not cancel within timeout")

        # Clear any remaining items in queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Cancel any pending progress callbacks
        if self._progress_tasks:
            for task in list(self._progress_tasks):
                task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._progress_tasks, return_exceptions=True),
                    timeout=PROGRESS_TASK_CANCELLATION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("⚠️  Some progress callbacks did not cancel in time")
            finally:
                self._progress_tasks.clear()

    async def add_work(self, work_item: LLMWorkItem[TInput, TOutput, TContext]):
        """
        Add a work item to the processing queue.

        Args:
            work_item: Work item to process

        Raises:
            RuntimeError: If called after process_all() has started
        """
        if self._processing_started:
            raise RuntimeError(
                "Cannot add work after process_all() has started. "
                "Create a new processor instance for additional batches."
            )

        await self._queue.put(work_item)
        self._stats.total += 1

    async def process_all(self) -> BatchResult[TOutput, TContext]:
        """
        Process all work items in the queue.

        Returns:
            BatchResult containing all results and statistics
        """
        # Mark processing as started to prevent add_work() calls (v0.4.0)
        self._processing_started = True

        # Clear results and reinitialize stats for this run
        # This ensures each call to process_all() starts fresh
        self._results = []
        self._stats = ProcessingStats(total=self._queue.qsize())

        # Record start time for rate calculation
        self._stats.start_time = time.time()
        self._is_processing = True

        # Start workers and store them for cleanup
        self._workers = [
            asyncio.create_task(self._worker(worker_id)) for worker_id in range(self.max_workers)
        ]

        # Wait for all work to complete
        try:
            await self._queue.join()
        finally:
            # Unblock workers even if queue.join() is cancelled or fails
            for _ in range(self.max_workers):
                await self._queue.put(None)

        logger.info("✓ Queue processing complete, waiting for workers to finish...")

        # Wait for workers to finish with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers),
                timeout=WORKER_SHUTDOWN_TIMEOUT,
            )
            logger.info(f"✓ All {len(self._workers)} workers finished successfully")
        except TimeoutError:
            logger.error(
                f"⚠️  Workers did not finish within {WORKER_SHUTDOWN_TIMEOUT}s after queue.join(). "
                "Cancelling workers and proceeding..."
            )
            # Cancel any workers that are still running
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()
            # Wait briefly for cancellations to complete
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=WORKER_CANCELLATION_TIMEOUT * 2.5,  # Allow more time during shutdown
                )
            except TimeoutError:
                logger.error("⚠️  Some workers could not be cancelled")

        self._is_processing = False

        # Snapshot results before returning to prevent contamination if processor is reused
        # Create a copy so the returned BatchResult is independent of future runs
        results_snapshot = list(self._results)
        return BatchResult(results=results_snapshot)

    @abstractmethod
    async def _worker(self, worker_id: int):
        """
        Worker coroutine that processes items from the queue.

        Each implementation defines its own worker strategy.

        Args:
            worker_id: Unique identifier for this worker
        """
        pass

    @abstractmethod
    async def _process_item(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        """
        Process a single work item.

        Each implementation defines how to execute the agent and handle results.

        Args:
            work_item: Work item to process

        Returns:
            Result of processing the work item
        """
        pass

    async def _run_post_processor(self, result: WorkItemResult[TOutput, TContext]) -> None:
        """
        Run the post-processor callback if provided.

        Args:
            result: Work item result to post-process
        """
        if self.post_processor is None:
            return

        try:
            await_result = self.post_processor(result)
            # Handle both async and sync post-processors
            if asyncio.iscoroutine(await_result):
                # Inner timeout for the post-processor execution itself
                # (Outer timeout at worker level handles semaphore waits)
                await asyncio.wait_for(await_result, timeout=POST_PROCESSOR_EXECUTION_TIMEOUT)
        except TimeoutError:
            logger.error(
                f"✗ Post-processor execution timed out after {POST_PROCESSOR_EXECUTION_TIMEOUT}s for {result.item_id}"
            )
        except Exception as e:
            # Log error with full details - this is critical for debugging
            import traceback

            logger.error(
                f"✗ Post-processor failed for {result.item_id}:\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error message: {str(e)}\n"
                f"  Full traceback:\n{traceback.format_exc()}"
            )

    async def _run_progress_callback(self, completed: int, total: int, current_item: str) -> None:
        """Invoke progress callback with timeout and non-blocking handling."""
        if self.progress_callback is None:
            return
        if self._progress_callback_is_async:
            callback_awaitable = self.progress_callback(completed, total, current_item)
        else:
            # Cast to sync callable since we know it's not async at this point
            callback_awaitable = asyncio.to_thread(
                self.progress_callback,  # type: ignore[arg-type]
                completed,
                total,
                current_item,
            )

        callback_task: asyncio.Task[None] = asyncio.create_task(callback_awaitable)  # type: ignore[arg-type]
        self._track_progress_task(callback_task, log_exceptions=True)

        if self.progress_callback_timeout is not None:

            async def monitor() -> None:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(callback_task),
                        timeout=self.progress_callback_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "⚠️  Progress callback exceeded timeout of %.2fs; continuing without waiting.",
                        self.progress_callback_timeout,
                    )
                    callback_task.cancel()
                except asyncio.CancelledError:
                    pass

            monitor_task = asyncio.create_task(monitor())
            self._track_progress_task(monitor_task, log_exceptions=False)

    def _track_progress_task(self, task: asyncio.Task[Any], *, log_exceptions: bool) -> None:
        """Track background tasks for progress callbacks and clean up on completion."""
        self._progress_tasks.add(task)

        def _cleanup(completed_task: asyncio.Task[Any]) -> None:
            self._progress_tasks.discard(completed_task)
            try:
                completed_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if log_exceptions:
                    logger.warning(f"⚠️  Progress callback failed: {exc}")

        task.add_done_callback(_cleanup)
