"""Base classes and interfaces for batch LLM processing."""

import asyncio
import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypedDict  # noqa: F401

from typing_extensions import TypeVar  # PEP 696 defaults on Python < 3.13

# Conditional imports for type checking
if TYPE_CHECKING:
    from .llm_strategies import LLMCallStrategy

# Type variables for generic typing. PEP 696 defaults let common usage drop
# trailing parameters: ``ParallelBatchProcessor[str, MyOutput]`` (context
# defaults to None), or even ``ParallelBatchProcessor[str]``.
# Note: TInput is currently unused by the framework (prompts are always str);
# it is kept for backward compatibility and slated for removal in the next
# major release.
TInput = TypeVar("TInput", default=str)  # Input data type
TOutput = TypeVar("TOutput", default=Any)  # Agent output type
TContext = TypeVar("TContext", default=None)  # Optional context passed through

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
        cached_input_tokens: Number of input tokens served from cache
            (Gemini/OpenAI/DeepSeek/pydantic-ai all populate this)
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int


@dataclass
class LLMResponse:
    """
    Normalized response from any LLM provider.

    Returned by LLMModel.generate(). Provides a provider-agnostic interface
    so strategies don't need to know about Gemini, OpenAI, etc. response formats.

    Attributes:
        text: The response text content.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        total_tokens: Total tokens used.
        cached_input_tokens: Input tokens served from cache (0 if no caching).
        metadata: Provider-specific metadata (safety ratings, finish reason, etc.).
        raw: The raw provider response object, for edge cases.

    Added in v0.6.0.
    """

    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0
    metadata: dict[str, Any] | None = None
    raw: Any = None

    @property
    def token_usage(self) -> TokenUsage:
        """Return token counts as a TokenUsage dict."""
        result: TokenUsage = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_input_tokens:
            result["cached_input_tokens"] = self.cached_input_tokens
        return result


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
        if self.strategy is None:
            raise ValueError(
                "strategy must not be None. "
                "Pass an LLMCallStrategy instance (e.g., PydanticAIStrategy, GeminiStrategy, "
                "or your custom subclass)."
            )
        if not isinstance(self.prompt, str):
            raise TypeError(
                f"prompt must be a string (got {type(self.prompt).__name__}: {repr(self.prompt)[:80]}). "
                f"If you need to pass structured data, serialize it to a string first."
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
        metadata: Provider-specific metadata returned alongside the response —
            e.g. ``{"provider": "Anthropic", "finish_reason": "stop",
            "model": "anthropic/claude-haiku-4-5"}``. Populated when the
            strategy returns a 3-tuple ``(output, tokens, metadata)`` from
            ``execute()``; ``None`` for legacy 2-tuple strategies.
            Added in v0.10.0. (For Gemini safety ratings specifically, this
            replaces the older ``gemini_safety_ratings`` field — see below.)
        gemini_safety_ratings: **Deprecated.** Use ``metadata['safety_ratings']``
            instead. Still populated when the underlying model surfaces them,
            for backward compat. To be removed in a future release.
    """

    item_id: str
    success: bool
    output: TOutput | None = None
    error: str | None = None
    context: TContext | None = None
    token_usage: TokenUsage = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    metadata: dict[str, Any] | None = None
    # Deprecated — derived from metadata; excluded from repr/eq so the
    # deprecation warning on read (see the property attached below the class)
    # doesn't fire on every repr()/comparison.
    gemini_safety_ratings: dict[str, str] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Backfill ``gemini_safety_ratings`` from ``metadata['safety_ratings']``
        for backward compatibility. Once ``gemini_safety_ratings`` is removed,
        this method goes away. (Reads/writes go through ``__dict__`` directly
        so the framework itself never triggers the deprecation warning.)
        """
        if (
            self.__dict__.get("gemini_safety_ratings") is None
            and self.metadata is not None
            and "safety_ratings" in self.metadata
        ):
            ratings = self.metadata["safety_ratings"]
            if isinstance(ratings, dict):
                self.__dict__["gemini_safety_ratings"] = ratings


def _get_gemini_safety_ratings(self: "WorkItemResult") -> dict[str, str] | None:
    warnings.warn(
        "WorkItemResult.gemini_safety_ratings is deprecated and will be removed "
        "in a future release; read result.metadata['safety_ratings'] instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.__dict__.get("gemini_safety_ratings")


def _set_gemini_safety_ratings(self: "WorkItemResult", value: dict[str, str] | None) -> None:
    self.__dict__["gemini_safety_ratings"] = value


# Replace the plain dataclass attribute with a property AFTER the decorator
# has generated __init__: reads emit a DeprecationWarning, writes (including
# the generated __init__'s assignment and the __post_init__ backfill) go
# straight to the instance __dict__.
WorkItemResult.gemini_safety_ratings = property(  # type: ignore[assignment,method-assign]
    _get_gemini_safety_ratings, _set_gemini_safety_ratings
)


class CachedTokenRates:
    """Named cached-token price rates for the providers we support.

    Each constant is the **fraction of the normal input-token price** you
    pay for tokens served from cache (so 0.10 = "10% of normal" =
    "90% discount"). Pass these to
    :meth:`BatchResult.effective_input_tokens` to compute provider-aware
    billable token estimates:

    .. code-block:: python

        result.effective_input_tokens(CachedTokenRates.OPENAI)

    Rates are accurate as of v0.9.0 (early 2026); confirm with each
    provider's pricing page before using these for invoicing.

    Notes:
        - **Anthropic prompt caching** is asymmetric: cache *reads* are at
          ``ANTHROPIC_READ`` (10% of normal), but cache *writes* are billed
          at a 25% premium over normal input price. This helper covers
          read-side savings only; the write premium is a separate cost
          best computed at billing time from your usage logs.
        - **OpenRouter** routes to many upstream providers, each with its
          own rate. Pick the constant for the upstream that actually
          served your request (visible in ``LLMResponse.metadata['provider']``).
    """

    GEMINI: float = 0.10
    """Gemini context cache: cached tokens cost 10% of normal."""

    OPENAI: float = 0.50
    """OpenAI prompt caching: cached tokens cost 50% of normal (chat completions)."""

    ANTHROPIC_READ: float = 0.10
    """Anthropic prompt cache reads: 10% of normal (cache writes are
    billed at 1.25× normal — not modeled here)."""

    DEEPSEEK: float = 0.10
    """DeepSeek context cache: cached tokens cost 10% of normal."""


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
    # Derived fields — always recomputed from `results` in __post_init__.
    # init=False so the constructor signature doesn't advertise arguments
    # that would be silently discarded.
    total_items: int = field(init=False, default=0)
    succeeded: int = field(init=False, default=0)
    failed: int = field(init=False, default=0)
    total_input_tokens: int = field(init=False, default=0)
    total_output_tokens: int = field(init=False, default=0)
    total_cached_tokens: int = field(init=False, default=0)  # v0.2.0

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

    def effective_input_tokens(self, cached_token_rate: float | None = None) -> int:
        """
        Estimate billable input tokens after the cache discount.

        ``cached_token_rate`` is the fraction of the normal input-token price
        you pay for tokens served from cache. For example, Gemini charges
        10% of the normal price (rate = 0.10), so 1000 cached tokens cost
        the same as 100 uncached tokens.

        Use the named constants on :class:`CachedTokenRates` to avoid
        hardcoding magic numbers:

        .. code-block:: python

            result.effective_input_tokens(CachedTokenRates.OPENAI)
            result.effective_input_tokens(CachedTokenRates.GEMINI)

        Args:
            cached_token_rate: Fraction (0.0–1.0) of the normal input price
                paid for cached tokens. When omitted (``None``) it defaults to
                ``CachedTokenRates.GEMINI`` (0.10) for backward compatibility —
                pre-v0.9.0 versions hardcoded this value. **Pass an explicit
                rate when working with non-Gemini providers** to get accurate
                numbers; relying on the implicit default while cached tokens
                are present emits a ``UserWarning``, since the Gemini rate is
                wrong for e.g. OpenAI (~0.50).

        Returns:
            Effective input tokens billed. The discount is computed by
            truncating ``cached_tokens * (1 - rate)`` toward zero with
            ``int()``, which means the returned billable estimate is
            rounded **up** when the discount would have a fractional
            part — a deliberately conservative choice for cost reporting
            (your real bill is at most this number, never more).

        Raises:
            ValueError: If ``cached_token_rate`` is not in [0.0, 1.0].
        """
        if cached_token_rate is None:
            # Implicit default. Only nudge when it actually changes the answer
            # (i.e. there are cached tokens to discount) — silent for the common
            # no-cache case so we don't cry wolf.
            if self.total_cached_tokens > 0:
                import warnings

                warnings.warn(
                    "effective_input_tokens() called without an explicit "
                    "cached_token_rate; defaulting to the Gemini rate "
                    "(CachedTokenRates.GEMINI = 0.10). This is wrong for other "
                    "providers (OpenAI is ~0.50). Pass an explicit "
                    "CachedTokenRates constant to silence this warning.",
                    UserWarning,
                    stacklevel=2,
                )
            cached_token_rate = CachedTokenRates.GEMINI

        if not 0.0 <= cached_token_rate <= 1.0:
            raise ValueError(
                f"cached_token_rate must be in [0.0, 1.0]; got {cached_token_rate}. "
                f"This is the fraction of normal price paid for cached tokens "
                f"(0.0 = free, 1.0 = no discount). For named provider rates, "
                f"use CachedTokenRates."
            )
        # cached_token_rate is what you PAY; (1 - rate) is the discount.
        # int() floors the discount toward zero -> conservative (over-)estimate
        # of effective billable tokens. See the Returns docstring.
        discount = int(self.total_cached_tokens * (1.0 - cached_token_rate))
        return self.total_input_tokens - discount


def _unpack_strategy_result(
    result: Any,
) -> tuple[Any, "TokenUsage", dict[str, Any] | None]:
    """Compat-shim for the LLMCallStrategy.execute() return contract.

    Strategies may return either:

    - ``(output, token_usage)`` — legacy 2-tuple shape (pre-v0.10.0).
    - ``(output, token_usage, metadata)`` — current 3-tuple shape; ``metadata``
      is forwarded into ``WorkItemResult.metadata``.

    Returns the normalized 3-tuple. Raises ``ValueError`` for any other shape
    (clearer than letting Python's tuple-unpacking error reach the caller).

    The 2-tuple path will be removed in a future release; custom strategies
    should migrate to the 3-tuple shape.
    """
    if not isinstance(result, tuple):
        raise ValueError(
            f"Strategy.execute() must return a tuple of (output, tokens) or "
            f"(output, tokens, metadata); got {type(result).__name__}."
        )
    if len(result) == 3:
        output, tokens, metadata = result
        return output, tokens, metadata
    if len(result) == 2:
        output, tokens = result
        return output, tokens, None
    raise ValueError(
        f"Strategy.execute() must return a 2- or 3-tuple "
        f"(output, tokens [, metadata]); got a tuple of length {len(result)}."
    )


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
    cached_input_tokens: int = 0  # Input tokens served from cache (any provider)

    def copy(self) -> dict[str, Any]:
        """Return a dictionary copy of the stats for backwards compatibility.

        The dict exposes the cached-token count under both keys:
        ``cached_input_tokens`` (the stored field) and ``total_cached_tokens``
        (the preferred alias, matching ``BatchResult.total_cached_tokens``).
        """
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
            "total_cached_tokens": self.cached_input_tokens,
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
            except (TimeoutError, asyncio.TimeoutError):  # distinct classes on Python 3.10
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
            except (TimeoutError, asyncio.TimeoutError):  # distinct classes on Python 3.10
                logger.warning("[WARN]Some progress callbacks did not cancel in time")
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

    async def _on_batch_started(self) -> None:
        """Hook for subclasses to run logic when a batch starts."""
        return

    async def _on_batch_completed(self) -> None:
        """Hook for subclasses to run logic after a batch completes."""
        return

    async def process_all(self) -> BatchResult[TOutput, TContext]:
        """
        Process all work items in the queue.

        Processor instances are one-shot: after processing starts, add_work()
        raises RuntimeError. Create a new processor for additional batches.

        Returns:
            BatchResult containing all results and statistics
        """
        # Mark processing as started to prevent add_work() calls (v0.4.0)
        self._processing_started = True

        # Initialize result and stats containers for this one-shot batch.
        self._results = []
        self._stats = ProcessingStats(total=self._queue.qsize())

        # Record start time for rate calculation
        self._stats.start_time = time.time()
        self._is_processing = True

        await self._on_batch_started()

        # Start workers and store them for cleanup
        self._workers = [
            asyncio.create_task(self._worker(worker_id)) for worker_id in range(self.max_workers)
        ]

        # Wait for all work to complete, watching worker tasks so that a
        # crashed worker surfaces as an error instead of hanging queue.join()
        # on items nobody can process anymore.
        join_task = asyncio.create_task(self._queue.join())
        try:
            await self._watch_workers_until_drained(join_task)
        finally:
            join_task.cancel()
            try:
                await join_task
            except asyncio.CancelledError:
                pass
            # Unblock workers even if queue.join() was cancelled or a worker
            # crashed. put_nowait: a full bounded queue must never block exit
            # (workers exiting via cancellation don't need sentinels).
            for _ in range(self.max_workers):
                try:
                    self._queue.put_nowait(None)
                except asyncio.QueueFull:
                    break

        logger.info("[OK]Queue processing complete, waiting for workers to finish...")

        # Wait for workers to finish with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers),
                timeout=WORKER_SHUTDOWN_TIMEOUT,
            )
            logger.info(f"[OK]All {len(self._workers)} workers finished successfully")
        except (TimeoutError, asyncio.TimeoutError):  # distinct classes on Python 3.10
            logger.error(
                f"[WARN]Workers did not finish within {WORKER_SHUTDOWN_TIMEOUT}s after queue.join(). "
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
            except (TimeoutError, asyncio.TimeoutError):  # distinct classes on Python 3.10
                logger.error("[WARN]Some workers could not be cancelled")

        self._is_processing = False

        await self._on_batch_completed()

        # Snapshot results before returning so callers receive an independent list.
        results_snapshot = list(self._results)
        return BatchResult(results=results_snapshot)

    async def _watch_workers_until_drained(self, join_task: "asyncio.Task[None]") -> None:
        """Wait for ``queue.join()``, failing fast if a worker dies first.

        Workers only exit normally after consuming a sentinel, and sentinels
        are queued only after ``join()`` returns — so any worker task that
        completes while the queue is still draining has crashed. Without this
        watch, ``queue.join()`` would wait forever on items the dead worker
        can no longer process.
        """
        watched = set(self._workers)
        while True:
            done, _ = await asyncio.wait({join_task, *watched}, return_when=asyncio.FIRST_COMPLETED)
            if join_task in done:
                return
            finished = done & watched
            watched -= finished
            for task in finished:
                try:
                    exc = task.exception()
                except asyncio.CancelledError:
                    exc = None  # Externally cancelled worker; treated below.
                if exc is not None:
                    for worker in self._workers:
                        if not worker.done():
                            worker.cancel()
                    raise RuntimeError(
                        f"Worker crashed with {type(exc).__name__} before the queue "
                        f"was drained; aborting batch: {exc}"
                    ) from exc
            if not watched:
                raise RuntimeError(
                    "All workers exited before the queue was drained; aborting batch."
                )

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

        Timeout enforcement lives in the caller (the worker loop wraps this in
        ``config.post_processor_timeout``); this method only isolates
        post-processor exceptions so a buggy callback can't fail the item.

        Args:
            result: Work item result to post-process
        """
        if self.post_processor is None:
            return

        try:
            await_result = self.post_processor(result)
            # Handle both async and sync post-processors
            if asyncio.iscoroutine(await_result):
                await await_result
        except Exception as e:
            # Log error with full details - this is critical for debugging
            import traceback

            logger.error(
                f"[FAIL]Post-processor failed for {result.item_id}:\n"
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

        callback_task: asyncio.Task[None] = asyncio.create_task(callback_awaitable)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        self._track_progress_task(callback_task, log_exceptions=True)

        if self.progress_callback_timeout is not None:

            async def monitor() -> None:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(callback_task),
                        timeout=self.progress_callback_timeout,
                    )
                except (TimeoutError, asyncio.TimeoutError):  # distinct classes on Python 3.10
                    logger.warning(
                        "[WARN]Progress callback exceeded timeout of %.2fs; continuing without waiting.",
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
                    logger.warning(f"[WARN]Progress callback failed: {exc}")

        task.add_done_callback(_cleanup)
