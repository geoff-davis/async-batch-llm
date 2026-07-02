"""Per-item execution engine: retries, error classification, rate-limit
coordination, and token accounting for a single work item.

Extracted from :class:`ParallelBatchProcessor` so the exact same execution
semantics can back three surfaces: the batch worker loop, the single-call
helper (:mod:`async_batch_llm.single`), and the rate-limited gateway
(:mod:`async_batch_llm.gateway`). The processor delegates its per-item methods
here; the queue-less surfaces drive :meth:`ItemExecutor.execute` directly.

The executor reads its dependencies live from a *host* (the processor passes
``self``; the gateway passes a lightweight host) because two of them —
``error_classifier`` (auto-resolved at batch start) and ``_stats`` (rebound by
``start()``) — are reassigned after construction.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, cast

from ..base import (
    LLMWorkItem,
    RetryState,
    TContext,
    TInput,
    TokenUsage,
    TOutput,
    WorkItemResult,
    _unpack_strategy_result,
)
from ..observers import ProcessingEvent
from ..strategies import FrameworkTimeoutError, RateLimitRetriesExceeded
from .error_logging import log_retryable_error, log_validation_error

if TYPE_CHECKING:
    from aiolimiter import AsyncLimiter

    from ..base import ProcessingStats
    from ..core import ProcessorConfig
    from ..llm_strategies import LLMCallStrategy
    from ..strategies import ErrorClassifier
    from ..token_extractor import TokenExtractor
    from .event_dispatcher import EventDispatcher
    from .rate_limit_coordinator import RateLimitCoordinator
    from .strategy_lifecycle import StrategyLifecycle

logger = logging.getLogger(__name__)

# Kept in sync with parallel.py (single source would create an import cycle).
ERROR_MESSAGE_MAX_LENGTH = 200
ERROR_MESSAGE_DETAILED_LENGTH = 500

_E = TypeVar("_E", bound=BaseException)


def _detach_traceback(exc: _E) -> _E:
    """Clear tracebacks along an exception's cause/context chain before it is
    stored on a ``WorkItemResult``.

    A traceback pins every frame's locals — strategies, clients, raw responses —
    for as long as the result is held, which for a large accumulated batch of
    failures can retain far more memory than before ``WorkItemResult.exception``
    existed. The full failure (type, message, stack) is already logged at the
    point it happens, so the stored exception keeps its type/message/args (enough
    for ``call()`` to re-raise the provider's type) but drops the frame-pinning
    tracebacks. A re-raise gets a fresh traceback from the raise site.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        cur.__traceback__ = None
        cur = cur.__cause__ or cur.__context__
    return exc


class ExecutorHostProtocol(Protocol[TInput, TOutput, TContext]):
    """The surface :class:`ItemExecutor` reads from its host.

    Both hosts — :class:`~async_batch_llm.parallel.ParallelBatchProcessor` and
    the lightweight :class:`~async_batch_llm._internal.executor_host.ExecutorHost`
    — expose exactly these. ``error_classifier`` and ``_stats`` are read live (not
    snapshotted) because the processor reassigns them after construction.

    ``_extract_token_usage``, ``_process_item``, and ``_handle_execution_error``
    are invoked *through the host* (not as the executor's own methods) so that a
    ``ParallelBatchProcessor`` subclass overriding any of them — ``_process_item``
    is abstract on ``BatchProcessor``; ``_extract_token_usage`` is a documented
    override point — still takes effect during batch runs. The processor's
    versions delegate back to the executor's implementations, so the base case is
    one extra hop with no behavior change.
    """

    config: ProcessorConfig
    error_classifier: ErrorClassifier
    _token_extractor: TokenExtractor
    _rate_limit_coord: RateLimitCoordinator
    _proactive_rate_limiter: AsyncLimiter | None
    _events: EventDispatcher[TInput, TOutput, TContext]
    _stats: ProcessingStats
    _stats_lock: asyncio.Lock
    _strategy_lifecycle: StrategyLifecycle[TOutput]

    def _extract_token_usage(self, exception: Exception) -> dict[str, int]: ...

    async def _process_item(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int = 1,
        strategy: LLMCallStrategy[TOutput] | None = None,
        retry_state: RetryState | None = None,
    ) -> WorkItemResult[TOutput, TContext]: ...

    async def _handle_execution_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
    ) -> WorkItemResult[TOutput, TContext]: ...


class ItemExecutor(Generic[TInput, TOutput, TContext]):
    """Executes one work item with the full resilience pipeline.

    ``host`` must satisfy :class:`ExecutorHostProtocol`.
    """

    def __init__(self, host: ExecutorHostProtocol[TInput, TOutput, TContext]) -> None:
        self._host = host

    # ── Dependencies (read live from host) ───────────────────────
    @property
    def config(self) -> ProcessorConfig:
        return self._host.config

    @property
    def error_classifier(self) -> ErrorClassifier:
        return self._host.error_classifier

    @property
    def _token_extractor(self) -> TokenExtractor:
        return self._host._token_extractor

    @property
    def _rate_limit_coord(self) -> RateLimitCoordinator:
        return self._host._rate_limit_coord

    @property
    def _proactive_rate_limiter(self) -> AsyncLimiter | None:
        return self._host._proactive_rate_limiter

    @property
    def _events(self) -> EventDispatcher[TInput, TOutput, TContext]:
        return self._host._events

    @property
    def _stats(self) -> ProcessingStats:
        return self._host._stats

    @property
    def _stats_lock(self) -> asyncio.Lock:
        return self._host._stats_lock

    @property
    def _strategy_lifecycle(self) -> StrategyLifecycle[TOutput]:
        return self._host._strategy_lifecycle

    # ── Thin delegators (so moved bodies stay verbatim) ──────────
    async def _emit_event(self, event: ProcessingEvent, data: dict | None = None) -> None:
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

    async def _ensure_strategy_prepared(self, strategy) -> None:
        await self._strategy_lifecycle.ensure_prepared(strategy)

    async def _handle_rate_limit(
        self, worker_id, observed_generation=None, suggested_wait=None
    ) -> None:
        await self._rate_limit_coord.handle_rate_limit(
            worker_id, observed_generation, suggested_wait
        )

    def _log_retryable_error(
        self, exception, work_item, attempt_number, failed_token_usage
    ) -> None:
        log_retryable_error(exception, work_item.item_id, attempt_number, failed_token_usage)

    def _log_validation_error(self, exception, work_item, attempt_number, token_msg) -> None:
        log_validation_error(exception, work_item.item_id, attempt_number, token_msg)

    # ── Queue-less entry points (used by gateway + single) ───────
    async def wait_for_capacity(self) -> None:
        """Respect the shared cooldown + slow-start ramp before an item."""
        await self._rate_limit_coord.wait_if_paused()
        delay = await self._rate_limit_coord.apply_slow_start()
        if delay > 0:
            await asyncio.sleep(delay)

    async def execute(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], worker_id: int = 0
    ) -> WorkItemResult[TOutput, TContext]:
        """Run one item end-to-end, always returning a WorkItemResult.

        Waits out any active cooldown, runs the retry pipeline, and converts an
        exhausted/unhandled failure into a failed result (never raises for
        business errors; CancelledError still propagates).
        """
        await self.wait_for_capacity()
        try:
            return await self._process_item_with_retries(work_item, worker_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return await self.build_failure_result(work_item, e, worker_id)

    async def build_failure_result(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        e: Exception,
        worker_id: int = 0,
    ) -> WorkItemResult[TOutput, TContext]:
        """Build the failed result for an exhausted/unhandled error.

        Relocated verbatim from the worker loop so both the batch worker and the
        queue-less surfaces produce identical failure results and ITEM_FAILED
        events.
        """
        # All retries exhausted or unhandled exception
        # Create a failed result so the item is recorded

        # Extract token usage from exception if available
        failed_tokens = {}
        if hasattr(e, "__dict__") and "_failed_token_usage" in e.__dict__:
            failed_tokens = e.__dict__["_failed_token_usage"]

        token_msg = ""
        if failed_tokens.get("total_tokens", 0) > 0:
            token_msg = f" (consumed {failed_tokens['total_tokens']} tokens across all attempts)"

        logger.error(
            f"[FAIL]Worker {worker_id} failed to process {work_item.item_id} after all retries: "
            f"{type(e).__name__}: {str(e)[:ERROR_MESSAGE_MAX_LENGTH]}{token_msg}"
        )

        # Try middleware error handlers
        middleware_result = await self._run_middlewares_on_error(work_item, e)
        result: WorkItemResult[TOutput, TContext]
        if middleware_result is not None:
            result = middleware_result
        else:
            # Annotated above: ty infers unannotated constructions against
            # the PEP 696 defaults ([Any, None]) instead of the executor's
            # type parameters.
            result = WorkItemResult(
                item_id=work_item.item_id,
                success=False,
                error=f"{type(e).__name__}: {str(e)[:ERROR_MESSAGE_MAX_LENGTH]}",
                context=work_item.context,
                token_usage=cast(TokenUsage, failed_tokens),
                exception=_detach_traceback(e),
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

        return result

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
                # Through the host so a processor subclass override takes effect.
                result = await self._host._process_item(
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
                attempt_tokens = self._host._extract_token_usage(e)
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

    async def _process_item(
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

            # Create result (annotated: ty infers unannotated constructions
            # against the PEP 696 defaults instead of the executor's params)
            work_result: WorkItemResult[TOutput, TContext] = WorkItemResult(
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
            # Through the host so a processor subclass override takes effect.
            return await self._host._handle_execution_error(e, work_item, worker_id, attempt_number)

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
        failed_token_usage = self._host._extract_token_usage(exception)
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
            exception=_detach_traceback(exception),
        )
