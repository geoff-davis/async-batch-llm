"""A lightweight host that backs :class:`ItemExecutor` without a worker pool.

The :class:`~async_batch_llm.parallel.ParallelBatchProcessor` is one host for
the executor; this is the other. It builds the same per-item dependencies
(rate-limit coordinator, strategy lifecycle, token extractor, classifier,
stats) but no queue, workers, or result stream — so the single-call helper and
the gateway can run ``executor.execute(work_item)`` directly.

Observers and middlewares are intentionally empty here: these surfaces are the
request path, not a batch with progress reporting. A single shared host (hence
a single shared ``RateLimitCoordinator``) is what makes many concurrent callers
respect one coordinated cooldown.
"""

from __future__ import annotations

import asyncio
from typing import Generic

from ..base import (
    LLMWorkItem,
    ProcessingStats,
    RetryState,
    TContext,
    TInput,
    TOutput,
    WorkItemResult,
)
from ..core import ProcessorConfig
from ..llm_strategies import LLMCallStrategy
from ..strategies import (
    DefaultErrorClassifier,
    ErrorClassifier,
    ExponentialBackoffStrategy,
    RateLimitStrategy,
)
from ..token_extractor import TokenExtractor
from .capacity import CapacityLimiter
from .event_dispatcher import EventDispatcher
from .guardrails import AbortController
from .item_executor import ItemExecutor
from .rate_limit_coordinator import RateLimitCoordinator
from .strategy_lifecycle import StrategyLifecycle


def _resolve_classifier(
    strategy: LLMCallStrategy | None, explicit: ErrorClassifier | None
) -> ErrorClassifier:
    """Use the caller's classifier, else the strategy's recommendation, else default."""
    if explicit is not None:
        return explicit
    if strategy is not None:
        recommend = getattr(strategy, "recommended_error_classifier", None)
        if recommend is not None:
            recommended: ErrorClassifier | None
            try:
                recommended = recommend()
            except Exception:
                recommended = None
            if recommended is not None:
                return recommended
    return DefaultErrorClassifier()


class ExecutorHost(Generic[TInput, TOutput, TContext]):
    """Owns the per-item dependencies for queue-less execution.

    Exposes exactly the attribute surface :class:`ItemExecutor` reads, plus a
    ready-built ``executor``. Call :meth:`aclose` to run strategy ``cleanup()``.
    """

    def __init__(
        self,
        config: ProcessorConfig,
        *,
        strategy: LLMCallStrategy | None = None,
        error_classifier: ErrorClassifier | None = None,
        rate_limit_strategy: RateLimitStrategy | None = None,
    ) -> None:
        self.config = config
        self.error_classifier = _resolve_classifier(strategy, error_classifier)
        self.rate_limit_strategy = rate_limit_strategy or ExponentialBackoffStrategy(
            initial_cooldown=config.rate_limit.cooldown_seconds,
            max_cooldown=config.rate_limit.max_cooldown_seconds,
            backoff_multiplier=config.rate_limit.backoff_multiplier,
            slow_start_items=config.rate_limit.slow_start_items,
            slow_start_initial_delay=config.rate_limit.slow_start_initial_delay,
            slow_start_final_delay=config.rate_limit.slow_start_final_delay,
        )

        # No observers/middlewares on the request path. (Parameterized
        # explicitly: with PEP 696 defaults a bare `EventDispatcher` would
        # resolve to [str, Any, None] and break ExecutorHostProtocol
        # conformance.)
        self._events: EventDispatcher[TInput, TOutput, TContext] = EventDispatcher(
            observers=[], middlewares=[]
        )
        # One shared coordinator → coordinated cooldown across all callers.
        self._rate_limit_coord = RateLimitCoordinator(
            rate_limit_strategy=self.rate_limit_strategy,
            events=self._events,
        )
        self._strategy_lifecycle: StrategyLifecycle[TOutput] = StrategyLifecycle()
        self._capacity_limiter = CapacityLimiter(
            config.max_provider_concurrency,
            max_workers=config.max_workers,
            startup_ramp=config.startup_ramp,
        )
        self._stats = ProcessingStats()
        self._stats_lock = asyncio.Lock()
        self._token_extractor = TokenExtractor()
        # Queue-less call/gateway surfaces have item deadlines but no shared
        # batch abort controller.
        self._abort_controller: AbortController | None = None

        if config.max_requests_per_minute:
            from aiolimiter import AsyncLimiter

            self._proactive_rate_limiter: AsyncLimiter | None = AsyncLimiter(
                max_rate=config.max_requests_per_minute,
                time_period=60,
            )
        else:
            self._proactive_rate_limiter = None

        self.executor: ItemExecutor[TInput, TOutput, TContext] = ItemExecutor(self)

    # These three satisfy ExecutorHostProtocol's override-point hooks. On the
    # queue-less path there's no subclass to override them, so they delegate
    # straight back to the executor (the processor's versions do the same).
    def _extract_token_usage(self, exception: Exception) -> dict[str, int]:
        return self._token_extractor.extract_from_exception(exception)

    async def _process_item(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int = 1,
        strategy: LLMCallStrategy[TOutput] | None = None,
        retry_state: RetryState | None = None,
    ) -> WorkItemResult[TOutput, TContext]:
        return await self.executor._process_item(
            work_item, worker_id, attempt_number, strategy, retry_state
        )

    async def _process_item_with_retries(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        deadline: float | None = None,
    ) -> WorkItemResult[TOutput, TContext]:
        return await self.executor._process_item_with_retries(work_item, worker_id, deadline)

    async def _handle_execution_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
    ) -> WorkItemResult[TOutput, TContext]:
        return await self.executor._handle_execution_error(
            exception, work_item, worker_id, attempt_number
        )

    async def aclose(self) -> None:
        """Run cleanup() on every strategy this host prepared."""
        await self._strategy_lifecycle.cleanup_all()
