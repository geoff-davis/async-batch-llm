"""Queue-less shared call pool: many callers, one shared cooldown.

A long-lived service object for a web app's request path. Many concurrent
callers each :meth:`submit` one prompt; an :class:`asyncio.Semaphore` caps
global concurrency against the provider, and a single shared
``RateLimitCoordinator`` (inside the host) gives one coordinated cooldown — one
caller's 429 briefly throttles everyone, then slow-starts.

There is no queue, no worker pool, no background dispatcher, and no per-request
Future demux: each caller's coroutine runs :meth:`ItemExecutor.execute`
directly under the semaphore and returns its own result. A cancelled caller
(client disconnect) simply releases its slot — nothing is orphaned.

For load, two opt-in knobs bound the request path: ``max_pending`` (an admission
cap that rejects instantly instead of growing an unbounded waiter list) and
``submit_timeout`` (a per-caller latency budget). Both are off by default.

    from contextlib import asynccontextmanager
    from async_batch_llm import OpenAIModel, OpenAIStrategy, ProcessorConfig
    from async_batch_llm import LLMCallPool

    @asynccontextmanager
    async def lifespan(app):
        strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-6-luna"))
        async with LLMCallPool(strategy, config=ProcessorConfig(max_workers=5)) as pool:
            app.state.llm = pool
            yield

    # in a handler:  summary = await request.app.state.llm.submit(prompt)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Generic, TypeVar, cast

from ._internal.capacity import warn_if_worker_capacity_exceeded
from ._internal.cleanup import CleanupAction, CleanupPhase, CleanupStep, SharedCloser
from ._internal.executor_host import ExecutorHost
from ._internal.guardrails import AbortCause, AbortController
from ._internal.input_validation import validate_strategy
from .base import BatchTermination, LLMWorkItem, WorkItemResult
from .cache_groups import CacheGroupTracker
from .core import ProcessorConfig
from .llm_strategies import LLMCallStrategy
from .single import unwrap_result

TOutput = TypeVar("TOutput")

logger = logging.getLogger(__name__)


class LLMCallPool(Generic[TOutput]):
    """An in-process shared call pool for single LLM calls from many callers.

    Create one at startup and call :meth:`submit` from any number of concurrent
    handlers. ``config.max_workers`` is the global concurrency budget. The
    strategy is shared, so use a stateless strategy or one whose ``prepare()``
    yields a reusable resource.

    A long ``rate_limit.cooldown_seconds`` stalls *all* callers during a 429 —
    tune it via ``config`` to trade upstream protection against latency.

    Two opt-in knobs bound the request path under load (both off by default, so
    the default is pure backpressure — callers beyond ``max_workers`` wait):

    - ``max_pending`` — an admission cap. With it set, at most
      ``max_workers + max_pending`` requests may be in flight (running *or*
      waiting on the semaphore); further submits are rejected *instantly* with a
      failed result instead of growing an unbounded waiter list. Bounds memory.
    - ``submit_timeout`` — a per-caller latency bound (seconds). An admitted
      request that hasn't completed within the budget (e.g. stuck behind a
      cooldown) is cancelled and returns a failed result. A firing timeout
      cancels an in-flight request that might have been about to succeed — the
      right trade for a web latency budget where the client is already gone.

    ``cache_tracker`` (provisional) is the :class:`~async_batch_llm.CacheGroupTracker`
    behind :meth:`cache_stats`. By default the pool creates one with
    ``config.guardrails.cache_hit_window``; pass one restored with
    ``CacheGroupTracker.from_dict`` to continue a previous pool's statistics. Its
    own ``window`` then applies.

    ``config.guardrails.abort_on_error_categories`` stops the pool on the first
    request that finishes failed in one of those categories, as it stops a
    processor run. From then on every submit, including those waiting for a slot,
    returns a ``batch_aborted`` result without calling the provider; calls already
    running drain or are cancelled per ``config.guardrails.abort_mode``. The stop
    is permanent for this pool: :attr:`termination` says why, and recovering
    means building a new pool.
    """

    def __init__(
        self,
        strategy: LLMCallStrategy[TOutput],
        *,
        config: ProcessorConfig | None = None,
        error_classifier: Any = None,
        max_pending: int | None = None,
        submit_timeout: float | None = None,
        cache_tracker: CacheGroupTracker | None = None,
    ) -> None:
        if cache_tracker is not None and not isinstance(cache_tracker, CacheGroupTracker):
            raise TypeError(
                f"cache_tracker must be a CacheGroupTracker (got {type(cache_tracker).__name__})"
            )
        if max_pending is not None and max_pending < 0:
            raise ValueError(f"max_pending must be >= 0 (got {max_pending})")
        if submit_timeout is not None and submit_timeout <= 0:
            raise ValueError(f"submit_timeout must be > 0 (got {submit_timeout})")

        validate_strategy(strategy)
        cfg = config or ProcessorConfig(max_workers=5)
        # Always an int after ProcessorConfig.__post_init__ resolution.
        gateway_workers = cast(int, cfg.max_workers)
        warn_if_worker_capacity_exceeded(
            strategy=strategy,
            max_workers=gateway_workers,
            surface="LLMCallPool",
            stacklevel=3,
        )
        self._strategy = strategy
        self._cache_tracker = (
            cache_tracker
            if cache_tracker is not None
            else CacheGroupTracker(window=cfg.guardrails.cache_hit_window)
        )
        self._abort_categories = cfg.guardrails.abort_on_error_categories
        self._abort: AbortController | None = (
            AbortController(cfg.guardrails.abort_mode) if self._abort_categories else None
        )
        self._host: ExecutorHost[Any, TOutput, Any] = ExecutorHost(
            cfg,
            strategy=strategy,
            error_classifier=error_classifier,
            cache_tracker=self._cache_tracker,
            abort_controller=self._abort,
        )
        self._sem = asyncio.Semaphore(gateway_workers)
        self._seq = 0
        self._closed = False

        # Admission cap: max requests in flight (running + waiting). None = off.
        self._max_inflight = None if max_pending is None else gateway_workers + max_pending
        self._inflight = 0
        self._submit_timeout = submit_timeout
        # Set whenever no request is in flight; the close drains it before
        # cleaning up the shared strategy.
        self._idle = asyncio.Event()
        self._idle.set()
        # Concurrent aclose() callers share one ordered close attempt.
        self._closer = SharedCloser(self._cleanup_steps, name="LLMCallPool", logger=logger)

    async def __aenter__(self) -> LLMCallPool[TOutput]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        self._closed = True
        report = await self._closer.close(primary_exception=exc_val)
        if exc_val is None:
            report.raise_first()
        return False

    async def submit(self, prompt: str, *, timeout: float | None = None) -> TOutput:
        """Submit one prompt and await its output, raising on failure.

        Blocks on the semaphore when the pool is saturated (backpressure), unless
        ``max_pending`` is set (then an over-cap submit raises immediately). A
        rejected/timed-out submit raises :class:`LLMCallError`. ``timeout``
        overrides the pool's ``submit_timeout`` for this call.
        """
        return unwrap_result(await self.submit_result(prompt, timeout=timeout))

    async def submit_result(
        self, prompt: str, *, timeout: float | None = None
    ) -> WorkItemResult[TOutput, Any]:
        """Like :meth:`submit` but returns the WorkItemResult instead of raising.

        Gives access to ``token_usage`` / ``metadata`` and lets callers branch
        on ``.success`` without exception handling. A request rejected by the
        admission cap or cut off by the timeout comes back as a failed result
        (``success=False``) rather than raising.

        ``timeout`` overrides the pool's ``submit_timeout`` for this call.
        """
        if self._closed:
            raise RuntimeError("Shared call pool is closed")

        self._seq += 1
        item_id = f"req-{self._seq}"

        if self._abort is not None and self._abort.aborted:
            return self._abort.result_for(
                LLMWorkItem(item_id=item_id, strategy=self._strategy, prompt=prompt)
            )

        # Admission cap. The check + increment below run with no `await` between
        # them, so in asyncio this is race-free and correctly counts waiters
        # toward the bound.
        if self._max_inflight is not None and self._inflight >= self._max_inflight:
            return WorkItemResult(
                item_id=item_id, success=False, error="shared call pool saturated", context=None
            )

        effective_timeout = self._submit_timeout if timeout is None else timeout
        work_item: LLMWorkItem[Any, TOutput, Any] = LLMWorkItem(
            item_id=item_id, strategy=self._strategy, prompt=prompt
        )

        self._inflight += 1
        self._idle.clear()
        try:
            run = self._run(work_item)
            if effective_timeout is None:
                return await run
            # Wrap the whole `async with self._sem: execute(...)` coroutine — NOT
            # `wait_for(sem.acquire())`, which leaks the permit. Cancellation here
            # unwinds through __aexit__, releasing the slot.
            try:
                return await asyncio.wait_for(run, effective_timeout)
            except (TimeoutError, asyncio.TimeoutError):
                return WorkItemResult(
                    item_id=item_id, success=False, error="submit timed out", context=None
                )
        finally:
            self._inflight -= 1
            if self._inflight == 0:
                self._idle.set()

    async def _run(self, work_item: LLMWorkItem[Any, TOutput, Any]) -> WorkItemResult[TOutput, Any]:
        """Acquire a concurrency slot, run the item, release the slot."""
        async with self._sem:
            if self._abort is not None and self._abort.aborted:
                return self._abort.result_for(work_item)
            result = await self._host.executor.execute(work_item)
        if (
            self._abort is not None
            and not result.success
            and result.error_category in self._abort_categories
        ):
            await self._trip_abort(result)
        return result

    async def _trip_abort(self, result: WorkItemResult[TOutput, Any]) -> None:
        assert self._abort is not None
        reason = (
            f"Fail-fast triggered by item {result.item_id!r} "
            f"with category {result.error_category!r}"
        )
        hint = self._hint_for(result)
        if hint:
            reason = f"{reason}: {hint}"
        if await self._abort.trip(
            AbortCause(
                kind="fail_fast",
                reason=reason,
                error_category=result.error_category,
                triggering_item_id=result.item_id,
            )
        ):
            logger.error("LLMCallPool stopped: %s", reason)

    def _hint_for(self, result: WorkItemResult[TOutput, Any]) -> str | None:
        # The operator hint lives on the classification, not the result.
        if not isinstance(result.exception, Exception):
            return None
        classifier = self._host._classifier_resolver.resolve(self._strategy)
        try:
            return classifier.classify(result.exception).hint
        except Exception:
            return None

    @property
    def termination(self) -> BatchTermination | None:
        """Why the pool stopped, or ``None`` while it is accepting work (provisional).

        Set when a request fails in one of
        ``config.guardrails.abort_on_error_categories``; ``kind`` is
        ``"fail_fast"`` and ``reason`` names the item, the category and any
        classifier hint.
        """
        if self._abort is None or not self._abort.aborted:
            return None
        return self._abort.termination()

    def cache_stats(self) -> dict[str, Any]:
        """Prompt-cache reads and writes per cache group since creation or the last reset.

        Returns ``{"cache_groups": [...], "cache_groups_overflow": ...}`` in the
        shape ``ParallelBatchProcessor.get_stats()`` uses. Each group also reports
        ``window_hit_rate`` over its last ``cache_tracker.window`` measured calls
        (``None`` until that many), since a long-lived pool's
        cumulative ``hit_rate`` changes slowly. Statistics only: the pool has no
        cache hit-rate floor. Call it from the pool's event loop; it doesn't await.
        """
        return self._cache_tracker.stats()

    def reset_cache_stats(self) -> None:
        """Clear the per-group cache statistics, including the 64-group cap's overflow.

        A call still running at the reset is counted when it returns.
        """
        self._cache_tracker.reset()

    @property
    def cache_tracker(self) -> CacheGroupTracker:
        """The pool's live :class:`~async_batch_llm.CacheGroupTracker` (provisional).

        Save it with ``pool.cache_tracker.to_dict()`` and pass
        ``CacheGroupTracker.from_dict(saved)`` as ``cache_tracker=`` to a later
        pool to carry the statistics across a restart, or call
        ``pool.cache_tracker.below_floor(...)`` to apply a floor yourself.
        """
        return self._cache_tracker

    async def aclose(self) -> None:
        """Stop accepting work, drain in-flight requests, then run cleanup().

        Marks the pool closed (new submits raise immediately), then waits for
        already-admitted requests — running *or* still waiting on the semaphore —
        to drain before cleaning up the shared strategy, whose clients/caches may
        still be in use. The drain waits indefinitely; set ``submit_timeout`` to
        bound how long an admitted request can hold up shutdown.

        Concurrent callers share one ordered close attempt, so every
        ``await gw.aclose()`` returns only once cleanup has actually completed.
        A successful step is never repeated; a failed one is retried by the
        next call. Cancelling one waiter defers that cancellation until the
        shared attempt finishes; cancelling it a second time force-aborts.
        """
        # Set synchronously so new submits are rejected immediately, even before
        # the close attempt is scheduled.
        self._closed = True
        report = await self._closer.close()
        report.raise_first()

    def _cleanup_steps(self) -> list[CleanupAction]:
        return [
            CleanupStep("gateway in-flight drain", self._drain_inflight, barrier=True),
            CleanupPhase("host resources", self._host._cleanup_steps),
        ]

    async def _drain_inflight(self) -> None:
        # No new admissions once _closed is set, so _inflight only decreases.
        if self._inflight > 0:
            await self._idle.wait()


__all__ = ["LLMCallPool"]
