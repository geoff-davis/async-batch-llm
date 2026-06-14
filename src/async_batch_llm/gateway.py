"""Queue-less rate-limited gateway: many callers, one shared cooldown.

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
    from async_batch_llm.gateway import LLMGateway

    @asynccontextmanager
    async def lifespan(app):
        strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-4o-mini"))
        async with LLMGateway(strategy, config=ProcessorConfig(max_workers=5)) as gw:
            app.state.llm = gw
            yield

    # in a handler:  summary = await request.app.state.llm.submit(prompt)
"""

from __future__ import annotations

import asyncio
from typing import Any, Generic, TypeVar

from ._internal.executor_host import ExecutorHost
from .base import LLMWorkItem, WorkItemResult
from .core import ProcessorConfig
from .llm_strategies import LLMCallStrategy
from .single import unwrap_result

TOutput = TypeVar("TOutput")


class LLMGateway(Generic[TOutput]):
    """A shared, rate-limited entry point for single LLM calls from many callers.

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
    """

    def __init__(
        self,
        strategy: LLMCallStrategy[TOutput],
        *,
        config: ProcessorConfig | None = None,
        error_classifier: Any = None,
        max_pending: int | None = None,
        submit_timeout: float | None = None,
    ) -> None:
        if max_pending is not None and max_pending < 0:
            raise ValueError(f"max_pending must be >= 0 (got {max_pending})")
        if submit_timeout is not None and submit_timeout <= 0:
            raise ValueError(f"submit_timeout must be > 0 (got {submit_timeout})")

        cfg = config or ProcessorConfig(max_workers=5)
        self._strategy = strategy
        self._host: ExecutorHost[Any, TOutput, Any] = ExecutorHost(
            cfg, strategy=strategy, error_classifier=error_classifier
        )
        self._sem = asyncio.Semaphore(cfg.max_workers)
        self._seq = 0
        self._closed = False

        # Admission cap: max requests in flight (running + waiting). None = off.
        self._max_inflight = None if max_pending is None else cfg.max_workers + max_pending
        self._inflight = 0
        self._submit_timeout = submit_timeout

    async def __aenter__(self) -> LLMGateway[TOutput]:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def submit(self, prompt: str, *, timeout: float | None = None) -> TOutput:
        """Submit one prompt and await its output, raising on failure.

        Blocks on the semaphore when the pool is saturated (backpressure), unless
        ``max_pending`` is set (then an over-cap submit raises immediately). A
        rejected/timed-out submit raises :class:`LLMCallError`. ``timeout``
        overrides the gateway's ``submit_timeout`` for this call.
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

        ``timeout`` overrides the gateway's ``submit_timeout`` for this call.
        """
        if self._closed:
            raise RuntimeError("LLMGateway is closed")

        self._seq += 1
        item_id = f"req-{self._seq}"

        # Admission cap. The check + increment below run with no `await` between
        # them, so in asyncio this is race-free and correctly counts waiters
        # toward the bound.
        if self._max_inflight is not None and self._inflight >= self._max_inflight:
            return WorkItemResult(
                item_id=item_id, success=False, error="gateway saturated", context=None
            )

        effective_timeout = self._submit_timeout if timeout is None else timeout
        work_item: LLMWorkItem[Any, TOutput, Any] = LLMWorkItem(
            item_id=item_id, strategy=self._strategy, prompt=prompt
        )

        self._inflight += 1
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

    async def _run(self, work_item: LLMWorkItem[Any, TOutput, Any]) -> WorkItemResult[TOutput, Any]:
        """Acquire a concurrency slot, run the item, release the slot."""
        async with self._sem:
            return await self._host.executor.execute(work_item)

    async def aclose(self) -> None:
        """Stop accepting work and run the strategy's cleanup(). Idempotent."""
        if self._closed:
            return
        self._closed = True
        await self._host.aclose()
