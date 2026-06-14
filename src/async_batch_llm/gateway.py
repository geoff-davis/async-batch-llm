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
    """

    def __init__(
        self,
        strategy: LLMCallStrategy[TOutput],
        *,
        config: ProcessorConfig | None = None,
        error_classifier: Any = None,
    ) -> None:
        cfg = config or ProcessorConfig(max_workers=5)
        self._strategy = strategy
        self._host: ExecutorHost[Any, TOutput, Any] = ExecutorHost(
            cfg, strategy=strategy, error_classifier=error_classifier
        )
        self._sem = asyncio.Semaphore(cfg.max_workers)
        self._seq = 0
        self._closed = False

    async def __aenter__(self) -> LLMGateway[TOutput]:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def submit(self, prompt: str) -> TOutput:
        """Submit one prompt and await its output, raising on failure.

        Blocks on the semaphore when the pool is saturated (backpressure).
        """
        return unwrap_result(await self.submit_result(prompt))

    async def submit_result(self, prompt: str) -> WorkItemResult[TOutput, Any]:
        """Like :meth:`submit` but returns the WorkItemResult instead of raising.

        Gives access to ``token_usage`` / ``metadata`` and lets callers branch
        on ``.success`` without exception handling.
        """
        if self._closed:
            raise RuntimeError("LLMGateway is closed")
        self._seq += 1
        work_item: LLMWorkItem[Any, TOutput, Any] = LLMWorkItem(
            item_id=f"req-{self._seq}", strategy=self._strategy, prompt=prompt
        )
        async with self._sem:
            return await self._host.executor.execute(work_item)

    async def aclose(self) -> None:
        """Stop accepting work and run the strategy's cleanup(). Idempotent."""
        if self._closed:
            return
        self._closed = True
        await self._host.aclose()
