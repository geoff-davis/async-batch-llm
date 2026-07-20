"""Single-call convenience API: one resilient LLM call, no batch ceremony.

``call`` / ``call_result`` run one prompt through the full resilience pipeline —
error-type-aware retries, the coordinated rate-limit cooldown, and token
accounting — by driving :class:`ItemExecutor` directly over a lightweight host.
No worker pool, queue, or result stream is created.

    from async_batch_llm import OpenAIModel, OpenAIStrategy
    from async_batch_llm.single import call

    strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-4o-mini"))
    summary = await call(strategy, "Summarize: ...")          # output, or raises

    result = await call_result(strategy, "Summarize: ...")     # full WorkItemResult
    if result.success:
        print(result.output, result.token_usage)

For many calls that share context (e.g. a ``GeminiCachedModel``), prefer
:class:`~async_batch_llm.gateway.LLMCallPool` or ``process_prompts`` so the
prepared strategy and coordinator are reused across calls.
"""

from __future__ import annotations

from typing import Any, TypeVar

from ._internal.executor_host import ExecutorHost
from .base import LLMWorkItem, WorkItemResult
from .core import ProcessorConfig
from .llm_strategies import LLMCallStrategy

TOutput = TypeVar("TOutput")

_SINGLE_CALL_ITEM_ID = "single"


class LLMCallError(RuntimeError):
    """Raised by :func:`call` / ``LLMCallPool.submit`` when a request fails and no
    originating provider exception was preserved.

    When the failure carried a provider exception — the usual case for an
    exhausted-retry or permanent error — that exact exception is re-raised
    instead, preserving its type. ``LLMCallError`` is the fallback for failures
    that produce no exception: the pool's admission-cap (``max_pending``) and
    ``submit_timeout`` rejections.

    Carries the originating :class:`WorkItemResult` on ``.result`` so callers
    can still reach token usage, metadata, and the error string.
    """

    def __init__(self, message: str, *, result: WorkItemResult[Any, Any]) -> None:
        super().__init__(message)
        self.result = result


def unwrap_result(result: WorkItemResult[TOutput, Any]) -> TOutput:
    """Return ``result.output`` on success, else raise.

    Re-raises the underlying exception when the executor preserved it
    (``WorkItemResult.exception``); otherwise raises :class:`LLMCallError`.
    """
    if result.success:
        # non-None on success
        return result.output  # type: ignore[return-value]  # ty:ignore[invalid-return-type]
    if result.exception is not None:
        raise result.exception
    raise LLMCallError(result.error or "LLM call failed", result=result)


async def call_result(
    strategy: LLMCallStrategy[TOutput],
    prompt: str,
    *,
    config: ProcessorConfig | None = None,
    error_classifier: Any = None,
) -> WorkItemResult[TOutput, Any]:
    """Run one prompt through the full pipeline and return the WorkItemResult.

    Inspect ``.success`` / ``.output`` / ``.error`` / ``.token_usage``. Use this
    when you want token accounting or to branch on failure without exceptions;
    for the happy-path one-liner use :func:`call`.
    """
    host: ExecutorHost[Any, TOutput, Any] = ExecutorHost(
        config or ProcessorConfig(max_workers=1),
        strategy=strategy,
        error_classifier=error_classifier,
    )
    try:
        work_item: LLMWorkItem[Any, TOutput, Any] = LLMWorkItem(
            item_id=_SINGLE_CALL_ITEM_ID, strategy=strategy, prompt=prompt
        )
        return await host.executor.execute(work_item)
    finally:
        await host.aclose()


async def call(
    strategy: LLMCallStrategy[TOutput],
    prompt: str,
    *,
    config: ProcessorConfig | None = None,
    error_classifier: Any = None,
) -> TOutput:
    """Run one prompt and return its output, raising on failure.

    Set a timeout or retry policy via ``config`` (e.g.
    ``ProcessorConfig(attempt_timeout=20)``).
    """
    result = await call_result(strategy, prompt, config=config, error_classifier=error_classifier)
    return unwrap_result(result)
