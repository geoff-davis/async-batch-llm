"""High-level streaming API built on the processor's first-class streaming mode.

:func:`process_stream` yields each result as it completes; :func:`process_prompts`
collects them into a :class:`BatchResult`. Both accept a sync **or** async
iterable of prompts — bare strings get auto-generated item ids (``item_0``,
``item_1``, …), ``(item_id, prompt)`` pairs, or ``(item_id, prompt, context)``
triples (the context flows to ``WorkItemResult.context`` and your post_processor).

Because the processor runs workers while work is still being fed, a bounded
``ProcessorConfig.max_queue_size`` becomes backpressure: you can stream an
arbitrarily large (even unbounded) input through **constant memory**, since the
producer blocks on a full queue instead of buffering everything up front.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ._internal.guardrails import BatchAdmissionStopped
from .artifacts import ArtifactStore, ResumePolicy
from .base import BatchResult, BatchTermination, LLMWorkItem, WorkItemResult
from .core import ProcessorConfig
from .parallel import ParallelBatchProcessor

if TYPE_CHECKING:
    from .llm_strategies import LLMCallStrategy

TOutput = TypeVar("TOutput")

# A prompt entry is a bare prompt string, an (item_id, prompt) pair, or an
# (item_id, prompt, context) triple. The context is forwarded to
# LLMWorkItem.context — so it reaches WorkItemResult.context and any
# post_processor (handy for carrying DB ids, gold labels, routing keys, etc.).
PromptEntry = str | tuple[str, str] | tuple[str, str, Any]
PromptSource = Iterable[PromptEntry] | AsyncIterable[PromptEntry]


async def _aiter(source: PromptSource) -> AsyncIterator[Any]:
    """Iterate a sync OR async iterable uniformly.

    Yields ``Any`` because ``isinstance(source, AsyncIterable)`` narrows away the
    element type; callers pass each entry straight to :func:`_to_work_item`,
    which validates the shape.
    """
    if isinstance(source, AsyncIterable):
        try:
            async for entry in source:
                yield entry
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()
    else:
        for entry in source:
            yield entry


def _to_work_item(
    entry: PromptEntry, index: int, strategy: LLMCallStrategy[TOutput]
) -> LLMWorkItem[Any, TOutput, Any]:
    if isinstance(entry, tuple):
        # Cast to a variadic tuple so positional access doesn't depend on the
        # type checker narrowing the tuple length (which it can't do via len()).
        parts = cast("tuple[Any, ...]", entry)
        if len(parts) == 2:
            return LLMWorkItem(item_id=str(parts[0]), strategy=strategy, prompt=str(parts[1]))
        if len(parts) == 3:
            return LLMWorkItem(
                item_id=str(parts[0]),
                strategy=strategy,
                prompt=str(parts[1]),
                context=parts[2],
            )
        raise TypeError(
            "prompt tuple must be (item_id, prompt) or (item_id, prompt, context); "
            f"got a tuple of length {len(parts)} at index {index}."
        )
    return LLMWorkItem(item_id=f"item_{index}", strategy=strategy, prompt=str(entry))


async def _process_stream_impl(
    strategy: LLMCallStrategy[TOutput],
    prompts: PromptSource,
    *,
    config: ProcessorConfig | None = None,
    artifact_store: ArtifactStore | None = None,
    resume: ResumePolicy = ResumePolicy.NONE,
    termination_out: list[Any] | None = None,
    **processor_kwargs: Any,
) -> AsyncIterator[WorkItemResult[TOutput, Any]]:
    """Yield each :class:`WorkItemResult` as it completes, in completion order.

    Args:
        strategy: The LLM call strategy to run for every prompt.
        prompts: Sync or async iterable of bare prompt strings (ids
            auto-generated), ``(item_id, prompt)`` pairs, and/or
            ``(item_id, prompt, context)`` triples (context → result.context).
        config: Processor configuration. Defaults to ``ProcessorConfig()``. Set
            ``max_queue_size`` to bound memory for very large inputs.
        **processor_kwargs: Forwarded to ``ParallelBatchProcessor`` (observers,
            middlewares, ``error_classifier``, ``progress_callback``, …).

    Yields:
        Results (success *and* failure) in the order items finish — generally
        NOT the order submitted. A producer error (e.g. a failing async input)
        propagates to the consumer after already-queued results drain; breaking
        out of the loop early cancels the producer and tears down the workers.
    """
    processor = ParallelBatchProcessor(
        config=config or ProcessorConfig(),
        artifact_store=artifact_store,
        resume=resume,
        **processor_kwargs,
    )
    feed_error: list[BaseException] = []

    async def _feed() -> None:
        try:
            index = 0
            async for entry in _aiter(prompts):
                await processor.add_work(_to_work_item(entry, index, strategy))  # backpressure here
                index += 1
        except asyncio.CancelledError:
            if not processor.aborted:
                raise  # consumer broke out early; cleanup handles teardown
        except BatchAdmissionStopped:
            pass  # controlled batch timeout/fail-fast stopped source admission
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer below
            feed_error.append(exc)
        finally:
            # Normal completion, controlled abort, or producer error: close the
            # accepted stream so every queued item receives a terminal result.
            if not processor._finished:
                await processor.finish()

    async with processor:  # __aexit__ -> cleanup() cancels workers/finalize
        processor.start()
        producer = asyncio.create_task(_feed())

        async def _stop_producer_on_abort() -> None:
            await processor.wait_for_abort()
            if not producer.done():
                producer.cancel()

        abort_watcher = asyncio.create_task(_stop_producer_on_abort())
        try:
            async for result in processor.results():
                yield result
        finally:
            abort_watcher.cancel()
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer
            await asyncio.gather(abort_watcher, return_exceptions=True)

    if termination_out is not None:
        termination_out.append(processor.termination)

    # Reached only when results() ended normally (end-of-stream). If the
    # producer failed, surface its exception now (after draining results).
    if feed_error:
        raise feed_error[0]


async def process_stream(
    strategy: LLMCallStrategy[TOutput],
    prompts: PromptSource,
    *,
    config: ProcessorConfig | None = None,
    artifact_store: ArtifactStore | None = None,
    resume: ResumePolicy = ResumePolicy.NONE,
    **processor_kwargs: Any,
) -> AsyncIterator[WorkItemResult[TOutput, Any]]:
    """Yield terminal results in completion order.

    Ordered streaming is intentionally not offered: a slow early item would
    block later completed results and require a potentially unbounded reorder
    buffer. Use ``process_prompts(..., preserve_order=True)`` when collection is
    acceptable.
    """
    async for result in _process_stream_impl(
        strategy,
        prompts,
        config=config,
        artifact_store=artifact_store,
        resume=resume,
        **processor_kwargs,
    ):
        yield result


async def process_prompts(
    strategy: LLMCallStrategy[TOutput],
    prompts: PromptSource,
    *,
    config: ProcessorConfig | None = None,
    preserve_order: bool = False,
    artifact_store: ArtifactStore | None = None,
    resume: ResumePolicy = ResumePolicy.NONE,
    **processor_kwargs: Any,
) -> BatchResult[TOutput, Any]:
    """Run ``strategy`` over ``prompts`` and collect every result.

    The one-liner entry point — drains :func:`process_stream` into a
    :class:`BatchResult` (whose ``results`` are in completion order). For
    constant-memory processing of very large inputs, use :func:`process_stream`
    directly with a bounded ``config.max_queue_size``.
    """
    termination: list[Any] = []
    results = [
        result
        async for result in _process_stream_impl(
            strategy,
            prompts,
            config=config,
            artifact_store=artifact_store,
            resume=resume,
            termination_out=termination,
            **processor_kwargs,
        )
    ]
    batch = BatchResult(
        results=results,
        termination=termination[0] if termination else BatchTermination(),
    )
    return batch.in_input_order() if preserve_order else batch
