"""High-level convenience API for the common case.

Most batches are "run this strategy over these prompts and give me the results."
:func:`process_prompts` and :func:`process_stream` collapse the
add-work / process_all dance into one call, while the full
:class:`~async_batch_llm.ParallelBatchProcessor` API stays available for
post-processors, middleware, observers, and custom queueing.

Both helpers accept prompts as either bare strings (item ids are auto-generated
as ``item_0``, ``item_1``, …) or ``(item_id, prompt)`` pairs.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Iterable
from typing import TYPE_CHECKING, Any, TypeVar

from .base import BatchResult, LLMWorkItem, WorkItemResult
from .core import ProcessorConfig
from .parallel import ParallelBatchProcessor

if TYPE_CHECKING:
    from .base import PostProcessorFunc
    from .llm_strategies import LLMCallStrategy

TOutput = TypeVar("TOutput")

# Accepted prompt input: bare strings or (item_id, prompt) pairs.
PromptInput = Iterable[str] | Iterable[tuple[str, str]]


def _normalize_prompts(prompts: PromptInput) -> list[tuple[str, str]]:
    """Coerce the accepted prompt shapes into ``(item_id, prompt)`` pairs.

    - ``"..."``            -> ``("item_<i>", "...")`` (auto id by position)
    - ``("id", "...")``    -> used as-is
    """
    normalized: list[tuple[str, str]] = []
    for i, entry in enumerate(prompts):
        if isinstance(entry, str):
            normalized.append((f"item_{i}", entry))
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            item_id, prompt = entry
            normalized.append((str(item_id), str(prompt)))
        else:
            raise TypeError(
                "prompts must be strings or (item_id, prompt) pairs; got "
                f"{type(entry).__name__} {entry!r} at index {i}."
            )
    return normalized


async def process_prompts(
    strategy: LLMCallStrategy[TOutput],
    prompts: PromptInput,
    config: ProcessorConfig | None = None,
    **processor_kwargs: Any,
) -> BatchResult[TOutput, None]:
    """Run ``strategy`` over ``prompts`` and return the collected results.

    A one-call wrapper around :class:`ParallelBatchProcessor`: it builds the
    processor, queues every prompt, runs the batch, and returns the
    :class:`BatchResult`.

    Args:
        strategy: The LLM call strategy to run for every prompt.
        prompts: Bare strings (ids auto-generated) or ``(item_id, prompt)`` pairs.
        config: Processor configuration. Defaults to ``ProcessorConfig()``.
        **processor_kwargs: Forwarded to ``ParallelBatchProcessor`` (e.g.
            ``post_processor``, ``observers``, ``middlewares``,
            ``error_classifier``, ``progress_callback``). When
            ``error_classifier`` is omitted it is auto-selected from ``strategy``.

    Returns:
        A :class:`BatchResult` whose ``results`` are in completion order.

    Example:
        >>> result = await process_prompts(strategy, ["Summarize A", "Summarize B"])
        >>> print(result.succeeded, "/", result.total_items)
    """
    items = _normalize_prompts(prompts)
    proc_config = config if config is not None else ProcessorConfig()

    # try/finally rather than `async with` so the return type stays
    # BatchResult (an `async with` whose __aexit__ returns bool reads to the
    # type checker as "might swallow the exception and fall through").
    processor: ParallelBatchProcessor[None, TOutput, None] = ParallelBatchProcessor(
        config=proc_config, **processor_kwargs
    )
    try:
        for item_id, prompt in items:
            await processor.add_work(LLMWorkItem(item_id=item_id, strategy=strategy, prompt=prompt))
        return await processor.process_all()
    finally:
        await processor.shutdown()


async def process_stream(
    strategy: LLMCallStrategy[TOutput],
    prompts: PromptInput,
    config: ProcessorConfig | None = None,
    **processor_kwargs: Any,
) -> AsyncIterator[WorkItemResult[TOutput, None]]:
    """Yield each :class:`WorkItemResult` as it completes (streaming).

    Same inputs as :func:`process_prompts`, but instead of waiting for the whole
    batch it yields results (success *and* failure) the moment each item
    finishes — useful for writing out / displaying progress incrementally.
    Implemented by attaching a post-processor that pushes completed results onto
    an internal queue; any ``post_processor`` you pass is still called first.

    Example:
        >>> async for result in process_stream(strategy, prompts):
        ...     if result.success:
        ...         await save(result.item_id, result.output)
    """
    items = _normalize_prompts(prompts)
    proc_config = config if config is not None else ProcessorConfig()

    # None is the end-of-stream sentinel — work item results are always objects,
    # so it's unambiguous and keeps the queue strictly typed.
    out_queue: asyncio.Queue[WorkItemResult[TOutput, None] | None] = asyncio.Queue()
    user_post: PostProcessorFunc[TOutput, None] | None = processor_kwargs.pop(
        "post_processor", None
    )

    async def streaming_post(result: WorkItemResult[TOutput, None]) -> None:
        # Preserve a caller-supplied post-processor, then publish to the stream.
        if user_post is not None:
            maybe = user_post(result)
            if inspect.isawaitable(maybe):
                await maybe
        await out_queue.put(result)

    processor = ParallelBatchProcessor[None, TOutput, None](
        config=proc_config, post_processor=streaming_post, **processor_kwargs
    )

    async def _drive() -> None:
        # Always enqueue the sentinel so the consumer terminates even on error;
        # the exception (if any) is re-raised when the consumer awaits the task.
        try:
            async with processor:
                for item_id, prompt in items:
                    await processor.add_work(
                        LLMWorkItem(item_id=item_id, strategy=strategy, prompt=prompt)
                    )
                await processor.process_all()
        finally:
            await out_queue.put(None)

    driver = asyncio.create_task(_drive())
    try:
        while True:
            result = await out_queue.get()
            if result is None:
                break
            yield result
        # Surface any error raised inside the batch run.
        await driver
    finally:
        if not driver.done():
            driver.cancel()
            try:
                await driver
            except (asyncio.CancelledError, Exception):
                # Consumer is tearing down early; swallow driver teardown noise.
                pass
