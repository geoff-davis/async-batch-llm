"""High-level streaming API built on the processor's first-class streaming mode.

:func:`process_stream` yields each result as it completes; :func:`process_prompts`
collects them into a :class:`BatchResult`. Both accept a sync **or** async
iterable of prompts — bare strings get auto-generated item ids (``item_0``,
``item_1``, …), ``(item_id, prompt)`` pairs, or ``(item_id, prompt, context)``
triples (the context flows to ``WorkItemResult.context`` and your post_processor).

Because the processor runs workers while work is still being fed, a bounded
``ProcessorConfig.max_queue_size`` becomes input backpressure: the producer
blocks on a full work queue instead of buffering the complete source up front.
The result handoff queue is unbounded, so callers should consume results
promptly when streaming a large or unbounded source.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ._internal.guardrails import BatchAdmissionStopped
from .artifacts import ArtifactStore, ResumePolicy
from .base import BatchResult, BatchTermination, LLMWorkItem, WorkItemResult
from .core import ProcessorConfig
from .parallel import ParallelBatchProcessor

if TYPE_CHECKING:
    from .base import ProgressCallbackFunc
    from .llm_strategies import LLMCallStrategy

logger = logging.getLogger(__name__)

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


class _ProgressReporter:
    """Bundled ``progress=True`` reporter: tqdm when importable, else logging.

    Implements the ``progress_callback(completed, total, current_item_id)``
    contract. The tqdm bar's total follows the processor's running total, so
    streaming sources that keep adding work render correctly. Added in
    v0.19.0 (issue #100).
    """

    def __init__(self) -> None:
        self._bar: Any = None
        self._tqdm_failed = False
        self._last_logged = 0

    def __call__(self, completed: int, total: int, current_item_id: str) -> None:
        if not self._tqdm_failed:
            if self._bar is None:
                try:
                    from tqdm.auto import tqdm  # type: ignore[import-not-found]

                    self._bar = tqdm(total=total, unit="item", desc="batch")
                except ImportError:
                    self._tqdm_failed = True
                    logger.info(
                        "progress=True: tqdm is not installed "
                        "(pip install 'async-batch-llm[progress]'); "
                        "falling back to interval logging."
                    )
            if self._bar is not None:
                if self._bar.total != total:
                    self._bar.total = total
                self._bar.n = completed
                self._bar.refresh()
                return
        # Logging fallback: every 10 items and on completion.
        if completed - self._last_logged >= 10 or completed >= total:
            self._last_logged = completed
            logger.info("progress: %d/%d items completed", completed, total)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _resolve_progress(
    progress: bool | ProgressCallbackFunc,
    processor_kwargs: dict[str, Any],
) -> _ProgressReporter | None:
    """Fold the ``progress=`` option into ``processor_kwargs``.

    Returns the bundled reporter (so the caller can close it) when
    ``progress=True``; a user-supplied callable is passed through untouched.
    """
    if progress is False:
        return None
    if "progress_callback" in processor_kwargs:
        raise ValueError(
            "Pass either progress= or progress_callback=, not both. "
            "progress=True is a convenience wrapper that installs a bundled "
            "progress_callback."
        )
    if progress is True:
        reporter = _ProgressReporter()
        processor_kwargs["progress_callback"] = reporter
        return reporter
    processor_kwargs["progress_callback"] = progress
    return None


def _apply_concurrency_shorthand(
    config: ProcessorConfig | None, concurrency: int | None
) -> ProcessorConfig | None:
    """Fold a ``concurrency=N`` shorthand into the effective config.

    With no config, builds ``ProcessorConfig(concurrency=N)``. With a config
    whose ``concurrency`` is unset, returns a derived copy (a default-valued
    ``max_workers`` is treated as unset so the knob can size it). A config
    that already sets a different ``concurrency`` raises.
    """
    if concurrency is None:
        return config
    if config is None:
        return ProcessorConfig(concurrency=concurrency)
    if config.concurrency is not None:
        if config.concurrency == concurrency:
            return config
        raise ValueError(
            f"Conflicting concurrency: config.concurrency={config.concurrency} "
            f"but concurrency={concurrency} was also passed. Set it in one place."
        )
    overrides: dict[str, Any] = {"concurrency": concurrency}
    if config.max_workers == 5:
        # The historical default — treat as unset so derivation applies.
        overrides["max_workers"] = None
    return dataclasses.replace(config, **overrides)


async def _process_stream_impl(
    strategy: LLMCallStrategy[TOutput],
    prompts: PromptSource,
    *,
    config: ProcessorConfig | None = None,
    concurrency: int | None = None,
    progress: bool | ProgressCallbackFunc = False,
    artifact_store: ArtifactStore | None = None,
    resume: ResumePolicy = ResumePolicy.NONE,
    termination_out: list[BatchTermination] | None = None,
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
    config = _apply_concurrency_shorthand(config, concurrency)
    reporter = _resolve_progress(progress, processor_kwargs)
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
            if reporter is not None:
                reporter.close()

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
    concurrency: int | None = None,
    progress: bool | ProgressCallbackFunc = False,
    artifact_store: ArtifactStore | None = None,
    resume: ResumePolicy = ResumePolicy.NONE,
    **processor_kwargs: Any,
) -> AsyncIterator[WorkItemResult[TOutput, Any]]:
    """Yield terminal results in completion order.

    Ordered streaming is intentionally not offered: a slow early item would
    block later completed results and require a potentially unbounded reorder
    buffer. Use ``process_prompts(..., preserve_order=True)`` when collection is
    acceptable.

    ``concurrency=N`` is shorthand for ``ProcessorConfig(concurrency=N)`` —
    the single knob that coherently sizes workers, provider admission, and
    built-in model connection pools (v0.19.0).
    """
    async for result in _process_stream_impl(
        strategy,
        prompts,
        config=config,
        concurrency=concurrency,
        progress=progress,
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
    concurrency: int | None = None,
    progress: bool | ProgressCallbackFunc = False,
    preserve_order: bool = False,
    artifact_store: ArtifactStore | None = None,
    resume: ResumePolicy = ResumePolicy.NONE,
    **processor_kwargs: Any,
) -> BatchResult[TOutput, Any]:
    """Run ``strategy`` over ``prompts`` and collect every result.

    The one-liner entry point — drains :func:`process_stream` into a
    :class:`BatchResult` (whose ``results`` are in completion order). For
    lazy, bounded-input processing of very large inputs, use
    :func:`process_stream` directly with a positive
    ``config.max_queue_size`` and consume results promptly.

    ``concurrency=N`` is shorthand for ``ProcessorConfig(concurrency=N)`` —
    the single knob that coherently sizes workers, provider admission, and
    built-in model connection pools (v0.19.0).

    ``progress=True`` renders a tqdm bar when tqdm is installed
    (``pip install 'async-batch-llm[progress]'``) and falls back to interval
    logging otherwise; pass a callable ``(completed, total, current_item_id)``
    for a custom reporter. Works for both collected and streamed runs
    (v0.19.0).
    """
    termination: list[BatchTermination] = []
    started = time.monotonic()
    results = [
        result
        async for result in _process_stream_impl(
            strategy,
            prompts,
            config=config,
            concurrency=concurrency,
            progress=progress,
            artifact_store=artifact_store,
            resume=resume,
            termination_out=termination,
            **processor_kwargs,
        )
    ]
    batch = BatchResult(
        results=results,
        termination=termination[0] if termination else BatchTermination(),
        wall_time_seconds=time.monotonic() - started,
    )
    return batch.in_input_order() if preserve_order else batch
