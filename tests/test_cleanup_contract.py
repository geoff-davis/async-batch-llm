"""Executable conformance tests for the cleanup and stream-terminal contract.

Each test is derived from one clause of ``docs/cleanup-lifecycle-contract.md``
and drives a public surface (``async with``, ``shutdown()``, ``process_all()``,
``start()``/``finish()``/``results()``, ``aclose()``, ``call_result()``,
``process_prompts()``/``process_stream()``). Tests never call private
finalizers directly and never rely on fakes that swallow cancellation.

Clause tags in test names: c1 timing/ordering, c2 cancellation deliveries,
c3 portable classification, c4 retry/idempotency, c5 stream finalization.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import subprocess
import sys
import textwrap
import time
import weakref
from pathlib import Path
from typing import Any

import pytest

import async_batch_llm.base as base_module
import async_batch_llm.streaming as streaming_module
from async_batch_llm import (
    ArtifactIdentity,
    ArtifactIOError,
    BaseObserver,
    CleanupInterruptedError,
    JsonlArtifactStore,
    LLMGateway,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessingEvent,
    ProcessorConfig,
    RateLimitConfig,
    RetryConfig,
    RetryState,
    SqliteArtifactStore,
    StreamFinalizationError,
    call_result,
    process_prompts,
    process_stream,
)
from async_batch_llm._internal import cleanup as cleanup_module
from async_batch_llm._internal.admission import QuotaGate
from async_batch_llm._internal.executor_host import ExecutorHost
from async_batch_llm.base import TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy

_TOKENS: TokenUsage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
_SRC = Path(__file__).resolve().parents[1] / "src" / "async_batch_llm"


# --------------------------------------------------------------------------- #
# Fakes that behave like real user code (no cancellation swallowing).
# --------------------------------------------------------------------------- #


class _Strategy(LLMCallStrategy[str]):
    def __init__(
        self,
        name: str = "s",
        log: list[str] | None = None,
        *,
        cleanup_delay: float = 0.0,
        cleanup_error: BaseException | None = None,
        cleanup_error_factory: Any = None,
        execute_delay: float = 0.0,
    ) -> None:
        self.name = name
        self.log = log if log is not None else []
        self.cleanup_delay = cleanup_delay
        self.cleanup_error = cleanup_error
        self.cleanup_error_factory = cleanup_error_factory
        self.execute_delay = execute_delay
        self.cleanup_calls = 0
        self.cleaned = False

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        del attempt, timeout, state
        if self.execute_delay:
            await asyncio.sleep(self.execute_delay)
        return prompt, _TOKENS

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        self.log.append(f"{self.name}:cleanup:start")
        if self.cleanup_delay:
            await asyncio.sleep(self.cleanup_delay)
        if self.cleanup_error_factory is not None:
            raise self.cleanup_error_factory()
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.cleaned = True
        self.log.append(f"{self.name}:cleanup:done")


class _StubbornStrategy(_Strategy):
    """A worker running this keeps going for ``linger`` seconds after cancel."""

    def __init__(self, log: list[str], *, linger: float) -> None:
        super().__init__("stubborn", log)
        self.linger = linger

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        del attempt, timeout, state
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.log.append("worker:cancel-received")
            await asyncio.sleep(self.linger)
            self.log.append("worker:done")
            raise
        return prompt, _TOKENS


class _Store:
    """Fake ArtifactStore with observable close ordering and injectable errors."""

    def __init__(
        self,
        log: list[str] | None = None,
        *,
        close_delay: float = 0.0,
        close_error: BaseException | None = None,
        append_error: BaseException | None = None,
        append_error_for: set[str] | None = None,
    ) -> None:
        self.log = log if log is not None else []
        self.close_delay = close_delay
        self.close_error = close_error
        self.append_error = append_error
        self.append_error_for = append_error_for
        self.close_calls = 0
        self.closed = False

    async def prepare_item(self, work_item: Any) -> Any:
        return work_item.item_id

    async def lookup(self, work_item: Any, prepared_item: Any, policy: Any) -> Any:
        return None

    async def append(self, work_item: Any, prepared_item: Any, result: Any) -> None:
        if self.append_error is not None and (
            self.append_error_for is None or work_item.item_id in self.append_error_for
        ):
            raise self.append_error

    def iter_results(self, *, successes_only: bool = False) -> Any:
        raise NotImplementedError

    async def close(self) -> None:
        self.close_calls += 1
        self.log.append("store:close:start")
        if self.close_delay:
            await asyncio.sleep(self.close_delay)
        if self.close_error is not None:
            raise self.close_error
        self.closed = True
        self.log.append("store:close:done")


def _processor(**kwargs: Any) -> ParallelBatchProcessor[Any, str, Any]:
    kwargs.setdefault("config", ProcessorConfig(max_workers=1, attempt_timeout=5.0))
    return ParallelBatchProcessor(**kwargs)


async def _run_one(processor: ParallelBatchProcessor[Any, str, Any], strategy: _Strategy) -> None:
    await processor.add_work(LLMWorkItem("one", strategy, "prompt"))
    result = await processor.process_all()
    assert result.succeeded == 1


def _warnings(caplog: pytest.LogCaptureFixture, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


async def _collect(processor: ParallelBatchProcessor[Any, str, Any]) -> list[Any]:
    return [r async for r in processor.results()]


# --------------------------------------------------------------------------- #
# Clause 1 — timing and ordering
# --------------------------------------------------------------------------- #


async def test_c1_no_global_deadline_slow_cleanup_runs_to_completion() -> None:
    assert not hasattr(cleanup_module, "CleanupTimeoutError")
    strategy = _Strategy(cleanup_delay=0.3)
    started = time.perf_counter()
    async with _processor() as processor:
        await _run_one(processor, strategy)
    assert strategy.cleaned
    assert time.perf_counter() - started >= 0.3


async def test_c1_slow_step_warns_once_without_changing_behavior(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cleanup_module, "CLEANUP_SLOW_WARNING_SECONDS", 0.05)
    caplog.set_level(logging.WARNING)
    strategy = _Strategy(cleanup_delay=0.2)
    async with _processor() as processor:
        await _run_one(processor, strategy)
    assert strategy.cleaned
    assert len(_warnings(caplog, "still running after")) == 1


async def test_c1_shutdown_order_runtime_then_admission_then_strategies_then_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    strategy = _StubbornStrategy(log, linger=0.1)
    store = _Store(log)
    processor = _processor(artifact_store=store)
    original = processor._admission_registry.shutdown

    async def recording_shutdown() -> None:
        log.append("admission:shutdown")
        await original()

    monkeypatch.setattr(processor._admission_registry, "shutdown", recording_shutdown)

    processor.start()
    await processor.add_work(LLMWorkItem("one", strategy, "prompt"))
    await asyncio.sleep(0.05)  # the worker is now inside execute()
    await processor.shutdown()

    assert log.index("worker:done") < log.index("admission:shutdown")
    assert log.index("admission:shutdown") < log.index("stubborn:cleanup:start")
    assert log.index("stubborn:cleanup:done") < log.index("store:close:start")
    assert store.closed


async def test_c1_ordinary_failure_does_not_prevent_sibling_cleanup() -> None:
    log: list[str] = []
    bad = _Strategy("bad", log, cleanup_error=ValueError("bad cleanup"))
    good = _Strategy("good", log)
    store = _Store(log)
    processor = _processor(
        config=ProcessorConfig(max_workers=2, attempt_timeout=5.0), artifact_store=store
    )
    await processor.add_work(LLMWorkItem("a", bad, "a"))
    await processor.add_work(LLMWorkItem("b", good, "b"))
    await processor.process_all()

    with pytest.raises(ValueError, match="bad cleanup"):
        await processor.shutdown()

    assert good.cleaned
    assert store.closed


async def test_c1_secondary_cleanup_failures_are_logged_with_tracebacks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    first = _Strategy("first", cleanup_error=ValueError("first failure"))
    second = _Strategy("second", cleanup_error=ValueError("second failure"))
    processor = _processor(config=ProcessorConfig(max_workers=2, attempt_timeout=5.0))
    await processor.add_work(LLMWorkItem("a", first, "a"))
    await processor.add_work(LLMWorkItem("b", second, "b"))
    await processor.process_all()

    with pytest.raises(ValueError):
        await processor.shutdown()

    logged = [r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info]
    messages = " ".join(r.getMessage() for r in logged)
    assert "first failure" in messages
    assert "second failure" in messages


async def test_c1_gateway_drain_waits_for_inflight_and_warns_when_slow(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cleanup_module, "CLEANUP_SLOW_WARNING_SECONDS", 0.02)
    caplog.set_level(logging.WARNING)
    strategy = _Strategy(execute_delay=0.15)
    gateway = LLMGateway(strategy, config=ProcessorConfig(max_workers=1))
    running = asyncio.create_task(gateway.submit_result("slow"))
    await asyncio.sleep(0.01)

    await gateway.aclose()

    result = await running
    assert result.success
    assert strategy.cleaned
    assert len(_warnings(caplog, "still running after")) == 1


async def test_c1_live_async_progress_callback_is_a_barrier_in_batch_mode() -> None:
    log: list[str] = []
    store = _Store(log)

    async def progress(completed: int, total: int, item_id: str) -> None:
        log.append("progress:start")
        await asyncio.sleep(0.15)
        log.append("progress:done")

    processor = _processor(
        config=ProcessorConfig(max_workers=1, attempt_timeout=5.0, progress_callback_timeout=None),
        artifact_store=store,
        progress_callback=progress,
    )
    await _run_one(processor, _Strategy())

    assert log.index("progress:done") < log.index("store:close:start")


async def test_c1_live_async_progress_callback_is_a_barrier_in_streaming_mode() -> None:
    log: list[str] = []
    store = _Store(log)

    async def progress(completed: int, total: int, item_id: str) -> None:
        log.append("progress:start")
        await asyncio.sleep(0.15)
        log.append("progress:done")

    processor = _processor(
        config=ProcessorConfig(max_workers=1, attempt_timeout=5.0, progress_callback_timeout=None),
        artifact_store=store,
        progress_callback=progress,
    )
    processor.start()
    await processor.add_work(LLMWorkItem("one", _Strategy(), "prompt"))
    await processor.finish()
    results = await _collect(processor)
    await processor.shutdown()

    assert len(results) == 1
    assert log.index("progress:done") < log.index("store:close:start")


async def test_c1_synchronous_progress_thread_is_a_barrier() -> None:
    log: list[str] = []
    store = _Store(log)

    def progress(completed: int, total: int, item_id: str) -> None:
        log.append("progress:start")
        time.sleep(0.15)
        log.append("progress:done")

    processor = _processor(
        config=ProcessorConfig(max_workers=1, attempt_timeout=5.0, progress_callback_timeout=0.01),
        artifact_store=store,
        progress_callback=progress,
    )
    await _run_one(processor, _Strategy())

    assert log.index("progress:done") < log.index("store:close:start")


async def test_c1_synchronous_post_processor_thread_is_a_barrier() -> None:
    log: list[str] = []
    store = _Store(log)

    def post(result: Any) -> None:
        log.append("post:start")
        time.sleep(0.15)
        log.append("post:done")

    processor = _processor(
        config=ProcessorConfig(max_workers=1, attempt_timeout=5.0, post_processor_timeout=0.01),
        artifact_store=store,
        post_processor=post,
    )
    await _run_one(processor, _Strategy())

    assert log.index("post:done") < log.index("store:close:start")


async def test_c1_worker_threshold_is_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(base_module, "WORKER_CANCELLATION_TIMEOUT", 0.05)
    caplog.set_level(logging.WARNING)
    log: list[str] = []
    strategy = _StubbornStrategy(log, linger=0.2)
    store = _Store(log)
    processor = _processor(artifact_store=store)
    processor.start()
    await processor.add_work(LLMWorkItem("one", strategy, "prompt"))
    await asyncio.sleep(0.05)

    await processor.shutdown()

    assert log.index("worker:done") < log.index("store:close:start")
    assert len(_warnings(caplog, "still stopping after")) == 1


async def test_c1_progress_threshold_is_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(base_module, "PROGRESS_TASK_CANCELLATION_TIMEOUT", 0.05)
    caplog.set_level(logging.WARNING)
    log: list[str] = []
    cancelled = False

    async def progress(completed: int, total: int, item_id: str) -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            cancelled = True
            raise
        log.append("progress:done")

    store = _Store(log)
    processor = _processor(
        config=ProcessorConfig(max_workers=1, attempt_timeout=5.0, progress_callback_timeout=None),
        artifact_store=store,
        progress_callback=progress,
    )
    processor.start()
    await processor.add_work(LLMWorkItem("one", _Strategy(), "prompt"))
    await asyncio.sleep(0.05)  # the item is done; its progress callback is running

    await processor.shutdown()

    assert not cancelled
    assert log.index("progress:done") < log.index("store:close:start")
    assert len(_warnings(caplog, "still running after")) == 1


# --------------------------------------------------------------------------- #
# Clause 2 — cancellation deliveries
# --------------------------------------------------------------------------- #


async def _context_body(
    processor: ParallelBatchProcessor[Any, str, Any], strategy: _Strategy
) -> None:
    async with processor:
        await _run_one(processor, strategy)
        await asyncio.sleep(30)


async def test_c2_first_cancellation_defers_until_teardown_completes() -> None:
    log: list[str] = []
    strategy = _Strategy("s", log, cleanup_delay=0.2)
    store = _Store(log)
    processor = _processor(artifact_store=store)
    task = asyncio.create_task(_context_body(processor, strategy))
    await asyncio.sleep(0.05)

    cancelled_at = time.perf_counter()
    task.cancel()
    await asyncio.wait([task], timeout=2)

    assert task.done() and task.cancelled()
    assert strategy.cleaned
    assert store.closed
    assert time.perf_counter() - cancelled_at >= 0.2


async def test_c2_cancellation_stays_primary_over_ordinary_cleanup_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    strategy = _Strategy(cleanup_error=ValueError("cleanup boom"))
    store = _Store()
    processor = _processor(artifact_store=store)
    task = asyncio.create_task(_context_body(processor, strategy))
    await asyncio.sleep(0.05)

    task.cancel()
    await asyncio.wait([task], timeout=2)

    assert task.cancelled()
    assert store.closed
    assert _warnings(caplog, "cleanup boom")


async def test_c2_second_cancellation_force_aborts_and_skips_dependents(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    log: list[str] = []
    strategy = _Strategy("s", log, cleanup_delay=1.0)
    store = _Store(log)
    processor = _processor(artifact_store=store)

    async def streaming_body() -> None:
        # A started-but-unfinished stream: the strategy is prepared and the
        # store is still open, so both are pending when cleanup begins.
        async with processor:
            processor.start()
            await processor.add_work(LLMWorkItem("one", strategy, "prompt"))
            await asyncio.sleep(30)

    task = asyncio.create_task(streaming_body())
    await asyncio.sleep(0.05)

    task.cancel()
    await asyncio.sleep(0.05)  # first delivery: strategy cleanup is now in progress
    assert not task.done()
    second_at = time.perf_counter()
    task.cancel()
    await asyncio.wait([task], timeout=2)

    assert task.cancelled()
    assert time.perf_counter() - second_at < 0.5
    assert not strategy.cleaned
    assert not store.closed
    assert _warnings(caplog, "abandon")
    assert _warnings(caplog, "skipped")


async def test_c2_shutdown_call_cancelled_once_finishes_then_reraises() -> None:
    strategy = _Strategy(cleanup_delay=0.2)
    store = _Store()
    processor = _processor(artifact_store=store)
    await _run_one(processor, strategy)

    closing = asyncio.create_task(processor.shutdown())
    await asyncio.sleep(0.05)
    closing.cancel()
    await asyncio.wait([closing], timeout=2)

    assert closing.cancelled()
    assert strategy.cleaned
    assert store.closed


async def test_c2_cancelled_single_call_finishes_cleanup_and_stays_cancelled() -> None:
    strategy = _Strategy(execute_delay=30, cleanup_delay=0.1)
    task = asyncio.create_task(call_result(strategy, "prompt"))
    await asyncio.sleep(0.05)

    task.cancel()
    await asyncio.wait([task], timeout=2)

    assert task.cancelled()
    assert strategy.cleaned


# --------------------------------------------------------------------------- #
# Clause 3 — portable classification
# --------------------------------------------------------------------------- #


def test_c3_no_task_state_introspection_in_package() -> None:
    offenders = []
    for path in _SRC.rglob("*.py"):
        text = path.read_text()
        if ".cancelling()" in text or ".uncancel(" in text:
            offenders.append(path.name)
    assert offenders == []


async def test_c3_cancelled_error_raised_inside_cleanup_is_an_interruption_not_caller_cancel() -> (
    None
):
    strategy = _Strategy(cleanup_error_factory=lambda: asyncio.CancelledError("inner"))
    processor = _processor()
    await _run_one(processor, strategy)

    with pytest.raises(CleanupInterruptedError) as info:
        await processor.shutdown()

    assert isinstance(info.value.__cause__, asyncio.CancelledError)
    current = asyncio.current_task()
    assert current is not None and not current.cancelled()

    strategy.cleanup_error_factory = None
    await processor.shutdown()
    assert strategy.cleanup_calls == 2
    assert strategy.cleaned


async def test_c3_cleanup_task_cancelled_by_third_party_is_retryable() -> None:
    class SelfCancelling(_Strategy):
        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            if self.cleanup_calls == 1:
                task = asyncio.current_task()
                assert task is not None
                task.cancel()  # a third party cancelling the private step task
                await asyncio.sleep(0)
            self.cleaned = True

    strategy = SelfCancelling()
    processor = _processor()
    await _run_one(processor, strategy)

    with pytest.raises(CleanupInterruptedError):
        await processor.shutdown()
    await processor.shutdown()

    assert strategy.cleanup_calls == 2
    assert strategy.cleaned


async def test_c3_interruption_is_exception_not_base_exception() -> None:
    assert issubclass(CleanupInterruptedError, Exception)
    assert not issubclass(CleanupInterruptedError, asyncio.CancelledError)
    assert issubclass(StreamFinalizationError, Exception)
    assert not issubclass(StreamFinalizationError, asyncio.CancelledError)


# --------------------------------------------------------------------------- #
# Clause 4 — retry and idempotency
# --------------------------------------------------------------------------- #


async def test_c4_successful_steps_are_never_repeated_failed_steps_are_retried() -> None:
    good = _Strategy("good")
    bad = _Strategy("bad", cleanup_error=ValueError("bad"))
    store = _Store()
    processor = _processor(
        config=ProcessorConfig(max_workers=2, attempt_timeout=5.0), artifact_store=store
    )
    await processor.add_work(LLMWorkItem("a", good, "a"))
    await processor.add_work(LLMWorkItem("b", bad, "b"))
    await processor.process_all()

    with pytest.raises(ValueError):
        await processor.shutdown()
    assert bad.cleanup_calls == 1  # never retried inside the same close call

    bad.cleanup_error = None
    await processor.shutdown()
    await processor.shutdown()

    assert good.cleanup_calls == 1
    assert bad.cleanup_calls == 2
    assert store.close_calls == 1


async def test_c4_concurrent_close_calls_share_one_attempt() -> None:
    strategy = _Strategy(cleanup_delay=0.1)
    store = _Store(close_delay=0.05)
    processor = _processor(artifact_store=store)
    await _run_one(processor, strategy)

    await asyncio.gather(processor.shutdown(), processor.shutdown(), processor.shutdown())

    assert strategy.cleanup_calls == 1
    assert store.close_calls == 1


async def test_c4_context_exit_then_shutdown_and_shutdown_then_context_exit_are_safe() -> None:
    first = _Strategy("first")
    async with _processor() as processor:
        await _run_one(processor, first)
    await processor.shutdown()
    assert first.cleanup_calls == 1

    second = _Strategy("second")
    processor = _processor()
    await _run_one(processor, second)
    await processor.shutdown()
    async with processor:
        pass
    assert second.cleanup_calls == 1


async def test_c4_failed_outcomes_and_tracebacks_are_not_retained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Transient(Exception):
        pass

    # pytest's log capture would itself retain the exception via exc_info;
    # silence the logger so only framework retention is measured.
    caplog.set_level(logging.CRITICAL, logger="async_batch_llm.base")
    strategy = _Strategy(cleanup_error_factory=lambda: Transient("transient"))
    processor = _processor()
    await _run_one(processor, strategy)

    with pytest.raises(Transient) as info:
        await processor.shutdown()
    ref = weakref.ref(info.value)
    del info
    gc.collect()

    assert ref() is None


async def test_c4_new_preparation_is_rejected_once_close_starts() -> None:
    processor = _processor()
    await _run_one(processor, _Strategy())
    await processor.shutdown()

    with pytest.raises(RuntimeError):
        await processor._ensure_strategy_prepared(_Strategy("late"))


async def test_c4_user_cleanup_idempotency_is_documented() -> None:
    api = (Path(__file__).resolve().parents[1] / "docs" / "API.md").read_text()
    assert "idempotent" in api
    assert "idempotent" in (LLMCallStrategy.cleanup.__doc__ or "")


# --------------------------------------------------------------------------- #
# Clause 5 — stream finalization
# --------------------------------------------------------------------------- #


async def test_c5_worker_crash_raises_original_exception_never_clean_eos() -> None:
    error = ArtifactIOError("append failed")
    store = _Store(append_error=error)
    processor = _processor(artifact_store=store)
    processor.start()
    await processor.add_work(LLMWorkItem("one", _Strategy(), "prompt"))
    await processor.finish()

    with pytest.raises(ArtifactIOError) as info:
        await asyncio.wait_for(_collect(processor), timeout=2)
    assert info.value is error
    await processor.shutdown()


async def test_c5_consumer_receives_queued_results_then_the_failure() -> None:
    error = ArtifactIOError("append failed")
    store = _Store(append_error=error, append_error_for={"two"})
    processor = _processor(artifact_store=store)
    processor.start()
    await processor.add_work(LLMWorkItem("one", _Strategy(), "one"))
    await processor.add_work(LLMWorkItem("two", _Strategy(), "two"))
    await processor.finish()
    await asyncio.sleep(0.05)  # both items are handled before the consumer reads

    seen: list[str] = []
    with pytest.raises(ArtifactIOError):
        async for result in processor.results():
            seen.append(result.item_id)
    assert seen == ["one"]
    await processor.shutdown()


async def test_c5_shutdown_before_finish_reports_failure_not_clean_eos() -> None:
    processor = _processor()
    processor.start()
    await processor.add_work(LLMWorkItem("one", _Strategy(), "prompt"))
    await asyncio.sleep(0.05)
    await processor.shutdown()

    with pytest.raises(StreamFinalizationError):
        await asyncio.wait_for(_collect(processor), timeout=1)


async def test_c5_finalizer_cancelled_by_shutdown_reports_failure_with_cause() -> None:
    log: list[str] = []
    strategy = _Strategy("s", log, execute_delay=30)
    processor = _processor()
    processor.start()
    await processor.add_work(LLMWorkItem("one", strategy, "prompt"))
    await processor.finish()
    consumer = asyncio.create_task(_collect(processor))
    await asyncio.sleep(0.05)

    await processor.shutdown()

    with pytest.raises(StreamFinalizationError) as info:
        await asyncio.wait_for(consumer, timeout=1)
    assert isinstance(info.value.__cause__, asyncio.CancelledError)


async def test_c5_terminal_is_durable_for_concurrent_and_later_consumers() -> None:
    error = ArtifactIOError("append failed")
    store = _Store(append_error=error)
    processor = _processor(artifact_store=store)
    processor.start()
    first = asyncio.create_task(_collect(processor))
    second = asyncio.create_task(_collect(processor))
    await asyncio.sleep(0)
    await processor.add_work(LLMWorkItem("one", _Strategy(), "prompt"))
    await processor.finish()

    done, _ = await asyncio.wait({first, second}, timeout=2)
    assert done == {first, second}
    assert isinstance(first.exception(), ArtifactIOError)
    assert isinstance(second.exception(), ArtifactIOError)
    with pytest.raises(ArtifactIOError):
        await asyncio.wait_for(_collect(processor), timeout=1)
    await processor.shutdown()
    with pytest.raises(ArtifactIOError):
        await asyncio.wait_for(_collect(processor), timeout=1)


async def test_c5_late_result_after_terminal_is_delivered_then_failure_raised() -> None:
    error = ArtifactIOError("append failed")
    store = _Store(append_error=error, append_error_for={"fast"})
    slow = _Strategy("slow", execute_delay=0.1)
    processor = _processor(
        config=ProcessorConfig(max_workers=2, attempt_timeout=5.0), artifact_store=store
    )
    processor.start()
    await processor.add_work(LLMWorkItem("slow", slow, "slow"))
    await processor.add_work(LLMWorkItem("fast", _Strategy(), "fast"))
    await asyncio.sleep(0.2)  # terminal decided by the crash; the slow result lands after it

    seen: list[str] = []
    with pytest.raises(ArtifactIOError):
        async for result in processor.results():
            seen.append(result.item_id)
    assert seen == ["slow"]
    await processor.shutdown()


async def test_c5_live_progress_callback_prevents_clean_end_of_stream() -> None:
    log: list[str] = []

    async def progress(completed: int, total: int, item_id: str) -> None:
        await asyncio.sleep(0.15)
        log.append("progress:done")

    processor = _processor(
        config=ProcessorConfig(max_workers=1, attempt_timeout=5.0, progress_callback_timeout=None),
        progress_callback=progress,
    )
    processor.start()
    await processor.add_work(LLMWorkItem("one", _Strategy(), "prompt"))
    await processor.finish()
    results = await _collect(processor)
    log.append("eos")
    await processor.shutdown()

    assert len(results) == 1
    assert log == ["progress:done", "eos"]


_PROCESS_CONTROL_SCRIPT = textwrap.dedent(
    """
    import asyncio, sys
    from async_batch_llm import (
        BaseObserver, LLMWorkItem, ParallelBatchProcessor, ProcessingEvent, ProcessorConfig,
    )
    from async_batch_llm.llm_strategies import LLMCallStrategy

    class S(LLMCallStrategy):
        async def execute(self, prompt, attempt, timeout, state=None):
            return prompt, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    class Boom(BaseObserver):
        async def on_event(self, event, data):
            if event is ProcessingEvent.BATCH_COMPLETED:
                raise {EXC}

    async def main():
        processor = ParallelBatchProcessor(config=ProcessorConfig(max_workers=1), observers=[Boom()])
        processor.start()
        await processor.add_work(LLMWorkItem("one", S(), "prompt"))
        await processor.finish()
        try:
            async for _ in processor.results():
                pass
        except BaseException as exc:
            print("CONSUMER_SAW", type(exc).__name__, flush=True)
            raise
        print("CLEAN_EOS", flush=True)

    asyncio.run(main())
    """
)


@pytest.mark.parametrize(
    ("exc", "expected_code"),
    [("SystemExit(3)", 3), ("KeyboardInterrupt()", None)],
)
def test_c5_process_control_from_finalization_retains_type(
    exc: str, expected_code: int | None
) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _PROCESS_CONTROL_SCRIPT.replace("{EXC}", exc)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "CLEAN_EOS" not in proc.stdout
    if expected_code is not None:
        assert proc.returncode == expected_code
    else:
        assert proc.returncode != 0
        assert "KeyboardInterrupt" in proc.stderr


# --------------------------------------------------------------------------- #
# Public surfaces: manual shutdown closes real stores; high-level API policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
async def test_manual_shutdown_closes_real_artifact_store(tmp_path: Path, backend: str) -> None:
    identity = ArtifactIdentity(provider="test", model="test")
    if backend == "jsonl":
        store: Any = JsonlArtifactStore(tmp_path / "run.jsonl", identity=identity)
    else:
        store = SqliteArtifactStore(tmp_path / "run.sqlite", identity=identity)

    processor = _processor(artifact_store=store)
    await processor.add_work(LLMWorkItem("one", _Strategy(), "prompt"))
    await processor.shutdown()
    await processor.shutdown()

    assert store._closed is True
    if backend == "sqlite":
        assert store._executor_shutdown is True


async def test_process_stream_early_exit_closes_store_and_strategy() -> None:
    store = _Store()
    strategy = _Strategy()
    async with contextlib.aclosing(
        process_stream(strategy, ["a", "b", "c"], artifact_store=store)
    ) as stream:
        async for _ in stream:
            break
    assert store.closed
    assert strategy.cleaned


async def test_process_stream_input_error_is_primary_and_store_is_closed() -> None:
    store = _Store()
    strategy = _Strategy(cleanup_error=ValueError("cleanup boom"))

    async def prompts() -> Any:
        yield "a"
        raise LookupError("input source failed")

    with pytest.raises(LookupError):
        async for _ in process_stream(strategy, prompts(), artifact_store=store):
            pass
    assert store.closed
    assert strategy.cleanup_calls == 1


async def test_high_level_apis_agree_strategy_cleanup_failure_preserves_results(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    collected = await process_prompts(_Strategy(cleanup_error=ValueError("boom")), ["a", "b"])
    assert collected.succeeded == 2

    streamed = [
        r async for r in process_stream(_Strategy(cleanup_error=ValueError("boom")), ["a", "b"])
    ]
    assert len(streamed) == 2

    result = await call_result(_Strategy(cleanup_error=ValueError("boom")), "a")
    assert result.success
    assert len(_warnings(caplog, "boom")) >= 3


async def test_high_level_apis_raise_artifact_close_failure() -> None:
    close_error = ArtifactIOError("close failed")
    with pytest.raises(ArtifactIOError):
        await process_prompts(_Strategy(), ["a"], artifact_store=_Store(close_error=close_error))
    with pytest.raises(ArtifactIOError):
        async for _ in process_stream(
            _Strategy(), ["a"], artifact_store=_Store(close_error=close_error)
        ):
            pass


async def test_gateway_body_error_is_primary_and_close_is_idempotent() -> None:
    strategy = _Strategy(cleanup_error=ValueError("cleanup boom"))
    with pytest.raises(LookupError):
        async with LLMGateway(strategy, config=ProcessorConfig(max_workers=1)) as pool:
            await pool.submit("p")
            raise LookupError("body")
    assert strategy.cleanup_calls == 1

    with pytest.raises(ValueError, match="cleanup boom"):
        await pool.aclose()
    assert strategy.cleanup_calls == 2
    strategy.cleanup_error = None
    await pool.aclose()
    await pool.aclose()
    assert strategy.cleanup_calls == 3


async def test_context_body_exception_wins_over_cleanup_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    strategy = _Strategy(cleanup_error=ValueError("cleanup boom"))
    with pytest.raises(LookupError, match="body"):
        async with _processor() as processor:
            await _run_one(processor, strategy)
            raise LookupError("body")
    assert _warnings(caplog, "cleanup boom")


async def test_keyboard_interrupt_from_cleanup_retains_type_over_cancellation() -> None:
    class Interrupting(_Strategy):
        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            raise KeyboardInterrupt()

    strategy = Interrupting()
    processor = _processor()
    await _run_one(processor, strategy)
    # Cancel this task so the cancellation is already propagating when the
    # close begins (first delivery); the step's KeyboardInterrupt must still
    # win and keep its type. Awaited directly: a KeyboardInterrupt escaping
    # a separate task would tear down the event loop.
    current = asyncio.current_task()
    assert current is not None
    current.cancel()
    with pytest.raises(KeyboardInterrupt):
        await processor.shutdown()


async def test_batch_worker_late_failure_is_raised_after_finalization(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Observer(BaseObserver):
        def __init__(self) -> None:
            self.completed = False

        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
            if event is ProcessingEvent.BATCH_COMPLETED:
                self.completed = True

    observer = Observer()
    store = _Store()
    processor = _processor(artifact_store=store, observers=[observer])
    original = processor._worker

    async def crashing_worker(worker_id: int) -> None:
        await original(worker_id)
        raise RuntimeError("worker died after draining")

    processor._worker = crashing_worker  # type: ignore[method-assign]
    await processor.add_work(LLMWorkItem("one", _Strategy(), "prompt"))

    with pytest.raises(RuntimeError, match="worker died"):
        await processor.process_all()
    assert observer.completed
    assert store.closed


# --------------------------------------------------------------------------- #
# Review round 1 on this branch: paths the clause tests did not cover
# --------------------------------------------------------------------------- #


async def test_c1_cancelled_batch_keeps_callback_thread_barrier_for_later_shutdown() -> None:
    """Cancelling process_all() while it waits for a callback thread must not
    drop the thread pool: a later shutdown() still waits for the thread before
    closing the artifact store."""
    log: list[str] = []
    store = _Store(log)

    def progress(completed: int, total: int, item_id: str) -> None:
        log.append("progress:start")
        time.sleep(0.3)
        log.append("progress:done")

    processor = _processor(
        config=ProcessorConfig(max_workers=1, attempt_timeout=5.0, progress_callback_timeout=0.01),
        artifact_store=store,
        progress_callback=progress,
    )
    await processor.add_work(LLMWorkItem("one", _Strategy(), "prompt"))
    run = asyncio.create_task(processor.process_all())
    await asyncio.sleep(0.1)  # item done; process_all() is waiting on the callback thread
    run.cancel()
    await asyncio.wait([run], timeout=2)
    assert run.cancelled()

    await processor.shutdown()

    assert "progress:done" in log
    assert log.index("progress:done") < log.index("store:close:start")


async def test_c5_finalizer_cancelled_before_start_still_decides_failure() -> None:
    """A finalizer task cancelled before its coroutine ran never reached the
    code that publishes a terminal; the stop path must decide one so a
    consumer does not block forever. (White-box trigger: only a cancel-all
    sweep reaches this ordering through public calls.)"""
    processor = _processor()
    processor.start()
    await processor.add_work(LLMWorkItem("one", _Strategy(), "prompt"))
    await processor.finish()
    assert processor._finalize_task is not None
    processor._finalize_task.cancel()

    await processor.shutdown()

    with pytest.raises(StreamFinalizationError) as info:
        await asyncio.wait_for(_collect(processor), timeout=1)
    assert isinstance(info.value.__cause__, asyncio.CancelledError)


async def test_c1_failed_cooldown_shutdown_does_not_skip_same_scope_quota_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _processor(config=ProcessorConfig(max_workers=1, max_requests_per_minute=1000))
    strategy = _Strategy()
    await _run_one(processor, strategy)
    registry = processor._admission_registry
    state = registry.resolve(strategy)
    calls: list[str] = []
    original_cooldown = state.cooldown.shutdown
    original_gate = state.quota_gate.shutdown

    async def failing_cooldown() -> None:
        calls.append("cooldown")
        raise ValueError("cooldown boom")

    async def recording_gate() -> None:
        calls.append("gate")
        await original_gate()

    monkeypatch.setattr(state.cooldown, "shutdown", failing_cooldown)
    monkeypatch.setattr(state.quota_gate, "shutdown", recording_gate)

    with pytest.raises(ValueError, match="cooldown boom"):
        await processor.shutdown()
    assert calls == ["cooldown", "gate"]
    assert registry._scope_entries  # the failed scope stays for retry

    monkeypatch.setattr(state.cooldown, "shutdown", original_cooldown)
    await processor.shutdown()
    assert not registry._scope_entries


async def test_c3_coordinator_shutdown_does_not_swallow_caller_cancellation() -> None:
    host: ExecutorHost[Any, str, Any] = ExecutorHost(
        ProcessorConfig(
            max_workers=1,
            rate_limit=RateLimitConfig(cooldown_seconds=5.0, slow_start_items=0),
        )
    )
    coordinator = host._rate_limit_coord
    waiter = asyncio.create_task(coordinator.handle_rate_limit(worker_id=0))
    await asyncio.sleep(0.01)  # the owned cooldown task is now pending

    closing = asyncio.create_task(coordinator.shutdown())
    await asyncio.sleep(0)  # closing is awaiting the cancelled cooldown task
    closing.cancel()
    await asyncio.wait([closing], timeout=1)

    try:
        assert closing.cancelled(), "the caller's own cancellation was swallowed"
    finally:
        await asyncio.wait([waiter], timeout=1)
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await host.aclose()


async def test_process_stream_input_error_is_primary_over_reporter_close_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR)

    async def failing_aclose(self: Any, **kwargs: Any) -> None:
        raise RuntimeError("bar close failed")

    monkeypatch.setattr(streaming_module._ProgressReporter, "aclose", failing_aclose)

    async def prompts() -> Any:
        yield "a"
        raise LookupError("input failed")

    with pytest.raises(LookupError, match="input failed"):
        async for _ in process_stream(_Strategy(), prompts(), progress=True):
            pass
    assert _warnings(caplog, "bar close failed")

    # With no other error the reporter failure is a runtime failure and raises.
    with pytest.raises(RuntimeError, match="bar close failed"):
        await process_prompts(_Strategy(), ["a"], progress=True)


# --------------------------------------------------------------------------- #
# Review round 2 on this branch: owned-task handles and outcomes
# --------------------------------------------------------------------------- #


def _cooldown_host() -> ExecutorHost[Any, str, Any]:
    return ExecutorHost(
        ProcessorConfig(
            max_workers=1,
            rate_limit=RateLimitConfig(cooldown_seconds=5.0, slow_start_items=0),
        )
    )


async def test_c4_cancelled_coordinator_shutdown_rejoins_finalization_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry after a cancelled shutdown must wait for the owned *finalization*
    to complete (not merely for the cancelled wrapper task to settle) and must
    not re-cancel the wrapper, which would cut the shielded finalizer loose."""
    host = _cooldown_host()
    coordinator = host._rate_limit_coord
    original_finalize = coordinator._finalize_cooldown
    finalization_finished = False

    async def slow_finalize(*args: Any, **kwargs: Any) -> None:
        nonlocal finalization_finished
        await asyncio.sleep(0.3)
        await original_finalize(*args, **kwargs)
        finalization_finished = True

    monkeypatch.setattr(coordinator, "_finalize_cooldown", slow_finalize)
    waiter = asyncio.create_task(coordinator.handle_rate_limit(worker_id=0))
    await asyncio.sleep(0.01)

    closing = asyncio.create_task(coordinator.shutdown())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.wait([closing], timeout=1)
    assert closing.cancelled()
    assert not finalization_finished

    try:
        await coordinator.shutdown()  # the retry
        assert finalization_finished, "retry returned before the finalization completed"
        assert not coordinator._in_cooldown
        await asyncio.wait([waiter], timeout=1)
        assert waiter.done(), "the reporting worker was never released"
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await host.aclose()


async def test_c4_cooldown_task_cancelled_before_start_is_still_finalized() -> None:
    """Cancelling the owned task before its first instruction means its own
    cancellation handler never runs; shutdown() must finalize the generation
    itself so the paused worker is released."""
    host = _cooldown_host()
    coordinator = host._rate_limit_coord
    waiter = asyncio.create_task(coordinator.handle_rate_limit(worker_id=0))
    await asyncio.sleep(0)  # the cooldown task exists but has not run yet
    assert coordinator._cooldown_task is not None
    assert coordinator._in_cooldown

    try:
        await coordinator.shutdown()
        assert not coordinator._in_cooldown
        await asyncio.wait([waiter], timeout=1)
        assert waiter.done(), "the reporting worker was never released"
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await host.aclose()


async def test_c4_failed_cooldown_finalization_stays_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _cooldown_host()
    coordinator = host._rate_limit_coord
    original_finalize = coordinator._finalize_cooldown
    attempts = 0

    async def flaky_finalize(*args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("finalization failed")
        await original_finalize(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_finalize_cooldown", flaky_finalize)
    waiter = asyncio.create_task(coordinator.handle_rate_limit(worker_id=0))
    await asyncio.sleep(0.01)

    try:
        with pytest.raises(RuntimeError, match="finalization failed"):
            await coordinator.shutdown()
        assert coordinator._in_cooldown, "a failed finalization must not be checkpointed"

        await coordinator.shutdown()  # the retry re-runs finalization
        assert not coordinator._in_cooldown
        await asyncio.wait([waiter], timeout=1)
        assert waiter.done(), "the reporting worker was never released"
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await host.aclose()


async def test_c1_owned_cooldown_task_failure_is_surfaced_by_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _cooldown_host()
    coordinator = host._rate_limit_coord

    original_finalize = coordinator._finalize_cooldown

    async def failing_finalize(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("cooldown teardown failed")

    monkeypatch.setattr(coordinator, "_finalize_cooldown", failing_finalize)
    waiter = asyncio.create_task(coordinator.handle_rate_limit(worker_id=0))
    await asyncio.sleep(0.01)

    try:
        with pytest.raises(RuntimeError, match="cooldown teardown failed"):
            await coordinator.shutdown()
        assert coordinator._in_cooldown, "a failed finalization must not be checkpointed"
    finally:
        # The failure is retryable: with a working finalizer the host close
        # finalizes the generation and releases the waiter.
        monkeypatch.setattr(coordinator, "_finalize_cooldown", original_finalize)
        await host.aclose()
        await asyncio.wait([waiter], timeout=1)
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
    assert not coordinator._in_cooldown


async def _pending_wake_gate(sleep: Any) -> tuple[QuotaGate, asyncio.Task[None]]:
    gate = QuotaGate(max_requests_per_minute=1, sleep=sleep)
    gate._request_available = 0.0  # a full deficit: the wake sleeps ~60s
    gate._schedule_wake(None)
    assert gate._wake_task is not None
    await asyncio.sleep(0)  # let the wake task enter its sleep
    return gate, gate._wake_task


async def test_c4_cancelled_gate_shutdown_rejoins_wake_task_on_retry() -> None:
    cleanup_finished = False

    async def lingering_sleep(delay: float) -> None:
        nonlocal cleanup_finished
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)  # a second cancel would cut this short
            cleanup_finished = True
            raise

    gate, owned = await _pending_wake_gate(lingering_sleep)
    closing = asyncio.create_task(gate.shutdown())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.wait([closing], timeout=1)
    assert closing.cancelled()
    assert not owned.done()

    await gate.shutdown()  # the retry re-joins; it must not re-cancel
    assert owned.done()
    assert cleanup_finished, "retry re-cancelled the wake task instead of re-joining it"


async def test_c1_owned_wake_task_failure_is_surfaced_by_gate_shutdown() -> None:
    async def failing_sleep(delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise RuntimeError("wake teardown failed") from None

    gate, _ = await _pending_wake_gate(failing_sleep)
    with pytest.raises(RuntimeError, match="wake teardown failed"):
        await gate.shutdown()


# --------------------------------------------------------------------------- #
# Round 4 — cancellation the coordinator did not send, and retained failures
# --------------------------------------------------------------------------- #


async def test_c3_observer_cancellation_during_cooldown_end_is_an_interruption() -> None:
    """A ``CancelledError`` raised by an observer while COOLDOWN_ENDED is being
    delivered was not sent by shutdown(): the owned task must not finalize the
    generation a second time (which would deliver COOLDOWN_ENDED twice), and
    shutdown() must report the interruption instead of accepting it as a
    successful teardown. The next explicit close is clean."""
    ended = 0

    class _CancelOnEnd(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
            nonlocal ended
            if event is ProcessingEvent.COOLDOWN_ENDED:
                ended += 1
                if ended == 1:
                    raise asyncio.CancelledError()

    class _RateLimitOnce(_Strategy):
        calls = 0

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 throttled")
            return prompt, _TOKENS

    processor = _processor(
        config=ProcessorConfig(
            max_workers=1,
            retry=RetryConfig(max_attempts=1, max_rate_limit_retries=1, initial_wait=0.01),
            rate_limit=RateLimitConfig(
                cooldown_seconds=0, max_cooldown_seconds=0, slow_start_items=0
            ),
        ),
        observers=[_CancelOnEnd()],
    )
    with pytest.raises(CleanupInterruptedError):
        async with processor:
            await processor.add_work(LLMWorkItem("one", _RateLimitOnce(), "prompt"))
            result = await processor.process_all()
            assert result.succeeded == 1
    assert ended == 1, "the interrupted generation was finalized a second time"
    await processor.shutdown()  # the interruption was reported once; nothing is left to redo


async def test_c3_third_party_cancel_of_cooldown_task_is_surfaced_then_finalized_later() -> None:
    """An owned cooldown task cancelled by someone other than shutdown() is an
    interruption: the first close reports it without finalizing on the task's
    behalf inside the same call, and the next explicit close finalizes."""
    host = _cooldown_host()
    coordinator = host._rate_limit_coord
    waiter = asyncio.create_task(coordinator.handle_rate_limit(worker_id=0))
    await asyncio.sleep(0.01)
    owned = coordinator._cooldown_task
    assert owned is not None
    owned.cancel()  # not shutdown's cancellation
    await asyncio.wait([owned], timeout=1)
    assert owned.cancelled()

    try:
        with pytest.raises(CleanupInterruptedError):
            await coordinator.shutdown()
        assert coordinator._in_cooldown, "an interrupted step must not be finalized in-call"
        assert not waiter.done()

        await coordinator.shutdown()  # the later explicit close retries
        assert not coordinator._in_cooldown
        await asyncio.wait([waiter], timeout=1)
        assert waiter.done(), "the reporting worker was never released"
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await host.aclose()


async def test_c4_interrupted_cooldown_finalization_is_not_retried_within_one_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owned task can be cancelled by a third party while it is finalizing
    the generation, before the state transition. That is an interruption: the
    same close must report it rather than run a second finalization, and the
    next explicit close finalizes."""
    host = _cooldown_host()
    coordinator = host._rate_limit_coord
    original_finalize = coordinator._finalize_cooldown
    release = asyncio.Event()
    attempts = 0

    async def gated_finalize(*args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        await release.wait()
        await original_finalize(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_finalize_cooldown", gated_finalize)
    waiter = asyncio.create_task(coordinator.handle_rate_limit(worker_id=0))
    await asyncio.sleep(0.01)
    owned = coordinator._cooldown_task
    assert owned is not None

    closing = asyncio.create_task(coordinator.shutdown())
    for _ in range(20):
        await asyncio.sleep(0)
        if attempts == 1:
            break
    assert attempts == 1
    owned.cancel()  # a third party interrupts the owned finalization

    try:
        with pytest.raises(CleanupInterruptedError):
            await closing
        assert attempts == 1, "the interrupted finalization was retried within one close"
        assert coordinator._in_cooldown

        release.set()
        await coordinator.shutdown()  # the later explicit close retries
        assert attempts == 2
        assert not coordinator._in_cooldown
        await asyncio.wait([waiter], timeout=1)
        assert waiter.done(), "the reporting worker was never released"
    finally:
        release.set()
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await host.aclose()


async def test_c4_surfaced_cooldown_failure_is_not_retained_after_successful_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.CRITICAL, logger="async_batch_llm")
    host = _cooldown_host()
    coordinator = host._rate_limit_coord
    original_finalize = coordinator._finalize_cooldown
    held: list[weakref.ref[Any]] = []

    class _Sentinel:
        pass

    async def flaky_finalize(*args: Any, **kwargs: Any) -> None:
        if not held:
            sentinel = _Sentinel()  # reachable only through the failure's traceback
            held.append(weakref.ref(sentinel))
            raise RuntimeError("finalization failed")
        await original_finalize(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_finalize_cooldown", flaky_finalize)
    waiter = asyncio.create_task(coordinator.handle_rate_limit(worker_id=0))
    await asyncio.sleep(0.01)

    try:
        failed = False
        try:
            await coordinator.shutdown()
        except RuntimeError:
            failed = True
        assert failed
        gc.collect()
        assert held[0]() is None, "the failure is retained after the call that raised it"
        assert coordinator._in_cooldown
        await coordinator.shutdown()  # the successful retry
        assert not coordinator._in_cooldown
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await host.aclose()


# --------------------------------------------------------------------------- #
# Round 5 — pending third-party cancellations and delayed retries
# --------------------------------------------------------------------------- #


async def test_c3_pending_third_party_cancel_of_cooldown_task_is_not_relabelled() -> None:
    """A third-party cancellation that is already pending when shutdown()
    starts must still be classified as an interruption. Shutdown may not send
    a cancellation of its own that makes the outcome look expected."""
    host = _cooldown_host()
    coordinator = host._rate_limit_coord
    waiter = asyncio.create_task(coordinator.handle_rate_limit(worker_id=0))
    await asyncio.sleep(0.01)
    owned = coordinator._cooldown_task
    assert owned is not None
    owned.cancel()  # pending: not yet delivered when shutdown() begins

    try:
        with pytest.raises(CleanupInterruptedError):
            await coordinator.shutdown()
        assert coordinator._in_cooldown
        await coordinator.shutdown()
        assert not coordinator._in_cooldown
        await asyncio.wait([waiter], timeout=1)
        assert waiter.done(), "the reporting worker was never released"
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await host.aclose()


async def test_c3_pending_third_party_cancel_of_wake_task_is_not_relabelled() -> None:
    gate, owned = await _pending_wake_gate(asyncio.sleep)
    owned.cancel()  # pending: not yet delivered when shutdown() begins
    with pytest.raises(CleanupInterruptedError):
        await gate.shutdown()
    await gate.shutdown()  # reported once


async def test_c1_wake_task_outcome_survives_a_delayed_retry() -> None:
    """If a shutdown call is cancelled and the wake task settles before the
    retry, the retry must still observe the task's outcome."""

    async def failing_sleep(delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise RuntimeError("wake teardown failed") from None

    gate, owned = await _pending_wake_gate(failing_sleep)
    closing = asyncio.create_task(gate.shutdown())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.wait([closing], timeout=1)
    assert closing.cancelled()
    await asyncio.wait([owned], timeout=1)
    assert owned.done()

    with pytest.raises(RuntimeError, match="wake teardown failed"):
        await gate.shutdown()  # the delayed retry
    await gate.shutdown()  # reported once


async def test_c1_settled_third_party_cancel_of_wake_task_survives_a_delayed_retry() -> None:
    async def lingering_sleep(delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            raise

    gate, owned = await _pending_wake_gate(lingering_sleep)
    closing = asyncio.create_task(gate.shutdown())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.wait([closing], timeout=1)
    assert closing.cancelled()
    owned.cancel()  # a third party, after shutdown has started
    await asyncio.wait([owned], timeout=1)
    assert owned.cancelled()

    with pytest.raises(CleanupInterruptedError):
        await gate.shutdown()  # the delayed retry
    await gate.shutdown()  # reported once


async def test_c4_gate_failure_is_not_retained_after_the_call_that_raised_it() -> None:
    held: list[weakref.ref[Any]] = []

    class _Sentinel:
        pass

    async def failing_sleep(delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            sentinel = _Sentinel()  # reachable only through the failure's traceback
            held.append(weakref.ref(sentinel))
            raise RuntimeError("wake teardown failed") from None

    gate = (await _pending_wake_gate(failing_sleep))[0]  # hold no reference to the task
    failed = False
    try:
        await gate.shutdown()
    except RuntimeError:
        failed = True
    assert failed
    gc.collect()
    assert held[0]() is None, "the failure is retained after the call that raised it"


# --------------------------------------------------------------------------- #
# Round 6 — overlapping cooldown generations and the stop-aware sleep's children
# --------------------------------------------------------------------------- #


async def test_c1_older_cooldown_task_still_delivering_events_is_waited_for_and_reported() -> None:
    """Workers are released before COOLDOWN_ENDED observers finish, so an
    immediately retried item can start a newer generation while the older
    owned task is still delivering. Shutdown must wait for every live owned
    task, not only the newest handle, and report the older task's failure."""
    release = asyncio.Event()
    ended = 0

    class _BlockThenCancelFirstEnd(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
            nonlocal ended
            if event is ProcessingEvent.COOLDOWN_ENDED:
                ended += 1
                if ended == 1:
                    await release.wait()
                    raise asyncio.CancelledError()

    class _RateLimitTwice(_Strategy):
        calls = 0

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("429 throttled")
            return prompt, _TOKENS

    processor = _processor(
        config=ProcessorConfig(
            max_workers=1,
            retry=RetryConfig(max_attempts=1, max_rate_limit_retries=2, initial_wait=0.01),
            rate_limit=RateLimitConfig(
                cooldown_seconds=0, max_cooldown_seconds=0, slow_start_items=0
            ),
        ),
        observers=[_BlockThenCancelFirstEnd()],
    )
    await processor.add_work(LLMWorkItem("one", _RateLimitTwice(), "prompt"))
    result = await processor.process_all()
    assert result.succeeded == 1
    assert ended == 2, "the second generation should have finalized while the first blocked"

    closing = asyncio.create_task(processor.shutdown())
    await asyncio.sleep(0.05)
    assert not closing.done(), "shutdown returned while an owned cooldown task was still live"
    release.set()
    with pytest.raises(CleanupInterruptedError):
        await closing
    await processor.shutdown()  # reported once


async def test_c3_third_party_cancel_of_the_sleep_child_is_an_interruption() -> None:
    """The stop-aware sleep must not treat a sleep it did not cancel as
    stopped: with the stop unset, a cancelled sleep is an interruption."""

    async def self_cancelling_sleep(delay: float) -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()  # a third party cancels the private sleep child
        await asyncio.sleep(delay)

    stop = asyncio.Event()
    with pytest.raises(CleanupInterruptedError):
        await cleanup_module.sleep_unless_stopped(self_cancelling_sleep(1), stop, name="sleep")
    assert not stop.is_set()


async def test_c3_third_party_cancel_of_the_stop_watcher_is_an_interruption() -> None:
    """A cancelled stop watcher must not be mistaken for a requested stop."""

    class _SabotagedEvent(asyncio.Event):
        async def wait(self) -> bool:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()  # a third party cancels the private stop watcher
            return await super().wait()

    slept = False

    async def sleep(delay: float) -> None:
        nonlocal slept
        await asyncio.sleep(delay)
        slept = True

    stop = _SabotagedEvent()
    with pytest.raises(CleanupInterruptedError):
        await cleanup_module.sleep_unless_stopped(sleep(1), stop, name="sleep")
    assert not slept and not stop.is_set()


async def test_c3_interrupted_cooldown_sleep_does_not_resume_workers_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interruption of the cooldown sleep is not an ordinary sleep error:
    the generation stays paused, shutdown reports the interruption, and the
    next explicit close finalizes."""
    import async_batch_llm._internal.rate_limit_coordinator as coordinator_module

    async def interrupted_sleep(sleep: Any, *args: Any, **kwargs: Any) -> None:
        sleep.close()  # the coroutine the coordinator hands over is never run here
        raise CleanupInterruptedError("cooldown sleep")

    monkeypatch.setattr(coordinator_module, "sleep_unless_stopped", interrupted_sleep)
    host = _cooldown_host()
    coordinator = host._rate_limit_coord
    waiter = asyncio.create_task(coordinator.handle_rate_limit(worker_id=0))
    await asyncio.sleep(0.05)

    try:
        assert coordinator._in_cooldown, "the interrupted sleep resumed workers early"
        assert not waiter.done()
        with pytest.raises(CleanupInterruptedError):
            await coordinator.shutdown()
        assert coordinator._in_cooldown
        await coordinator.shutdown()
        assert not coordinator._in_cooldown
        await asyncio.wait([waiter], timeout=1)
        assert waiter.done()
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await host.aclose()
