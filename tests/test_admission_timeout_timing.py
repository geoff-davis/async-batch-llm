"""A capacity wait ended by a deadline or abort keeps its timing (#181)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from async_batch_llm import (
    AbortMode,
    GuardrailConfig,
    LLMCallStrategy,
    ProcessorConfig,
    RetryConfig,
    process_prompts,
)
from async_batch_llm.base import RetryState, TokenUsage, WorkItemResult
from async_batch_llm.observers import BaseObserver, ProcessingEvent

WAIT_FLOOR = 0.02  # every scenario holds capacity for well over this


class Holder(LLMCallStrategy[str]):
    """One capacity slot. ``hold`` keeps it until ``release`` is set, even when
    cancelled, so a waiting item can only leave through its own guardrail."""

    max_concurrency = 1

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.release = asyncio.Event()
        self.held = asyncio.Event()
        self.fail_once: set[str] = set()

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls.append(prompt)
        if prompt in self.fail_once:
            self.fail_once.discard(prompt)
            raise ConnectionError("transient")
        if prompt.startswith("quick"):
            await asyncio.sleep(0.03)
            return prompt, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None
        self.held.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.wait_for(self.release.wait(), timeout=2)
            raise
        raise AssertionError("unreachable")


class ReleaseWhenFailed(BaseObserver):
    def __init__(self, strategy: Holder, item_id: str) -> None:
        self.strategy = strategy
        self.item_id = item_id

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        if event is ProcessingEvent.ITEM_FAILED and data["item_id"] == self.item_id:
            self.strategy.release.set()


class FailAfter(LLMCallStrategy[str]):
    """Unlimited-capacity strategy that fails with an abort category."""

    async def execute(self, prompt, attempt, timeout, state=None):
        await asyncio.sleep(0.04)
        raise ValueError("abort trigger")


def _by_id(results: list[WorkItemResult]) -> dict[str, WorkItemResult]:
    return {result.item_id: result for result in results}


def _assert_wait_recorded(result: WorkItemResult) -> None:
    assert result.timing is not None
    assert result.admission_wait_seconds >= WAIT_FLOOR
    assert result.admission_wait_seconds == pytest.approx(result.timing.admission_wait_seconds)
    assert result.timing.attempts[-1].admission_wait_seconds >= WAIT_FLOOR


@pytest.mark.asyncio
async def test_item_deadline_in_capacity_wait_keeps_wait_and_category() -> None:
    strategy = Holder()
    try:
        batch = await process_prompts(
            strategy,
            [("hold", "hold"), ("wait", "wait")],
            config=ProcessorConfig(
                max_workers=2, guardrails=GuardrailConfig(total_timeout_per_item=0.06)
            ),
            observers=[ReleaseWhenFailed(strategy, "wait")],
        )
    finally:
        strategy.release.set()
    results = _by_id(batch.results)
    waiting = results["wait"]
    assert strategy.calls == ["hold"]
    assert waiting.error_category == "framework_total_item_timeout"
    _assert_wait_recorded(waiting)
    assert waiting.timing.timeout_category == "admission_timeout"
    assert waiting.timing.attempts[-1].timeout_category == "admission_timeout"
    # The holder timed out inside its provider call: unchanged classification.
    assert results["hold"].timing.timeout_category == "framework_total_item_timeout"
    assert results["hold"].admission_wait_seconds < WAIT_FLOOR


@pytest.mark.asyncio
async def test_batch_deadline_in_capacity_wait_keeps_wait_and_category() -> None:
    strategy = Holder()
    try:
        batch = await process_prompts(
            strategy,
            [("hold", "hold"), ("wait", "wait")],
            config=ProcessorConfig(
                max_workers=2,
                guardrails=GuardrailConfig(batch_timeout=0.06, abort_mode=AbortMode.CANCEL_ACTIVE),
            ),
            observers=[ReleaseWhenFailed(strategy, "wait")],
        )
    finally:
        strategy.release.set()
    waiting = _by_id(batch.results)["wait"]
    assert strategy.calls == ["hold"]
    assert waiting.error_category == "batch_deadline_exceeded"
    _assert_wait_recorded(waiting)
    assert waiting.timing.timeout_category == "admission_timeout"


@pytest.mark.asyncio
async def test_abort_in_capacity_wait_keeps_wait_without_timeout_category() -> None:
    strategy = Holder()
    try:
        batch = await _abort_run(strategy)
    finally:
        strategy.release.set()
    waiting = _by_id(batch.results)["wait"]
    assert strategy.calls == ["hold"]
    assert waiting.error_category == "batch_aborted"
    _assert_wait_recorded(waiting)
    assert waiting.timing.timeout_category is None


async def _abort_run(strategy: Holder):
    from async_batch_llm import LLMWorkItem, ParallelBatchProcessor

    config = ProcessorConfig(
        max_workers=3,
        guardrails=GuardrailConfig(
            abort_on_error_categories=frozenset({"logic_error"}),
            abort_mode=AbortMode.CANCEL_ACTIVE,
        ),
    )
    async with ParallelBatchProcessor(
        config=config, observers=[ReleaseWhenFailed(strategy, "wait")]
    ) as processor:
        await processor.add_work(LLMWorkItem("hold", strategy, "hold"))
        await processor.add_work(LLMWorkItem("wait", strategy, "wait"))
        await processor.add_work(LLMWorkItem("trigger", FailAfter(), "trigger"))
        return await processor.process_all()


@pytest.mark.asyncio
async def test_capacity_waits_accumulate_across_attempts() -> None:
    strategy = Holder()
    strategy.fail_once.add("wait")
    first_failure = asyncio.Event()

    class SignalFirstFailure(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
            if event is ProcessingEvent.ITEM_ADMITTED and data["item_id"] == "wait":
                first_failure.set()

    async def prompts():
        yield "quick", "quick"  # holds capacity ~30 ms, so attempt 1 waits
        yield "wait", "wait"
        await first_failure.wait()
        yield "hold", "hold"  # takes capacity during the retry backoff

    try:
        batch = await process_prompts(
            strategy,
            prompts(),
            config=ProcessorConfig(
                max_workers=3,
                retry=RetryConfig(max_attempts=3, initial_wait=0.02, max_wait=0.02, jitter=False),
                guardrails=GuardrailConfig(total_timeout_per_item=0.2),
            ),
            observers=[SignalFirstFailure(), ReleaseWhenFailed(strategy, "wait")],
        )
    finally:
        strategy.release.set()
    waiting = _by_id(batch.results)["wait"]
    assert strategy.calls[:2] == ["quick", "wait"]
    assert waiting.error_category == "framework_total_item_timeout"
    attempts = waiting.timing.attempts
    assert len(attempts) == 2
    assert attempts[0].admission_wait_seconds >= 0.01
    assert attempts[1].admission_wait_seconds >= WAIT_FLOOR
    assert attempts[1].timeout_category == "admission_timeout"
    assert waiting.admission_wait_seconds == pytest.approx(
        attempts[0].admission_wait_seconds + attempts[1].admission_wait_seconds
    )
