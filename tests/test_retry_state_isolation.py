"""Regression coverage for framework runtime state isolation (v0.24.0).

Originally planned as v0.23.1, Session A; that working version was never published.

``RetryState.data`` belongs to application strategies. Framework deadlines,
try counters, quota accounting and timing live in a private sidecar that no
public ``RetryState`` operation can observe or erase.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import pickle
from typing import Any

from async_batch_llm import (
    BaseObserver,
    GuardrailConfig,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessingEvent,
    ProcessorConfig,
    RateLimitConfig,
    RetryConfig,
    RetryState,
    TokenEstimate,
    call_result,
)
from async_batch_llm._internal.execution_state import reset_attempt_runtime, runtime_state
from async_batch_llm.base import TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy

_TOKENS: TokenUsage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class _ClearThenBlock(LLMCallStrategy[str]):
    def __init__(self, first_error: Exception) -> None:
        self.first_error = first_error
        self.calls = 0
        self.second_cancelled = asyncio.Event()

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        del prompt, attempt, timeout
        self.calls += 1
        assert state is not None
        # A user key that resembles an old implementation key is still just
        # application data and cannot affect the private deadline.
        state.set("_abl_total_item_deadline", None)
        if self.calls == 1:
            raise self.first_error
        try:
            await asyncio.Event().wait()
        finally:
            self.second_cancelled.set()
        return "unreachable", _TOKENS

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ) -> None:
        del exception, attempt
        assert state is not None
        state.clear()


class _SuccessfulStrategy(LLMCallStrategy[str]):
    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        del attempt, timeout, state
        return prompt, _TOKENS


async def test_retry_state_clear_cannot_remove_total_item_deadline() -> None:
    strategy = _ClearThenBlock(RuntimeError("transient"))
    result = await asyncio.wait_for(
        call_result(
            strategy,
            "prompt",
            config=ProcessorConfig(
                attempt_timeout=0.5,
                retry=RetryConfig(max_attempts=2, initial_wait=0.001, jitter=False),
                guardrails=GuardrailConfig(total_timeout_per_item=0.05),
            ),
        ),
        timeout=0.5,
    )

    assert strategy.calls == 2
    assert strategy.second_cancelled.is_set()
    assert result.error_category == "framework_total_item_timeout"
    assert [attempt.try_number for attempt in result.timing.attempts] == [1, 2]


async def test_retry_state_clear_during_rate_limit_keeps_deadline() -> None:
    strategy = _ClearThenBlock(RuntimeError("429 rate limit"))
    result = await call_result(
        strategy,
        "prompt",
        config=ProcessorConfig(
            attempt_timeout=0.5,
            retry=RetryConfig(max_attempts=1, max_rate_limit_retries=2),
            rate_limit=RateLimitConfig(
                cooldown_seconds=0.01,
                max_cooldown_seconds=0.01,
                slow_start_items=0,
            ),
            guardrails=GuardrailConfig(total_timeout_per_item=0.05),
        ),
    )

    assert strategy.calls == 2
    assert result.error_category == "framework_total_item_timeout"
    assert [attempt.try_number for attempt in result.timing.attempts] == [1, 2]


def test_retry_state_public_operations_hide_framework_runtime() -> None:
    state = RetryState({"application": 1, "_abl_total_item_deadline": "user value"})
    item = runtime_state(state)
    item.total_deadline = 123.0
    item.cumulative_admission_wait_seconds = 4.5
    reset_attempt_runtime(state, 7).reserved_tokens = 99

    assert state.data == {"application": 1, "_abl_total_item_deadline": "user value"}
    assert dataclasses.asdict(state) == {"data": state.data}
    assert "123" not in repr(state)
    assert "application" in state

    state.clear()
    assert state.data == {}
    assert item.total_deadline == 123.0
    assert item.cumulative_admission_wait_seconds == 4.5
    assert item.current_attempt.try_number == 7
    assert item.current_attempt.reserved_tokens == 99


def test_retry_state_copy_and_pickle_exclude_private_runtime() -> None:
    state = RetryState({"nested": {"value": 1}})
    runtime_state(state).total_deadline = 123.0

    shallow = copy.copy(state)
    assert shallow.data == state.data
    assert shallow.data is not state.data
    assert runtime_state(shallow) is not runtime_state(state)
    assert runtime_state(shallow).total_deadline is None

    payload = pickle.dumps(state)
    restored = pickle.loads(payload)
    assert b"_internal.execution_state" not in payload
    assert restored.data == state.data
    assert runtime_state(restored).total_deadline is None


def test_retry_state_deepcopy_preserves_self_reference_identity() -> None:
    state = RetryState()
    state.data["self"] = state

    duplicate = copy.deepcopy(state)

    assert duplicate is not state
    assert duplicate.data["self"] is duplicate


async def test_concurrent_retry_states_have_distinct_private_runtime() -> None:
    seen: list[tuple[int, int]] = []

    class Capture(LLMCallStrategy[str]):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            del attempt, timeout
            assert state is not None
            seen.append((id(state), id(runtime_state(state))))
            await asyncio.sleep(0)
            return prompt, _TOKENS

    processor = ParallelBatchProcessor(config=ProcessorConfig(max_workers=2))
    strategy = Capture()
    await processor.add_work(LLMWorkItem("one", strategy, "one"))
    await processor.add_work(LLMWorkItem("two", strategy, "two"))
    result = await processor.process_all()
    await processor.shutdown()

    assert result.succeeded == 2
    assert len({state_id for state_id, _ in seen}) == 2
    assert len({runtime_id for _, runtime_id in seen}) == 2


async def test_quota_events_use_none_before_physical_try_is_assigned() -> None:
    class Estimated(_SuccessfulStrategy):
        def estimate_tokens(
            self, prompt: str, attempt: int, state: RetryState | None
        ) -> TokenEstimate:
            return TokenEstimate(input_tokens=1)

    class Recorder(BaseObserver):
        def __init__(self) -> None:
            self.events: list[tuple[ProcessingEvent, dict[str, Any]]] = []

        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
            self.events.append((event, data))

    observer = Recorder()
    strategy = Estimated()
    processor = ParallelBatchProcessor(
        config=ProcessorConfig(max_workers=1, max_tokens_per_minute=100),
        observers=[observer],
    )
    result = await processor._process_item(
        LLMWorkItem("one", strategy, "prompt"),
        worker_id=0,
        attempt_number=1,
        strategy=strategy,
        retry_state=RetryState(),
    )
    await processor.shutdown()

    assert result.success
    quota_events = [
        data
        for event, data in observer.events
        if event in {ProcessingEvent.QUOTA_ADMITTED, ProcessingEvent.QUOTA_RECONCILED}
    ]
    assert quota_events
    assert all(data["try_number"] is None for data in quota_events)
