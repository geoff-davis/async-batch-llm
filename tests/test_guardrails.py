"""End-to-end item deadlines, batch deadlines, and fail-fast guardrails."""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections import Counter
from typing import Any

import httpx
import pytest
from openai import APIStatusError

from async_batch_llm import (
    AbortMode,
    DefaultErrorClassifier,
    ErrorInfo,
    GuardrailConfig,
    ItemDeadlineExceeded,
    LLMCallStrategy,
    OpenAIErrorClassifier,
    ProcessorConfig,
    RateLimitConfig,
    RetryConfig,
    call_result,
    process_prompts,
    process_stream,
)
from async_batch_llm.base import RetryState, TokenUsage, WorkItemResult
from async_batch_llm.gateway import LLMGateway
from async_batch_llm.observers import BaseObserver, ProcessingEvent

_TOKENS: TokenUsage = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}


class _TransientWithTokens(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self._failed_token_usage = dict(_TOKENS)


class _RetryThenSleep(LLMCallStrategy[str]):
    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls += 1
        raise _TransientWithTokens("retry me")


@pytest.mark.asyncio
async def test_total_item_deadline_spans_retry_backoff_and_preserves_usage() -> None:
    strategy = _RetryThenSleep()
    result = await call_result(
        strategy,
        "x",
        config=ProcessorConfig(
            retry=RetryConfig(max_attempts=3, initial_wait=0.2, max_wait=0.2, jitter=False),
            guardrails=GuardrailConfig(total_timeout_per_item=0.04),
        ),
    )

    assert strategy.calls == 1
    assert not result.success
    assert result.error_category == "framework_total_item_timeout"
    assert isinstance(result.exception, ItemDeadlineExceeded)
    assert result.token_usage["total_tokens"] == 3
    assert result.timing.attempts[0].retry_backoff_seconds >= 0.02
    assert result.timing.total_seconds >= 0.03


class _RateLimitThenSuccess(LLMCallStrategy[str]):
    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 rate limit")
        return "ok", _TOKENS, None


@pytest.mark.asyncio
async def test_total_item_deadline_spans_coordinated_cooldown() -> None:
    strategy = _RateLimitThenSuccess()
    result = await call_result(
        strategy,
        "x",
        config=ProcessorConfig(
            retry=RetryConfig(max_attempts=1, max_rate_limit_retries=3),
            rate_limit=RateLimitConfig(
                cooldown_seconds=0.2,
                slow_start_items=0,
                slow_start_initial_delay=0,
                slow_start_final_delay=0,
                backoff_multiplier=1,
            ),
            guardrails=GuardrailConfig(total_timeout_per_item=0.04),
        ),
    )

    assert strategy.calls == 1
    assert result.error_category == "framework_total_item_timeout"
    assert result.timing.attempts[0].cooldown_wait_seconds >= 0.02


@pytest.mark.asyncio
async def test_total_item_deadline_applies_to_gateway_execution() -> None:
    strategy = _SlowStrategy(0.2)
    async with LLMGateway(
        strategy,
        config=ProcessorConfig(
            max_workers=1,
            guardrails=GuardrailConfig(total_timeout_per_item=0.03),
        ),
    ) as gateway:
        result = await gateway.submit_result("gateway")

    assert result.error_category == "framework_total_item_timeout"
    assert isinstance(result.exception, ItemDeadlineExceeded)
    assert strategy.cancelled == 1


@pytest.mark.asyncio
async def test_total_item_deadline_applies_to_dry_run_execution() -> None:
    class SlowDryRun(LLMCallStrategy[str]):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage, None]:
            raise AssertionError("execute must not be called in dry-run mode")

        async def dry_run(self, prompt: str) -> tuple[str, TokenUsage]:
            await asyncio.sleep(0.2)
            return prompt, _TOKENS

    result = await call_result(
        SlowDryRun(),
        "dry-run",
        config=ProcessorConfig(
            dry_run=True,
            guardrails=GuardrailConfig(total_timeout_per_item=0.03),
        ),
    )

    assert result.error_category == "framework_total_item_timeout"
    assert isinstance(result.exception, ItemDeadlineExceeded)


class _CapacityStrategy(LLMCallStrategy[str]):
    max_concurrency = 1

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls.append(prompt)
        await asyncio.sleep(0.2)
        return prompt, _TOKENS, None


@pytest.mark.asyncio
async def test_total_item_deadline_covers_capacity_and_starts_no_late_call() -> None:
    strategy = _CapacityStrategy()
    result = await process_prompts(
        strategy,
        [("a", "a"), ("b", "b")],
        config=ProcessorConfig(
            max_workers=2,
            guardrails=GuardrailConfig(total_timeout_per_item=0.04),
        ),
        preserve_order=True,
    )
    assert len(strategy.calls) == 1
    assert result.failed == 2
    assert all(item.error_category == "framework_total_item_timeout" for item in result.results)


class _SlowStrategy(LLMCallStrategy[str]):
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls: list[str] = []
        self.cancelled = 0

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls.append(prompt)
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return prompt, _TOKENS, None


@pytest.mark.asyncio
async def test_batch_deadline_drain_active_preserves_active_success_and_aborts_queue() -> None:
    strategy = _SlowStrategy(0.06)
    result = await process_prompts(
        strategy,
        [str(index) for index in range(4)],
        config=ProcessorConfig(
            max_workers=1,
            guardrails=GuardrailConfig(
                batch_timeout=0.02,
                abort_mode=AbortMode.DRAIN_ACTIVE,
            ),
        ),
        preserve_order=True,
    )

    assert strategy.calls == ["0"]
    assert result.results[0].success
    assert all(item.error_category == "batch_deadline_exceeded" for item in result.results[1:])
    assert [item.submission_index for item in result.results] == [0, 1, 2, 3]
    assert result.termination.kind == "batch_timeout"
    assert result.total_items == 4


@pytest.mark.asyncio
async def test_batch_deadline_cancel_active_cancels_provider_and_finishes_stream() -> None:
    strategy = _SlowStrategy(1.0)
    result = await asyncio.wait_for(
        process_prompts(
            strategy,
            [str(index) for index in range(4)],
            config=ProcessorConfig(
                max_workers=1,
                guardrails=GuardrailConfig(
                    batch_timeout=0.02,
                    abort_mode=AbortMode.CANCEL_ACTIVE,
                ),
            ),
        ),
        timeout=0.5,
    )

    assert strategy.cancelled == 1
    assert result.total_items == 4
    assert all(item.error_category == "batch_deadline_exceeded" for item in result.results)
    assert result.termination.kind == "batch_timeout"


@pytest.mark.asyncio
async def test_batch_deadline_stops_async_producer_without_task_leaks() -> None:
    producer_closed = asyncio.Event()
    produced = 0

    async def source():
        nonlocal produced
        try:
            while True:
                produced += 1
                yield f"p{produced}"
        finally:
            producer_closed.set()

    before = {task for task in asyncio.all_tasks() if not task.done()}
    results = [
        item
        async for item in process_stream(
            _SlowStrategy(0.03),
            source(),
            config=ProcessorConfig(
                max_workers=1,
                max_queue_size=2,
                guardrails=GuardrailConfig(
                    batch_timeout=0.02,
                    abort_mode=AbortMode.CANCEL_ACTIVE,
                ),
            ),
        )
    ]
    await asyncio.sleep(0)
    after = {task for task in asyncio.all_tasks() if not task.done()}
    assert producer_closed.is_set()
    # At most one blocked put may win concurrently with the abort; if it did,
    # that now-accepted item must also receive a terminal result.
    assert len(results) <= 4
    assert produced <= 5
    assert after - before == set()


class _CategoryError(Exception):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class _CategoryClassifier:
    def classify(self, exception: Exception) -> ErrorInfo:
        if isinstance(exception, _CategoryError):
            return ErrorInfo(False, False, False, exception.category)
        return DefaultErrorClassifier().classify(exception)


class _CategoryStrategy(LLMCallStrategy[str]):
    def __init__(self, category: str | None) -> None:
        self.category = category
        self.calls: list[str] = []

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls.append(prompt)
        if self.category is not None and prompt == "trigger":
            raise _CategoryError(self.category)
        return prompt, _TOKENS, None


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["insufficient_balance", "authentication"])
async def test_configured_terminal_category_triggers_fail_fast(category: str) -> None:
    strategy = _CategoryStrategy(category)
    result = await process_prompts(
        strategy,
        [("trigger", "trigger"), ("later-1", "later-1"), ("later-2", "later-2")],
        config=ProcessorConfig(
            max_workers=1,
            guardrails=GuardrailConfig(abort_on_error_categories=frozenset({category})),
        ),
        error_classifier=_CategoryClassifier(),
        preserve_order=True,
    )

    assert strategy.calls == ["trigger"]
    assert result.results[0].error_category == category
    assert [item.error_category for item in result.results[1:]] == [
        "batch_aborted",
        "batch_aborted",
    ]
    assert result.termination.kind == "fail_fast"
    assert result.termination.triggering_item_id == "trigger"


@pytest.mark.asyncio
async def test_unconfigured_item_error_does_not_abort_batch() -> None:
    strategy = _CategoryStrategy("validation_error")
    result = await process_prompts(
        strategy,
        [("trigger", "trigger"), ("later", "later")],
        config=ProcessorConfig(max_workers=1),
        error_classifier=_CategoryClassifier(),
    )
    assert strategy.calls == ["trigger", "later"]
    assert result.termination.kind == "completed"
    assert result.total_items == 2


class _AbortObserver(BaseObserver):
    def __init__(self) -> None:
        self.events: Counter[ProcessingEvent] = Counter()

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        self.events[event] += 1


@pytest.mark.asyncio
async def test_concurrent_terminal_failures_trip_abort_once() -> None:
    ready = 0
    release = asyncio.Event()

    class ConcurrentFailure(LLMCallStrategy[str]):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage, None]:
            nonlocal ready
            ready += 1
            if ready == 4:
                release.set()
            await release.wait()
            raise _CategoryError("authentication")

    observer = _AbortObserver()
    result = await process_prompts(
        ConcurrentFailure(),
        [str(index) for index in range(4)],
        config=ProcessorConfig(
            max_workers=4,
            guardrails=GuardrailConfig(abort_on_error_categories=frozenset({"authentication"})),
        ),
        error_classifier=_CategoryClassifier(),
        observers=[observer],
    )
    assert result.total_items == 4
    assert observer.events[ProcessingEvent.BATCH_ABORTED] == 1


@pytest.mark.asyncio
async def test_active_retry_backoff_stops_after_other_item_triggers_abort() -> None:
    release = asyncio.Event()
    started: set[str] = set()
    calls: Counter[str] = Counter()

    class Mixed(LLMCallStrategy[str]):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage, None]:
            calls[prompt] += 1
            started.add(prompt)
            if len(started) == 2:
                release.set()
            await release.wait()
            if prompt == "fatal":
                raise _CategoryError("insufficient_balance")
            raise ConnectionError("retry later")

    result = await process_prompts(
        Mixed(),
        [("fatal", "fatal"), ("retrying", "retrying")],
        config=ProcessorConfig(
            max_workers=2,
            retry=RetryConfig(max_attempts=3, initial_wait=0.2, max_wait=0.2, jitter=False),
            guardrails=GuardrailConfig(
                abort_on_error_categories=frozenset({"insufficient_balance"})
            ),
        ),
        error_classifier=_CategoryClassifier(),
        preserve_order=True,
    )
    assert calls == Counter({"fatal": 1, "retrying": 1})
    assert result.results[1].error_category == "batch_aborted"


@pytest.mark.asyncio
async def test_external_cancellation_still_propagates() -> None:
    task = asyncio.create_task(
        call_result(
            _SlowStrategy(1),
            "x",
            config=ProcessorConfig(guardrails=GuardrailConfig(total_timeout_per_item=10)),
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["total_timeout_per_item", "batch_timeout"])
def test_guardrail_timeout_validation(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        GuardrailConfig(**{field: value})


def test_openai_classifier_distinguishes_authentication_and_permission() -> None:
    classifier = OpenAIErrorClassifier()
    for status, category in ((401, "authentication"), (403, "permission_denied")):
        response = httpx.Response(
            status,
            request=httpx.Request("POST", "https://api.example.test/v1"),
        )
        exception = APIStatusError("denied", response=response, body=None)
        assert classifier.classify(exception).error_category == category


@pytest.mark.asyncio
async def test_item_deadline_excludes_postprocessing() -> None:
    postprocessed = asyncio.Event()

    async def postprocess(result: WorkItemResult) -> None:
        await asyncio.sleep(0.04)
        postprocessed.set()

    result = await process_prompts(
        _CategoryStrategy(None),
        ["ok"],
        config=ProcessorConfig(guardrails=GuardrailConfig(total_timeout_per_item=0.02)),
        post_processor=postprocess,
    )
    assert result.succeeded == 1
    assert postprocessed.is_set()


@pytest.mark.asyncio
async def test_stream_can_be_closed_externally_without_guardrail_misclassification() -> None:
    stream = process_stream(_SlowStrategy(1), ["one", "two"])
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with contextlib.suppress(Exception):
        await stream.aclose()
