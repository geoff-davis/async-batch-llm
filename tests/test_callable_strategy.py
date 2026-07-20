"""CallableStrategy integration through the existing execution surfaces."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from async_batch_llm import (
    ArtifactError,
    ArtifactIdentity,
    CallableStrategy,
    CallOutcome,
    JsonlArtifactStore,
    LLMGateway,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    ResumePolicy,
    TokenTrackingError,
    call,
    call_result,
    process_prompts,
    process_stream,
)
from async_batch_llm.base import RetryState
from async_batch_llm.core import RateLimitConfig, RetryConfig
from async_batch_llm.strategies import ErrorClassifier, ErrorInfo


def _config(*, attempts: int = 3, workers: int = 2, timeout: float = 1.0) -> ProcessorConfig:
    return ProcessorConfig(
        max_workers=workers,
        attempt_timeout=timeout,
        retry=RetryConfig(
            max_attempts=attempts,
            initial_wait=0.001,
            max_wait=0.001,
            jitter=False,
            max_rate_limit_retries=3,
        ),
        rate_limit=RateLimitConfig(
            cooldown_seconds=0,
            slow_start_items=0,
            slow_start_initial_delay=0,
            slow_start_final_delay=0,
        ),
    )


async def _echo(
    prompt: str, *, attempt: int, timeout: float, state: RetryState | None
) -> CallOutcome[str]:
    assert timeout > 0
    assert state is not None
    return CallOutcome(
        f"{prompt}:{attempt}",
        {"input_tokens": 2, "output_tokens": 3},
        {"provider": "local", "attempt": attempt},
    )


class _Classifier(ErrorClassifier):
    def __init__(self, *, retryable: bool) -> None:
        self.retryable = retryable

    def classify(self, exception: Exception) -> ErrorInfo:
        return ErrorInfo(
            is_retryable=self.retryable,
            is_rate_limit=False,
            is_timeout=False,
            error_category="custom",
        )


@pytest.mark.asyncio
async def test_success_usage_metadata_and_provider_timing() -> None:
    result = await call_result(CallableStrategy(_echo), "hello", config=_config())

    assert result.success
    assert result.output == "hello:1"
    assert result.token_usage == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    assert result.metadata == {"provider": "local", "attempt": 1}
    assert result.timing is not None
    assert result.timing.attempts[0].provider_seconds is not None
    assert result.timing.attempts[0].provider_seconds >= 0


@pytest.mark.asyncio
async def test_empty_usage_is_not_estimated_and_mappings_are_copied() -> None:
    usage: dict[str, int] = {}
    metadata: dict[str, Any] = {"nested": "original"}

    async def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        return CallOutcome(prompt, usage, metadata)

    result = await call_result(CallableStrategy(invoke), "p", config=_config())
    usage["total_tokens"] = 99
    metadata["nested"] = "changed"

    assert result.token_usage == {}
    assert result.metadata == {"nested": "original"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage", "message"),
    [
        ({"other": 1}, "Unknown.*other"),
        ({"input_tokens": -1}, "non-negative integer"),
        ({"input_tokens": True}, "non-negative integer"),
        ({"input_tokens": 1.5}, "non-negative integer"),
    ],
)
async def test_invalid_token_usage_is_actionable(usage: Mapping[str, Any], message: str) -> None:
    async def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        return CallOutcome(prompt, usage)  # type: ignore[arg-type]

    result = await call_result(CallableStrategy(invoke), "p", config=_config(attempts=1))
    assert not result.success
    assert isinstance(result.exception, (TypeError, ValueError))
    assert result.error is not None
    assert __import__("re").search(message, result.error)


def test_obviously_synchronous_invoke_is_rejected() -> None:
    def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        return CallOutcome(prompt)

    with pytest.raises(TypeError, match="async def"):
        CallableStrategy(invoke)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_runtime_non_awaitable_and_invalid_outcome_are_rejected() -> None:
    strategy = CallableStrategy(_echo)
    strategy._invoke = lambda *args, **kwargs: CallOutcome("no-await")  # type: ignore[assignment]
    result = await call_result(strategy, "p", config=_config(attempts=1))
    assert isinstance(result.exception, TypeError)
    assert "non-awaitable" in str(result.exception)

    async def invalid(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> Any:
        return (prompt, {})

    result = await call_result(CallableStrategy(invalid), "p", config=_config(attempts=1))
    assert isinstance(result.exception, TypeError)
    assert "CallOutcome" in str(result.exception)


@pytest.mark.asyncio
async def test_original_terminal_exception_is_preserved_and_reraised() -> None:
    terminal = RuntimeError("original provider failure")

    async def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        raise terminal

    strategy = CallableStrategy(invoke)
    result = await call_result(strategy, "p", config=_config(attempts=1))
    assert result.exception is terminal
    with pytest.raises(RuntimeError) as raised:
        await call(strategy, "p", config=_config(attempts=1))
    assert raised.value is terminal


@pytest.mark.asyncio
async def test_per_item_retry_state_is_isolated_under_interleaving() -> None:
    first_attempts_ready = asyncio.Event()
    release_first_attempts = asyncio.Event()
    first_count = 0
    seen_feedback: dict[str, object] = {}

    async def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        nonlocal first_count
        assert state is not None
        if attempt == 1:
            first_count += 1
            if first_count == 4:
                first_attempts_ready.set()
            await release_first_attempts.wait()
            raise RuntimeError(f"invalid:{prompt}")
        seen_feedback[prompt] = state.get("feedback")
        return CallOutcome(prompt)

    async def on_error(error: Exception, attempt: int, state: RetryState | None) -> None:
        assert state is not None
        state.set("feedback", str(error))

    strategy = CallableStrategy(invoke, on_error=on_error)
    task = asyncio.create_task(
        process_prompts(strategy, ["a", "b", "c", "d"], config=_config(workers=4))
    )
    await first_attempts_ready.wait()
    release_first_attempts.set()
    batch = await task

    assert batch.succeeded == 4
    assert seen_feedback == {key: f"invalid:{key}" for key in "abcd"}


@pytest.mark.asyncio
async def test_rate_limit_reuses_attempt_but_content_retry_advances_it() -> None:
    attempts: list[int] = []

    async def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        attempts.append(attempt)
        if len(attempts) == 1:
            raise RuntimeError("429 rate limit")
        if len(attempts) == 2:
            raise RuntimeError("invalid content")
        return CallOutcome("ok")

    result = await call_result(CallableStrategy(invoke), "p", config=_config())
    assert result.success
    assert attempts == [1, 1, 2]


@pytest.mark.asyncio
async def test_failed_attempt_token_usage_is_accumulated() -> None:
    calls = 0

    async def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TokenTrackingError(
                "billed response did not parse",
                token_usage={"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
            )
        return CallOutcome("recovered", {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})

    result = await call_result(CallableStrategy(invoke), "p", config=_config())
    assert result.success
    assert result.token_usage == {"input_tokens": 6, "output_tokens": 9, "total_tokens": 15}


@pytest.mark.asyncio
async def test_classifier_recommendation_and_explicit_precedence() -> None:
    recommended = _Classifier(retryable=False)
    explicit = _Classifier(retryable=True)
    calls = 0

    async def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        nonlocal calls
        calls += 1
        raise RuntimeError("retry me")

    strategy = CallableStrategy(invoke, error_classifier=recommended)
    assert strategy.recommended_error_classifier() is recommended
    result = await call_result(strategy, "p", config=_config(attempts=2))
    assert not result.success
    assert calls == 1

    calls = 0
    result = await call_result(strategy, "p", config=_config(attempts=2), error_classifier=explicit)
    assert not result.success
    assert calls == 2


def test_capacity_configuration_and_scope() -> None:
    scope = object()
    strategy = CallableStrategy(_echo, max_concurrency=3, concurrency_scope=scope)
    assert strategy.max_concurrency == 3
    assert strategy.concurrency_scope is scope
    assert CallableStrategy(_echo).concurrency_scope is not strategy
    for value in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            CallableStrategy(_echo, max_concurrency=value)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_shared_concurrency_scope_limits_distinct_strategies() -> None:
    scope = object()
    active = 0
    peak = 0
    release = asyncio.Event()

    async def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        release.set()
        await asyncio.sleep(0.01)
        active -= 1
        return CallOutcome(prompt)

    first = CallableStrategy(invoke, max_concurrency=1, concurrency_scope=scope)
    second = CallableStrategy(invoke, max_concurrency=1, concurrency_scope=scope)
    processor = ParallelBatchProcessor[None, str, None](config=_config(workers=2))
    async with processor:
        await processor.add_work(LLMWorkItem(item_id="a", strategy=first, prompt="a"))
        await processor.add_work(LLMWorkItem(item_id="b", strategy=second, prompt="b"))
        batch = await processor.process_all()
    assert batch.succeeded == 2
    assert release.is_set()
    assert peak == 1


@pytest.mark.asyncio
async def test_request_concurrency_callback_runs_once() -> None:
    requested: list[int] = []

    def resize(value: int) -> bool:
        requested.append(value)
        return True

    strategy = CallableStrategy(_echo, request_concurrency=resize)
    batch = await process_prompts(strategy, ["a", "b"], config=ProcessorConfig(concurrency=3))
    assert batch.succeeded == 2
    assert requested == [3]


@pytest.mark.asyncio
async def test_sync_and_async_lifecycle_callbacks_run_once_for_shared_strategy() -> None:
    calls: list[str] = []

    def prepare() -> None:
        calls.append("prepare")

    async def cleanup() -> None:
        calls.append("cleanup")

    strategy = CallableStrategy(_echo, prepare=prepare, cleanup=cleanup)
    batch = await process_prompts(strategy, ["a", "b", "c"], config=_config(workers=3))
    assert batch.succeeded == 3
    assert calls == ["prepare", "cleanup"]


@pytest.mark.asyncio
async def test_cleanup_after_failure_and_caller_cancellation() -> None:
    failure_cleanup = 0

    async def fail(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        raise RuntimeError("provider failed")

    def cleaned() -> None:
        nonlocal failure_cleanup
        failure_cleanup += 1

    await call_result(CallableStrategy(fail, cleanup=cleaned), "p", config=_config(attempts=1))
    assert failure_cleanup == 1

    started = asyncio.Event()
    cancelled_cleanup = asyncio.Event()

    async def wait_forever(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def cleanup() -> None:
        cancelled_cleanup.set()

    task = asyncio.create_task(call(CallableStrategy(wait_forever, cleanup=cleanup), "p"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled_cleanup.is_set()


@pytest.mark.asyncio
async def test_failed_prepare_is_not_cleaned_and_callback_cancellation_propagates() -> None:
    cleanup_called = False

    async def prepare() -> None:
        raise RuntimeError("prepare failed")

    def cleanup() -> None:
        nonlocal cleanup_called
        cleanup_called = True

    with pytest.raises(RuntimeError, match="prepare failed"):
        await call(CallableStrategy(_echo, prepare=prepare, cleanup=cleanup), "p")
    assert not cleanup_called

    async def cancelled_prepare() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await call(CallableStrategy(_echo, prepare=cancelled_prepare), "p")


@pytest.mark.asyncio
async def test_dry_run_callback_and_fallback() -> None:
    def dry_run(prompt: str) -> CallOutcome[str]:
        return CallOutcome(f"dry:{prompt}", {"total_tokens": 7}, {"ignored": True})

    strategy = CallableStrategy(_echo, dry_run=dry_run)
    result = await call_result(strategy, "p", config=ProcessorConfig(max_workers=1, dry_run=True))
    assert result.output == "dry:p"
    assert result.token_usage == {"total_tokens": 7}

    fallback = await call_result(
        CallableStrategy(_echo), "p", config=ProcessorConfig(max_workers=1, dry_run=True)
    )
    assert fallback.success
    assert str(fallback.output).startswith("[DRY-RUN]")


@pytest.mark.asyncio
async def test_batch_stream_single_and_gateway_surfaces() -> None:
    strategy = CallableStrategy(_echo)
    batch = await process_prompts(strategy, ["a", "b"], config=_config())
    assert [result.output for result in batch.results] == ["a:1", "b:1"]

    streamed = [
        result.output async for result in process_stream(strategy, ["c", "d"], config=_config())
    ]
    assert set(streamed) == {"c:1", "d:1"}
    assert await call(strategy, "e", config=_config()) == "e:1"

    async with LLMGateway(strategy, config=_config()) as pool:
        assert await pool.submit("f") == "f:1"


@pytest.mark.asyncio
async def test_artifact_identity_replay_and_identity_change(tmp_path: Any) -> None:
    path = tmp_path / "callable.jsonl"
    first_calls = 0

    async def first_invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        nonlocal first_calls
        first_calls += 1
        return CallOutcome("first")

    first = CallableStrategy(
        first_invoke,
        identity=ArtifactIdentity(provider="app", model="route", application_version="1"),
    )
    batch = await process_prompts(first, [("id", "p")], artifact_store=JsonlArtifactStore(path))
    assert batch.results[0].output == "first"
    assert first_calls == 1

    replay_calls = 0

    async def replay_invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        nonlocal replay_calls
        replay_calls += 1
        return CallOutcome("should not run")

    replay = CallableStrategy(
        replay_invoke,
        identity=ArtifactIdentity(provider="app", model="route", application_version="1"),
    )
    batch = await process_prompts(
        replay,
        [("id", "p")],
        artifact_store=JsonlArtifactStore(path),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert batch.results[0].output == "first"
    assert batch.results[0].replayed_from_artifact
    assert replay_calls == 0

    changed_calls = 0

    async def changed_invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        nonlocal changed_calls
        changed_calls += 1
        return CallOutcome("changed")

    changed = CallableStrategy(
        changed_invoke,
        identity=ArtifactIdentity(provider="app", model="route", application_version="2"),
    )
    batch = await process_prompts(
        changed,
        [("id", "p")],
        artifact_store=JsonlArtifactStore(path),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert batch.results[0].output == "changed"
    assert not batch.results[0].replayed_from_artifact
    assert changed_calls == 1


@pytest.mark.asyncio
async def test_callable_artifact_requires_identity_before_invocation(tmp_path: Any) -> None:
    invoked = False

    async def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        nonlocal invoked
        invoked = True
        return CallOutcome(prompt)

    with pytest.raises(ArtifactError, match="ArtifactIdentity"):
        await process_prompts(
            CallableStrategy(invoke),
            ["p"],
            artifact_store=JsonlArtifactStore(tmp_path / "unsafe.jsonl"),
        )
    assert not invoked


@pytest.mark.asyncio
async def test_explicit_store_identity_takes_precedence(tmp_path: Any) -> None:
    batch = await process_prompts(
        CallableStrategy(_echo),
        ["p"],
        artifact_store=JsonlArtifactStore(
            tmp_path / "explicit.jsonl", identity=ArtifactIdentity(provider="owner")
        ),
    )
    assert batch.succeeded == 1


@pytest.mark.asyncio
async def test_attempt_timeout_and_external_cancellation_propagate() -> None:
    started = asyncio.Event()

    async def slow(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    result = await call_result(
        CallableStrategy(slow), "p", config=_config(attempts=1, timeout=0.01)
    )
    assert not result.success
    assert result.error_category == "framework_timeout"

    task = asyncio.create_task(call(CallableStrategy(slow), "p", config=_config(timeout=1)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
