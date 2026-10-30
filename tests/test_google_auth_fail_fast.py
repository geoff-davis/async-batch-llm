"""Expired Google credentials fail fast: classification and LLMCallPool abort (#256)."""

from __future__ import annotations

import asyncio

import pytest

from async_batch_llm import (
    AbortMode,
    LLMCallPool,
    LLMCallStrategy,
    ProcessorConfig,
    RetryConfig,
    process_prompts,
)
from async_batch_llm.classifiers.gemini import GeminiErrorClassifier
from async_batch_llm.classifiers.openai import OpenAIErrorClassifier
from async_batch_llm.classifiers.openrouter import OpenRouterErrorClassifier
from async_batch_llm.classifiers.pydantic_ai import PydanticAIErrorClassifier
from async_batch_llm.core import GuardrailConfig
from async_batch_llm.strategies.errors import BatchAbortedError, DefaultErrorClassifier

google_auth_exceptions = pytest.importorskip("google.auth.exceptions")
DefaultCredentialsError = google_auth_exceptions.DefaultCredentialsError
ReauthFailError = google_auth_exceptions.ReauthFailError
RefreshError = google_auth_exceptions.RefreshError
TransportError = google_auth_exceptions.TransportError

CLASSIFIERS = [
    GeminiErrorClassifier,
    OpenAIErrorClassifier,
    OpenRouterErrorClassifier,
    PydanticAIErrorClassifier,
    DefaultErrorClassifier,
]
AUTH = frozenset({"authentication"})


@pytest.mark.parametrize("classifier", CLASSIFIERS)
@pytest.mark.parametrize(
    "exception",
    [
        RefreshError("Reauthentication is needed."),
        ReauthFailError("reauth failed"),
        DefaultCredentialsError("Your default credentials were not found."),
    ],
)
def test_credential_failures_are_non_retryable_authentication(classifier, exception):
    info = classifier().classify(exception)
    assert info.error_category == "authentication"
    assert not info.is_retryable
    assert not info.is_rate_limit
    assert info.hint is not None
    assert "gcloud auth application-default login" in info.hint


@pytest.mark.parametrize("classifier", CLASSIFIERS)
@pytest.mark.parametrize(
    "exception",
    [RefreshError("token endpoint returned 503", retryable=True), TransportError("reset")],
)
def test_transient_google_auth_failures_stay_retryable(classifier, exception):
    info = classifier().classify(exception)
    assert info.is_retryable
    assert info.error_category != "authentication"


class _ExpiredCredentials(LLMCallStrategy[str]):
    """Raises a non-retryable RefreshError, like google-genai with expired ADC."""

    def __init__(
        self, *, delays: dict[str, float] | None = None, succeed: frozenset[str] = frozenset()
    ) -> None:
        self.delays = delays or {}
        self.succeed = succeed
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def execute(self, prompt, attempt, timeout, state=None):
        self.calls.append(prompt)
        self.started.set()
        await self.release.wait()
        if prompt in self.delays:
            await asyncio.sleep(self.delays[prompt])
        if prompt in self.succeed:
            return f"ok:{prompt}", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None
        raise RefreshError("Reauthentication is needed.")


def _config(fast_retry: RetryConfig, *, workers: int = 1, **guardrails) -> ProcessorConfig:
    return ProcessorConfig(
        max_workers=workers,
        retry=fast_retry,
        guardrails=GuardrailConfig(**guardrails),
    )


@pytest.mark.asyncio
async def test_pool_stops_after_first_authentication_failure(fast_retry):
    strategy = _ExpiredCredentials()
    config = _config(fast_retry, abort_on_error_categories=AUTH)
    async with LLMCallPool(strategy, config=config) as pool:
        assert pool.termination is None
        first = await pool.submit_result("a")
        rest = [await pool.submit_result(prompt) for prompt in ("b", "c")]
        with pytest.raises(BatchAbortedError):
            await pool.submit("d")

    assert first.error_category == "authentication"
    # Non-retryable: one call, no retries.
    assert strategy.calls == ["a"]
    assert [r.error_category for r in rest] == ["batch_aborted", "batch_aborted"]
    assert all(r.exception is not None for r in rest)
    termination = pool.termination
    assert termination is not None
    assert termination.kind == "fail_fast"
    assert termination.error_category == "authentication"
    assert termination.triggering_item_id == first.item_id
    assert termination.reason is not None
    assert "'authentication'" in termination.reason
    assert "gcloud auth application-default login" in termination.reason
    assert termination.reason in rest[0].error


@pytest.mark.asyncio
async def test_pool_submits_waiting_for_a_slot_abort_without_calls(fast_retry):
    strategy = _ExpiredCredentials()
    strategy.release.clear()
    config = _config(fast_retry, abort_on_error_categories=AUTH)
    async with LLMCallPool(strategy, config=config) as pool:
        first = asyncio.create_task(pool.submit_result("a"))
        await strategy.started.wait()
        # One worker: these wait on the pool's slot while "a" runs.
        waiting = [asyncio.create_task(pool.submit_result(p)) for p in ("b", "c")]
        await asyncio.sleep(0)
        strategy.release.set()
        results = await asyncio.gather(first, *waiting)

    assert strategy.calls == ["a"]
    assert [r.error_category for r in results] == [
        "authentication",
        "batch_aborted",
        "batch_aborted",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (AbortMode.DRAIN_ACTIVE, None),
        (AbortMode.CANCEL_ACTIVE, "batch_aborted"),
    ],
)
async def test_pool_abort_mode_governs_calls_already_running(fast_retry, mode, expected):
    # "b" is still running when "a" fails and stops the pool; draining lets it
    # finish and keep its success, cancelling ends it batch_aborted.
    strategy = _ExpiredCredentials(delays={"a": 0.05, "b": 0.5}, succeed=frozenset({"b"}))
    config = _config(fast_retry, workers=2, abort_on_error_categories=AUTH, abort_mode=mode)
    async with LLMCallPool(strategy, config=config) as pool:
        results = await asyncio.gather(pool.submit_result("a"), pool.submit_result("b"))

    assert sorted(strategy.calls) == ["a", "b"]
    assert results[0].error_category == "authentication"
    assert results[1].error_category == expected
    assert results[1].success is (expected is None)


@pytest.mark.asyncio
async def test_pool_ignores_failures_outside_the_abort_categories(fast_retry):
    strategy = _ExpiredCredentials()
    config = _config(fast_retry, abort_on_error_categories=frozenset({"insufficient_balance"}))
    async with LLMCallPool(strategy, config=config) as pool:
        results = [await pool.submit_result(p) for p in ("a", "b")]
        assert pool.termination is None

    assert strategy.calls == ["a", "b"]
    assert [r.error_category for r in results] == ["authentication", "authentication"]


@pytest.mark.asyncio
async def test_pool_without_guardrail_keeps_serving(fast_retry):
    strategy = _ExpiredCredentials(succeed=frozenset({"a"}))
    async with LLMCallPool(strategy, config=_config(fast_retry)) as pool:
        ok = await pool.submit("a")
        failed = [await pool.submit_result(p) for p in ("b", "c")]
        assert pool.termination is None

    assert ok == "ok:a"
    # Each request is tried once (no longer retried) and none is aborted.
    assert strategy.calls == ["a", "b", "c"]
    assert [r.error_category for r in failed] == ["authentication", "authentication"]


@pytest.mark.asyncio
async def test_processor_run_stops_on_expired_credentials(fast_retry):
    strategy = _ExpiredCredentials()
    result = await process_prompts(
        strategy,
        ["a", "b", "c"],
        config=_config(fast_retry, abort_on_error_categories=AUTH),
    )

    assert strategy.calls == ["a"]
    assert result.termination.kind == "fail_fast"
    assert result.termination.error_category == "authentication"
    assert sorted(r.error_category for r in result.results) == [
        "authentication",
        "batch_aborted",
        "batch_aborted",
    ]


@pytest.mark.asyncio
async def test_pool_resume_accepts_work_again(fast_retry):
    strategy = _ExpiredCredentials(succeed=frozenset({"c", "d"}))
    config = _config(fast_retry, abort_on_error_categories=AUTH)
    async with LLMCallPool(strategy, config=config) as pool:
        assert pool.resume() is None  # not stopped: a no-op
        await pool.submit_result("a")
        stopped = pool.termination
        assert (await pool.submit_result("b")).error_category == "batch_aborted"

        cleared = pool.resume()
        assert cleared == stopped
        assert pool.termination is None
        assert await pool.submit("c") == "ok:c"
        assert await pool.submit("d") == "ok:d"

        # Still failing after the resume: the pool stops again.
        await pool.submit_result("e")
        assert pool.termination is not None
        assert pool.termination.triggering_item_id == "req-5"
        assert (await pool.submit_result("f")).error_category == "batch_aborted"

    assert strategy.calls == ["a", "c", "d", "e"]


@pytest.mark.asyncio
async def test_failure_started_before_resume_does_not_stop_resumed_pool(fast_retry):
    # "b" is running when "a" stops the pool, and fails only after the resume.
    strategy = _ExpiredCredentials(delays={"a": 0.01, "b": 0.2}, succeed=frozenset({"c"}))
    config = _config(fast_retry, workers=2, abort_on_error_categories=AUTH)
    async with LLMCallPool(strategy, config=config) as pool:
        slow = asyncio.create_task(pool.submit_result("b"))
        await pool.submit_result("a")
        assert pool.termination is not None
        assert pool.resume() is not None
        late = await slow
        assert pool.termination is None
        assert await pool.submit("c") == "ok:c"

    assert late.error_category == "authentication"
    assert sorted(strategy.calls) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_pool_resume_without_abort_categories_is_a_no_op(fast_retry):
    async with LLMCallPool(_ExpiredCredentials(), config=_config(fast_retry)) as pool:
        await pool.submit_result("a")
        assert pool.resume() is None
        assert pool.termination is None
