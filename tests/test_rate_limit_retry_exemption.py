"""Tests for rate-limit errors being exempt from the max_attempts retry budget."""

import pytest

from async_batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RateLimitConfig,
    RateLimitRetriesExceeded,
    RetryConfig,
)
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy

_TOKENS: TokenUsage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 1}


class _ValidationFailure(Exception):
    """Name contains 'validation' so it's treated as a (retryable) content error."""


class _SequenceStrategy(LLMCallStrategy[str]):
    """Drive a fixed sequence of outcomes, recording the attempt# each call saw.

    Sequence entries: ``"429"`` (rate limit), ``"val"`` (validation error),
    ``"ok"`` (success). Also records the attempt# passed to ``on_error``.
    """

    def __init__(self, sequence: list[str]) -> None:
        self.sequence = sequence
        self.calls = 0
        self.execute_attempts: list[int] = []
        self.on_error_attempts: list[int] = []

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.execute_attempts.append(attempt)
        kind = self.sequence[self.calls]
        self.calls += 1
        if kind == "429":
            raise Exception("429 Too Many Requests")
        if kind == "val":
            raise _ValidationFailure("bad output")
        return "ok", _TOKENS, None

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ) -> None:
        self.on_error_attempts.append(attempt)


def _fast_config(*, max_attempts: int = 3, max_rate_limit_retries: int = 20) -> ProcessorConfig:
    # cooldown_seconds=0 so coordinated cooldowns don't actually sleep in tests.
    return ProcessorConfig(
        max_workers=1,
        retry=RetryConfig(
            max_attempts=max_attempts,
            initial_wait=0.01,
            jitter=False,
            max_rate_limit_retries=max_rate_limit_retries,
        ),
        rate_limit=RateLimitConfig(cooldown_seconds=0.0),
    )


async def _run_one(strategy: LLMCallStrategy, config: ProcessorConfig):
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="x", strategy=strategy, prompt="hi"))
        return await processor.process_all()


# ── (a) rate limits don't consume attempts ──────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limits_do_not_consume_attempt_budget():
    # 5 rate limits then success, with only max_attempts=3.
    strategy = _SequenceStrategy(["429", "429", "429", "429", "429", "ok"])
    result = await _run_one(strategy, _fast_config(max_attempts=3))

    assert result.succeeded == 1
    assert result.failed == 0
    # execute() saw attempt=1 on every call — throttling never advanced it.
    assert strategy.execute_attempts == [1, 1, 1, 1, 1, 1]


# ── (b) mixed failures: only non-rate-limit errors advance the attempt# ──────


@pytest.mark.asyncio
async def test_mixed_failures_attempt_numbering():
    # 429 (no consume), validation (consume -> attempt 2), 429 (no consume), success.
    strategy = _SequenceStrategy(["429", "val", "429", "ok"])
    result = await _run_one(strategy, _fast_config(max_attempts=3))

    assert result.succeeded == 1
    assert strategy.execute_attempts == [1, 1, 2, 2]
    # on_error fires for each failure (not the success), with the logical attempt.
    assert strategy.on_error_attempts == [1, 1, 2]


# ── (c) exceeding max_rate_limit_retries fails the item ──────────────────────


@pytest.mark.asyncio
async def test_exceeding_max_rate_limit_retries_fails_item():
    strategy = _SequenceStrategy(["429"] * 10)
    result = await _run_one(strategy, _fast_config(max_rate_limit_retries=2))

    assert result.failed == 1
    assert result.succeeded == 0
    # max_rate_limit_retries=2 -> retried twice, the 3rd 429 trips the cap.
    assert strategy.calls == 3
    assert strategy.execute_attempts == [1, 1, 1]
    item = result.results[0]
    assert item.error is not None
    assert "RateLimitRetriesExceeded" in item.error
    assert "rate-limit retries" in item.error


@pytest.mark.asyncio
async def test_max_rate_limit_retries_zero_fails_on_first_rate_limit():
    strategy = _SequenceStrategy(["429"] * 5)
    result = await _run_one(strategy, _fast_config(max_rate_limit_retries=0))

    assert result.failed == 1
    assert strategy.calls == 1  # first 429 immediately fails
    assert "RateLimitRetriesExceeded" in result.results[0].error


# ── token accounting on rate-limit exhaustion ───────────────────────────────


class _RateLimitWithTokens(LLMCallStrategy[str]):
    """Always rate-limited, but each attempt reports token usage."""

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        err = Exception("429 quota exceeded")
        err.__dict__["_failed_token_usage"] = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "cached_input_tokens": 0,
        }
        raise err


@pytest.mark.asyncio
async def test_rate_limit_exhaustion_includes_cumulative_tokens():
    strategy = _RateLimitWithTokens()
    result = await _run_one(strategy, _fast_config(max_rate_limit_retries=2))

    assert result.failed == 1
    # 3 billed attempts (2 retries + the one that trips the cap) * 30 tokens.
    assert result.total_input_tokens == 30
    assert result.results[0].token_usage["total_tokens"] == 90


# ── config validation ───────────────────────────────────────────────────────


def test_max_rate_limit_retries_must_be_non_negative():
    with pytest.raises(ValueError, match="max_rate_limit_retries must be >= 0"):
        RetryConfig(max_rate_limit_retries=-1)


def test_max_rate_limit_retries_default_is_20():
    assert RetryConfig().max_rate_limit_retries == 20


@pytest.mark.asyncio
async def test_raises_rate_limit_retries_exceeded_type():
    """The dedicated exception is what propagates (catchable by users)."""
    strategy = _SequenceStrategy(["429"] * 5)
    result = await _run_one(strategy, _fast_config(max_rate_limit_retries=1))
    assert result.failed == 1
    # Sanity: the exception class is importable and named in the error string.
    assert RateLimitRetriesExceeded.__name__ in result.results[0].error
