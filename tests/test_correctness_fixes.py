"""Regression tests for the v0.12.x correctness fixes.

Each test pins one of the bugs fixed in this batch:

1. add_work() raises (instead of deadlocking) on a full bounded queue.
2. tokens from failed attempts are merged into an eventually-successful result.
3. ITEM_FAILED is emitted when retries are exhausted (metrics == BatchResult).
5. "429" is matched on word boundaries, not as a bare substring.
7. PydanticAIStrategy works against pydantic-ai 1.x usage / output_type names.
"""

import asyncio
import warnings

import pytest
from pydantic import BaseModel

from async_batch_llm import (
    LLMWorkItem,
    MetricsObserver,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,
)
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.classifiers.gemini import GeminiErrorClassifier
from async_batch_llm.classifiers.openai import OpenAIErrorClassifier
from async_batch_llm.core import RetryConfig
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.strategies.errors import DefaultErrorClassifier


class _EchoStrategy(LLMCallStrategy[str]):
    """Trivial strategy that echoes the prompt back."""

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        return prompt, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None


# ── Fix 1: bounded-queue deadlock ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_work_raises_on_full_bounded_queue_instead_of_hanging():
    """A bounded queue that fills before processing must raise, not block forever.

    Workers only start in process_all() and add_work() is rejected afterwards,
    so a full bounded queue can never drain while we're still adding — blocking
    would deadlock. asyncio.wait_for proves add_work() returns/raises promptly.
    """
    config = ProcessorConfig(max_workers=1, max_queue_size=2)
    strategy = _EchoStrategy()

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await asyncio.wait_for(
            processor.add_work(LLMWorkItem(item_id="1", strategy=strategy, prompt="a")),
            timeout=1.0,
        )
        await asyncio.wait_for(
            processor.add_work(LLMWorkItem(item_id="2", strategy=strategy, prompt="b")),
            timeout=1.0,
        )

        # Third add must raise ValueError quickly (not hang). The wait_for would
        # raise TimeoutError if the old blocking put() behavior regressed.
        with pytest.raises(ValueError, match="queue is full"):
            await asyncio.wait_for(
                processor.add_work(LLMWorkItem(item_id="3", strategy=strategy, prompt="c")),
                timeout=1.0,
            )


# ── Fix 2: failed-attempt tokens merged into a later success ────────────────


class _FailValidationThenSucceed(LLMCallStrategy[str]):
    """Fail validation (with tokens attached) on attempt 1, succeed on attempt 2."""

    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls += 1
        if self.calls == 1:
            # Name contains "validation" so the framework retries immediately
            # (no backoff sleep). Tokens for the (billed) failed attempt are
            # attached the way strategies do on a parse failure.
            class FakeValidationError(Exception):
                pass

            err = FakeValidationError("bad output on attempt 1")
            err.__dict__["_failed_token_usage"] = {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            }
            raise err
        return "ok", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}, None


@pytest.mark.asyncio
async def test_failed_attempt_tokens_merged_into_successful_result():
    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=5.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.01, jitter=False),
    )
    strategy = _FailValidationThenSucceed()

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item", strategy=strategy, prompt="go"))
        result = await processor.process_all()

    assert result.succeeded == 1
    assert result.failed == 0

    item = result.results[0]
    assert item.success
    # Success attempt (10/5/15) PLUS the failed attempt (100/20/120).
    assert item.token_usage == {
        "input_tokens": 110,
        "output_tokens": 25,
        "total_tokens": 135,
    }
    # And it aggregates up through BatchResult.
    assert result.total_input_tokens == 110
    assert result.total_output_tokens == 25


# ── Fix 3: ITEM_FAILED emitted on exhausted retries ─────────────────────────


class _AlwaysFails(LLMCallStrategy[str]):
    """Always raises a retryable error so the item exhausts its retries."""

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        raise Exception("transient boom")


@pytest.mark.asyncio
async def test_metrics_counts_match_batch_result_on_exhausted_retries():
    metrics = MetricsObserver()
    config = ProcessorConfig(
        max_workers=2,
        timeout_per_item=5.0,
        retry=RetryConfig(max_attempts=2, initial_wait=0.01, jitter=False),
    )
    strategy = _AlwaysFails()

    async with ParallelBatchProcessor[str, str, None](
        config=config, observers=[metrics]
    ) as processor:
        for i in range(3):
            await processor.add_work(LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt="x"))
        result = await processor.process_all()

    m = await metrics.get_metrics()
    assert result.failed == 3
    assert result.succeeded == 0
    # The observer must agree with BatchResult — previously items that exhausted
    # retries emitted no ITEM_FAILED and were undercounted here.
    assert m["items_failed"] == result.failed
    assert m["items_succeeded"] == result.succeeded
    assert m["items_processed"] == result.total_items


# ── Fix 5: 429 word-boundary matching ───────────────────────────────────────


@pytest.mark.parametrize(
    "classifier",
    [DefaultErrorClassifier(), OpenAIErrorClassifier(), GeminiErrorClassifier()],
)
def test_429_substring_does_not_false_positive(classifier):
    # "4290 tokens" must NOT look like a 429 rate limit (would otherwise trigger
    # a coordinated cooldown of every worker).
    assert classifier.classify(Exception("Expected 4290 tokens, got 12")).is_rate_limit is False
    assert classifier.classify(Exception("error code 1429 here")).is_rate_limit is False


@pytest.mark.parametrize(
    "classifier",
    [DefaultErrorClassifier(), OpenAIErrorClassifier(), GeminiErrorClassifier()],
)
def test_real_429_still_detected(classifier):
    assert classifier.classify(Exception("HTTP 429 Too Many Requests")).is_rate_limit is True
    # Non-numeric rate-limit words still match as before.
    assert classifier.classify(Exception("quota exceeded")).is_rate_limit is True


# ── Fix 7: pydantic-ai 1.x compatibility ────────────────────────────────────


class _Usage1xOnly:
    """Usage exposing ONLY the pydantic-ai 1.x names."""

    input_tokens = 11
    output_tokens = 7
    total_tokens = 18


class _UsageDeprecating:
    """Usage where touching the 0.x names emits a DeprecationWarning."""

    input_tokens = 11
    output_tokens = 7
    total_tokens = 18

    @property
    def request_tokens(self) -> int:
        warnings.warn("request_tokens is deprecated", DeprecationWarning, stacklevel=2)
        return 11

    @property
    def response_tokens(self) -> int:
        warnings.warn("response_tokens is deprecated", DeprecationWarning, stacklevel=2)
        return 7


class _FakeResult:
    def __init__(self, output, usage) -> None:
        self.output = output
        self._usage = usage

    def usage(self):
        return self._usage


class _FakeAgent:
    def __init__(self, output, usage, *, output_type=None) -> None:
        self._output = output
        self._usage = usage
        if output_type is not None:
            self.output_type = output_type

    async def run(self, prompt, **kwargs):
        return _FakeResult(self._output, self._usage)


@pytest.mark.asyncio
async def test_pydantic_ai_strategy_reads_1x_usage_field_names():
    strategy = PydanticAIStrategy(agent=_FakeAgent("hello", _Usage1xOnly()))
    output, tokens, metadata = await strategy.execute("prompt", 1, 10.0)
    assert output == "hello"
    assert tokens == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    assert metadata is None


@pytest.mark.asyncio
async def test_pydantic_ai_strategy_avoids_deprecated_usage_names():
    strategy = PydanticAIStrategy(agent=_FakeAgent("hi", _UsageDeprecating()))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Must not access request_tokens/response_tokens (which would raise here).
        output, tokens, _ = await strategy.execute("prompt", 1, 10.0)
    assert output == "hi"
    assert tokens == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}


class _DryRunOutput(BaseModel):
    value: str = "default"


@pytest.mark.asyncio
async def test_pydantic_ai_dry_run_uses_output_type():
    # Agent exposes output_type (1.x) and NO result_type attribute.
    agent = _FakeAgent("unused", _Usage1xOnly(), output_type=_DryRunOutput)
    assert not hasattr(agent, "result_type")

    strategy = PydanticAIStrategy(agent=agent)
    output, tokens = await strategy.dry_run("hello world")
    assert isinstance(output, _DryRunOutput)
    assert tokens["total_tokens"] > 0
