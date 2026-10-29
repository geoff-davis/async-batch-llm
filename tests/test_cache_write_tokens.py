"""Prompt-cache write tokens (issue #235).

Providers report tokens written to a prompt cache separately from cache reads,
and some bill them above the normal input price. These tests cover extraction
from each built-in surface, aggregation, and pricing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from async_batch_llm import (
    BatchResult,
    CachedTokenRates,
    CallableStrategy,
    CallOutcome,
    DeepSeekModel,
    EmptyResponseError,
    LLMResponse,
    LLMWorkItem,
    OpenAIModel,
    OpenRouterModel,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,
    WorkItemResult,
)
from async_batch_llm.serialization import work_item_result_from_dict, work_item_result_to_dict
from async_batch_llm.token_extractor import TokenExtractor, _coerce_usage

# ── Provider extraction ──────────────────────────────────────────────────────


def _chat_completion(
    *, cache_write_tokens: int | None, content: str | None = "ok", cached: int = 12146
) -> Any:
    """A real SDK ChatCompletion, matching the usage quoted in the issue."""
    ChatCompletion = pytest.importorskip("openai.types.chat").ChatCompletion

    details: dict[str, Any] = {"cached_tokens": cached}
    if cache_write_tokens is not None:
        details["cache_write_tokens"] = cache_write_tokens
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 12486,
                "completion_tokens": 342,
                "total_tokens": 12828,
                "prompt_tokens_details": details,
            },
        }
    )


def _responses_response(*, cache_write_tokens: int | None, text: str = "ok") -> Any:
    Response = pytest.importorskip("openai.types.responses").Response

    details: dict[str, Any] = {"cached_tokens": 100}
    if cache_write_tokens is not None:
        details["cache_write_tokens"] = cache_write_tokens
    output = (
        [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
        if text
        else []
    )
    return Response.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-test",
            "status": "completed",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 20,
                "total_tokens": 1020,
                "input_tokens_details": details,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


def _chat_client(response: Any) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _responses_client(response: Any) -> MagicMock:
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=response)
    return client


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda c: OpenAIModel("gpt-test", c, api_surface="chat_completions"),
        lambda c: OpenRouterModel("anthropic/claude-test", c),
    ],
    ids=["openai-chat", "openrouter"],
)
async def test_chat_completions_report_cache_writes(model_factory) -> None:
    model = model_factory(_chat_client(_chat_completion(cache_write_tokens=337)))
    response = await model.generate("hello")
    assert response.input_tokens == 12486
    assert response.cached_input_tokens == 12146
    assert response.cache_write_input_tokens == 337
    assert response.token_usage["cache_write_input_tokens"] == 337


async def test_chat_completions_without_cache_writes_keep_usage_shape() -> None:
    model = OpenAIModel(
        "gpt-test",
        _chat_client(_chat_completion(cache_write_tokens=None)),
        api_surface="chat_completions",
    )
    response = await model.generate("hello")
    assert response.cache_write_input_tokens == 0
    assert "cache_write_input_tokens" not in response.token_usage


async def test_chat_completions_empty_response_carries_cache_writes() -> None:
    model = OpenAIModel(
        "gpt-test",
        _chat_client(_chat_completion(cache_write_tokens=337, content=None)),
        api_surface="chat_completions",
    )
    with pytest.raises(EmptyResponseError) as excinfo:
        await model.generate("hello")
    assert TokenExtractor().extract_from_exception(excinfo.value)["cache_write_input_tokens"] == 337


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda c: OpenAIModel("gpt-test", c),
        lambda c: DeepSeekModel("deepseek-test", c, api_surface="responses"),
    ],
    ids=["openai-responses", "deepseek-responses"],
)
async def test_responses_api_reports_cache_writes(model_factory) -> None:
    model = model_factory(_responses_client(_responses_response(cache_write_tokens=250)))
    response = await model.generate("hello")
    assert response.cached_input_tokens == 100
    assert response.cache_write_input_tokens == 250


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda c: OpenAIModel("gpt-test", c),
        lambda c: DeepSeekModel("deepseek-test", c, api_surface="responses"),
    ],
    ids=["openai-responses", "deepseek-responses"],
)
async def test_responses_api_empty_response_carries_cache_writes(model_factory) -> None:
    model = model_factory(_responses_client(_responses_response(cache_write_tokens=250, text="")))
    with pytest.raises(EmptyResponseError) as excinfo:
        await model.generate("hello")
    assert TokenExtractor().extract_from_exception(excinfo.value)["cache_write_input_tokens"] == 250


def test_mapping_details_from_older_sdks_are_read() -> None:
    from async_batch_llm.models import _openai_cache_write_tokens

    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens_details={"cached_tokens": 1, "cache_write_tokens": 9})
    )
    assert _openai_cache_write_tokens(response) == 9


async def test_pydantic_ai_strategy_reports_cache_reads_and_writes() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RequestUsage

    def respond(messages, info) -> ModelResponse:
        usage = RequestUsage(
            input_tokens=1000, output_tokens=10, cache_read_tokens=600, cache_write_tokens=300
        )
        return ModelResponse(parts=[TextPart("ok")], usage=usage)

    strategy = PydanticAIStrategy(agent=Agent(FunctionModel(respond)))
    output, tokens, _ = await strategy.execute("hi", attempt=1, timeout=10.0)
    assert output == "ok"
    assert tokens["input_tokens"] == 1000
    assert tokens["cached_input_tokens"] == 600
    assert tokens["cache_write_input_tokens"] == 300


# ── Token extraction (failure paths) ─────────────────────────────────────────


def test_coerce_reads_pydantic_ai_and_openai_shapes() -> None:
    pydantic_ai_usage = SimpleNamespace(
        input_tokens=100, output_tokens=5, cache_read_tokens=40, cache_write_tokens=30
    )
    assert _coerce_usage(pydantic_ai_usage)["cache_write_input_tokens"] == 30

    chat_usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=5,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40, cache_write_tokens=12),
    )
    assert _coerce_usage(chat_usage)["cache_write_input_tokens"] == 12


def test_coerce_leaves_out_zero_cache_writes_and_anthropic_native_name() -> None:
    assert "cache_write_input_tokens" not in _coerce_usage(
        SimpleNamespace(input_tokens=100, output_tokens=5, cache_write_tokens=0)
    )
    # Anthropic's input_tokens excludes cache_creation_input_tokens; other SDKs
    # reuse the name for an included count, so it is never guessed at.
    assert "cache_write_input_tokens" not in _coerce_usage(
        SimpleNamespace(input_tokens=100, output_tokens=5, cache_creation_input_tokens=50)
    )
    # Mocks answer every attribute; a non-integer is not a write count.
    assert "cache_write_input_tokens" not in _coerce_usage(MagicMock())


def test_accumulate_adds_cache_writes_only_once_reported() -> None:
    cumulative: dict[str, int] = {}
    TokenExtractor.accumulate(cumulative, {"input_tokens": 10, "total_tokens": 10})
    assert "cache_write_input_tokens" not in cumulative
    TokenExtractor.accumulate(cumulative, {"input_tokens": 10, "cache_write_input_tokens": 7})
    TokenExtractor.accumulate(cumulative, {"input_tokens": 10, "cache_write_input_tokens": 3})
    assert cumulative["cache_write_input_tokens"] == 10
    assert cumulative["input_tokens"] == 30


def test_strategy_mapping_keeps_cache_writes() -> None:
    observation = TokenExtractor.observe_result(
        {"input_tokens": 10, "output_tokens": 1, "cache_write_input_tokens": 4}
    )
    assert observation.usage["cache_write_input_tokens"] == 4


# ── End to end ───────────────────────────────────────────────────────────────


async def test_batch_totals_stats_and_failed_attempts_include_cache_writes(fast_retry) -> None:
    calls: dict[str, int] = {}

    async def invoke(prompt: str, *, attempt: int, timeout: float, state: Any) -> CallOutcome[str]:
        calls[prompt] = calls.get(prompt, 0) + 1
        usage = {"input_tokens": 100, "output_tokens": 10, "cache_write_input_tokens": 40}
        if prompt == "retry" and attempt == 1:
            # A retryable failure the provider still billed.
            error = ConnectionError("dropped")
            error.__dict__["_failed_token_usage"] = usage
            raise error
        return CallOutcome(prompt, token_usage=usage)

    strategy = CallableStrategy(invoke)
    config = ProcessorConfig(max_workers=2, retry=fast_retry)
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        for prompt in ("once", "retry"):
            await processor.add_work(LLMWorkItem(item_id=prompt, strategy=strategy, prompt=prompt))
        result = await processor.process_all()
        stats = await processor.get_stats()

    assert result.succeeded == 2
    by_id = result.by_id()
    assert by_id["once"].token_usage["cache_write_input_tokens"] == 40
    # The failed first attempt's writes were billed too.
    assert by_id["retry"].token_usage["cache_write_input_tokens"] == 80
    assert result.total_cache_write_tokens == 120
    assert stats["cache_write_input_tokens"] == 120
    assert "written 120" in result.summary()


def test_cache_writes_survive_result_serialization() -> None:
    result = WorkItemResult(
        item_id="a",
        success=True,
        output="x",
        token_usage={"input_tokens": 10, "cache_write_input_tokens": 4},
    )
    decoded = work_item_result_from_dict(work_item_result_to_dict(result))
    assert decoded.token_usage["cache_write_input_tokens"] == 4


# ── Pricing ──────────────────────────────────────────────────────────────────


def _batch(input_tokens: int, cached: int, written: int, output: int = 0) -> BatchResult:
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": written,
    }
    return BatchResult(results=[WorkItemResult(item_id="a", success=True, token_usage=usage)])


def test_cache_write_rate_adds_the_write_premium() -> None:
    # The issue's example: 12,486 input, 12,146 read from cache, 337 written.
    batch = _batch(12486, 12146, 337)
    without = batch.effective_input_tokens(CachedTokenRates.OPENAI)
    with_writes = batch.effective_input_tokens(
        CachedTokenRates.OPENAI, cache_write_rate=CachedTokenRates.OPENAI_WRITE
    )
    assert without == 12486 - 6073
    # 337 * 0.25 = 84.25, rounded up.
    assert with_writes == without + 85


def test_cache_writes_without_a_rate_keep_the_old_price() -> None:
    batch = _batch(1000, 0, 1000)
    assert batch.effective_input_tokens(CachedTokenRates.OPENAI) == 1000


def test_estimated_cost_prices_cache_writes() -> None:
    batch = _batch(1_000_000, 0, 1_000_000, output=0)
    cost = batch.estimated_cost(
        2.0,
        8.0,
        cached_token_rate=CachedTokenRates.ANTHROPIC_READ,
        cache_write_rate=CachedTokenRates.ANTHROPIC_WRITE_1H,
    )
    assert cost == pytest.approx(4.0)


@pytest.mark.parametrize("rate", [-0.1, float("nan"), float("inf")])
def test_invalid_cache_write_rate_raises(rate: float) -> None:
    with pytest.raises(ValueError, match="cache_write_rate"):
        _batch(10, 0, 10).effective_input_tokens(0.5, cache_write_rate=rate)


def test_llm_response_positional_construction_is_unchanged() -> None:
    response = LLMResponse("t", 10, 1, 11, 4, {"k": "v"}, "raw")
    assert (response.cached_input_tokens, response.raw) == (4, "raw")
    assert response.cache_write_input_tokens == 0
