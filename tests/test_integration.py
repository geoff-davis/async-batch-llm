"""Real-provider integration tests (deselected by default; they make paid API calls).

Each provider's tests skip unless its key is set. Calls are tiny (a few tokens,
cheap models), so a full run costs cents. Run them with the keys in the
environment, for example from a ``.env`` file:

    set -a; source .env; set +a
    uv run pytest -m integration tests/test_integration.py -v

Keys: ``GOOGLE_API_KEY`` (Gemini, PydanticAI), ``OPENAI_API_KEY``,
``OPENROUTER_API_KEY``, ``DEEPSEEK_API_KEY``. The tests go through the public
entry points (``llm()``, ``process_prompts``) and check what unit tests mock:
token accounting, provider metadata, structured output, and how each provider's
authentication failure is classified.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import BaseModel, Field

from async_batch_llm import (
    BatchResult,
    ErrorCategory,
    LLMCallStrategy,
    ProcessorConfig,
    RetryConfig,
    llm,
    process_prompts,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

needs_google = pytest.mark.skipif(not GOOGLE_API_KEY, reason="needs GOOGLE_API_KEY")
needs_openai = pytest.mark.skipif(not OPENAI_API_KEY, reason="needs OPENAI_API_KEY")
needs_openrouter = pytest.mark.skipif(not OPENROUTER_API_KEY, reason="needs OPENROUTER_API_KEY")
needs_deepseek = pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="needs DEEPSEEK_API_KEY")

GEMINI_MODEL = "gemini-3.5-flash-lite"
OPENAI_MODEL = "gpt-6-luna"
OPENAI_REASONING_MODEL = "gpt-6-luna"  # reasons by default (medium effort)
OPENROUTER_MODEL = "openai/gpt-6-luna"
DEEPSEEK_MODEL = "deepseek-v4-flash"

PROMPTS = [f"Reply with the number {n} and nothing else." for n in (1, 2, 3)]
# Failures that should fail fast, without retries, when the key is wrong.
AUTH_FAILURES = {
    ErrorCategory.AUTHENTICATION,
    ErrorCategory.PERMISSION_DENIED,
    ErrorCategory.CLIENT_ERROR,
}


class Pick(BaseModel):
    value: int = Field(ge=1, le=10)


def _config(**kwargs: Any) -> ProcessorConfig:
    return ProcessorConfig(
        max_workers=3, attempt_timeout=60.0, retry=RetryConfig(max_attempts=2), **kwargs
    )


async def _run(strategy: LLMCallStrategy[Any], prompts: list[str] = PROMPTS) -> BatchResult:
    return await process_prompts(strategy, prompts, config=_config(), preserve_order=True)


def _assert_all_succeeded(result: BatchResult) -> None:
    errors = [(r.item_id, r.error_category, r.error) for r in result.results if not r.success]
    assert result.failed == 0, errors
    assert result.succeeded == result.total_items > 0
    for r in result.results:
        assert r.token_usage["input_tokens"] > 0, r.token_usage
        assert r.token_usage["output_tokens"] > 0, r.token_usage
        assert r.token_usage["total_tokens"] > 0, r.token_usage


def _assert_numbers_echoed(result: BatchResult) -> None:
    outputs = [str(r.output).strip() for r in result.results]
    for n, output in zip((1, 2, 3), outputs, strict=True):
        assert str(n) in output, outputs


async def _assert_auth_failure(strategy: LLMCallStrategy[Any]) -> None:
    result = await _run(strategy, ["Reply with ok."])
    [item] = result.results
    assert not item.success
    assert item.error_category in AUTH_FAILURES, (item.error_category, item.error)
    # A rejected key is not worth retrying.
    assert len(item.timing.attempts) == 1, item.timing.attempts


# Gemini


@needs_google
async def test_gemini_factory_batch():
    result = await _run(llm(f"gemini:{GEMINI_MODEL}"))
    _assert_all_succeeded(result)
    _assert_numbers_echoed(result)
    metadata = result.results[0].metadata or {}
    # Plain enum names, not SDK enum objects (v0.27).
    assert metadata.get("finish_reason") == "STOP", metadata


@needs_google
async def test_gemini_cached_model_reports_cached_tokens():
    from google import genai
    from google.genai.types import Content, Part

    from async_batch_llm import GeminiCachedModel, GeminiStrategy

    # Explicit caching needs a minimum prompt size; ~3,000 tokens clears it.
    facts = "\n".join(f"Fact {i}: the code word for item {i} is word{i}." for i in range(250))
    client = genai.Client(api_key=GOOGLE_API_KEY)
    model = GeminiCachedModel(
        model=GEMINI_MODEL,
        client=client,
        cached_content=[Content(role="user", parts=[Part(text=facts)])],
        cache_ttl_seconds=300,
        cache_renewal_buffer_seconds=60,
    )
    try:
        result = await _run(
            GeminiStrategy(model=model),
            [f"What is the code word for item {i}? Reply with the word only." for i in (3, 7)],
        )
        _assert_all_succeeded(result)
        assert [str(r.output).strip() for r in result.results] == ["word3", "word7"]
        cached = [r.token_usage.get("cached_input_tokens", 0) for r in result.results]
        assert all(tokens > 0 for tokens in cached), cached
        assert result.total_cached_tokens == sum(cached)
    finally:
        # The test owns this client, not the model: close both transports.
        try:
            await model.delete_cache()
        finally:
            try:
                await client.aio.aclose()
            finally:
                client.close()


@needs_google
async def test_gemini_bad_key_fails_fast():
    await _assert_auth_failure(llm(f"gemini:{GEMINI_MODEL}", api_key="invalid-key"))


@needs_google
async def test_pydantic_ai_structured_output():
    from pydantic_ai import Agent
    from pydantic_ai.models.google import GoogleModel

    from async_batch_llm import PydanticAIStrategy

    agent = Agent(GoogleModel(GEMINI_MODEL), output_type=Pick)
    result = await _run(
        PydanticAIStrategy(agent=agent), ["Pick the number seven.", "Pick the number two."]
    )
    _assert_all_succeeded(result)
    assert [r.output for r in result.results] == [Pick(value=7), Pick(value=2)]


# OpenAI


@needs_openai
async def test_openai_responses_default():
    result = await _run(llm(f"openai:{OPENAI_MODEL}"))
    _assert_all_succeeded(result)
    _assert_numbers_echoed(result)
    metadata = result.results[0].metadata or {}
    assert metadata.get("api_surface") == "responses", metadata
    assert metadata.get("provider_request_id"), metadata


@needs_openai
async def test_openai_chat_completions_opt_out():
    from async_batch_llm import OpenAIModel, OpenAIStrategy

    model = OpenAIModel.from_api_key(OPENAI_MODEL, api_surface="chat_completions")
    result = await _run(OpenAIStrategy(model))
    _assert_all_succeeded(result)
    _assert_numbers_echoed(result)
    assert (result.results[0].metadata or {}).get("api_surface") != "responses"


@needs_openai
async def test_openai_reasoning_model_reports_reasoning_tokens():
    result = await _run(llm(f"openai:{OPENAI_REASONING_MODEL}"), ["What is 17 + 25?"])
    _assert_all_succeeded(result)
    assert "42" in str(result.results[0].output)
    reasoning = (result.results[0].metadata or {}).get("reasoning_tokens")
    assert isinstance(reasoning, int) and reasoning >= 0, reasoning


@needs_openai
async def test_openai_bad_key_fails_fast():
    await _assert_auth_failure(llm(f"openai:{OPENAI_MODEL}", api_key="sk-invalid"))


# OpenRouter


@needs_openrouter
async def test_openrouter_factory_batch():
    result = await _run(llm(f"openrouter:{OPENROUTER_MODEL}"))
    _assert_all_succeeded(result)
    _assert_numbers_echoed(result)
    metadata = result.results[0].metadata or {}
    assert metadata.get("provider"), metadata  # the upstream OpenRouter routed to


@needs_openrouter
async def test_openrouter_bad_key_fails_fast():
    await _assert_auth_failure(llm(f"openrouter:{OPENROUTER_MODEL}", api_key="sk-or-invalid"))


# DeepSeek and generic OpenAI-compatible


@needs_deepseek
async def test_deepseek_chat_batch():
    result = await _run(llm(f"deepseek:{DEEPSEEK_MODEL}", thinking=False))
    _assert_all_succeeded(result)
    _assert_numbers_echoed(result)


@needs_deepseek
async def test_deepseek_responses_structured_output():
    from async_batch_llm import DeepSeekModel, DeepSeekStrategy

    model = DeepSeekModel.from_api_key(
        DEEPSEEK_MODEL, api_surface="responses", response_schema=Pick, thinking=False
    )
    strategy = DeepSeekStrategy(model, response_parser=lambda r: Pick.model_validate_json(r.text))
    result = await _run(strategy, ["Pick the number seven.", "Pick the number two."])
    _assert_all_succeeded(result)
    assert [r.output for r in result.results] == [Pick(value=7), Pick(value=2)]
    assert (result.results[0].metadata or {}).get("api_surface") == "responses"


@needs_deepseek
async def test_openai_compatible_factory_against_deepseek():
    strategy = llm(
        f"openai-compatible:{DEEPSEEK_MODEL}",
        base_url="https://api.deepseek.com",
        api_key=DEEPSEEK_API_KEY,
    )
    result = await _run(strategy)
    _assert_all_succeeded(result)
    _assert_numbers_echoed(result)


@needs_deepseek
async def test_deepseek_bad_key_fails_fast():
    await _assert_auth_failure(llm(f"deepseek:{DEEPSEEK_MODEL}", api_key="sk-invalid"))
