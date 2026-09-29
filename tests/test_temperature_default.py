"""Built-in temperature defaults omit the parameter (v0.27)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from async_batch_llm import (
    DeepSeekModel,
    DeepSeekStrategy,
    GeminiCachedModel,
    GeminiModel,
    GeminiStrategy,
    ModelStrategy,
    OpenAIModel,
    OpenAIStrategy,
    OpenRouterStrategy,
    llm,
)
from async_batch_llm.core.protocols import LLMModel
from async_batch_llm.models import OpenAICompatibleModel

_TEMPERATURE_SITES = [
    LLMModel.generate,
    GeminiModel.generate,
    GeminiCachedModel.generate,
    OpenAICompatibleModel.generate,
    DeepSeekModel.generate,
    ModelStrategy.__init__,
    GeminiStrategy.__init__,
    OpenAIStrategy.__init__,
    OpenRouterStrategy.__init__,
    DeepSeekStrategy.__init__,
    llm,
]


@pytest.mark.parametrize("site", _TEMPERATURE_SITES, ids=lambda site: site.__qualname__)
def test_every_built_in_temperature_default_is_none(site) -> None:
    assert inspect.signature(site).parameters["temperature"].default is None


def _chat_client() -> MagicMock:
    message = SimpleNamespace(content="ok", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
    usage = SimpleNamespace(
        prompt_tokens=2, completion_tokens=1, total_tokens=3, prompt_tokens_details=None
    )
    response = SimpleNamespace(choices=[choice], usage=usage, model="m", id="r")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_openai_strategy_default_omits_and_explicit_zero_is_sent() -> None:
    client = _chat_client()
    await OpenAIStrategy(
        OpenAIModel("gpt-4o-mini", client, api_surface="chat_completions")
    ).execute("hi", 1, 10.0)
    assert "temperature" not in client.chat.completions.create.call_args.kwargs

    await OpenAIStrategy(
        OpenAIModel("gpt-4o-mini", client, api_surface="chat_completions"), temperature=0.0
    ).execute("hi", 1, 10.0)
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.0


def _gemini_client() -> MagicMock:
    response = MagicMock()
    response.text = "ok"
    response.usage_metadata = SimpleNamespace(
        prompt_token_count=2,
        candidates_token_count=1,
        total_token_count=3,
        cached_content_token_count=0,
        thoughts_token_count=0,
    )
    response.candidates = []
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_gemini_strategy_default_omits_and_generation_config_still_applies() -> None:
    client = _gemini_client()
    await GeminiStrategy(GeminiModel("gemini-test", client)).execute("hi", 1, 10.0)
    assert "temperature" not in client.aio.models.generate_content.call_args.kwargs["config"]

    await GeminiStrategy(
        GeminiModel("gemini-test", client), generation_config={"temperature": 0.7}
    ).execute("hi", 1, 10.0)
    assert client.aio.models.generate_content.call_args.kwargs["config"]["temperature"] == 0.7
