"""Tests for OpenRouterModel and OpenRouterStrategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest

from async_batch_llm import OpenRouterModel, OpenRouterStrategy
from async_batch_llm.base import LLMResponse


def _build_response(
    *,
    content: str = "out",
    provider: str | None = None,
    model: str = "anthropic/claude-haiku-4-5",
) -> MagicMock:
    response = MagicMock()
    response.model = model
    if provider is not None:
        response.provider = provider
    else:
        # Strip the attribute so getattr() returns None.
        del response.provider

    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = content
    response.choices = [choice]

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    usage.prompt_tokens_details = None
    response.usage = usage

    return response


def _build_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


class TestOpenRouterModelFromApiKey:
    def test_default_base_url_points_to_openrouter(self):
        model = OpenRouterModel.from_api_key("anthropic/claude-haiku-4-5", api_key="sk-or-fake")
        # Parse and compare host + path exactly rather than a substring check
        # (a substring `in` test on a URL is bypass-prone — CWE-20).
        parsed = urlparse(str(model._client.base_url))
        assert parsed.hostname == "openrouter.ai"
        assert parsed.path.rstrip("/") == "/api/v1"
        assert model._owns_client is True

    def test_referer_and_title_become_default_headers(self):
        model = OpenRouterModel.from_api_key(
            "anthropic/claude-haiku-4-5",
            api_key="sk-or-fake",
            referer="https://my-app.example.com",
            title="My App",
        )
        assert model._default_extra_headers == {
            "HTTP-Referer": "https://my-app.example.com",
            "X-Title": "My App",
        }

    def test_extra_headers_preserved_alongside_referer_title(self):
        model = OpenRouterModel.from_api_key(
            "anthropic/claude-haiku-4-5",
            api_key="sk-or-fake",
            extra_headers={"X-Other": "value"},
            referer="https://x.example",
        )
        assert model._default_extra_headers == {
            "X-Other": "value",
            "HTTP-Referer": "https://x.example",
        }

    def test_reads_openrouter_api_key_env_var(self, monkeypatch):
        # The OpenAI SDK doesn't know about OPENROUTER_API_KEY; we read it
        # ourselves. Make sure the env var ends up on the SDK client.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-env")
        model = OpenRouterModel.from_api_key("anthropic/claude-haiku-4-5")
        assert model._client.api_key == "sk-or-from-env"

    def test_raises_when_no_api_key_resolvable(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterModel.from_api_key("anthropic/claude-haiku-4-5")

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        model = OpenRouterModel.from_api_key("anthropic/claude-haiku-4-5", api_key="explicit-key")
        assert model._client.api_key == "explicit-key"


class TestOpenRouterMetadata:
    @pytest.mark.asyncio
    async def test_provider_field_surfaced_in_metadata(self):
        response = _build_response(provider="Anthropic")
        client = _build_client(response)

        model = OpenRouterModel("anthropic/claude-haiku-4-5", client)
        result = await model.generate("hi")

        assert result.metadata is not None
        assert result.metadata["provider"] == "Anthropic"
        assert result.metadata["model"] == "anthropic/claude-haiku-4-5"
        assert result.metadata["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_no_provider_field_omitted(self):
        response = _build_response(provider=None)
        client = _build_client(response)

        model = OpenRouterModel("anthropic/claude-haiku-4-5", client)
        result = await model.generate("hi")

        assert result.metadata is not None
        assert "provider" not in result.metadata


class TestOpenRouterStrategy:
    """OpenRouterStrategy mirrors OpenAIStrategy; smoke-test the core flow."""

    @pytest.mark.asyncio
    async def test_default_parser_returns_text(self):
        m = AsyncMock()
        m.generate = AsyncMock(
            return_value=LLMResponse(
                text="hello",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            )
        )
        strategy = OpenRouterStrategy(m)

        out, tokens, _metadata = await strategy.execute("p", 1, 10.0)

        assert out == "hello"
        assert tokens["total_tokens"] == 15
