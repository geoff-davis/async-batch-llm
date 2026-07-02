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
    error: dict | None = None,
) -> MagicMock:
    response = MagicMock()
    response.model = model
    # Mirror a real ChatCompletion: no error field unless the body had one
    # (MagicMock would otherwise auto-create a truthy .error attribute).
    response.error = error
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


class TestErrorInResponseBody:
    """OpenRouter reports upstream failures as HTTP 200 + `error` object.

    Regression: these used to fall through to the "No choices returned"
    ValueError and classify as a non-retryable logic error, even though
    they're precisely the transient failures the classifier should retry.
    """

    @pytest.mark.asyncio
    async def test_error_body_raises_provider_response_error(self):
        from async_batch_llm import ProviderResponseError

        response = _build_response(
            error={"message": "Upstream provider is overloaded", "code": 502}
        )
        response.choices = []
        client = _build_client(response)

        model = OpenRouterModel("anthropic/claude-haiku-4-5", client)
        with pytest.raises(ProviderResponseError, match="overloaded") as exc_info:
            await model.generate("hi")

        assert exc_info.value.code == 502
        assert exc_info.value.provider_error == {
            "message": "Upstream provider is overloaded",
            "code": 502,
        }

    @pytest.mark.asyncio
    async def test_no_error_body_generates_normally(self):
        response = _build_response(content="fine")
        client = _build_client(response)

        model = OpenRouterModel("anthropic/claude-haiku-4-5", client)
        result = await model.generate("hi")

        assert result.text == "fine"

    def test_classifier_retries_provider_response_error(self):
        from async_batch_llm import ProviderResponseError
        from async_batch_llm.classifiers import OpenRouterErrorClassifier

        classifier = OpenRouterErrorClassifier()
        info = classifier.classify(
            ProviderResponseError("Upstream provider error (code=502)", code=502)
        )
        assert info.is_retryable is True
        assert info.is_rate_limit is False
        assert info.error_category == "upstream_error"

    def test_classifier_flags_embedded_429_as_rate_limit(self):
        from async_batch_llm import ProviderResponseError
        from async_batch_llm.classifiers import OpenRouterErrorClassifier

        classifier = OpenRouterErrorClassifier()
        info = classifier.classify(
            ProviderResponseError("Provider returned error (code=429)", code=429)
        )
        assert info.is_rate_limit is True
        assert info.is_retryable is True
        assert info.error_category == "rate_limit"

    def test_classifier_flags_no_provider_as_network_error(self):
        from async_batch_llm import ProviderResponseError
        from async_batch_llm.classifiers import OpenRouterErrorClassifier

        classifier = OpenRouterErrorClassifier()
        info = classifier.classify(
            ProviderResponseError("OpenRouter returned an error: no_provider_available", code=None)
        )
        assert info.is_retryable is True
        assert info.error_category == "network_error"


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
