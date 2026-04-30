"""Tests for OpenAIModel and OpenAIStrategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from async_batch_llm import OpenAIModel, OpenAIStrategy
from async_batch_llm.base import LLMResponse


class TestOpenAIModelFromApiKey:
    def test_uses_default_base_url(self):
        # OpenAIModel default base url is None — the SDK uses its own default.
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-fake-key")
        assert isinstance(model, OpenAIModel)
        # Underlying client has SDK default base URL (api.openai.com).
        assert "api.openai.com" in str(model._client.base_url)

    def test_explicit_base_url_overrides(self):
        model = OpenAIModel.from_api_key(
            "gpt-4o-mini",
            api_key="sk-fake-key",
            base_url="https://custom.example.com/v1",
        )
        assert "custom.example.com" in str(model._client.base_url)


class TestOpenAIStrategy:
    """OpenAIStrategy delegates to model.generate; exercise via mocked LLMModel."""

    def _mock_model(self, *, text: str = "out", cached: int = 0) -> MagicMock:
        m = AsyncMock()
        m.generate = AsyncMock(
            return_value=LLMResponse(
                text=text,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cached_input_tokens=cached,
            )
        )
        return m

    @pytest.mark.asyncio
    async def test_default_parser_returns_text(self):
        model = self._mock_model(text="hello")
        strategy = OpenAIStrategy(model)

        out, tokens = await strategy.execute("p", 1, 10.0)

        assert out == "hello"
        assert tokens["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_custom_response_parser(self):
        from pydantic import BaseModel

        class Out(BaseModel):
            t: str

        model = self._mock_model(text="parsed")
        strategy = OpenAIStrategy(model, response_parser=lambda r: Out(t=r.text))

        out, _ = await strategy.execute("p", 1, 10.0)

        assert isinstance(out, Out)
        assert out.t == "parsed"

    @pytest.mark.asyncio
    async def test_parser_failure_attaches_failed_token_usage(self):
        model = self._mock_model(text="bad")

        def bad_parser(_):
            raise ValueError("can't parse")

        strategy = OpenAIStrategy(model, response_parser=bad_parser)

        with pytest.raises(ValueError) as exc_info:
            await strategy.execute("p", 1, 10.0)

        # Framework should be able to recover token usage from the failure.
        assert exc_info.value.__dict__.get("_failed_token_usage") == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

    @pytest.mark.asyncio
    async def test_prepare_cleanup_noop_on_stateless_model(self):
        model = self._mock_model()
        strategy = OpenAIStrategy(model)

        # No lifecycle methods on the mock — these should silently skip.
        await strategy.prepare()
        await strategy.cleanup()

    @pytest.mark.asyncio
    async def test_cached_tokens_propagate(self):
        model = self._mock_model(text="x", cached=42)
        strategy = OpenAIStrategy(model)

        _, tokens = await strategy.execute("p", 1, 10.0)

        assert tokens["cached_input_tokens"] == 42
