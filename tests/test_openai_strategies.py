"""Tests for OpenAIModel and OpenAIStrategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest

from async_batch_llm import OpenAIModel, OpenAIStrategy
from async_batch_llm.base import LLMResponse


class TestOpenAIModelFromApiKey:
    def test_uses_default_base_url(self):
        # OpenAIModel default base url is None — the SDK uses its own default.
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-fake-key")
        assert isinstance(model, OpenAIModel)
        # Underlying client has SDK default base URL (api.openai.com).
        # Parse and compare the host exactly rather than a substring check
        # (a substring `in` test on a URL is bypass-prone — CWE-20).
        assert urlparse(str(model._client.base_url)).hostname == "api.openai.com"
        # from_api_key takes ownership of the client.
        assert model._owns_client is True

    def test_explicit_base_url_overrides(self):
        model = OpenAIModel.from_api_key(
            "gpt-4o-mini",
            api_key="sk-fake-key",
            base_url="https://custom.example.com/v1",
        )
        assert urlparse(str(model._client.base_url)).hostname == "custom.example.com"

    def test_api_key_optional_lets_sdk_resolve_env_var(self, monkeypatch):
        # When api_key is None, the SDK reads OPENAI_API_KEY itself.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        model = OpenAIModel.from_api_key("gpt-4o-mini")
        assert model._client.api_key == "sk-from-env"

    def test_direct_constructor_does_not_own_client(self):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="sk-test")
        model = OpenAIModel("gpt-4o-mini", client)
        assert model._owns_client is False


class TestOpenAIModelLifecycle:
    @pytest.mark.asyncio
    async def test_cleanup_closes_owned_client(self):
        # Build via from_api_key, then swap the client for a mock so we can
        # verify .close() is awaited.
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test")
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        model._client = mock_client

        await model.cleanup()

        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_skips_user_provided_client(self):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="sk-test")
        model = OpenAIModel("gpt-4o-mini", client)

        # Spy on close so we can confirm we DON'T touch it.
        original_close = client.close
        close_called = False

        async def tracked_close(*args, **kwargs):
            nonlocal close_called
            close_called = True
            return await original_close(*args, **kwargs)

        client.close = tracked_close  # type: ignore[method-assign]

        await model.cleanup()

        assert close_called is False, (
            "cleanup() must not close clients that weren't created via from_api_key()"
        )

    @pytest.mark.asyncio
    async def test_cleanup_idempotent_when_close_fails(self, caplog):
        import logging

        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test")
        mock_client = MagicMock()
        mock_client.close = AsyncMock(side_effect=RuntimeError("boom"))
        model._client = mock_client

        caplog.set_level(logging.WARNING)
        # Should not propagate — cleanup is best-effort.
        await model.cleanup()
        assert any("Failed to close" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_strategy_cleanup_delegates_to_model(self):
        # OpenAICompatibleModel implements ManagedLLMModel (prepare + cleanup),
        # so OpenAIStrategy.cleanup() should propagate to it.
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test")
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        model._client = mock_client

        strategy = OpenAIStrategy(model)
        await strategy.cleanup()

        mock_client.close.assert_awaited_once()


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

        out, tokens, _metadata = await strategy.execute("p", 1, 10.0)

        assert out == "hello"
        assert tokens["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_custom_response_parser(self):
        from pydantic import BaseModel

        class Out(BaseModel):
            t: str

        model = self._mock_model(text="parsed")
        strategy = OpenAIStrategy(model, response_parser=lambda r: Out(t=r.text))

        out, _tokens, _metadata = await strategy.execute("p", 1, 10.0)

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

        _output, tokens, _metadata = await strategy.execute("p", 1, 10.0)

        assert tokens["cached_input_tokens"] == 42
