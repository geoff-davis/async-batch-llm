"""Tests for the OpenAICompatibleModel base class.

Exercises shared behavior (message coercion, system instruction handling,
extra_headers/body forwarding, token/metadata extraction, and the missing-SDK
ImportError) via the concrete subclasses, since instantiating the base class
itself is uninteresting.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from async_batch_llm.models import (
    DeepSeekModel,
    OpenAICompatibleModel,
    OpenAIModel,
    _coerce_to_messages,
    _has_system_message,
)


def _build_response(
    *,
    content: str | None = "hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    cached_tokens: int | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4o-mini",
) -> MagicMock:
    """Build a MagicMock that quacks like an openai ChatCompletion response."""
    response = MagicMock()
    response.model = model

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    response.choices = [choice]

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens
    if cached_tokens is None:
        # Older models / non-cached calls — no prompt_tokens_details at all.
        usage.prompt_tokens_details = None
    else:
        details = MagicMock()
        details.cached_tokens = cached_tokens
        usage.prompt_tokens_details = details
    response.usage = usage

    return response


def _build_client(response: MagicMock) -> MagicMock:
    """Build a mock AsyncOpenAI whose chat.completions.create returns ``response``."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


class TestMessageCoercion:
    def test_string_prompt_becomes_user_message(self):
        assert _coerce_to_messages("hi") == [{"role": "user", "content": "hi"}]

    def test_list_prompt_passes_through(self):
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        assert _coerce_to_messages(msgs) == msgs

    def test_has_system_message_detects(self):
        assert _has_system_message([{"role": "system", "content": "x"}]) is True
        assert _has_system_message([{"role": "user", "content": "x"}]) is False
        assert _has_system_message([]) is False


class TestOpenAICompatibleGenerate:
    """Generate-side behavior, exercised via OpenAIModel."""

    @pytest.mark.asyncio
    async def test_basic_text_response(self):
        response = _build_response(content="output text")
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client)
        result = await model.generate("hello")

        assert result.text == "output text"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.total_tokens == 15
        assert result.cached_input_tokens == 0
        assert result.metadata == {"finish_reason": "stop", "model": "gpt-4o-mini"}
        assert result.raw is response

    @pytest.mark.asyncio
    async def test_string_prompt_becomes_single_user_message(self):
        response = _build_response()
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client)
        await model.generate("hello")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_list_prompt_passes_through_unchanged(self):
        response = _build_response()
        client = _build_client(response)

        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        model = OpenAIModel("gpt-4o-mini", client)
        await model.generate(msgs)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["messages"] == msgs

    @pytest.mark.asyncio
    async def test_default_system_instruction_prepended(self):
        response = _build_response()
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client, system_instruction="be brief")
        await model.generate("hello")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["messages"][0] == {"role": "system", "content": "be brief"}
        assert kwargs["messages"][1] == {"role": "user", "content": "hello"}

    @pytest.mark.asyncio
    async def test_per_call_system_instruction_overrides_default(self):
        response = _build_response()
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client, system_instruction="default")
        await model.generate("hello", system_instruction="override")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["messages"][0] == {"role": "system", "content": "override"}

    @pytest.mark.asyncio
    async def test_system_instruction_skipped_when_messages_already_have_one(self):
        response = _build_response()
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client, system_instruction="default")
        msgs = [
            {"role": "system", "content": "from caller"},
            {"role": "user", "content": "hi"},
        ]
        await model.generate(msgs)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["messages"][0]["content"] == "from caller"
        # Default should NOT be prepended.
        assert all(m["content"] != "default" for m in kwargs["messages"])

    @pytest.mark.asyncio
    async def test_extra_headers_forwarded(self):
        response = _build_response()
        client = _build_client(response)

        model = OpenAIModel(
            "gpt-4o-mini",
            client,
            extra_headers={"X-Foo": "bar"},
        )
        await model.generate("hi")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_headers"] == {"X-Foo": "bar"}

    @pytest.mark.asyncio
    async def test_extra_body_default_and_override(self):
        response = _build_response()
        client = _build_client(response)

        model = OpenAIModel(
            "gpt-4o-mini",
            client,
            extra_body={"max_tokens": 100, "top_p": 0.9},
        )
        await model.generate("hi", config={"max_tokens": 200})

        kwargs = client.chat.completions.create.call_args.kwargs
        # config overrides instance default for max_tokens; top_p preserved.
        assert kwargs["extra_body"] == {"max_tokens": 200, "top_p": 0.9}

    @pytest.mark.asyncio
    async def test_temperature_sent_by_default(self):
        response = _build_response()
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client)
        await model.generate("hi")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_temperature_none_omits_param(self):
        # Reasoning models (o1/o3) reject an explicit temperature; None drops it.
        response = _build_response()
        client = _build_client(response)

        model = OpenAIModel("o1-mini", client)
        await model.generate("hi", temperature=None)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_cached_tokens_extracted(self):
        response = _build_response(cached_tokens=42)
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client)
        result = await model.generate("hi")

        assert result.cached_input_tokens == 42

    @pytest.mark.asyncio
    async def test_no_usage_returns_zeros(self):
        response = _build_response()
        response.usage = None
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client)
        result = await model.generate("hi")

        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0
        assert result.cached_input_tokens == 0

    @pytest.mark.asyncio
    async def test_none_content_raises_with_finish_reason(self):
        response = _build_response(content=None, finish_reason="length")
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client)
        with pytest.raises(ValueError, match="finish_reason='length'"):
            await model.generate("hi")

    @pytest.mark.asyncio
    async def test_no_choices_raises(self):
        response = _build_response()
        response.choices = []
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client)
        with pytest.raises(ValueError, match="No choices returned"):
            await model.generate("hi")

    @pytest.mark.asyncio
    async def test_extract_tokens_overridable(self):
        response = _build_response()
        client = _build_client(response)

        class CustomModel(OpenAICompatibleModel):
            _default_base_url = None

            def _extract_tokens(self, response):
                # Pretend we read DeepSeek-style fields.
                return 99, 88, 77, 66

        model = CustomModel("custom", client)
        result = await model.generate("hi")

        assert result.input_tokens == 99
        assert result.output_tokens == 88
        assert result.total_tokens == 77
        assert result.cached_input_tokens == 66

    @pytest.mark.asyncio
    async def test_extract_metadata_overridable(self):
        response = _build_response()
        client = _build_client(response)

        class CustomModel(OpenAICompatibleModel):
            _default_base_url = None

            def _extract_metadata(self, response):
                return {"custom": "metadata"}

        model = CustomModel("custom", client)
        result = await model.generate("hi")

        assert result.metadata == {"custom": "metadata"}


class TestDeepSeekModel:
    """DeepSeek reports cache hits at the top level of usage, not nested."""

    @pytest.mark.asyncio
    async def test_reads_prompt_cache_hit_tokens(self):
        # No nested prompt_tokens_details (cached_tokens=None), so the OpenAI
        # path would report 0 cached — DeepSeek's override must pick up the
        # top-level prompt_cache_hit_tokens instead.
        response = _build_response(prompt_tokens=100)
        response.usage.prompt_cache_hit_tokens = 30
        response.usage.prompt_cache_miss_tokens = 70
        client = _build_client(response)

        model = DeepSeekModel("deepseek-chat", client)
        result = await model.generate("hi")

        assert result.input_tokens == 100
        assert result.cached_input_tokens == 30

    @pytest.mark.asyncio
    async def test_no_cache_field_yields_zero(self):
        response = _build_response()
        response.usage.prompt_cache_hit_tokens = None
        client = _build_client(response)

        model = DeepSeekModel("deepseek-chat", client)
        result = await model.generate("hi")

        assert result.cached_input_tokens == 0

    def test_base_url_and_env_var(self):
        assert DeepSeekModel._default_base_url == "https://api.deepseek.com"
        assert DeepSeekModel._api_key_env_var == "DEEPSEEK_API_KEY"


class TestImportError:
    def test_constructor_raises_when_sdk_missing(self):
        with patch("async_batch_llm.models.AsyncOpenAI", None):
            with pytest.raises(ImportError) as exc_info:
                OpenAIModel("gpt-4o-mini", MagicMock())
            assert "openai is required" in str(exc_info.value)
            assert "[openai]" in str(exc_info.value)

    def test_from_api_key_raises_when_sdk_missing(self):
        with patch("async_batch_llm.models.AsyncOpenAI", None):
            with pytest.raises(ImportError) as exc_info:
                OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-x")
            assert "openai is required" in str(exc_info.value)
