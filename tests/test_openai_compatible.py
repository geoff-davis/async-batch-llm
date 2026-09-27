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
    _build_openai_http_client,
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
        with pytest.raises(ValueError, match="finish_reason='length'") as exc_info:
            await model.generate("hi")

        # The provider billed the call even though it produced no content —
        # the raised error must carry the usage for failed-attempt accounting.
        usage = exc_info.value._failed_token_usage
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_no_choices_raises(self):
        response = _build_response()
        response.choices = []
        client = _build_client(response)

        model = OpenAIModel("gpt-4o-mini", client)
        with pytest.raises(ValueError, match="No choices returned") as exc_info:
            await model.generate("hi")

        usage = exc_info.value._failed_token_usage
        assert usage["total_tokens"] == 15

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


class TestDeepSeekThinkingToggle:
    """thinking=True/False maps to DeepSeek's extra_body field (issue #27)."""

    def test_thinking_false_disables(self):
        model = DeepSeekModel("deepseek-v4-flash", MagicMock(), thinking=False)
        assert model._default_extra_body == {"thinking": {"type": "disabled"}}

    def test_thinking_true_enables(self):
        model = DeepSeekModel("deepseek-v4-flash", MagicMock(), thinking=True)
        assert model._default_extra_body == {"thinking": {"type": "enabled"}}

    def test_thinking_none_leaves_extra_body_untouched(self):
        model = DeepSeekModel("deepseek-chat", MagicMock())
        assert model._default_extra_body is None

    def test_thinking_merges_with_existing_extra_body(self):
        model = DeepSeekModel(
            "deepseek-v4-flash",
            MagicMock(),
            extra_body={"max_tokens": 100},
            thinking=False,
        )
        assert model._default_extra_body == {
            "max_tokens": 100,
            "thinking": {"type": "disabled"},
        }

    def test_explicit_thinking_in_extra_body_wins(self):
        model = DeepSeekModel(
            "deepseek-v4-flash",
            MagicMock(),
            extra_body={"thinking": {"type": "enabled"}},
            thinking=False,
        )
        assert model._default_extra_body == {"thinking": {"type": "enabled"}}

    def test_from_api_key_thinking_forwarded(self):
        with patch("async_batch_llm.models.AsyncOpenAI"):
            model = DeepSeekModel.from_api_key("deepseek-v4-flash", api_key="sk-x", thinking=False)
        assert model._default_extra_body == {"thinking": {"type": "disabled"}}

    @pytest.mark.asyncio
    async def test_thinking_forwarded_to_call(self):
        response = _build_response()
        client = _build_client(response)
        model = DeepSeekModel("deepseek-v4-flash", client, thinking=False)
        await model.generate("hi")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}


class TestJsonMode:
    """json_mode=True injects response_format into extra_body (issue #26)."""

    def test_json_mode_sets_response_format(self):
        with patch("async_batch_llm.models.AsyncOpenAI"):
            model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-x", json_mode=True)
        assert model._default_extra_body == {"response_format": {"type": "json_object"}}

    def test_json_mode_off_by_default(self):
        with patch("async_batch_llm.models.AsyncOpenAI"):
            model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-x")
        assert model._default_extra_body is None

    def test_explicit_response_format_wins_over_json_mode(self):
        with patch("async_batch_llm.models.AsyncOpenAI"):
            model = OpenAIModel.from_api_key(
                "gpt-4o-mini",
                api_key="sk-x",
                json_mode=True,
                extra_body={"response_format": {"type": "json_schema"}, "top_p": 0.9},
            )
        assert model._default_extra_body == {
            "response_format": {"type": "json_schema"},
            "top_p": 0.9,
        }

    @pytest.mark.asyncio
    async def test_json_mode_forwarded_to_call(self):
        response = _build_response()
        client = _build_client(response)
        with patch("async_batch_llm.models.AsyncOpenAI", return_value=client):
            model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-x", json_mode=True)
        await model.generate("return json")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["response_format"] == {"type": "json_object"}


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


class TestConnectionPoolSizing:
    """max_connections sizes the OpenAI SDK's matching HTTP pool (issue #25)."""

    class LegacyHttpxLimits:
        def __init__(
            self,
            *,
            max_connections: int = 100,
            max_keepalive_connections: int = 20,
        ) -> None:
            self.max_connections = max_connections
            self.max_keepalive_connections = max_keepalive_connections

    class Httpx2Limits(LegacyHttpxLimits):
        pass

    @pytest.mark.parametrize("limits_type", [LegacyHttpxLimits, Httpx2Limits])
    def test_http_client_uses_sdk_transport_limits(self, limits_type):
        default_limits = limits_type()
        with (
            patch("openai.DEFAULT_CONNECTION_LIMITS", default_limits),
            patch("openai.DefaultAsyncHttpxClient") as mock_http_client,
        ):
            result = _build_openai_http_client(150)

        limits = mock_http_client.call_args.kwargs["limits"]
        assert type(limits) is limits_type
        assert limits.max_connections == 150
        assert limits.max_keepalive_connections == 150
        assert result is mock_http_client.return_value

    def test_max_connections_builds_sized_http_client(self):
        with (
            patch("async_batch_llm.models.AsyncOpenAI") as mock_client_cls,
            patch("async_batch_llm.models._build_openai_http_client") as mock_http_client,
        ):
            OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-x", max_connections=150)

        mock_http_client.assert_called_once_with(150)
        _, kwargs = mock_client_cls.call_args
        assert kwargs["http_client"] is mock_http_client.return_value

    def test_no_max_connections_leaves_http_client_unset(self):
        with patch("async_batch_llm.models.AsyncOpenAI") as mock_client_cls:
            OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-x")
        _, kwargs = mock_client_cls.call_args
        assert "http_client" not in kwargs

    def test_max_connections_and_http_client_conflict(self):
        with patch("async_batch_llm.models.AsyncOpenAI"):
            with pytest.raises(ValueError, match="not both"):
                OpenAIModel.from_api_key(
                    "gpt-4o-mini",
                    api_key="sk-x",
                    max_connections=150,
                    http_client=MagicMock(),
                )

    def test_max_connections_must_be_positive(self):
        with patch("async_batch_llm.models.AsyncOpenAI"):
            with pytest.raises(ValueError, match=">= 1"):
                OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-x", max_connections=0)

    def test_deepseek_max_connections_forwarded(self):
        with (
            patch("async_batch_llm.models.AsyncOpenAI") as mock_client_cls,
            patch("async_batch_llm.models._build_openai_http_client") as mock_http_client,
        ):
            DeepSeekModel.from_api_key("deepseek-chat", api_key="sk-x", max_connections=300)
        mock_http_client.assert_called_once_with(300)
        _, kwargs = mock_client_cls.call_args
        assert kwargs["http_client"] is mock_http_client.return_value


@pytest.mark.parametrize("surface", ["call", "batch", "direct"])
async def test_prov1_owned_model_reopens_after_cleanup(surface, monkeypatch):
    import httpx
    from openai import AsyncOpenAI as SDKClient

    import async_batch_llm.models as models
    from async_batch_llm import OpenAIStrategy, call, process_prompts

    def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    created = []

    def construct(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = SDKClient(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(models, "AsyncOpenAI", construct)
    model = OpenAIModel.from_api_key("fake", api_key="key", max_retries=0)
    strategy = OpenAIStrategy(model)
    for _ in range(2):
        if surface == "call":
            assert await call(strategy, "x") == "ok"
        elif surface == "batch":
            batch = await process_prompts(strategy, [("x", "x")])
            assert batch.succeeded == 1
        else:
            assert (await model.generate("x")).text == "ok"
            await model.cleanup()
    assert len(created) == 2
    assert all(client.is_closed() for client in created)


async def test_prov1_caller_transport_survives_model_cleanup():
    import httpx

    transport = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    try:
        model = OpenAIModel.from_api_key("fake", api_key="key", http_client=transport)
        await model.cleanup()
        assert not transport.is_closed
        assert (await transport.get("https://test.invalid")).status_code == 200
    finally:
        await transport.aclose()


async def test_prov1_two_strategies_share_owned_model_until_last_release(monkeypatch):
    import asyncio

    import httpx
    from openai import AsyncOpenAI

    from async_batch_llm import OpenAIStrategy, ProcessorConfig, RetryConfig, call_result, models

    slow_started, fast_closed = asyncio.Event(), asyncio.Event()
    in_flight = 0
    close_counts = []

    async def handle(request):
        nonlocal in_flight
        in_flight += 1
        try:
            if b"slow" in request.content:
                slow_started.set()
                await fast_closed.wait()
            else:
                await slow_started.wait()
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        finally:
            in_flight -= 1

    def factory(**kwargs):
        client = AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        )
        original_close = client.close

        async def close():
            close_counts.append(in_flight)
            await original_close()

        client.close = close
        return client

    monkeypatch.setattr(models, "AsyncOpenAI", factory)
    model = OpenAIModel.from_api_key("m", api_key="test", max_retries=0)
    config = ProcessorConfig(max_workers=1, retry=RetryConfig(max_attempts=1))
    slow = asyncio.create_task(call_result(OpenAIStrategy(model), "slow", config=config))
    await slow_started.wait()
    try:
        fast = await call_result(OpenAIStrategy(model), "fast", config=config)
        assert fast.success
        assert close_counts == []
    finally:
        fast_closed.set()
        result = await slow
    assert result.success
    assert close_counts == [0]


async def test_prov1_abandoned_host_does_not_pin_retained_strategys_model_lease():
    import gc
    import weakref

    from async_batch_llm import OpenAIStrategy
    from async_batch_llm._internal.strategy_lifecycle import StrategyLifecycle

    client = _build_client(_build_response())
    client.close = AsyncMock()
    model = OpenAIModel("m", client=client)
    model._owns_client = True
    retained_strategy, peer = OpenAIStrategy(model), OpenAIStrategy(model)
    abandoned, remaining = StrategyLifecycle(), StrategyLifecycle()
    await abandoned.ensure_prepared(retained_strategy)
    await remaining.ensure_prepared(peer)
    abandoned_ref = weakref.ref(abandoned)
    del abandoned
    gc.collect()
    assert abandoned_ref() is None
    await remaining.cleanup_all()
    client.close.assert_awaited_once()


@pytest.mark.parametrize("negotiated", [False, True])
async def test_prov1_reopen_preserves_pool_limit(negotiated):
    import asyncio

    model = OpenAIModel.from_api_key(
        "m", api_key="test", **({} if negotiated else {"max_connections": 17})
    )
    if negotiated:
        assert await model.request_concurrency(17)
    original = model._client
    await model.cleanup()
    await asyncio.gather(*(model.prepare() for _ in range(10)))
    try:
        assert model._client is not original
        assert model.max_concurrency == 17
        assert model._client._client._transport._pool._max_connections == 17
    finally:
        await model.cleanup()


async def test_prov1_resize_does_not_close_active_shared_client():
    model = OpenAIModel.from_api_key("m", api_key="test")
    original = model._client
    model._client_used = True
    try:
        assert not await model.request_concurrency(17)
        assert model._client is original
        assert not original.is_closed()
    finally:
        await model.cleanup()


async def test_prov1_none_http_client_keeps_sdk_transport_owned():
    model = OpenAIModel.from_api_key("m", api_key="test", http_client=None)
    first = model._client
    await model.cleanup()
    assert first.is_closed()
    await model.prepare()
    try:
        assert model._client is not first
        assert not model._client.is_closed()
    finally:
        await model.cleanup()


@pytest.mark.parametrize("host_managed", [False, True])
async def test_prov1_redundant_manual_cleanup_preserves_peer_model_lease(host_managed):
    import gc

    from async_batch_llm import OpenAIStrategy
    from async_batch_llm._internal.strategy_lifecycle import StrategyLifecycle

    model = OpenAIModel.from_api_key("m", api_key="test")
    first, peer = OpenAIStrategy(model), OpenAIStrategy(model)
    host = StrategyLifecycle()
    await peer.prepare()
    try:
        if host_managed:
            await host.ensure_prepared(first)
            await host.cleanup_all()
        else:
            await first.prepare()
            await first.cleanup()
        gc.collect()
        assert not model._client.is_closed()
        await first.cleanup()
        assert not model._client.is_closed()
    finally:
        await peer.cleanup()
    assert model._client.is_closed()
    # Reusing the same strategy must acquire a fresh lease after the no-op close.
    await first.prepare()
    try:
        assert not model._client.is_closed()
    finally:
        await first.cleanup()
    assert model._client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("retries", [None, 3])
async def test_prov6_sdk_retry_setting_survives_reopen(retries):
    kwargs = {} if retries is None else {"max_retries": retries}
    model = OpenAIModel.from_api_key("fake", api_key="test", **kwargs)
    expected = 0 if retries is None else retries
    try:
        assert model._client.max_retries == expected
        assert await model.request_concurrency(3)
        assert model._client.max_retries == expected
        await model.cleanup()
        await model.prepare()
        assert model._client.max_retries == expected
    finally:
        await model.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("auth", ["admin_env", "admin_kwarg", "workload", "missing", "invalid"])
async def test_sc1_sdk_credentials_resolution(monkeypatch, auth):
    from openai import OpenAIError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    kwargs = {}
    if auth == "admin_env":
        monkeypatch.setenv("OPENAI_ADMIN_KEY", "test-admin")
    elif auth == "admin_kwarg":
        kwargs["admin_api_key"] = "test-admin"
    elif auth == "workload":
        kwargs["workload_identity"] = {"provider": "test-provider"}
    elif auth == "invalid":
        kwargs.update(api_key="test", workload_identity={"provider": "test-provider"})
    if auth == "missing":
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIModel.from_api_key("test", **kwargs)
    elif auth == "invalid":
        with pytest.raises(OpenAIError, match="mutually exclusive"):
            OpenAIModel.from_api_key("test", **kwargs)
    else:
        model = OpenAIModel.from_api_key("test", **kwargs)
        try:
            assert model._client.max_retries == 0
            await model.cleanup()
            await model.prepare()
            assert model._client.max_retries == 0
        finally:
            await model.cleanup()
