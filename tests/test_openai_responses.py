"""OpenAIModel on the Responses API (v0.27): translation, wire format, results."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest

import async_batch_llm.models as models
from async_batch_llm import (
    DeepSeekModel,
    DeepSeekStrategy,
    EmptyResponseError,
    GuardrailConfig,
    JsonlArtifactStore,
    ModelStrategy,
    OpenAIModel,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    ProviderResponseError,
    ResumePolicy,
    SqliteArtifactStore,
    process_prompts,
)
from async_batch_llm._internal.responses_translation import (
    translate_input,
    translate_request_config,
)
from async_batch_llm.artifacts import infer_artifact_identity
from async_batch_llm.base import LLMWorkItem
from async_batch_llm.classifiers.openai import OpenAIErrorClassifier

# ── Translation (pure functions) ─────────────────────────────────────────────


def test_tools_and_tool_choice_are_flattened_with_explicit_strict() -> None:
    config = {
        "tools": [
            {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "b", "strict": True}},
            {"type": "web_search"},
        ],
        "tool_choice": {"type": "function", "function": {"name": "a"}},
    }
    original = copy.deepcopy(config)
    out = translate_request_config(config)
    assert out["tools"] == [
        {"type": "function", "name": "a", "parameters": {"type": "object"}, "strict": False},
        {"type": "function", "name": "b", "strict": True},
        {"type": "web_search"},
    ]
    assert out["tool_choice"] == {"type": "function", "name": "a"}
    assert translate_request_config({"tool_choice": "required"})["tool_choice"] == "required"
    assert config == original  # never mutated


def test_message_translation_covers_parts_tool_calls_and_native_items() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "https://x/i.png", "detail": "low"}},
                {"type": "input_file", "file_id": "file_1"},
            ],
        },
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        {"type": "function_call_output", "call_id": "call_0", "output": "native"},
    ]
    original = copy.deepcopy(messages)
    assert translate_input(messages) == [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "https://x/i.png", "detail": "low"},
                {"type": "input_file", "file_id": "file_1"},
            ],
        },
        {"role": "assistant", "content": "calling"},
        {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        {"type": "function_call_output", "call_id": "call_0", "output": "native"},
    ]
    assert messages == original
    assert translate_input([{"role": "assistant", "content": [{"type": "text", "text": "a"}]}]) == [
        {"role": "assistant", "content": [{"type": "input_text", "text": "a"}]}
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}],
        [{"role": "user", "content": [{"type": "file", "file": {}}]}],
        [{"role": "function", "name": "f", "content": "x"}],
        [{"role": "tool", "content": "no id"}],
        [{"role": "user", "content": b"bytes"}],
        ["not a mapping"],
    ],
)
def test_unsupported_input_shapes_raise_before_the_request(messages: list[Any]) -> None:
    with pytest.raises(ValueError, match='api_surface="chat_completions"'):
        translate_input(messages)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"max_tokens": 5}, {"max_output_tokens": 5}),
        ({"max_completion_tokens": 5, "max_tokens": 5}, {"max_output_tokens": 5}),
        ({"max_output_tokens": 7, "max_tokens": 7}, {"max_output_tokens": 7}),
        ({"reasoning_effort": "low"}, {"reasoning": {"effort": "low"}}),
        (
            {"reasoning_effort": "low", "reasoning": {"summary": "auto"}},
            {"reasoning": {"summary": "auto", "effort": "low"}},
        ),
        (
            {"response_format": {"type": "json_object"}},
            {"text": {"format": {"type": "json_object"}}},
        ),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "V", "schema": {"type": "object"}, "strict": True},
                },
                "text": {"verbosity": "low"},
            },
            {
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "json_schema",
                        "name": "V",
                        "schema": {"type": "object"},
                        "strict": True,
                    },
                }
            },
        ),
        (
            {"logprobs": True, "top_logprobs": 2, "include": ["reasoning.encrypted_content"]},
            {
                "include": ["reasoning.encrypted_content", "message.output_text.logprobs"],
                "top_logprobs": 2,
            },
        ),
        ({"logprobs": False, "top_p": 0.5, "user": "u"}, {"top_p": 0.5, "user": "u"}),
        ({"stream": False, "store": True}, {"store": True}),
    ],
)
def test_request_aliases(config: dict[str, Any], expected: dict[str, Any]) -> None:
    assert translate_request_config(config) == expected


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"max_output_tokens": 5, "max_tokens": 6}, "conflicts"),
        ({"max_completion_tokens": 5, "max_tokens": 6}, "conflicts"),
        ({"reasoning_effort": "low", "reasoning": {"effort": "high"}}, "conflicts"),
        (
            {"response_format": {"type": "json_object"}, "text": {"format": {"type": "text"}}},
            "conflicts",
        ),
        ({"response_format": {"type": "weird"}}, "not supported"),
        ({"stream": True}, "stream=True"),
        ({"background": True}, "background=True"),
        ({"n": 2}, "n"),
        ({"stop": ["x"], "seed": 1}, "seed, stop"),
        ({"presence_penalty": 1, "frequency_penalty": 1, "logit_bias": {}}, "penalty"),
        ({"functions": [], "function_call": "auto"}, "function_call, functions"),
    ],
)
def test_untranslatable_requests_raise(config: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        translate_request_config(config)


# ── Wire format through the real SDK ─────────────────────────────────────────


def _usage(input_tokens: int = 10, output_tokens: int = 4, cached: int = 3, reasoning: int = 2):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }


def _message(*content: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": list(content),
    }


def _text(text: str, **extra: Any) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": [], **extra}


def _body(output: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-test",
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": _usage(),
    }
    body.update(overrides)
    return body


class _Server:
    """Captures JSON request bodies and headers; replies with a fixed body."""

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.headers: list[httpx.Headers] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.headers.append(request.headers)
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, json=self.reply)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch):
    from openai import AsyncOpenAI

    holder: dict[str, _Server] = {}

    def make(reply: dict[str, Any]) -> _Server:
        holder["server"] = _Server(reply)
        return holder["server"]

    def factory(**kwargs: Any) -> Any:
        transport = httpx.MockTransport(holder["server"].handle)
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=transport))

    monkeypatch.setattr(models, "AsyncOpenAI", factory)
    return make


async def test_default_request_body_uses_responses_with_store_false(server) -> None:
    srv = server(_body([_message(_text("hi"))]))
    model = OpenAIModel.from_api_key(
        "gpt-test", api_key="k", system_instruction="Be brief.", extra_headers={"X-T": "1"}
    )
    response = await model.generate("Hello")
    assert srv.paths == ["/v1/responses"]
    assert srv.requests[0] == {
        "model": "gpt-test",
        "input": "Hello",
        "instructions": "Be brief.",
        "store": False,
    }
    assert srv.headers[0]["x-t"] == "1"
    assert response.text == "hi"
    assert (response.input_tokens, response.output_tokens, response.total_tokens) == (10, 4, 14)
    assert response.cached_input_tokens == 3
    assert response.metadata is not None
    assert response.metadata["finish_reason"] == "stop"
    assert response.metadata["response_status"] == "completed"
    assert response.metadata["reasoning_tokens"] == 2
    assert response.metadata["api_surface"] == "responses"
    await model.cleanup()


async def test_translated_config_and_overrides_reach_the_wire(server) -> None:
    srv = server(_body([_message(_text("{}"))]))
    model = OpenAIModel.from_api_key("gpt-test", api_key="k", json_mode=True)
    config = {
        "max_tokens": 50,
        "reasoning_effort": "low",
        "store": True,
        "logprobs": True,
        "top_logprobs": 1,
        "custom_field": 3,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    }
    await model.generate([{"role": "user", "content": "go"}], temperature=0.2, config=config)
    body = srv.requests[0]
    assert body["input"] == [{"role": "user", "content": "go"}]
    assert body["temperature"] == 0.2
    assert body["max_output_tokens"] == 50
    assert body["reasoning"] == {"effort": "low"}
    assert body["text"] == {"format": {"type": "json_object"}}
    assert body["store"] is True
    assert body["include"] == ["message.output_text.logprobs"]
    assert body["top_logprobs"] == 1
    assert body["custom_field"] == 3
    assert body["tools"] == [{"type": "function", "name": "f", "parameters": {}, "strict": False}]
    assert config["max_tokens"] == 50 and "max_output_tokens" not in config
    await model.cleanup()


async def test_chat_only_parameter_fails_before_any_request(server) -> None:
    srv = server(_body([_message(_text("x"))]))
    model = OpenAIModel.from_api_key("gpt-test", api_key="k")
    with pytest.raises(ValueError, match="seed"):
        await model.generate("x", config={"seed": 1})
    assert srv.requests == []
    await model.cleanup()


async def test_chat_opt_out_keeps_chat_completions(server) -> None:
    srv = server(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }
    )
    model = OpenAIModel.from_api_key("gpt-test", api_key="k", api_surface="chat_completions")
    response = await model.generate("x", config={"seed": 1})
    assert srv.paths == ["/v1/chat/completions"]
    assert srv.requests[0]["seed"] == 1
    assert response.metadata is not None
    assert response.metadata["reasoning_tokens"] == 1
    assert "api_surface" not in response.metadata
    await model.cleanup()


# ── Result policy ────────────────────────────────────────────────────────────


def _function_call(name: str = "lookup") -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": name,
        "arguments": '{"q": 1}',
        "status": "completed",
    }


def _reasoning(content: list[str] | None = None, summary: list[str] | None = None):
    item: dict[str, Any] = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": text} for text in summary or []],
    }
    if content is not None:
        item["content"] = [{"type": "reasoning_text", "text": text} for text in content]
    return item


async def _generate(server, body: dict[str, Any]):
    server(body)
    model = OpenAIModel.from_api_key("gpt-test", api_key="k")
    try:
        return await model.generate("x")
    finally:
        await model.cleanup()


async def test_tool_only_completion_succeeds_with_tool_calls(server) -> None:
    response = await _generate(server, _body([_reasoning(summary=["plan"]), _function_call()]))
    assert response.text == ""
    assert response.metadata is not None
    assert response.metadata["finish_reason"] == "tool_calls"
    assert response.tool_calls is not None
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("call_1", "lookup", '{"q": 1}')
    ]
    assert response.reasoning == "plan"


async def test_reasoning_prefers_content_over_summary(server) -> None:
    from openai.types.responses import ResponseReasoningItem

    if "content" not in ResponseReasoningItem.model_fields:
        pytest.skip("this OpenAI SDK drops reasoning content; the summary fallback applies")
    response = await _generate(
        server,
        _body([_reasoning(content=["deep"], summary=["short"]), _message(_text("a"))]),
    )
    assert response.reasoning == "deep"


async def test_logprobs_are_exposed_verbatim(server) -> None:
    logprob = {"token": "a", "bytes": [97], "logprob": -0.1, "top_logprobs": []}
    response = await _generate(server, _body([_message(_text("a", logprobs=[logprob]))]))
    assert response.logprobs == [logprob]


async def test_incomplete_with_text_succeeds_with_mapped_finish_reason(server) -> None:
    for reason, finish in (("max_output_tokens", "length"), ("content_filter", "content_filter")):
        response = await _generate(
            server,
            _body(
                [_message(_text("partial"))],
                status="incomplete",
                incomplete_details={"reason": reason},
            ),
        )
        assert response.text == "partial"
        assert response.metadata is not None
        assert response.metadata["finish_reason"] == finish
        assert response.metadata["response_status"] == "incomplete"


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            _body([], status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
            "Incomplete OpenAI",
        ),
        (_body([_message({"type": "refusal", "refusal": "no"})]), "refused"),
        (
            _body([_message({"type": "refusal", "refusal": "no"}), _reasoning(summary=["s"])]),
            "refused",
        ),
        (_body([]), "No text returned"),
    ],
)
async def test_unusable_results_raise_with_usage(server, body: dict[str, Any], match: str) -> None:
    with pytest.raises(EmptyResponseError, match=match) as caught:
        await _generate(server, body)
    assert caught.value._failed_token_usage["total_tokens"] == 14


@pytest.mark.parametrize(
    ("code", "message", "category", "retryable"),
    [
        ("insufficient_quota", "You exceeded your current quota", "insufficient_balance", False),
        ("rate_limit_exceeded", "Rate limit reached", "rate_limit", True),
        ("server_error", "The server had an error", "unknown", True),
    ],
)
async def test_failed_responses_raise_and_classify(
    server, code: str, message: str, category: str, retryable: bool
) -> None:
    body = _body([], status="failed", error={"code": code, "message": message})
    with pytest.raises(ProviderResponseError) as caught:
        await _generate(server, body)
    assert caught.value._failed_token_usage["total_tokens"] == 14
    info = OpenAIErrorClassifier().classify(caught.value)
    assert (info.error_category, info.is_retryable) == (category, retryable)


def test_constructor_requires_a_responses_capable_client() -> None:
    class ChatOnly:
        class chat:  # noqa: N801
            completions = object()

    with pytest.raises(ValueError, match="responses.create"):
        OpenAIModel("gpt-test", ChatOnly())  # type: ignore[arg-type]
    OpenAIModel("gpt-test", ChatOnly(), api_surface="chat_completions")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="api_surface"):
        OpenAIModel("gpt-test", ChatOnly(), api_surface="realtime")  # type: ignore[arg-type]


# ── Identity and accounting ──────────────────────────────────────────────────


class _Client:
    class responses:  # noqa: N801
        @staticmethod
        async def create(**kwargs: Any) -> Any:  # pragma: no cover - never called
            raise AssertionError


def test_inferred_identity_marks_only_the_openai_responses_surface() -> None:
    client = _Client()
    chat = infer_artifact_identity(
        OpenAIStrategy(OpenAIModel("gpt-test", client, api_surface="chat_completions"))
    )
    responses = infer_artifact_identity(OpenAIStrategy(OpenAIModel("gpt-test", client)))
    generic = infer_artifact_identity(ModelStrategy(OpenAIModel("gpt-test", client)))
    assert (chat.provider, chat.model, dict(chat.extra)) == ("openai", "gpt-test", {})
    assert dict(responses.extra) == {"api_surface": "responses"}
    assert dict(generic.extra) == {"api_surface": "responses"}

    deepseek = DeepSeekModel("deepseek-v4-flash", client, api_surface="responses")
    assert dict(infer_artifact_identity(ModelStrategy(deepseek)).extra) == {}
    assert dict(infer_artifact_identity(OpenAIStrategy(deepseek)).extra) == {}
    assert dict(infer_artifact_identity(DeepSeekStrategy(deepseek)).extra) == {
        "api_surface": "responses"
    }


async def test_responses_usage_counts_toward_the_budget(server) -> None:
    server(_body([_message(_text("ok"))]))
    model = OpenAIModel.from_api_key("gpt-test", api_key="k")
    strategy = OpenAIStrategy(model)
    config = ProcessorConfig(max_workers=1, guardrails=GuardrailConfig(max_total_tokens=20))
    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for index in range(3):
            await processor.add_work(LLMWorkItem(item_id=str(index), strategy=strategy, prompt="x"))
        result = await processor.process_all()
        stats = await processor.get_stats()
    assert stats["budget_tokens_used"] == 28
    assert result.termination.kind == "budget_exceeded"


# ── v0.26 checkpoints ────────────────────────────────────────────────────────

_FIXTURES = Path(__file__).parent / "fixtures" / "v0_26"


@pytest.mark.parametrize(
    ("fixture", "store_type"),
    [("openai_chat.jsonl", JsonlArtifactStore), ("openai_chat.sqlite", SqliteArtifactStore)],
)
@pytest.mark.parametrize(
    ("api_surface", "replayed", "paths"),
    [
        ("chat_completions", True, []),
        ("responses", False, ["/v1/responses", "/v1/responses"]),
    ],
)
async def test_v026_checkpoints_replay_only_with_the_chat_opt_out(
    server, tmp_path, fixture, store_type, api_surface, replayed, paths
) -> None:
    """Artifacts written by v0.26.0's OpenAIModel (Chat Completions).

    The opt-out keeps the v0.26 identity, so results replay with no request.
    The Responses default changes the inferred identity, so both items run again
    rather than raising.
    """
    path = tmp_path / fixture
    shutil.copy(_FIXTURES / fixture, path)
    srv = server(_body([_message(_text("v027 answer"))]))
    strategy = OpenAIStrategy(
        OpenAIModel.from_api_key("gpt-test", api_key="k", api_surface=api_surface)
    )
    batch = await process_prompts(
        strategy,
        [("a", "prompt a"), ("b", "prompt b")],
        artifact_store=store_type(path),
        resume=ResumePolicy.REUSE_ALL,
        preserve_order=True,
    )
    assert [item.replayed_from_artifact for item in batch.results] == [replayed, replayed]
    expected = "v026 answer" if replayed else "v027 answer"
    assert [item.output for item in batch.results] == [expected, expected]
    assert srv.paths == paths


# ── Success is decided from provider output, not extractor metadata (OR-1) ────


async def _generate_with(server, body: dict[str, Any], extractor):
    server(body)
    model = OpenAIModel.from_api_key("gpt-test", api_key="k", metadata_extractors=[extractor])
    try:
        return await model.generate("x")
    finally:
        await model.cleanup()


async def test_extractor_cannot_turn_empty_output_into_success(server) -> None:
    fake_call = {"id": "c", "name": "fake", "arguments": "{}"}
    with pytest.raises(EmptyResponseError, match="No text returned"):
        await _generate_with(server, _body([]), lambda response: {"tool_calls": [fake_call]})


async def test_extractor_cannot_turn_a_valid_call_into_failure(server) -> None:
    response = await _generate_with(
        server, _body([_function_call()]), lambda response: {"tool_calls": None}
    )
    assert response.text == ""
    # The user's override still wins in the returned metadata.
    assert response.metadata is not None
    assert response.metadata["tool_calls"] is None
    assert response.metadata["finish_reason"] == "tool_calls"


async def test_extractor_cannot_turn_a_refusal_into_success(server) -> None:
    body = _body([_message({"type": "refusal", "refusal": "no"}), _function_call()])
    with pytest.raises(EmptyResponseError, match="OpenAI refused the request"):
        await _generate_with(server, body, lambda response: {"refusal": None})


@pytest.mark.parametrize("status", ["incomplete", "in_progress", "queued"])
async def test_textless_function_calls_need_completed_status(server, status: str) -> None:
    body = _body([_function_call()], status=status, incomplete_details={"reason": "other"})
    with pytest.raises(EmptyResponseError):
        await _generate(server, body)


def test_translated_messages_validate_against_the_sdk_input_schema() -> None:
    pydantic = pytest.importorskip("pydantic")
    from openai.types.responses import ResponseFunctionToolCallParam, ResponseInputParam

    try:
        pydantic.TypeAdapter(ResponseFunctionToolCallParam).validate_python(
            {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}
        )
    except pydantic.ValidationError:
        pytest.skip("this OpenAI SDK's input schema still requires function_call ids")

    translated = translate_input(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]
    )
    pydantic.TypeAdapter(ResponseInputParam).validate_python(translated)
