"""Tests for pluggable metadata extractors and the Gemini grounding extractor.

Covers the v0.15.0 (issue #52) additions:
- ``_run_extractors`` merge/isolation/empty semantics
- ``grounding_metadata_extractor`` parsing of a grounded Gemini response
- ``metadata_extractors`` wired through GeminiModel and OpenAICompatibleModel
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from async_batch_llm import MetadataExtractor, grounding_metadata_extractor
from async_batch_llm.models import (
    GeminiModel,
    OpenAIModel,
    _run_extractors,
)

# ── _run_extractors ──────────────────────────────────────────────────────────


def test_run_extractors_merges_on_top_of_base():
    base = {"finish_reason": "STOP"}
    out = _run_extractors(object(), [lambda r: {"extra": 1}], base)
    assert out == {"finish_reason": "STOP", "extra": 1}
    # Base is not mutated.
    assert base == {"finish_reason": "STOP"}


def test_run_extractors_user_key_wins():
    base = {"finish_reason": "STOP"}
    out = _run_extractors(object(), [lambda r: {"finish_reason": "OVERRIDDEN"}], base)
    assert out == {"finish_reason": "OVERRIDDEN"}


def test_run_extractors_later_extractor_wins():
    out = _run_extractors(
        object(),
        [lambda r: {"k": "first"}, lambda r: {"k": "second"}],
        None,
    )
    assert out == {"k": "second"}


def test_run_extractors_isolates_failures():
    def boom(_response):
        raise RuntimeError("extractor blew up")

    out = _run_extractors(object(), [boom, lambda r: {"ok": True}], {"base": 1})
    # The failing extractor is skipped; the rest still contribute.
    assert out == {"base": 1, "ok": True}


def test_run_extractors_empty_returns_none():
    assert _run_extractors(object(), None, None) is None
    assert _run_extractors(object(), [lambda r: None], None) is None
    assert _run_extractors(object(), [lambda r: {}], {}) is None


def test_run_extractors_none_base_no_extractors():
    assert _run_extractors(object(), None, {"a": 1}) == {"a": 1}


# ── grounding_metadata_extractor ─────────────────────────────────────────────


def _grounded_response():
    web1 = SimpleNamespace(uri="https://example.com/a", title="Source A")
    web2 = SimpleNamespace(uri="https://example.com/b", title=None)
    gm = SimpleNamespace(
        grounding_chunks=[SimpleNamespace(web=web1), SimpleNamespace(web=web2)],
        web_search_queries=["what is x", "how does x work"],
        grounding_supports=[
            SimpleNamespace(
                segment=SimpleNamespace(text="X is a thing.", start_index=0, end_index=12),
                grounding_chunk_indices=[0, 1],
            )
        ],
    )
    return SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=gm)])


def test_grounding_extractor_parses_full_response():
    out = grounding_metadata_extractor(_grounded_response())
    assert out is not None
    grounding = out["grounding"]
    assert grounding["sources"] == [
        {"uri": "https://example.com/a", "title": "Source A"},
        {"uri": "https://example.com/b", "title": None},
    ]
    assert grounding["queries"] == ["what is x", "how does x work"]
    assert grounding["supports"] == [
        {
            "text": "X is a thing.",
            "start_index": 0,
            "end_index": 12,
            "chunk_indices": [0, 1],
        }
    ]


def test_grounding_extractor_no_candidates():
    assert grounding_metadata_extractor(SimpleNamespace(candidates=[])) is None
    assert grounding_metadata_extractor(SimpleNamespace(candidates=None)) is None


def test_grounding_extractor_no_grounding_metadata():
    resp = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=None)])
    assert grounding_metadata_extractor(resp) is None


def test_grounding_extractor_empty_grounding_returns_none():
    gm = SimpleNamespace(grounding_chunks=[], web_search_queries=[], grounding_supports=[])
    resp = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=gm)])
    assert grounding_metadata_extractor(resp) is None


def test_grounding_extractor_skips_chunks_without_web():
    gm = SimpleNamespace(
        grounding_chunks=[
            SimpleNamespace(web=None),
            SimpleNamespace(web=SimpleNamespace(uri="https://kept.com", title="Kept")),
        ],
        web_search_queries=None,
        grounding_supports=None,
    )
    resp = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=gm)])
    out = grounding_metadata_extractor(resp)
    assert out == {"grounding": {"sources": [{"uri": "https://kept.com", "title": "Kept"}]}}


def test_metadata_extractor_type_alias_is_exported():
    # Smoke-check the public type alias is importable and usable as an annotation.
    extractors: list[MetadataExtractor] = [grounding_metadata_extractor]
    assert extractors


# ── GeminiModel wiring ───────────────────────────────────────────────────────


def _mock_gemini_response(*, with_grounding: bool = False):
    response = MagicMock()
    response.text = "answer"
    response.usage_metadata = MagicMock()
    response.usage_metadata.prompt_token_count = 5
    response.usage_metadata.candidates_token_count = 7
    response.usage_metadata.total_token_count = 12
    response.usage_metadata.cached_content_token_count = 0

    candidate = MagicMock()
    candidate.safety_ratings = None
    candidate.finish_reason = "STOP"
    if with_grounding:
        web = SimpleNamespace(uri="https://src.com", title="Src")
        candidate.grounding_metadata = SimpleNamespace(
            grounding_chunks=[SimpleNamespace(web=web)],
            web_search_queries=["q"],
            grounding_supports=None,
        )
    else:
        # Explicit None: a bare MagicMock auto-attribute would be truthy and
        # only stays out of metadata because MagicMock iterates empty.
        candidate.grounding_metadata = None
    response.candidates = [candidate]
    return response


def _mock_gemini_client(response):
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_gemini_model_user_extractor_merges_with_builtin():
    response = _mock_gemini_response()
    model = GeminiModel(
        "gemini-test",
        _mock_gemini_client(response),
        metadata_extractors=[lambda r: {"custom_key": "custom_value"}],
    )
    llm_response = await model.generate("prompt")

    assert llm_response.metadata is not None
    # Built-in finish_reason still present...
    assert llm_response.metadata["finish_reason"] == "STOP"
    # ...alongside the user-contributed key.
    assert llm_response.metadata["custom_key"] == "custom_value"


@pytest.mark.asyncio
async def test_gemini_model_without_extractors_unchanged():
    response = _mock_gemini_response()
    model = GeminiModel("gemini-test", _mock_gemini_client(response))
    llm_response = await model.generate("prompt")

    assert llm_response.metadata == {"finish_reason": "STOP"}


@pytest.mark.asyncio
async def test_gemini_model_grounding_extractor_end_to_end():
    response = _mock_gemini_response(with_grounding=True)
    model = GeminiModel(
        "gemini-test",
        _mock_gemini_client(response),
        metadata_extractors=[grounding_metadata_extractor],
    )
    llm_response = await model.generate("prompt")

    assert llm_response.metadata is not None
    assert llm_response.metadata["grounding"]["sources"] == [
        {"uri": "https://src.com", "title": "Src"}
    ]
    assert llm_response.metadata["grounding"]["queries"] == ["q"]


@pytest.mark.asyncio
async def test_gemini_model_emits_grounding_by_default():
    """Since v0.16.0, no extractor is needed: a grounded response lands in
    metadata['grounding'] by default, readable through the typed view."""
    response = _mock_gemini_response(with_grounding=True)
    model = GeminiModel("gemini-test", _mock_gemini_client(response))
    llm_response = await model.generate("prompt")

    assert llm_response.metadata is not None
    assert llm_response.metadata["grounding"]["sources"] == [
        {"uri": "https://src.com", "title": "Src"}
    ]
    assert llm_response.grounding is not None
    assert llm_response.grounding.sources[0].uri == "https://src.com"
    assert llm_response.grounding.queries == ["q"]
    json.dumps(llm_response.metadata)


@pytest.mark.asyncio
async def test_gemini_explicit_grounding_extractor_is_redundant_not_conflicting():
    """Phase-1 callers passing the extractor explicitly get the identical
    single 'grounding' key — same value, no duplication."""
    response = _mock_gemini_response(with_grounding=True)
    with_extractor = GeminiModel(
        "gemini-test",
        _mock_gemini_client(response),
        metadata_extractors=[grounding_metadata_extractor],
    )
    without = GeminiModel("gemini-test", _mock_gemini_client(response))

    assert (await with_extractor.generate("p")).metadata == (await without.generate("p")).metadata


@pytest.mark.asyncio
async def test_gemini_user_extractor_overrides_builtin_grounding():
    response = _mock_gemini_response(with_grounding=True)
    model = GeminiModel(
        "gemini-test",
        _mock_gemini_client(response),
        metadata_extractors=[lambda r: {"grounding": {"queries": ["custom"]}}],
    )
    llm_response = await model.generate("prompt")

    assert llm_response.metadata is not None
    assert llm_response.metadata["grounding"] == {"queries": ["custom"]}


# ── OpenAICompatibleModel wiring ─────────────────────────────────────────────


def _mock_openai_response():
    message = SimpleNamespace(content="hi", reasoning_content="because")
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=3, completion_tokens=4, total_tokens=7, prompt_tokens_details=None
    )
    return SimpleNamespace(choices=[choice], model="gpt-4o-mini", usage=usage)


@pytest.mark.asyncio
async def test_openai_model_user_extractor_merges_with_builtin():
    response = _mock_openai_response()
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    def reasoning_extractor(resp):
        rc = getattr(resp.choices[0].message, "reasoning_content", None)
        return {"reasoning_content": rc} if rc else None

    model = OpenAIModel("gpt-4o-mini", client, metadata_extractors=[reasoning_extractor])
    llm_response = await model.generate("prompt")

    assert llm_response.metadata is not None
    # Built-in keys preserved.
    assert llm_response.metadata["finish_reason"] == "stop"
    assert llm_response.metadata["model"] == "gpt-4o-mini"
    # User extractor key present.
    assert llm_response.metadata["reasoning_content"] == "because"


# ── OpenAI-compatible reserved keys (reasoning / tool_calls / logprobs) ──────


def _openai_client(response):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _openai_response(message, *, logprobs=None):
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=logprobs)
    usage = SimpleNamespace(
        prompt_tokens=3, completion_tokens=4, total_tokens=7, prompt_tokens_details=None
    )
    return SimpleNamespace(choices=[choice], model="gpt-4o-mini", usage=usage)


@pytest.mark.asyncio
async def test_openai_reasoning_content_emitted_by_default():
    """DeepSeek-style reasoning_content lands in metadata['reasoning']."""
    message = SimpleNamespace(content="hi", reasoning_content="because")
    response = _openai_response(message)
    model = OpenAIModel("gpt-4o-mini", _openai_client(response))
    llm_response = await model.generate("prompt")

    assert llm_response.metadata is not None
    assert llm_response.metadata["reasoning"] == "because"
    assert llm_response.reasoning == "because"


@pytest.mark.asyncio
async def test_openai_reasoning_fallback_field():
    """OpenRouter-style message.reasoning is the fallback."""
    message = SimpleNamespace(content="hi", reasoning="thought about it")
    response = _openai_response(message)
    model = OpenAIModel("gpt-4o-mini", _openai_client(response))
    llm_response = await model.generate("prompt")

    assert llm_response.metadata is not None
    assert llm_response.metadata["reasoning"] == "thought about it"


@pytest.mark.asyncio
async def test_openai_tool_calls_emitted_as_plain_dicts():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="lookup", arguments='{"q": "a"}'),
    )
    message = SimpleNamespace(content="calling a tool", tool_calls=[tool_call])
    response = _openai_response(message)
    model = OpenAIModel("gpt-4o-mini", _openai_client(response))
    llm_response = await model.generate("prompt")

    assert llm_response.metadata is not None
    assert llm_response.metadata["tool_calls"] == [
        {"id": "call_1", "name": "lookup", "arguments": '{"q": "a"}'}
    ]
    assert llm_response.tool_calls is not None
    assert llm_response.tool_calls[0].name == "lookup"
    # arguments stays the raw JSON string, deliberately unparsed.
    assert llm_response.tool_calls[0].arguments == '{"q": "a"}'
    json.dumps(llm_response.metadata)


@pytest.mark.asyncio
async def test_openai_logprobs_emitted_via_model_dump():
    class FakeLogprobs:
        def model_dump(self):
            return {"content": [{"token": "hi", "logprob": -0.5}]}

    message = SimpleNamespace(content="hi")
    response = _openai_response(message, logprobs=FakeLogprobs())
    model = OpenAIModel("gpt-4o-mini", _openai_client(response))
    llm_response = await model.generate("prompt")

    assert llm_response.metadata is not None
    assert llm_response.metadata["logprobs"] == {"content": [{"token": "hi", "logprob": -0.5}]}
    json.dumps(llm_response.metadata)


@pytest.mark.asyncio
async def test_magicmock_response_emits_no_reserved_keys():
    """Guard regression: MagicMock auto-attributes are truthy — the isinstance
    guards must keep them out of metadata (and keep it JSON-serializable).
    Mirrors the MagicMock response builder in test_openai_compatible.py."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "hi"
    response.choices[0].finish_reason = "stop"
    response.model = "gpt-4o-mini"
    response.usage.prompt_tokens = 3
    response.usage.completion_tokens = 4
    response.usage.total_tokens = 7
    response.usage.prompt_tokens_details = None
    model = OpenAIModel("gpt-4o-mini", _openai_client(response))
    llm_response = await model.generate("prompt")

    assert llm_response.metadata is not None
    assert not {"reasoning", "tool_calls", "logprobs"} & set(llm_response.metadata)
    json.dumps(llm_response.metadata)
