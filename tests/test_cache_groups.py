"""Per-cache-group hit rates and the cache hit-rate floor (issue #244)."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from async_batch_llm import (
    AbortMode,
    DeepSeekModel,
    ErrorCategory,
    GeminiCachedModel,
    GeminiModel,
    GuardrailConfig,
    JsonlArtifactStore,
    LLMCallPool,
    LLMCallStrategy,
    OpenAICompatibleModel,
    OpenAIModel,
    OpenRouterModel,
    ParallelBatchProcessor,
    ProcessorConfig,
    RateLimitConfig,
    ResumePolicy,
    RetryConfig,
    SqliteArtifactStore,
    call,
    call_result,
    process_prompts,
)
from async_batch_llm._internal.cache_groups import (
    MAX_CACHE_GROUPS,
    CacheHitTracker,
    normalize_cache_group,
)
from async_batch_llm.base import BatchResult, BatchTermination, LLMWorkItem, RetryState
from async_batch_llm.observers import BaseObserver, ProcessingEvent
from async_batch_llm.serialization import batch_result_from_dict, batch_result_to_dict

# ── Helpers ──────────────────────────────────────────────────────────────────

_FAST_RETRY = RetryConfig(max_attempts=3, initial_wait=0.001, max_wait=0.001, jitter=False)
_NO_COOLDOWN = RateLimitConfig(
    cooldown_seconds=0,
    max_cooldown_seconds=0,
    slow_start_items=0,
    slow_start_initial_delay=0,
    slow_start_final_delay=0,
    backoff_multiplier=1,
)


def _config(workers: int = 1, **guardrails: Any) -> ProcessorConfig:
    return ProcessorConfig(
        max_workers=workers,
        retry=_FAST_RETRY,
        rate_limit=_NO_COOLDOWN,
        guardrails=GuardrailConfig(**guardrails),
    )


def _si_group(text: str) -> str:
    return "si:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class _Cached(LLMCallStrategy[str]):
    """Each prompt is ``"<group>|<cached>/<input>"``; ``-`` as group means none."""

    def __init__(self, *, reason: str | None = None) -> None:
        self.calls: list[str] = []
        self.reason = reason

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, dict[str, int], dict[str, Any]]:
        self.calls.append(prompt)
        group, ratio = prompt.split("|")[:2]
        cached, total = (int(part) for part in ratio.split("/"))
        metadata: dict[str, Any] = {} if group == "-" else {"cache_group": group}
        if self.reason is not None:
            metadata["cache_miss_reason"] = self.reason
        usage = {
            "input_tokens": total,
            "output_tokens": 1,
            "total_tokens": total + 1,
            "cached_input_tokens": cached,
        }
        return prompt, usage, metadata


def _prompts(group: str, cached: int, total: int, count: int, start: int = 0) -> list[str]:
    # A unique suffix keeps item ids distinct; the strategy ignores it.
    return [f"{group}|{cached}/{total}|{i}" for i in range(start, start + count)]


async def _run(
    strategy: LLMCallStrategy[str], prompts: list[str], config: ProcessorConfig, **kwargs: Any
) -> tuple[BatchResult[str, None], dict[str, Any]]:
    async with ParallelBatchProcessor[str, str, None](config=config, **kwargs) as processor:
        for prompt in prompts:
            await processor.add_work(LLMWorkItem(item_id=prompt, strategy=strategy, prompt=prompt))
        result = await processor.process_all()
        stats = await processor.get_stats()
    return result, stats


def _group(stats: dict[str, Any], label: str | None) -> dict[str, Any]:
    matches = [entry for entry in stats["cache_groups"] if entry["group"] == label]
    assert len(matches) == 1, stats["cache_groups"]
    return matches[0]


def _tracker(floor: float | None = None, window: int = 3) -> CacheHitTracker:
    return CacheHitTracker(
        GuardrailConfig(min_cache_hit_rate=floor, cache_hit_window=window),
        trip_now=lambda cause: True,
        announce=AsyncMock(),
    )


def _usage(input_tokens: int, cached: int = 0, written: int = 0) -> dict[str, int]:
    usage = {"input_tokens": input_tokens, "output_tokens": 1, "cached_input_tokens": cached}
    if written:
        usage["cache_write_input_tokens"] = written
    return usage


def _record(
    tracker: CacheHitTracker,
    group: object = "g",
    usage: dict[str, Any] | None = None,
    *,
    known: bool = True,
    reason: object = None,
    enforce: bool = True,
) -> Any:
    metadata: dict[str, Any] = {"cache_group": group}
    if reason is not None:
        metadata["cache_miss_reason"] = reason
    return tracker.record(item_id="i", metadata=metadata, usage=usage, known=known, enforce=enforce)


# ── Built-in models stamp a group ────────────────────────────────────────────


def _chat_completion() -> Any:
    ChatCompletion = pytest.importorskip("openai.types.chat").ChatCompletion
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
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
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
    )


def _responses_response() -> Any:
    Response = pytest.importorskip("openai.types.responses").Response
    return Response.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 1,
                "total_tokens": 11,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


def _chat_client() -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_chat_completion())
    return client


def _responses_client() -> MagicMock:
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=_responses_response())
    return client


CHAT_MODELS = [
    lambda client, **kw: OpenAIModel(
        "gpt-test", client=client, api_surface="chat_completions", **kw
    ),
    lambda client, **kw: OpenRouterModel("openai/gpt-test", client=client, **kw),
    lambda client, **kw: DeepSeekModel("deepseek-test", client=client, **kw),
]
RESPONSES_MODELS = [
    lambda client, **kw: OpenAIModel("gpt-test", client=client, **kw),
    lambda client, **kw: DeepSeekModel(
        "deepseek-test", client=client, api_surface="responses", **kw
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", CHAT_MODELS)
async def test_chat_models_stamp_system_instruction_hash_or_key(factory: Any) -> None:
    model = factory(_chat_client(), system_instruction="Static rules")
    response = await model.generate("item")
    assert response.metadata["cache_group"] == _si_group("Static rules")

    keyed = factory(
        _chat_client(),
        system_instruction="Static rules",
        extra_body={"prompt_cache_key": "classify-v3"},
    )
    assert (await keyed.generate("item")).metadata["cache_group"] == "classify-v3"
    # Per-call config wins over the instance's extra_body, as it does on the wire.
    per_call = await keyed.generate("item", config={"prompt_cache_key": "other"})
    assert per_call.metadata["cache_group"] == "other"

    bare = factory(_chat_client())
    assert "cache_group" not in ((await bare.generate("item")).metadata or {})
    messages = [{"role": "system", "content": "From messages"}, {"role": "user", "content": "x"}]
    assert (await bare.generate(messages)).metadata["cache_group"] == _si_group("From messages")


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", RESPONSES_MODELS)
async def test_responses_models_stamp_system_instruction_hash_or_key(factory: Any) -> None:
    model = factory(_responses_client(), system_instruction="Static rules")
    assert (await model.generate("item")).metadata["cache_group"] == _si_group("Static rules")

    keyed = factory(_responses_client(), extra_body={"prompt_cache_key": "classify-v3"})
    assert (await keyed.generate("item")).metadata["cache_group"] == "classify-v3"

    bare = factory(_responses_client())
    assert "cache_group" not in ((await bare.generate("item")).metadata or {})

    # Instructions in per-call config go out in extra_body and replace the default.
    overridden = await model.generate("item", config={"instructions": "Other rules"})
    assert overridden.metadata["cache_group"] == _si_group("Other rules")
    only_config = await bare.generate("item", config={"instructions": "Other rules"})
    assert only_config.metadata["cache_group"] == _si_group("Other rules")
    cleared = await model.generate("item", config={"instructions": None})
    assert "cache_group" not in (cleared.metadata or {})
    # Null instructions clear the top-level field; a system message in the input stays.
    in_input = [{"role": "system", "content": "Rules in input"}, {"role": "user", "content": "x"}]
    kept = await model.generate(in_input, config={"instructions": None})
    assert kept.metadata["cache_group"] == _si_group("Rules in input")


def _gemini_response() -> MagicMock:
    response = MagicMock()
    response.text = "ok"
    response.usage_metadata.prompt_token_count = 10
    response.usage_metadata.candidates_token_count = 1
    response.usage_metadata.tool_use_prompt_token_count = None
    response.usage_metadata.thoughts_token_count = None
    response.usage_metadata.total_token_count = 11
    response.usage_metadata.cached_content_token_count = 0
    response.candidates = []
    return response


def _gemini_client() -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=_gemini_response())
    cache = MagicMock()
    cache.name = "cachedContents/rotating-name"
    cache.expire_time = None
    cache.create_time = None
    client.aio.caches.create = AsyncMock(return_value=cache)
    client.aio.caches.list = AsyncMock(return_value=[])
    return client


@pytest.mark.asyncio
async def test_gemini_models_stamp_a_group() -> None:
    model = GeminiModel("gemini-test", _gemini_client(), system_instruction="Static rules")
    assert (await model.generate("item")).metadata["cache_group"] == _si_group("Static rules")
    bare = GeminiModel("gemini-test", _gemini_client())
    assert "cache_group" not in ((await bare.generate("item")).metadata or {})

    # The group follows the instruction actually sent, after per-call config merges.
    overridden = await model.generate("item", config={"system_instruction": "Other rules"})
    assert overridden.metadata["cache_group"] == _si_group("Other rules")
    only_config = await bare.generate("item", config={"system_instruction": "Other rules"})
    assert only_config.metadata["cache_group"] == _si_group("Other rules")
    cleared = await model.generate("item", config={"system_instruction": None})
    assert "cache_group" not in (cleared.metadata or {})
    per_call = await model.generate("item", system_instruction="Per call")
    assert per_call.metadata["cache_group"] == _si_group("Per call")

    groups = []
    for _ in range(2):
        cached = GeminiCachedModel(
            model="gemini-test",
            client=_gemini_client(),
            cached_content=["shared context"],
            cache_ttl_seconds=3600,
        )
        await cached.prepare()
        groups.append((await cached.generate("item")).metadata["cache_group"])
    # Derived from the cached content, not the server-side name, so it's stable.
    assert groups[0] == groups[1]
    assert groups[0].startswith("gemini-cache:")
    assert "rotating-name" not in groups[0]


def _sent_body(client: MagicMock, surface: str) -> dict[str, Any]:
    create = client.chat.completions.create if surface == "chat" else client.responses.create
    return dict(create.call_args.kwargs.get("extra_body") or {})


def _auto_key(text: str) -> str:
    return "abl-si-" + hashlib.sha256(text.encode()).hexdigest()[:16]


SURFACES = [
    pytest.param("chat", CHAT_MODELS[0], id="openai-chat"),
    pytest.param("chat", CHAT_MODELS[1], id="openrouter"),
    pytest.param("responses", RESPONSES_MODELS[0], id="openai-responses"),
]


def _client(surface: str) -> MagicMock:
    return _chat_client() if surface == "chat" else _responses_client()


@pytest.mark.asyncio
@pytest.mark.parametrize(("surface", "factory"), SURFACES)
async def test_prompt_cache_key_option_is_sent_and_names_the_group(
    surface: str, factory: Any
) -> None:
    client = _client(surface)
    model = factory(client, system_instruction="Static rules", prompt_cache_key="classify-v3")
    response = await model.generate("item")
    assert _sent_body(client, surface)["prompt_cache_key"] == "classify-v3"
    assert response.metadata["cache_group"] == "classify-v3"

    # A per-call key wins, on the wire and in the stats.
    per_call = await model.generate("item", config={"prompt_cache_key": "other"})
    assert _sent_body(client, surface)["prompt_cache_key"] == "other"
    assert per_call.metadata["cache_group"] == "other"

    with pytest.raises(ValueError, match="not both"):
        factory(_client(surface), prompt_cache_key="a", extra_body={"prompt_cache_key": "b"})


@pytest.mark.asyncio
@pytest.mark.parametrize(("surface", "factory"), SURFACES)
async def test_auto_prompt_cache_key_follows_the_instruction_sent(
    surface: str, factory: Any
) -> None:
    client = _client(surface)
    model = factory(client, system_instruction="Static rules", prompt_cache_key="auto")
    response = await model.generate("item")
    assert _sent_body(client, surface)["prompt_cache_key"] == _auto_key("Static rules")
    assert response.metadata["cache_group"] == _auto_key("Static rules")

    override = {"instructions": "Other rules"} if surface == "responses" else None
    if override is not None:
        await model.generate("item", config=override)
        assert _sent_body(client, surface)["prompt_cache_key"] == _auto_key("Other rules")
    messages = [{"role": "system", "content": "From input"}, {"role": "user", "content": "x"}]
    bare_client = _client(surface)
    bare = factory(bare_client, prompt_cache_key="auto")
    await bare.generate(messages)
    assert _sent_body(bare_client, surface)["prompt_cache_key"] == _auto_key("From input")

    # No system instruction: no key is sent and no group stamped.
    unkeyed = await bare.generate("item")
    assert "prompt_cache_key" not in _sent_body(bare_client, surface)
    assert "cache_group" not in (unkeyed.metadata or {})


@pytest.mark.asyncio
@pytest.mark.parametrize(("surface", "factory"), SURFACES)
async def test_cache_group_label_precedence(surface: str, factory: Any) -> None:
    client = _client(surface)
    model = factory(
        client,
        system_instruction="Static rules",
        prompt_cache_key="classify-v3",
        cache_group="tier1",
    )
    # The model's label beats the key.
    assert (await model.generate("item")).metadata["cache_group"] == "tier1"
    # A per-call label beats the model's and never reaches the provider.
    per_call = await model.generate("item", config={"cache_group": "tier1-d"})
    assert per_call.metadata["cache_group"] == "tier1-d"
    assert "cache_group" not in _sent_body(client, surface)
    assert _sent_body(client, surface)["prompt_cache_key"] == "classify-v3"

    # An extractor's key still wins over every label.
    extracted = factory(
        _client(surface),
        cache_group="tier1",
        metadata_extractors=[lambda r: {"cache_group": "from-extractor"}],
    )
    assert (await extracted.generate("item")).metadata["cache_group"] == "from-extractor"


@pytest.mark.asyncio
async def test_deepseek_takes_a_label_but_no_prompt_cache_key() -> None:
    for factory, surface in ((CHAT_MODELS[2], "chat"), (RESPONSES_MODELS[1], "responses")):
        client = _client(surface)
        model = factory(client, cache_group="ds")
        assert (await model.generate("item")).metadata["cache_group"] == "ds"
        assert "cache_group" not in _sent_body(client, surface)
        with pytest.raises(TypeError):
            factory(_client(surface), prompt_cache_key="k")
    keyed = DeepSeekModel.from_api_key("deepseek-test", api_key="sk-test", cache_group="ds")
    assert keyed._cache_group_label == "ds"


def test_from_api_key_forwards_cache_options() -> None:
    openai_model = OpenAIModel.from_api_key(
        "gpt-test", api_key="sk-test", prompt_cache_key="auto", cache_group="g"
    )
    assert (openai_model._prompt_cache_key, openai_model._cache_group_label) == ("auto", "g")
    router = OpenRouterModel.from_api_key(
        "openai/gpt-test", api_key="sk-test", prompt_cache_key="k", cache_group="g"
    )
    assert (router._prompt_cache_key, router._cache_group_label) == ("k", "g")


def test_inherited_factory_works_for_subclasses_with_the_old_signature() -> None:
    class _Legacy(OpenAICompatibleModel):
        _default_base_url = "https://example.invalid/v1"

        def __init__(
            self,
            model: str,
            client: Any,
            *,
            system_instruction: str | None = None,
            extra_headers: dict[str, str] | None = None,
            extra_body: dict[str, Any] | None = None,
            metadata_extractors: Any = None,
        ) -> None:
            super().__init__(
                model,
                client,
                system_instruction=system_instruction,
                extra_headers=extra_headers,
                extra_body=extra_body,
                metadata_extractors=metadata_extractors,
            )

    legacy = _Legacy.from_api_key("m", api_key="sk-test")
    assert legacy._cache_group_label is None and legacy._prompt_cache_key is None
    # Asking for an option the subclass doesn't accept still fails loudly.
    with pytest.raises(TypeError, match="cache_group"):
        _Legacy.from_api_key("m", api_key="sk-test", cache_group="g")


@pytest.mark.parametrize("value", ["", 3, b"x"])
def test_invalid_cache_labels_and_keys_raise(value: Any) -> None:
    with pytest.raises(ValueError, match="cache_group"):
        OpenAIModel("gpt-test", client=_responses_client(), cache_group=value)
    with pytest.raises(ValueError, match="prompt_cache_key"):
        OpenAIModel("gpt-test", client=_responses_client(), prompt_cache_key=value)
    with pytest.raises(ValueError, match="cache_group"):
        GeminiModel("gemini-test", _gemini_client(), cache_group=value)


@pytest.mark.asyncio
async def test_invalid_per_call_label_raises() -> None:
    model = OpenAIModel("gpt-test", client=_responses_client())
    with pytest.raises(ValueError, match="cache_group"):
        await model.generate("item", config={"cache_group": ""})


@pytest.mark.asyncio
async def test_gemini_models_take_a_label() -> None:
    client = _gemini_client()
    model = GeminiModel("gemini-test", client, system_instruction="Rules", cache_group="g1")
    assert (await model.generate("item")).metadata["cache_group"] == "g1"
    per_call = await model.generate("item", config={"cache_group": "g2"})
    assert per_call.metadata["cache_group"] == "g2"
    sent = client.aio.models.generate_content.call_args.kwargs["config"]
    sent_dict = sent if isinstance(sent, dict) else sent.model_dump(exclude_none=True)
    assert "cache_group" not in sent_dict

    cached = GeminiCachedModel(
        model="gemini-test",
        client=_gemini_client(),
        cached_content=["shared context"],
        cache_ttl_seconds=3600,
        cache_group="docs",
    )
    await cached.prepare()
    assert (await cached.generate("item")).metadata["cache_group"] == "docs"
    per_call = await cached.generate("item", config={"cache_group": "docs-2"})
    assert per_call.metadata["cache_group"] == "docs-2"


def _with_diagnostics(diagnostics: dict[str, Any]) -> Any:
    response = _responses_response()
    try:
        return type(response).model_validate(
            {**response.model_dump(), "prompt_cache_diagnostics": diagnostics}
        )
    except Exception:  # SDK without the field: attach it as a plain mapping
        response.__dict__["prompt_cache_diagnostics"] = diagnostics
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("diagnostics", "reason"),
    [
        (
            {"type": "cache_miss", "reason": "input_changed", "cache_missed_tokens": 8},
            "input_changed",
        ),
        ({"type": "cache_hit"}, None),
        ({"type": "unavailable"}, None),
    ],
)
async def test_responses_miss_reason_comes_from_prompt_cache_diagnostics(
    diagnostics: dict[str, Any], reason: str | None
) -> None:
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=_with_diagnostics(diagnostics))
    response = await OpenAIModel("gpt-test", client=client, prompt_cache_key="k").generate("x")
    assert (response.metadata or {}).get("cache_miss_reason") == reason


def test_miss_reason_reads_mappings_too() -> None:
    from async_batch_llm.models import _diagnostics_miss_reason

    class _Resp:
        prompt_cache_diagnostics = {"type": "cache_miss", "reason": "prompt_cache_key_changed"}

    assert _diagnostics_miss_reason(_Resp()) == "prompt_cache_key_changed"
    _Resp.prompt_cache_diagnostics = {"type": "cache_miss", "reason": ""}
    assert _diagnostics_miss_reason(_Resp()) is None
    assert _diagnostics_miss_reason(object()) is None


# ── Group identity ───────────────────────────────────────────────────────────


def test_long_keys_are_hashed_not_truncated() -> None:
    a = "x" * 200 + "a"
    b = "x" * 200 + "b"
    assert normalize_cache_group(a) != normalize_cache_group(b)
    assert normalize_cache_group(a) == "sha256:" + hashlib.sha256(a.encode()).hexdigest()
    assert normalize_cache_group("x" * 128) == "x" * 128


@pytest.mark.parametrize("value", [None, "", 3, ["g"], b"g"])
def test_non_string_or_empty_groups_mean_ungrouped(value: object) -> None:
    assert normalize_cache_group(value) is None


def test_caller_labels_never_collide_with_internal_identities() -> None:
    tracker = _tracker()
    _record(tracker, None, _usage(10))
    _record(tracker, "(ungrouped)", _usage(10, 10))
    _record(tracker, "(other)", _usage(10, 10))
    stats = tracker.stats()
    labels = [entry["group"] for entry in stats["cache_groups"]]
    assert sorted(labels, key=str) == sorted([None, "(ungrouped)", "(other)"], key=str)
    assert _group(stats, None)["cached_input_tokens"] == 0
    assert stats["cache_groups_overflow"] is None


# ── What counts as measured ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("usage", "known"),
    [
        (None, False),
        ({}, True),
        ({"output_tokens": 5, "total_tokens": 5}, True),
        ({"input_tokens": 0, "output_tokens": 1, "total_tokens": 1}, True),
        (_usage(10, 10), False),
        ({"input_tokens": True, "cached_input_tokens": 0}, True),
    ],
)
def test_unknown_and_zero_input_are_unmeasured(usage: Any, known: bool) -> None:
    tracker = _tracker(floor=0.9, window=1)
    assert _record(tracker, "g", usage, known=known) is None
    entry = _group(tracker.stats(), "g")
    assert entry["measured_calls"] == 0
    assert entry["unmeasured_calls"] == 1
    assert entry["input_tokens"] == 0
    assert entry["window_hit_rate"] is None


@pytest.mark.asyncio
async def test_empty_usage_results_never_trip_the_floor() -> None:
    class _NoUsage(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            return prompt, {}, {"cache_group": "g"}

    prompts = [f"p{i}" for i in range(60)]
    result, stats = await _run(
        _NoUsage(), prompts, _config(min_cache_hit_rate=0.9, cache_hit_window=5)
    )
    assert result.termination.kind == "completed"
    assert _group(stats, "g")["unmeasured_calls"] == 60


def test_counters_and_rates() -> None:
    tracker = _tracker()
    _record(tracker, "g", _usage(100, 0, written=80))
    _record(tracker, "g", _usage(100, 90))
    entry = _group(tracker.stats(), "g")
    assert entry["measured_calls"] == 2
    assert entry["input_tokens"] == 200
    assert entry["cached_input_tokens"] == 90
    assert entry["cache_write_input_tokens"] == 80
    assert entry["hit_rate"] == pytest.approx(0.45)
    assert "window_hit_rate" not in entry  # no floor configured


# ── Miss reasons ─────────────────────────────────────────────────────────────


def test_latest_valid_miss_reason_is_kept_and_invalid_values_ignored() -> None:
    tracker = _tracker()
    _record(tracker, "g", _usage(10), reason="input_changed")
    _record(tracker, "g", _usage(10), reason="")
    _record(tracker, "g", _usage(10), reason=42)
    assert _group(tracker.stats(), "g")["last_miss_reason"] == "input_changed"
    _record(tracker, "g", _usage(10), reason="model_changed")
    assert _group(tracker.stats(), "g")["last_miss_reason"] == "model_changed"
    _record(tracker, "g", _usage(10), reason="r" * 100)
    assert _group(tracker.stats(), "g")["last_miss_reason"] == "r" * 64


def test_an_older_reason_survives_a_later_hit() -> None:
    # Historical diagnostic, not a claim about the whole window.
    tracker = _tracker()
    _record(tracker, "g", _usage(10), reason="input_changed")
    _record(tracker, "g", _usage(10, 10))
    assert _group(tracker.stats(), "g")["last_miss_reason"] == "input_changed"


def test_stop_reason_names_the_group_rate_and_last_miss_reason() -> None:
    tracker = _tracker(floor=0.8, window=2)
    assert _record(tracker, "g", _usage(10), reason="input_changed") is None
    cause = _record(tracker, "g", _usage(10))
    assert cause is not None
    assert cause.kind == "fail_fast"
    assert cause.error_category == "cache_hit_rate_below_floor"
    assert "'g'" in cause.reason
    assert "0.0%" in cause.reason and "80.0%" in cause.reason
    assert cause.reason.endswith("last miss reason: input_changed")


# ── The floor ────────────────────────────────────────────────────────────────


def test_no_trip_until_the_window_is_full() -> None:
    tracker = _tracker(floor=0.5, window=3)
    assert _record(tracker, "g", _usage(10)) is None
    assert _record(tracker, "g", _usage(10)) is None
    assert _record(tracker, "g", _usage(10)) is not None


def test_floor_is_inclusive() -> None:
    tracker = _tracker(floor=0.5, window=2)
    assert _record(tracker, "g", _usage(10, 5)) is None
    assert _record(tracker, "g", _usage(10, 5)) is None


def test_sliding_window_catches_a_late_drop_under_one_key() -> None:
    tracker = _tracker(floor=0.6, window=3)
    for _ in range(100):
        assert _record(tracker, "g", _usage(10, 10)) is None
    assert _record(tracker, "g", _usage(10)) is None  # window 2/3 = 66.7%
    cause = _record(tracker, "g", _usage(10))  # window 1/3 = 33.3%
    assert cause is not None
    # The cumulative rate is still far above the floor; only the window sees the drop.
    assert _group(tracker.stats(), "g")["hit_rate"] > 0.95


def test_other_groups_do_not_fill_a_groups_warm_up() -> None:
    tracker = _tracker(floor=0.5, window=3)
    for name in "abcdef":
        assert _record(tracker, name, _usage(10)) is None
    assert all(e["window_hit_rate"] is None for e in tracker.stats()["cache_groups"])


def test_ungrouped_attempts_are_checked_too() -> None:
    tracker = _tracker(floor=0.5, window=2)
    _record(tracker, None, _usage(10))
    cause = _record(tracker, None, _usage(10))
    assert cause is not None and "(ungrouped)" in cause.reason


def test_unenforced_attempt_is_still_counted() -> None:
    tracker = _tracker(floor=0.5, window=1)
    assert _record(tracker, "g", _usage(10), enforce=False) is None
    assert _group(tracker.stats(), "g")["measured_calls"] == 1


def test_trips_once() -> None:
    tracker = _tracker(floor=0.5, window=1)
    assert _record(tracker, "g", _usage(10)) is not None
    assert _record(tracker, "g", _usage(10)) is None


# ── Overflow ─────────────────────────────────────────────────────────────────


def test_overflow_aggregates_without_a_floor(caplog: pytest.LogCaptureFixture) -> None:
    tracker = _tracker()
    for i in range(MAX_CACHE_GROUPS):
        _record(tracker, f"g{i}", _usage(10))
    with caplog.at_level("WARNING"):
        assert _record(tracker, "late-1", _usage(10, 10)) is None
        assert _record(tracker, "late-2", _usage(10)) is None
        assert _record(tracker, "late-3", {}) is None
    stats = tracker.stats()
    assert len(stats["cache_groups"]) == MAX_CACHE_GROUPS
    overflow = stats["cache_groups_overflow"]
    assert overflow["measured_calls"] == 2
    assert overflow["unmeasured_calls"] == 1
    assert overflow["cached_input_tokens"] == 10
    assert sum("cache_groups_overflow" in r.message for r in caplog.records) == 1
    # A tracked group keeps its own counters after overflow starts.
    _record(tracker, "g0", _usage(10, 10))
    assert _group(tracker.stats(), "g0")["measured_calls"] == 2


def test_overflow_with_a_floor_fails_closed() -> None:
    # Alternating always-hit and always-miss groups would average 50% in an
    # aggregate bucket and pass a 40% floor; the tracker refuses instead.
    tracker = _tracker(floor=0.4, window=50)
    for i in range(MAX_CACHE_GROUPS):
        _record(tracker, f"hit{i}", _usage(10, 10))
    cause = _record(tracker, "miss", _usage(10))
    assert cause is not None
    assert cause.error_category == "cache_hit_rate_below_floor"
    assert f"More than {MAX_CACHE_GROUPS} cache groups" in cause.reason


def test_the_ungrouped_bucket_counts_toward_the_cap() -> None:
    tracker = _tracker(floor=0.4, window=50)
    _record(tracker, None, _usage(10, 10))
    for i in range(MAX_CACHE_GROUPS - 1):
        _record(tracker, f"g{i}", _usage(10, 10))
    assert _record(tracker, "one-more", _usage(10, 10)) is not None


# ── Processor runs ───────────────────────────────────────────────────────────


class _Recorder(BaseObserver):
    def __init__(self) -> None:
        self.events: list[tuple[ProcessingEvent, dict[str, Any]]] = []

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        self.events.append((event, dict(data)))


@pytest.mark.asyncio
async def test_stats_and_batch_completed_event_report_groups() -> None:
    recorder = _Recorder()
    prompts = _prompts("a", 90, 100, 3) + _prompts("b", 0, 50, 2) + _prompts("-", 0, 10, 1)
    result, stats = await _run(_Cached(), prompts, _config(), observers=[recorder])
    assert result.termination.kind == "completed"
    assert [entry["group"] for entry in stats["cache_groups"]] == ["a", "b", None]
    assert _group(stats, "a")["hit_rate"] == pytest.approx(0.9)
    assert _group(stats, "b")["measured_calls"] == 2
    assert stats["cache_groups_overflow"] is None
    completed = [
        data for event, data in recorder.events if event == ProcessingEvent.BATCH_COMPLETED
    ]
    assert completed[0]["cache_groups"] == stats["cache_groups"]
    assert completed[0]["cache_groups_overflow"] is None


@pytest.mark.asyncio
async def test_floor_stops_the_run_as_fail_fast() -> None:
    strategy = _Cached(reason="prompt_cache_key_changed")
    prompts = _prompts("g", 0, 100, 10)
    result, stats = await _run(
        strategy, prompts, _config(min_cache_hit_rate=0.5, cache_hit_window=3)
    )
    assert result.termination.kind == "fail_fast"
    assert result.termination.error_category == ErrorCategory.CACHE_HIT_RATE_BELOW_FLOOR
    assert result.termination.triggering_item_id == prompts[2]
    assert "last miss reason: prompt_cache_key_changed" in (result.termination.reason or "")
    assert len(strategy.calls) == 3
    unstarted = [r for r in result.results if not r.success]
    assert len(unstarted) == 7
    assert {r.error_category for r in unstarted} == {"batch_aborted"}
    # The crossing item itself succeeded and is kept.
    assert sum(r.success for r in result.results) == 3
    assert _group(stats, "g")["window_hit_rate"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [AbortMode.DRAIN_ACTIVE, AbortMode.CANCEL_ACTIVE])
async def test_abort_mode_applies_to_calls_in_flight(mode: AbortMode) -> None:
    class _Slow(_Cached):
        async def execute(self, prompt, attempt, timeout, state=None):
            if prompt.startswith("slow"):
                await asyncio.sleep(0.2)
                return prompt, {"input_tokens": 10, "output_tokens": 1}, {"cache_group": "s"}
            return await super().execute(prompt, attempt, timeout, state)

    prompts = ["slow|0/0|0", *_prompts("g", 0, 10, 2)]
    result, _ = await _run(
        _Slow(),
        prompts,
        _config(workers=2, min_cache_hit_rate=0.5, cache_hit_window=2, abort_mode=mode),
    )
    assert result.termination.kind == "fail_fast"
    slow = result.by_id()["slow|0/0|0"]
    assert slow.success is (mode == AbortMode.DRAIN_ACTIVE)


@pytest.mark.asyncio
async def test_failed_attempts_are_not_part_of_the_live_rate() -> None:
    class _FailThenHit(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            if attempt == 1:
                error = ConnectionError("billed then failed")
                error._failed_token_usage = {  # type: ignore[attr-defined]
                    "input_tokens": 100,
                    "output_tokens": 0,
                    "total_tokens": 100,
                }
                raise error
            usage = {
                "input_tokens": 100,
                "output_tokens": 1,
                "total_tokens": 101,
                "cached_input_tokens": 100,
            }
            return prompt, usage, {"cache_group": "g"}

    result, stats = await _run(_FailThenHit(), ["a"], _config())
    assert result.results[0].token_usage["input_tokens"] == 200
    entry = _group(stats, "g")
    assert entry["measured_calls"] == 1
    assert entry["hit_rate"] == 1.0


@pytest.mark.asyncio
async def test_only_the_attempt_that_returned_is_attributed() -> None:
    # The processor never retries after a strategy returns, so a retried item's
    # earlier attempts all raised. They carry no metadata and no group is charged.
    class _FailThenSwitch(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            if attempt == 1:
                error = ConnectionError("billed then failed")
                error._failed_token_usage = {  # type: ignore[attr-defined]
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "total_tokens": 10,
                }
                raise error
            usage = {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}
            return prompt, usage, {"cache_group": f"attempt-{attempt}"}

    result, stats = await _run(_FailThenSwitch(), ["a"], _config())
    assert result.results[0].metadata["cache_group"] == "attempt-2"
    assert [entry["group"] for entry in stats["cache_groups"]] == ["attempt-2"]
    assert _group(stats, "attempt-2")["measured_calls"] == 1


@pytest.mark.asyncio
async def test_dry_run_records_nothing() -> None:
    config = _config(min_cache_hit_rate=0.9, cache_hit_window=1)
    config.dry_run = True
    result, stats = await _run(_Cached(), _prompts("g", 0, 10, 3), config)
    assert result.termination.kind == "completed"
    assert stats["cache_groups"] == []


@pytest.mark.asyncio
async def test_budget_wins_on_the_same_attempt_but_the_attempt_still_counts() -> None:
    result, stats = await _run(
        _Cached(),
        _prompts("g", 0, 100, 3),
        _config(max_total_tokens=100, min_cache_hit_rate=0.5, cache_hit_window=1),
    )
    assert result.termination.kind == "budget_exceeded"
    assert _group(stats, "g")["measured_calls"] == 1


@pytest.mark.asyncio
async def test_config_reuse_gets_a_fresh_tracker() -> None:
    config = _config(min_cache_hit_rate=0.5, cache_hit_window=2)
    first, _ = await _run(_Cached(), _prompts("g", 0, 10, 1), config)
    second, stats = await _run(_Cached(), _prompts("g", 0, 10, 1), config)
    assert first.termination.kind == second.termination.kind == "completed"
    assert _group(stats, "g")["measured_calls"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("store_type", [JsonlArtifactStore, SqliteArtifactStore])
async def test_tripped_run_resumes_and_replays_are_not_counted(
    tmp_path: Any, store_type: Any
) -> None:
    path = tmp_path / "cache.artifact"
    prompts = _prompts("g", 0, 10, 5)
    first = await process_prompts(
        _Cached(),
        prompts,
        config=_config(min_cache_hit_rate=0.5, cache_hit_window=2),
        artifact_store=store_type(path),
    )
    assert first.termination.kind == "fail_fast"
    assert first.succeeded == 2

    second_strategy = _Cached()
    recorder = _Recorder()
    second = await process_prompts(
        second_strategy,
        prompts,
        config=_config(),
        artifact_store=store_type(path),
        resume=ResumePolicy.REUSE_SUCCESSES,
        observers=[recorder],
    )
    completed = [d for e, d in recorder.events if e == ProcessingEvent.BATCH_COMPLETED]
    stats = completed[0]
    assert second.termination.kind == "completed"
    assert second.succeeded == 5
    assert len(second_strategy.calls) == 3
    assert sum(r.replayed_from_artifact for r in second.results) == 2
    assert _group(stats, "g")["measured_calls"] == 3


def test_floor_termination_round_trips() -> None:
    result: BatchResult[str, None] = BatchResult(
        results=[],
        termination=BatchTermination(
            kind="fail_fast",
            reason="Cache hit rate for group 'g' fell to 0.0% over the last 3 measured calls",
            error_category="cache_hit_rate_below_floor",
            triggering_item_id="a",
        ),
    )
    assert batch_result_from_dict(batch_result_to_dict(result)).termination == result.termination


# ── Configuration ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_cache_hit_rate": float("nan")}, "finite"),
        ({"min_cache_hit_rate": float("inf")}, "finite"),
        ({"min_cache_hit_rate": True}, "finite"),
        ({"min_cache_hit_rate": "0.8"}, "finite"),
        ({"min_cache_hit_rate": -0.1}, "between 0 and 1"),
        ({"min_cache_hit_rate": 1.5}, "between 0 and 1"),
        ({"cache_hit_window": 0}, "positive int"),
        ({"cache_hit_window": -3}, "positive int"),
        ({"cache_hit_window": True}, "positive int"),
        ({"cache_hit_window": 2.0}, "positive int"),
    ],
)
def test_floor_config_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GuardrailConfig(**kwargs)


@pytest.mark.parametrize("rate", [0, 0.0, 1, 0.75])
def test_valid_floors(rate: float) -> None:
    assert GuardrailConfig(min_cache_hit_rate=rate).min_cache_hit_rate == rate


@pytest.mark.asyncio
async def test_single_call_and_pool_reject_a_floor() -> None:
    config = _config(min_cache_hit_rate=0.5)
    with pytest.raises(ValueError, match="min_cache_hit_rate"):
        await call(_Cached(), "g|0/10|0", config=config)
    with pytest.raises(ValueError, match="min_cache_hit_rate"):
        await call_result(_Cached(), "g|0/10|0", config=config)
    with pytest.raises(ValueError, match="min_cache_hit_rate"):
        LLMCallPool(_Cached(), config=config)


# ── LLMCallPool statistics ───────────────────────────────────────────────────


def test_window_without_a_floor_is_reported_but_never_trips() -> None:
    tracker = CacheHitTracker(GuardrailConfig(cache_hit_window=2), windowed=True)
    assert _record(tracker, "g", _usage(10)) is None
    assert _group(tracker.stats(), "g")["window_hit_rate"] is None
    assert _record(tracker, "g", _usage(10, 10)) is None
    assert _record(tracker, "g", _usage(10)) is None
    assert _group(tracker.stats(), "g")["window_hit_rate"] == pytest.approx(0.5)


def test_a_floor_needs_its_stop_callbacks() -> None:
    with pytest.raises(ValueError, match="trip_now and announce"):
        CacheHitTracker(GuardrailConfig(min_cache_hit_rate=0.5))


def test_reset_forgets_groups_overflow_and_trip() -> None:
    tracker = _tracker(floor=0.5, window=1)
    for i in range(MAX_CACHE_GROUPS + 1):
        _record(tracker, f"g{i}", _usage(10, 10))
    assert tracker.stats()["cache_groups_overflow"] is not None
    tracker.reset()
    assert tracker.stats() == {"cache_groups": [], "cache_groups_overflow": None}
    # Tripping is armed again after a reset.
    assert _record(tracker, "fresh", _usage(10)) is not None


@pytest.mark.asyncio
async def test_pool_reports_groups_with_a_rolling_window() -> None:
    config = ProcessorConfig(
        max_workers=4,
        retry=_FAST_RETRY,
        rate_limit=_NO_COOLDOWN,
        guardrails=GuardrailConfig(cache_hit_window=3),
    )
    async with LLMCallPool(_Cached(), config=config) as pool:
        assert pool.cache_stats() == {"cache_groups": [], "cache_groups_overflow": None}
        prompts = _prompts("a", 90, 100, 4) + _prompts("b", 0, 50, 2) + _prompts("-", 0, 10, 1)
        results = await asyncio.gather(*(pool.submit_result(p) for p in prompts))
        assert all(r.success for r in results)
        stats = pool.cache_stats()
        assert [entry["group"] for entry in stats["cache_groups"]] == ["a", "b", None]
        assert _group(stats, "a")["measured_calls"] == 4
        assert _group(stats, "a")["window_hit_rate"] == pytest.approx(0.9)
        # A 0% group is only reported: the pool has no floor.
        assert _group(stats, "b")["window_hit_rate"] is None
        await pool.submit_result("b|0/50|9")
        assert _group(pool.cache_stats(), "b")["window_hit_rate"] == 0.0

        pool.reset_cache_stats()
        assert pool.cache_stats()["cache_groups"] == []
        await pool.submit_result("a|100/100|9")
        assert _group(pool.cache_stats(), "a")["measured_calls"] == 1


@pytest.mark.asyncio
async def test_pool_counts_only_attempts_that_returned() -> None:
    class _Flaky(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            if prompt == "fail":
                raise ValueError("bad request")
            usage = {"input_tokens": 10, "output_tokens": 1, "cached_input_tokens": 10}
            return prompt, usage, {"cache_group": "g"}

    async with LLMCallPool(_Flaky(), config=_config()) as pool:
        assert not (await pool.submit_result("fail")).success
        assert (await pool.submit_result("ok")).success
        entry = _group(pool.cache_stats(), "g")
        assert entry["measured_calls"] == 1
        assert entry["unmeasured_calls"] == 0
