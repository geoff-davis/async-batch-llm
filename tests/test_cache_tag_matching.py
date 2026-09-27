"""Tests for cache tag matching (v0.3 feature, post display_name encoding)."""

import importlib.util
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _genai_or_skip() -> None:
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")


class _AsyncIterList:
    """Minimal async iterator wrapping a list — matches genai's `caches.list()` API."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __aiter__(self) -> "_AsyncIterList":
        self._iter = iter(self._items)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration as exc:  # noqa: PERF203
            raise StopAsyncIteration from exc


def _make_mock_cache(name: str, *, display_name: str | None = None) -> Any:
    cache = MagicMock()
    cache.name = name
    cache.model = "projects/test/models/gemini-test"
    cache.display_name = display_name
    cache.expire_time = None
    cache.create_time = MagicMock()
    cache.create_time.timestamp.return_value = time.time()
    return cache


def _make_mock_client(caches: list[Any]) -> Any:
    client = MagicMock()
    client.aio.caches.list = AsyncMock(return_value=_AsyncIterList(caches))
    client.aio.caches.create = AsyncMock(return_value=_make_mock_cache("new-cache"))
    client.aio.caches.delete = AsyncMock()
    return client


# ─── Encode/decode helpers ────────────────────────────────────────────


def test_encode_decode_round_trip():
    _genai_or_skip()
    from async_batch_llm.models import (
        _decode_tags_from_display_name,
        _encode_tags_to_display_name,
    )

    tags = {"customer": "acme", "version": "v1", "environment": "prod"}
    assert _decode_tags_from_display_name(_encode_tags_to_display_name(tags)) == tags


def test_encode_is_deterministic_regardless_of_key_order():
    """Equal tag sets must produce equal display_names — otherwise cache lookup
    would miss tagged caches created in a different insertion order."""
    _genai_or_skip()
    from async_batch_llm.models import _encode_tags_to_display_name

    a = _encode_tags_to_display_name({"a": "1", "b": "2"})
    b = _encode_tags_to_display_name({"b": "2", "a": "1"})
    assert a == b


def test_decode_returns_none_for_unprefixed_display_name():
    _genai_or_skip()
    from async_batch_llm.models import _decode_tags_from_display_name

    assert _decode_tags_from_display_name(None) is None
    assert _decode_tags_from_display_name("") is None
    assert _decode_tags_from_display_name("user-picked-name") is None


def test_decode_returns_none_for_malformed_prefix_payload():
    _genai_or_skip()
    from async_batch_llm.models import _decode_tags_from_display_name

    assert _decode_tags_from_display_name("abl-tags:not-json") is None
    assert _decode_tags_from_display_name('abl-tags:["list","not","dict"]') is None


def test_encoded_display_name_passes_genai_validation():
    """Regression for the bug report: `metadata` was rejected by
    CreateCachedContentConfig; `display_name` must be accepted."""
    _genai_or_skip()
    from google.genai.types import CreateCachedContentConfig

    from async_batch_llm.models import _encode_tags_to_display_name

    cfg = CreateCachedContentConfig(
        contents=[],
        ttl="60s",
        display_name=_encode_tags_to_display_name({"prompt_version": "v1"}),
    )
    assert cfg.display_name == 'abl-tags:{"prompt_version":"v1"}'


# ─── _find_or_create_cache tag matching via the real implementation ───


@pytest.mark.asyncio
async def test_find_existing_cache_via_display_name_tags():
    _genai_or_skip()
    from async_batch_llm.models import GeminiCachedModel, _encode_tags_to_display_name

    want = {"customer": "acme", "version": "v1"}
    other = {"customer": "globex", "version": "v1"}

    matching = _make_mock_cache("matching", display_name=_encode_tags_to_display_name(want))
    mismatching = _make_mock_cache("mismatching", display_name=_encode_tags_to_display_name(other))

    model = GeminiCachedModel(
        model="gemini-test",
        client=_make_mock_client([mismatching, matching]),
        cached_content=[],
        cache_tags=want,
    )

    matching.display_name = _encode_tags_to_display_name(model._cache_tags)
    await model.prepare()

    assert model._cache is matching


@pytest.mark.asyncio
async def test_cache_without_display_name_tags_is_skipped_when_model_has_tags():
    """A legacy cache (no abl-tags display_name) must NOT be reused by a tagged model —
    otherwise we'd silently reuse a cache whose contents may not match the intended tags."""
    _genai_or_skip()
    from async_batch_llm.models import GeminiCachedModel

    legacy = _make_mock_cache("legacy", display_name=None)
    client = _make_mock_client([legacy])

    model = GeminiCachedModel(
        model="gemini-test",
        client=client,
        cached_content=[],
        cache_tags={"version": "v1"},
    )

    await model.prepare()

    # Legacy cache skipped → create must be called.
    client.aio.caches.create.assert_called_once()
    assert model._cache is not legacy


@pytest.mark.asyncio
async def test_model_without_user_tags_rejects_cache_without_content_tag():
    """Content identity is required even when no user tags are supplied."""
    _genai_or_skip()
    from async_batch_llm.models import GeminiCachedModel, _encode_tags_to_display_name

    tagged = _make_mock_cache(
        "tagged", display_name=_encode_tags_to_display_name({"version": "v1"})
    )

    model = GeminiCachedModel(
        model="gemini-test",
        client=_make_mock_client([tagged]),
        cached_content=[],
        # no cache_tags
    )

    await model.prepare()
    assert model._cache is not tagged


@pytest.mark.asyncio
async def test_tag_match_is_case_and_type_sensitive():
    _genai_or_skip()
    from async_batch_llm.models import GeminiCachedModel, _encode_tags_to_display_name

    # Cache encodes uppercase + numeric-string.
    cache = _make_mock_cache(
        "c", display_name=_encode_tags_to_display_name({"customer": "ACME", "version": "1"})
    )
    client = _make_mock_client([cache])

    # Model wants lowercase + string "1".
    model = GeminiCachedModel(
        model="gemini-test",
        client=client,
        cached_content=[],
        cache_tags={"customer": "acme", "version": "1"},
    )

    await model.prepare()

    # Case mismatch → cache skipped, new one created.
    client.aio.caches.create.assert_called_once()


# ─── Write path: _create_new_cache passes display_name, not metadata ──


@pytest.mark.asyncio
async def test_create_cache_forwards_tags_as_display_name_only():
    """Regression for the bug report: must NOT pass `metadata=` to
    CreateCachedContentConfig (google-genai rejects it), but SHOULD pass
    `display_name=` encoded with our sentinel prefix."""
    _genai_or_skip()
    from async_batch_llm.models import GeminiCachedModel, _encode_tags_to_display_name

    tags = {"prompt_version": "v1"}

    # No existing caches → will create.
    client = _make_mock_client([])

    model = GeminiCachedModel(
        model="gemini-test",
        client=client,
        cached_content=[],
        cache_tags=tags,
    )

    await model.prepare()

    client.aio.caches.create.assert_called_once()
    call_kwargs = client.aio.caches.create.call_args.kwargs
    config = call_kwargs["config"]

    # The fix: tags appear in display_name, never in a metadata kwarg.
    assert config.display_name == _encode_tags_to_display_name(model._cache_tags)
    assert not hasattr(config, "metadata") or getattr(config, "metadata", None) is None


@pytest.mark.asyncio
async def test_create_cache_without_user_tags_includes_content_tag():
    _genai_or_skip()
    from async_batch_llm.models import GeminiCachedModel

    client = _make_mock_client([])

    model = GeminiCachedModel(
        model="gemini-test",
        client=client,
        cached_content=[],
        # no cache_tags
    )

    await model.prepare()

    config = client.aio.caches.create.call_args.kwargs["config"]
    assert "abl-content" in config.display_name


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["same", "different", "legacy", "different_model"])
async def test_prov2_cache_requires_exact_content_identity(case):
    from datetime import datetime, timedelta, timezone

    from google.genai.types import CachedContent, Content, Part

    from async_batch_llm.models import GeminiCachedModel

    stored = []
    client = _make_mock_client([])

    async def create(*, model, config):
        cache = CachedContent(
            name=f"cachedContents/{len(stored)}",
            model=f"models/{model}",
            display_name=config.display_name,
            create_time=datetime.now(timezone.utc),
            expire_time=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        stored.append(cache)
        return cache

    async def listing():
        return _AsyncIterList(stored)

    client.aio.caches.create = AsyncMock(side_effect=create)
    client.aio.caches.list = AsyncMock(side_effect=listing)
    content = [Content(role="user", parts=[Part(text="private context")])]
    first = GeminiCachedModel("gemini-test", client, content)
    await first.prepare()
    if case == "legacy":
        stored[0].display_name = "old cache"
    second = GeminiCachedModel(
        "different-gemini-test" if case == "different_model" else "gemini-test",
        client,
        [Content(role="user", parts=[Part(text="other")])] if case == "different" else content,
    )
    await second.prepare()
    assert client.aio.caches.create.await_count == (1 if case == "same" else 2)
    assert (first.cache_name == second.cache_name) is (case == "same")


@pytest.mark.parametrize("tags", [{"abl-content": "forged"}, {"long": "x" * 128}])
def test_prov2_cache_tag_budget_and_reserved_key(tags):
    from async_batch_llm.models import GeminiCachedModel

    with pytest.raises(ValueError, match="abl-content|128-character"):
        GeminiCachedModel("gemini-test", _make_mock_client([]), [], cache_tags=tags)


@pytest.mark.parametrize("kind", ["nested", "bytes", "image", "file", "equivalent"])
def test_sc2_sc3_cache_fingerprints_follow_request_content(kind):
    from google.genai import types

    from async_batch_llm.models import GeminiCachedModel

    if kind == "nested":
        left = [["doc", types.Part(text="more")]]
        right = [
            types.Content(role="user", parts=[types.Part(text="doc"), types.Part(text="more")])
        ]
    elif kind == "bytes":
        left = [
            {
                "role": "user",
                "parts": [{"inline_data": {"data": b"%PDF", "mime_type": "application/pdf"}}],
            }
        ]
        right = [
            types.Content(
                role="user",
                parts=[
                    types.Part(inline_data=types.Blob(data=b"%PDF", mime_type="application/pdf"))
                ],
            )
        ]
    elif kind == "image":
        Image = pytest.importorskip("PIL.Image")
        left = [Image.new("RGB", (2, 2), "red")]
        right = [Image.new("RGB", (2, 2), "red")]
    elif kind == "file":
        left = [
            types.File(uri="https://test/files/a", mime_type="application/pdf", state="PROCESSING")
        ]
        right = [
            types.File(uri="https://test/files/a", mime_type="application/pdf", state="ACTIVE")
        ]
    else:
        left = ["doc"]
        right = [types.Content(role="user", parts=[types.Part(text="doc")])]
    first = GeminiCachedModel("gemini-test", _make_mock_client([]), left)
    second = GeminiCachedModel("gemini-test", _make_mock_client([]), right)
    assert first._cache_tags["abl-content"] == second._cache_tags["abl-content"]
    other = GeminiCachedModel("gemini-test", _make_mock_client([]), ["different"])
    assert first._cache_tags["abl-content"] != other._cache_tags["abl-content"]


def test_sc2_sc3_cache_fallback_without_sdk_transformer(monkeypatch):
    from google.genai import _transformers, types

    from async_batch_llm.models import GeminiCachedModel

    monkeypatch.delattr(_transformers, "t_contents")
    file = types.File(uri="https://test/files/a", mime_type="application/pdf", state="PROCESSING")
    content = [
        ["doc", types.Part(text="more")],
        {"parts": [{"inline_data": {"data": b"%PDF", "mime_type": "application/pdf"}}]},
        file,
    ]
    first = GeminiCachedModel("gemini-test", _make_mock_client([]), content)
    file.state = "ACTIVE"
    second = GeminiCachedModel("gemini-test", _make_mock_client([]), content)
    assert first._cache_tags == second._cache_tags
    file.uri = "https://test/files/b"
    different = GeminiCachedModel("gemini-test", _make_mock_client([]), content)
    assert first._cache_tags != different._cache_tags
