"""Tests for GeminiCachedModel auto-renew (fix #4).

The old renewal path nuked the current cache reference and called
``_find_or_create_cache``, which re-listed and re-adopted the very same
near-expiry cache — extending nothing, and doing a ``caches.list()`` round-trip
on every ``generate()`` inside the renewal-buffer window. The fix prefers
``caches.update()`` to extend the TTL in place, and (on the fallback path)
skips candidates that are themselves within the renewal buffer.
"""

import importlib.util
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _genai_or_skip() -> None:
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")


class _AsyncIterList:
    """Minimal async iterator wrapping a list — matches genai's caches.list() API."""

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


def _make_mock_cache(name: str, *, expire_in: float = 3600.0) -> Any:
    cache = MagicMock()
    cache.name = name
    cache.model = "projects/test/models/gemini-test"
    cache.display_name = None
    cache.create_time = MagicMock()
    cache.create_time.timestamp.return_value = time.time()
    cache.expire_time = MagicMock()
    cache.expire_time.timestamp.return_value = time.time() + expire_in
    return cache


def _make_response(text: str) -> Any:
    response = MagicMock()
    response.text = text
    usage = MagicMock()
    usage.prompt_token_count = 5
    usage.candidates_token_count = 3
    usage.total_token_count = 8
    usage.cached_content_token_count = 4
    response.usage_metadata = usage
    # Empty candidates → _extract_metadata returns None without touching attrs.
    response.candidates = []
    return response


def _expired_cached_model(client: Any) -> Any:
    """Build a GeminiCachedModel whose current cache is already inside the buffer."""
    from async_batch_llm.models import GeminiCachedModel

    model = GeminiCachedModel(
        model="gemini-test",
        client=client,
        cached_content=[],
        cache_ttl_seconds=3600,
        cache_renewal_buffer_seconds=300,
        auto_renew=True,
    )
    # Install a near-expiry current cache directly (skip prepare()).
    model._cache = _make_mock_cache("current", expire_in=50)
    # Age 3400 > ttl(3600) - buffer(300) = 3300 → _is_cache_expired() True.
    model._cache_created_at = time.time() - 3400
    model._prepared = True
    return model


@pytest.mark.asyncio
async def test_renew_extends_ttl_in_place_via_update():
    """Renewal must call caches.update() and NOT re-list/re-adopt."""
    _genai_or_skip()

    client = MagicMock()
    client.aio.caches.list = AsyncMock(
        side_effect=AssertionError("list() must not be called during in-place renew")
    )
    client.aio.caches.update = AsyncMock(return_value=None)
    client.aio.caches.create = AsyncMock(
        side_effect=AssertionError("create() must not be called during in-place renew")
    )
    client.aio.models.generate_content = AsyncMock(return_value=_make_response("hi"))

    model = _expired_cached_model(client)

    response = await model.generate("do it")

    client.aio.caches.update.assert_awaited_once()
    client.aio.caches.list.assert_not_called()
    # TTL pushed out → no longer considered expired.
    assert not model._is_cache_expired()
    assert response.text == "hi"


@pytest.mark.asyncio
async def test_renew_loop_does_not_relist_on_every_call():
    """Once renewed in place, subsequent calls in the window do no extra work."""
    _genai_or_skip()

    client = MagicMock()
    client.aio.caches.list = AsyncMock(return_value=_AsyncIterList([]))
    client.aio.caches.update = AsyncMock(return_value=None)
    client.aio.caches.create = AsyncMock()
    client.aio.models.generate_content = AsyncMock(return_value=_make_response("ok"))

    model = _expired_cached_model(client)

    for _ in range(3):
        await model.generate("loop")

    # Exactly one renew across three calls; never a list() round-trip.
    assert client.aio.caches.update.await_count == 1
    client.aio.caches.list.assert_not_called()
    client.aio.caches.create.assert_not_called()


@pytest.mark.asyncio
async def test_renew_fallback_skips_near_expiry_candidates():
    """If update() fails, fall back to find-or-create — skipping near-expiry caches."""
    _genai_or_skip()

    near_expiry = _make_mock_cache("near", expire_in=100)  # within 300s buffer
    fresh = _make_mock_cache("fresh", expire_in=3600)

    client = MagicMock()
    client.aio.caches.list = AsyncMock(return_value=_AsyncIterList([near_expiry]))
    client.aio.caches.update = AsyncMock(side_effect=RuntimeError("update unsupported"))
    client.aio.caches.create = AsyncMock(return_value=fresh)
    client.aio.models.generate_content = AsyncMock(return_value=_make_response("hi"))

    model = _expired_cached_model(client)

    await model.generate("go")

    client.aio.caches.update.assert_awaited_once()
    # near-expiry candidate skipped → a fresh cache is created.
    client.aio.caches.create.assert_awaited_once()
    assert model._cache is fresh


def test_cache_within_renewal_buffer_helper():
    """Unit-test the expiry predicate directly."""
    _genai_or_skip()

    client = MagicMock()
    model = _expired_cached_model(client)

    assert model._cache_within_renewal_buffer(_make_mock_cache("x", expire_in=100)) is True
    assert model._cache_within_renewal_buffer(_make_mock_cache("y", expire_in=3600)) is False

    # Unknown expiry → don't skip.
    no_expiry = MagicMock()
    no_expiry.expire_time = None
    assert model._cache_within_renewal_buffer(no_expiry) is False
