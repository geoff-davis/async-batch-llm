"""Stress / race-condition tests for shared strategies and cached models.

These tests lock in invariants around the v0.6.0 model abstraction that
the existing suites only touch lightly:

- Under 20 concurrent workers and 100 items, prepare() is called exactly
  once for a shared LLMCallStrategy. (Existing tests cover 10 workers;
  this doubles the pressure on the double-checked locking path.)
- GeminiCachedModel.delete_cache() is safe under concurrent callers:
  the underlying client API fires exactly once and no RuntimeError
  surfaces to callers who arrive after the cache is gone.
- GeminiCachedModel exposes its cache only after _find_or_create_cache
  completes, so concurrent prepare() callers never observe a
  half-initialized state.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from async_batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryState,
)
from async_batch_llm.base import TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy


class _CountingSharedStrategy(LLMCallStrategy[str]):
    """Tracks prepare()/execute()/cleanup() call counts under concurrency."""

    def __init__(self):
        self.prepare_count = 0
        self.execute_count = 0
        self.cleanup_count = 0
        self._prepare_lock = asyncio.Lock()

    async def prepare(self) -> None:
        async with self._prepare_lock:
            self.prepare_count += 1
            # Simulate a slow cache/connection setup so races have time to manifest.
            await asyncio.sleep(0.05)

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        self.execute_count += 1
        return f"ok:{prompt}", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    async def cleanup(self) -> None:
        self.cleanup_count += 1


@pytest.mark.asyncio
async def test_shared_strategy_prepare_once_at_high_concurrency():
    """With 20 workers and 100 items sharing one strategy, prepare() must
    fire exactly once — the double-checked lock inside StrategyLifecycle
    is the only thing preventing 20 parallel prepare() calls here."""
    strategy = _CountingSharedStrategy()
    config = ProcessorConfig(max_workers=20, attempt_timeout=15.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(100):
            await processor.add_work(
                LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt=f"p{i}")
            )
        result = await processor.process_all()

    assert result.total_items == 100
    assert result.succeeded == 100
    assert strategy.prepare_count == 1, strategy.prepare_count
    assert strategy.execute_count == 100
    # Cleanup runs exactly once via __aexit__.
    assert strategy.cleanup_count == 1


# ─── GeminiCachedModel concurrent delete_cache ────────────────────────


class _ConcurrentDeleteClient:
    """Mock genai client that tracks concurrent delete calls."""

    def __init__(self):
        self.delete_calls = 0
        self._delete_lock = asyncio.Lock()

    class _Aio:
        def __init__(self, outer: _ConcurrentDeleteClient):
            self._outer = outer
            self.caches = _ConcurrentDeleteClient._Caches(outer)

    class _Caches:
        def __init__(self, outer: _ConcurrentDeleteClient):
            self._outer = outer

        async def delete(self, *, name: str) -> None:
            async with self._outer._delete_lock:
                self._outer.delete_calls += 1
                # Simulate API latency so concurrent callers race.
                await asyncio.sleep(0.02)

    @property
    def aio(self) -> Any:
        return _ConcurrentDeleteClient._Aio(self)


@pytest.mark.asyncio
async def test_concurrent_delete_cache_does_not_crash():
    """Calling GeminiCachedModel.delete_cache() from many tasks in parallel
    must not raise — the first caller deletes the cache; later callers
    observe `_cache is None` and return silently."""
    from async_batch_llm.models import GeminiCachedModel

    model = GeminiCachedModel.__new__(GeminiCachedModel)
    client = _ConcurrentDeleteClient()
    model._client = client
    cache_obj = type("C", (), {"name": "caches/race"})()
    model._cache = cache_obj
    model._cache_created_at = 0.0
    model._prepared = True

    await asyncio.gather(*(model.delete_cache() for _ in range(10)))

    # Exactly one delete fires; the nine other tasks observe _cache is None
    # after acquiring the cache lock and return silently. No AttributeError,
    # no duplicated API calls.
    assert client.delete_calls == 1, client.delete_calls
    assert model._cache is None
    assert model._prepared is False


# ─── Cache state visibility under concurrent prepare() ────────────────


class _SlowCachedModel:
    """Stand-in for GeminiCachedModel with a deliberately slow prepare()
    that exposes any half-initialized state via `_prepared` races."""

    def __init__(self):
        self._prepared = False
        self.prepare_calls = 0
        self._cache_value: str | None = None

    async def prepare(self) -> None:
        self.prepare_calls += 1
        # Intentionally yield between "starting prepare" and "done" so
        # concurrent workers can observe both states.
        await asyncio.sleep(0.05)
        self._cache_value = "ready"
        self._prepared = True


class _StrategyUsingSlowCache(LLMCallStrategy[str]):
    def __init__(self, model: _SlowCachedModel):
        self.model = model

    async def prepare(self) -> None:
        await self.model.prepare()

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        # Assertion lives here: by the time execute runs, the model MUST be
        # fully prepared — otherwise StrategyLifecycle is leaking pre-ready
        # state to workers.
        assert self.model._prepared is True
        assert self.model._cache_value == "ready"
        return f"ok:{prompt}", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


@pytest.mark.asyncio
async def test_workers_never_observe_half_prepared_model():
    """Workers must only execute after prepare() has fully completed.
    This pins the StrategyLifecycle guarantee that prepare() is awaited
    before any execute() call is scheduled on any worker."""
    slow_model = _SlowCachedModel()
    strategy = _StrategyUsingSlowCache(slow_model)
    config = ProcessorConfig(max_workers=10, attempt_timeout=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(40):
            await processor.add_work(
                LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt=f"p{i}")
            )
        result = await processor.process_all()

    assert result.succeeded == 40
    # prepare() should still have fired exactly once across all 10 workers.
    assert slow_model.prepare_calls == 1
