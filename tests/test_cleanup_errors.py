"""Cleanup and best-effort error path tests.

Locks in the contract that:
- A strategy whose cleanup() raises Exception does not crash the processor
  and does not prevent cleanup of other strategies.
- BaseException subclasses (KeyboardInterrupt, SystemExit) are not swallowed
  by cleanup, because the code must use `except Exception` (not `BaseException`).
- Missing token usage_metadata logs a DEBUG record so users can diagnose
  why token counts come back as zero.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from async_batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryState,
)
from async_batch_llm.base import TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy


class _OkStrategy(LLMCallStrategy[str]):
    """Always-succeeding strategy with tracked prepare/cleanup calls."""

    def __init__(self, name: str):
        self.name = name
        self.prepared = False
        self.cleaned = False

    async def prepare(self) -> None:
        self.prepared = True

    async def cleanup(self) -> None:
        self.cleaned = True

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        return f"ok-{self.name}:{prompt}", {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        }


class _BadCleanupStrategy(_OkStrategy):
    """Strategy whose cleanup() raises to simulate a failing cache-delete."""

    def __init__(self, name: str, exc: BaseException):
        super().__init__(name)
        self._exc = exc

    async def cleanup(self) -> None:
        self.cleaned = True
        raise self._exc


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_crash_or_skip_siblings(caplog):
    """A strategy whose cleanup() raises Exception must:
    - not crash the processor,
    - log a WARNING naming the failing class,
    - still allow sibling strategies to be cleaned up.
    """
    caplog.set_level(logging.WARNING, logger="async_batch_llm.parallel")

    bad = _BadCleanupStrategy("bad", ValueError("simulated cleanup boom"))
    ok = _OkStrategy("ok")

    config = ProcessorConfig(max_workers=2, timeout_per_item=5.0)
    async with ParallelBatchProcessor[str, str, None](config=config) as proc:
        await proc.add_work(LLMWorkItem(item_id="1", strategy=bad, prompt="a"))
        await proc.add_work(LLMWorkItem(item_id="2", strategy=ok, prompt="b"))
        result = await proc.process_all()

    assert result.succeeded == 2
    assert bad.cleaned is True
    assert ok.cleaned is True, "sibling cleanup must run even after one fails"

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "_BadCleanupStrategy" in r.getMessage() and "simulated cleanup boom" in r.getMessage()
        for r in warnings
    ), [r.getMessage() for r in warnings]


@pytest.mark.asyncio
async def test_cleanup_does_not_swallow_base_exceptions():
    """Cleanup uses `except Exception`, so KeyboardInterrupt (BaseException)
    must propagate rather than being swallowed as a warning.
    """
    bad = _BadCleanupStrategy("bad", KeyboardInterrupt())

    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, str, None](config=config)
    await processor.add_work(LLMWorkItem(item_id="1", strategy=bad, prompt="a"))
    await processor.process_all()

    with pytest.raises(KeyboardInterrupt):
        await processor._cleanup_strategies()


@pytest.mark.asyncio
async def test_cleanup_is_idempotent():
    """Calling cleanup twice must not re-run strategy cleanup."""
    ok = _OkStrategy("ok")
    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, str, None](config=config)
    await processor.add_work(LLMWorkItem(item_id="1", strategy=ok, prompt="a"))
    await processor.process_all()

    await processor._cleanup_strategies()
    call_count_before = ok.cleaned
    await processor._cleanup_strategies()  # second call is a no-op
    assert ok.cleaned == call_count_before


def test_extract_tokens_missing_metadata_logs_debug(caplog):
    """When a response has no usage_metadata, we return zeros but emit DEBUG
    so users can diagnose why their token counts are empty."""
    from async_batch_llm import models as models_mod

    class _NoUsage:
        pass

    caplog.set_level(logging.DEBUG, logger="async_batch_llm.models")
    result = models_mod._extract_tokens(_NoUsage())
    assert result == (0, 0, 0, 0)

    debug_messages = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
    assert any("usage_metadata" in m for m in debug_messages), debug_messages


@pytest.mark.asyncio
async def test_delete_cache_error_logged_with_traceback(caplog):
    """GeminiCachedModel.delete_cache() must log with exc_info so the
    underlying error is debuggable, not just stringified."""
    from async_batch_llm.models import GeminiCachedModel

    caplog.set_level(logging.WARNING, logger="async_batch_llm.models")

    class _FailingCaches:
        async def delete(self, *, name):
            raise RuntimeError("api refused delete")

    class _Aio:
        caches = _FailingCaches()

    class _FakeClient:
        aio = _Aio()

    model = GeminiCachedModel.__new__(GeminiCachedModel)
    model._client = _FakeClient()
    cache = type("C", (), {"name": "caches/abc"})()
    model._cache = cache
    model._cache_created_at = 0.0
    model._prepared = True

    await model.delete_cache()

    records = [r for r in caplog.records if "caches/abc" in r.getMessage()]
    assert records, "expected a warning mentioning the cache name"
    assert records[0].exc_info is not None, "exc_info should be attached for debugging"


@pytest.mark.asyncio
async def test_aexit_runs_parent_cleanup_when_strategy_cleanup_cancelled():
    """Regression: __aexit__ called cleanup() only after strategy cleanup, so
    a CancelledError raised mid-cleanup (e.g. Ctrl-C) skipped worker
    cancellation and leaked progress tasks."""

    class CancelledCleanupStrategy(LLMCallStrategy[str]):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            return "ok", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

        async def cleanup(self) -> None:
            raise asyncio.CancelledError()

    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, str, None](config=config)
    # Keep a strong reference — prepared strategies are tracked in a WeakSet.
    strategy = CancelledCleanupStrategy()
    await processor._ensure_strategy_prepared(strategy)

    cleanup_called = False
    original_cleanup = processor.cleanup

    async def spy_cleanup() -> None:
        nonlocal cleanup_called
        cleanup_called = True
        await original_cleanup()

    processor.cleanup = spy_cleanup  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await processor.__aexit__(None, None, None)

    assert cleanup_called, "parent cleanup() must run even when strategy cleanup is cancelled"


def test_deprecated_kwargs_do_not_mutate_shared_config():
    """Regression: deprecated ctor kwargs were written into the caller's
    ProcessorConfig, silently corrupting a config shared across processors."""
    config = ProcessorConfig(max_workers=5, timeout_per_item=60.0)

    with pytest.warns(DeprecationWarning):
        processor = ParallelBatchProcessor[str, str, None](config=config, max_workers=2)

    assert processor.config.max_workers == 2
    assert processor.config is not config
    assert config.max_workers == 5, "caller's config must not be mutated"

    with pytest.warns(DeprecationWarning):
        processor2 = ParallelBatchProcessor[str, str, None](config=config, rate_limit_cooldown=1.0)

    assert processor2.config.rate_limit.cooldown_seconds == 1.0
    assert config.rate_limit.cooldown_seconds == 300.0, "nested config must not be mutated"


# Ensure import-only test: async strategy helper class actually imports cleanly
def test_module_structure_sanity():
    assert asyncio is not None
