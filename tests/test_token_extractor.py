"""Tests for TokenExtractor — central place for pulling token usage out of
successful responses, custom framework attributes, and PydanticAI-style
exception chains.

These tests drive the extraction logic out of ParallelBatchProcessor
(currently parallel.py:_extract_token_usage) into a dedicated class so it
can be reused and tested in isolation.
"""

from __future__ import annotations

import logging

import pytest

from async_batch_llm.token_extractor import TokenExtractor


@pytest.fixture
def extractor() -> TokenExtractor:
    return TokenExtractor()


# ─── Custom framework attribute ───────────────────────────────────────


def test_extract_from_failed_token_usage_dict(extractor):
    """Strategies attach `_failed_token_usage` dict to exceptions so the
    framework can account for tokens consumed by failed attempts."""
    e = RuntimeError("boom")
    e.__dict__["_failed_token_usage"] = {
        "input_tokens": 10,
        "output_tokens": 3,
        "total_tokens": 13,
    }
    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 3
    assert result["total_tokens"] == 13


def test_extract_ignores_non_dict_failed_token_usage(extractor):
    e = RuntimeError("boom")
    e.__dict__["_failed_token_usage"] = "not-a-dict"  # corrupt / unexpected shape
    result = extractor.extract_from_exception(e)
    assert result == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
    }


# ─── Direct .usage attribute on the exception ─────────────────────────


class _Usage:
    request_tokens = 7
    response_tokens = 2
    total_tokens = 9


class _ExceptionWithUsage(Exception):
    usage = _Usage()


def test_extract_from_exception_usage_attribute(extractor):
    e = _ExceptionWithUsage("x")
    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 7
    assert result["output_tokens"] == 2
    assert result["total_tokens"] == 9


class _CallableUsage:
    def __call__(self):
        return _Usage()


class _ExceptionWithCallableUsage(Exception):
    usage = _CallableUsage()


def test_extract_from_callable_usage_attribute(extractor):
    e = _ExceptionWithCallableUsage("x")
    result = extractor.extract_from_exception(e)
    assert result["total_tokens"] == 9


# ─── PydanticAI-style exception chain (cause.result.usage()) ──────────


class _PydanticAICause(Exception):
    class _Result:
        @staticmethod
        def usage():
            return _Usage()

    result = _Result()


def test_extract_from_pydantic_ai_cause_chain(extractor):
    outer = RuntimeError("outer")
    outer.__cause__ = _PydanticAICause("inner")
    result = extractor.extract_from_exception(outer)
    assert result["input_tokens"] == 7
    assert result["output_tokens"] == 2
    assert result["total_tokens"] == 9


# ─── Missing info → zeros, DEBUG log ──────────────────────────────────


def test_extract_returns_zeros_for_plain_exception(extractor, caplog):
    caplog.set_level(logging.DEBUG, logger="async_batch_llm.token_extractor")
    result = extractor.extract_from_exception(ValueError("no tokens here"))
    assert result == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
    }


def test_extract_logs_debug_when_usage_shape_is_weird(extractor, caplog):
    """If a provider exception has a `usage` attr that blows up on access,
    we should log DEBUG and return zeros rather than crash."""

    class _BadUsage:
        @property
        def request_tokens(self):
            raise RuntimeError("simulated shape mismatch")

    class _ExcBadUsage(Exception):
        usage = _BadUsage()

    caplog.set_level(logging.DEBUG, logger="async_batch_llm.token_extractor")
    result = extractor.extract_from_exception(_ExcBadUsage("x"))
    assert result["total_tokens"] == 0
    debug_messages = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
    assert any("token usage" in m.lower() for m in debug_messages), debug_messages


# ─── Accumulation across retries ──────────────────────────────────────


def test_accumulate_adds_all_fields(extractor):
    acc = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_input_tokens": 0}
    extractor.accumulate(acc, {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8})
    extractor.accumulate(
        acc, {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "cached_input_tokens": 4}
    )
    assert acc == {
        "input_tokens": 7,
        "output_tokens": 4,
        "total_tokens": 11,
        "cached_input_tokens": 4,
    }


def test_accumulate_ignores_missing_fields(extractor):
    acc = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "cached_input_tokens": 0}
    extractor.accumulate(acc, {})  # no-op
    assert acc == {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "cached_input_tokens": 0}


# ─── CancelledError propagates (never swallowed) ──────────────────────


def test_cancelled_error_propagates(extractor):
    """Internal try/except must not swallow CancelledError."""
    import asyncio

    class _CancelOnAccess(Exception):
        @property
        def __cause__(self):  # type: ignore[override]
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        extractor.extract_from_exception(_CancelOnAccess("x"))
