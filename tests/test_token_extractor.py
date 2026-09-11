"""Tests for TokenExtractor — central place for pulling token usage out of
successful responses, custom framework attributes, and PydanticAI-style
exception chains.

These tests drive the extraction logic out of ParallelBatchProcessor
(currently parallel.py:_extract_token_usage) into a dedicated class so it
can be reused and tested in isolation.
"""

from __future__ import annotations

import logging
from collections import UserDict
from types import SimpleNamespace

import pytest

from async_batch_llm.token_extractor import TokenExtractor


@pytest.mark.parametrize("shape", [dict, UserDict, SimpleNamespace])
@pytest.mark.parametrize(
    "names", [("input_tokens", "output_tokens"), ("request_tokens", "response_tokens")]
)
@pytest.mark.parametrize("total", [None, 0, -1, True, 1.5])
def test_optional_exception_total_uses_components_only_when_absent(shape, names, total):
    error = RuntimeError("provider failed")
    error.usage = shape(**{names[0]: 7, names[1]: 3, "total_tokens": total})
    observed = TokenExtractor().observe_exception(error)
    expected = 10 if total is None else (0 if total == 0 else None)
    assert observed.reported_tokens == expected
    assert observed.known is (expected is not None)
    assert observed.usage["total_tokens"] == (expected or 0)


def test_optional_total_does_not_relax_successful_mapping_validation():
    with pytest.raises(ValueError, match="total_tokens"):
        TokenExtractor.observe_result({"input_tokens": 7, "output_tokens": 3, "total_tokens": None})


@pytest.mark.parametrize("shape", [dict, UserDict, SimpleNamespace])
@pytest.mark.parametrize("fallback", ["cache_read_tokens", "prompt_tokens_details"])
@pytest.mark.parametrize("cached", [None, 0])
def test_optional_cached_counter_falls_back_without_erasing_known_zero(shape, fallback, cached):
    fields = {"input_tokens": 7, "output_tokens": 3, "cached_input_tokens": cached}
    if fallback == "cache_read_tokens":
        fields[fallback] = 40
    else:
        fields["cache_read_tokens"] = None
        fields[fallback] = shape(cached_tokens=40)
    error = RuntimeError("provider failed")
    error.usage = shape(**fields)
    observed = TokenExtractor().observe_exception(error)
    assert observed.usage["cached_input_tokens"] == (40 if cached is None else 0)
    assert observed.reported_tokens == 10


@pytest.mark.parametrize("shape", [dict, UserDict, lambda **kw: SimpleNamespace(**kw)])
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, None),
        ({"unrelated": 12}, None),
        ({"cached_input_tokens": 40}, None),
        ({"total_tokens": None}, None),
        ({"input_tokens": None}, None),
        ({"input_tokens": 7, "output_tokens": None}, None),
        ({"total_tokens": 0}, 0),
        ({"input_tokens": 0}, 0),
        ({"input_tokens": 7}, 7),
        ({"input_tokens": 7, "output_tokens": 3}, 10),
        ({"input_tokens": 7, "output_tokens": 3, "total_tokens": 12}, 12),
    ],
)
def test_exception_usage_presence_and_canonical_total(shape, fields, expected):
    error = RuntimeError("provider failed")
    error.usage = shape(**fields)
    observed = TokenExtractor().observe_exception(error)
    assert observed.known is (expected is not None)
    assert observed.reported_tokens == expected
    assert observed.usage["total_tokens"] == (expected or 0)
    if "cached_input_tokens" in fields:
        assert observed.usage["cached_input_tokens"] == 40


@pytest.mark.parametrize("value", [None, True, -1, 1.5, float("nan"), float("inf"), "1.5"])
@pytest.mark.parametrize("key", ["input_tokens", "total_tokens"])
def test_invalid_exception_counter_is_not_known_zero(value, key):
    error = RuntimeError("provider failed")
    error._failed_token_usage = {key: value}
    observed = TokenExtractor().observe_exception(error)
    assert observed.known is False
    assert observed.reported_tokens is None


@pytest.mark.parametrize("stamp", [{}, {"total_tokens": None}, {"cached_input_tokens": 40}])
def test_empty_exact_stamp_does_not_hide_valid_direct_usage(stamp):
    error = RuntimeError("provider failed")
    error._failed_token_usage = stamp
    error.usage = {"input_tokens": 7, "output_tokens": 3}
    observed = TokenExtractor().observe_exception(error)
    assert observed.known and observed.reported_tokens == 10
    assert observed.usage["total_tokens"] == 10


@pytest.mark.parametrize("shape", [dict, UserDict, lambda **kw: SimpleNamespace(**kw)])
def test_legacy_usage_aliases_derive_canonical_total(shape):
    error = RuntimeError("provider failed")
    error.usage = shape(request_tokens=7, response_tokens=3)
    observed = TokenExtractor().observe_exception(error)
    assert observed.reported_tokens == observed.usage["total_tokens"] == 10


def test_modern_usage_attributes_do_not_touch_deprecated_aliases():
    class Usage:
        input_tokens = 7
        output_tokens = 3

        @property
        def request_tokens(self):
            raise AssertionError("deprecated alias accessed")

    error = RuntimeError("provider failed")
    error.usage = Usage()
    assert TokenExtractor().observe_exception(error).usage["total_tokens"] == 10


@pytest.mark.parametrize("source", ["direct", "cause"])
def test_sync_usage_accessor_is_called_once_and_normalized(source):
    calls = []

    class UsageOwner:
        def usage(self):
            calls.append(1)
            return UserDict(input_tokens=7, output_tokens=3)

    error = RuntimeError("provider failed")
    if source == "direct":
        error.usage = UsageOwner().usage
    else:
        cause = RuntimeError("cause")
        cause.result = UsageOwner()
        error.__cause__ = cause
    observed = TokenExtractor().observe_exception(error)
    assert observed.reported_tokens == observed.usage["total_tokens"] == 10
    assert calls == [1]


@pytest.mark.parametrize("source", ["async_method", "returned_coroutine", "coroutine"])
def test_async_usage_is_unknown_without_starting_or_leaking_coroutines(source, recwarn):
    import gc

    calls = []

    async def usage():
        calls.append(1)
        return {"total_tokens": 9}

    error = RuntimeError("provider failed")
    error.usage = {
        "async_method": usage,
        "returned_coroutine": lambda: usage(),
        "coroutine": None,
    }[source]
    if source == "coroutine":
        error.usage = usage()
    observed = TokenExtractor().observe_exception(error)
    del error
    gc.collect()
    assert observed.known is False
    assert calls == []
    assert not [warning for warning in recwarn if "was never awaited" in str(warning.message)]


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_usage_accessor_preserves_process_control(control):
    def usage():
        raise control()

    error = RuntimeError("provider failed")
    error.usage = usage
    with pytest.raises(control):
        TokenExtractor().observe_exception(error)


def test_result_total_is_canonical_without_mutating_input():
    usage = UserDict(input_tokens=7, output_tokens=3)
    observed = TokenExtractor.observe_result(usage)
    assert observed.reported_tokens == observed.usage["total_tokens"] == 10
    assert usage == {"input_tokens": 7, "output_tokens": 3}


@pytest.mark.parametrize("value", [None, True, -1, 1.5, float("nan"), float("inf"), "7"])
def test_successful_usage_keeps_strict_counter_validation(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        TokenExtractor.observe_result({"input_tokens": value})


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
    assert acc == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "cached_input_tokens": 0,
    }


# ─── OpenAI/OpenRouter usage shape ────────────────────────────────────


class _OpenAIPromptDetails:
    cached_tokens = 32


class _OpenAIUsage:
    """Mimics the openai SDK's CompletionUsage shape."""

    prompt_tokens = 12
    completion_tokens = 7
    total_tokens = 19
    prompt_tokens_details = _OpenAIPromptDetails()


class _ExceptionWithOpenAIUsage(Exception):
    usage = _OpenAIUsage()


def test_extract_from_openai_usage_shape(extractor):
    """OpenAI / OpenRouter usage uses prompt_tokens / completion_tokens names
    and surfaces cached counts via prompt_tokens_details.cached_tokens."""
    e = _ExceptionWithOpenAIUsage("x")
    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 12
    assert result["output_tokens"] == 7
    assert result["total_tokens"] == 19
    assert result["cached_input_tokens"] == 32


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


# ─── Precedence and coercion fixes ────────────────────────────────────


def test_failed_token_usage_takes_precedence_over_heuristics(extractor):
    """The framework-stamped exact count must win over heuristic paths.

    Regression: _failed_token_usage was checked last, so an exception that
    also exposed .usage (or a cause chain) had its exact count shadowed."""

    class _HeuristicUsage:
        prompt_tokens = 999
        completion_tokens = 999
        total_tokens = 1998

    e = Exception("boom")
    e.usage = _HeuristicUsage()  # type: ignore[attr-defined]
    e.__dict__["_failed_token_usage"] = {
        "input_tokens": 5,
        "output_tokens": 1,
        "total_tokens": 6,
    }

    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 5
    assert result["output_tokens"] == 1
    assert result["total_tokens"] == 6


def test_failed_token_usage_coerces_non_int_numerics(extractor):
    """Float/str counts are coerced instead of silently dropped to zero."""
    e = Exception("boom")
    e.__dict__["_failed_token_usage"] = {
        "input_tokens": 5.0,
        "output_tokens": "7",
        "total_tokens": 12,
    }

    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 5
    assert result["output_tokens"] == 7
    assert result["total_tokens"] == 12


def test_observation_distinguishes_explicit_zero_from_unknown(extractor):
    known_zero = Exception("known")
    known_zero.__dict__["_failed_token_usage"] = {"total_tokens": 0}

    explicit = extractor.observe_exception(known_zero)
    missing = extractor.observe_exception(Exception("unknown"))
    empty_success = extractor.observe_result({})

    assert explicit.known and explicit.reported_tokens == 0
    assert not missing.known and missing.reported_tokens is None
    assert not empty_success.known and empty_success.reported_tokens is None


def test_malformed_exception_usage_is_unknown_and_never_negative(extractor):
    malformed = Exception("provider failure")
    malformed.__dict__["_failed_token_usage"] = {"total_tokens": -5}

    observation = extractor.observe_exception(malformed)

    assert not observation.known
    assert observation.reported_tokens is None
    assert observation.usage["total_tokens"] == 0


def test_extract_pydantic_ai_v1_usage_shape(extractor):
    """pydantic-ai v1 renamed fields to input_tokens/output_tokens and
    surfaces cache hits as cache_read_tokens."""

    class _V1Usage:
        input_tokens = 10
        output_tokens = 5
        total_tokens = 15
        cache_read_tokens = 4

    e = Exception("x")
    e.usage = _V1Usage()  # type: ignore[attr-defined]

    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5
    assert result["cached_input_tokens"] == 4


def test_property_style_usage_on_cause_result(extractor):
    """pydantic-ai 1.x exposes result.usage as a property (not callable);
    the __cause__ path must read it directly."""

    class _V1Usage:
        input_tokens = 10
        output_tokens = 5
        total_tokens = 15

    class _Result:
        usage = _V1Usage()  # property-style: plain attribute, not a method

    cause = Exception("cause")
    cause.result = _Result()  # type: ignore[attr-defined]
    e = Exception("wrapper")
    e.__cause__ = cause

    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 10
    assert result["total_tokens"] == 15
