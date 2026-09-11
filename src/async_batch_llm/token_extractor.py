"""Centralized token-usage extraction for LLM responses and exceptions.

The framework needs to account for tokens consumed even by failed attempts
so users see accurate cost/usage telemetry. Different providers surface
usage in different ways:

1. **Custom framework attribute** — strategies attach `_failed_token_usage`
   to exceptions via `__dict__` when they know the count. Checked first:
   it's an exact per-attempt count, so it must win over the heuristics.
2. **PydanticAI-style** — exception's `__cause__` has a `.result` with a
   usage property (or legacy callable `.usage()`).
3. **Direct `.usage` attribute** on the exception (OpenAI-style wrappers).

Previously this logic lived inline on `ParallelBatchProcessor`. Extracting
it makes each path testable in isolation and keeps the processor lean.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

from .base import TokenUsage

logger = logging.getLogger(__name__)


_EMPTY_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "cached_input_tokens": 0,
}
_USAGE_KEYS = frozenset(_EMPTY_USAGE)
_MISSING = object()


@dataclass(frozen=True)
class TokenUsageObservation:
    """Internal usage plus whether a provider supplied an exact observation.

    ``reported_tokens=None`` means usage was unavailable. Zero is deliberately
    retained as a distinct known value for full reservation refunds.
    """

    usage: TokenUsage
    known: bool
    reported_tokens: int | None


_reused_observation: ContextVar[tuple[object, BaseException, TokenUsageObservation] | None] = (
    ContextVar("abl_reused_usage_observation", default=None)
)


class TokenExtractor:
    """Best-effort token-usage extraction from LLM exceptions."""

    def extract_from_exception(self, exception: BaseException) -> dict[str, int]:
        """Return a token-usage dict for a failed LLM call.

        Tries three strategies in order and returns the first match. Returns
        zeroed dict if no extraction succeeds. Never raises for normal
        extraction failures. Cancellation and process-control exceptions propagate.
        """
        return cast(dict[str, int], self.observe_exception(exception).usage).copy()

    def observe_exception(self, exception: BaseException) -> TokenUsageObservation:
        """Observe failed-attempt usage without collapsing unknown into zero."""
        reused = _reused_observation.get()
        if reused is not None and reused[0] is self and reused[1] is exception:
            return reused[2]
        fallback = TokenUsageObservation(cast(TokenUsage, dict(_EMPTY_USAGE)), False, None)
        try:
            # Strategy 1: Custom _failed_token_usage attribute (set by this
            # framework). Checked first — it carries the exact per-attempt
            # count and must not be shadowed by the heuristic paths below.
            exc_dict = getattr(exception, "__dict__", None)
            if isinstance(exc_dict, dict):
                failed = exc_dict.get("_failed_token_usage")
                if isinstance(failed, Mapping):
                    fallback = _coerce_usage_observation(failed)
                    if fallback.known:
                        return fallback

            # Strategy 2: PydanticAI-style exception with result in __cause__.
            # pydantic-ai 1.x exposes usage as a property; older versions and
            # test doubles expose a synchronous usage() method.
            cause = getattr(exception, "__cause__", None)
            if cause is not None:
                result = getattr(cause, "result", None)
                if result is not None:
                    usage_attr = getattr(result, "usage", None)
                    if usage_attr is not None:
                        return _accessor_observation(usage_attr)

            # Strategy 3: Direct .usage attribute on exception
            usage = getattr(exception, "usage", None)
            if usage is not None:
                return _accessor_observation(usage)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Extraction is best-effort; log for debugging.
            logger.debug(
                "Failed to extract token usage from %s: %s. Returning zero tokens.",
                type(exception).__name__,
                e,
            )

        return fallback

    @contextmanager
    def _reuse_observation(
        self, exception: BaseException, observation: TokenUsageObservation
    ) -> Iterator[None]:
        """Let a synchronous compatibility override call super without re-extraction."""
        token = _reused_observation.set((self, exception, observation))
        try:
            yield
        finally:
            _reused_observation.reset(token)

    def _observe_failed_stamp(self, exception: BaseException) -> TokenUsageObservation | None:
        """Read only an exact stamp, without invoking usage accessors or overrides.

        Recovery hooks may supply a later report for result accounting. As with
        exception extraction, ordinary malformed reports are best effort and
        control exceptions propagate. Normalization copies mutable stamp values.
        """
        try:
            exc_dict = getattr(exception, "__dict__", None)
            failed = exc_dict.get("_failed_token_usage") if isinstance(exc_dict, dict) else None
            if isinstance(failed, Mapping):
                return _coerce_usage_observation(failed)
        except Exception:
            logger.debug("Failed to read token usage stamp from %s", type(exception).__name__)
        return None

    @staticmethod
    def observe_result(usage: object) -> TokenUsageObservation:
        """Validate and observe a successful strategy usage mapping."""
        if not isinstance(usage, Mapping):
            raise TypeError(f"Strategy token usage must be a mapping (got {type(usage).__name__})")
        return _mapping_observation(
            cast(Mapping[object, object], usage),
            strict=True,
        )

    @staticmethod
    def accumulate(cumulative: dict[str, int], attempt_tokens: dict[str, int]) -> None:
        """Add per-attempt token counts into a running cumulative total.

        Missing fields on `attempt_tokens` are treated as zero.
        """
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
            cumulative[key] = cumulative.get(key, 0) + attempt_tokens.get(key, 0)


def _first_attr(usage: Any, *names: str, skip_none: bool = False) -> Any:
    """Return the first present field, preserving missing versus invalid values.

    Short-circuiting matters: pydantic-ai 1.x keeps ``request_tokens`` /
    ``response_tokens`` as *deprecated* aliases that emit a DeprecationWarning
    when touched, so we ask for the 1.x names (``input_tokens`` /
    ``output_tokens``) first and never read the deprecated ones when the new
    ones are present.
    """
    for name in names:
        value = (
            usage.get(name, _MISSING)
            if isinstance(usage, Mapping)
            else getattr(usage, name, _MISSING)
        )
        if value is not _MISSING and not (skip_none and value is None):
            return value
    return _MISSING


def _accessor_observation(usage: Any) -> TokenUsageObservation:
    if inspect.iscoroutinefunction(usage) or (
        callable(usage) and inspect.iscoroutinefunction(type(usage).__call__)
    ):
        return _mapping_observation({}, strict=False)
    if callable(usage):
        usage = usage()
    if inspect.isawaitable(usage):
        # Extraction never schedules or awaits provider work. Close an unstarted
        # coroutine handed to us so rejecting it does not leak a warning. Futures
        # and tasks belong to their caller and must not be cancelled here.
        if inspect.iscoroutine(usage) and inspect.getcoroutinestate(usage) == inspect.CORO_CREATED:
            usage.close()
        return _mapping_observation({}, strict=False)
    return _coerce_usage_observation(usage)


def _coerce_usage(usage: Any) -> dict[str, int]:
    """Convert a provider-specific usage object into our dict shape.

    Field-name aliasing covers the common providers:

    - PydanticAI 1.x / Anthropic / our normalized shape: ``input_tokens`` /
      ``output_tokens`` (PydanticAI 0.x ``request_tokens`` / ``response_tokens``
      are still read as a fallback).
    - OpenAI / OpenRouter: ``prompt_tokens`` / ``completion_tokens``
    """
    return cast(dict[str, int], _coerce_usage_observation(usage).usage).copy()


def _coerce_usage_observation(usage: Any) -> TokenUsageObservation:
    fields = {
        "input_tokens": _first_attr(usage, "input_tokens", "request_tokens", "prompt_tokens"),
        "output_tokens": _first_attr(
            usage, "output_tokens", "response_tokens", "completion_tokens"
        ),
        "total_tokens": _first_attr(usage, "total_tokens"),
        "cached_input_tokens": _first_attr(
            usage, "cached_input_tokens", "cache_read_tokens", skip_none=True
        ),
    }
    if fields["cached_input_tokens"] is _MISSING:
        fields["cached_input_tokens"] = _first_attr(
            _first_attr(usage, "prompt_tokens_details"), "cached_tokens"
        )
    return _mapping_observation(
        {key: value for key, value in fields.items() if value is not _MISSING}, strict=False
    )


def _mapping_observation(
    usage: Mapping[object, object],
    *,
    strict: bool,
) -> TokenUsageObservation:
    recognized = {key for key in usage if isinstance(key, str) and key in _USAGE_KEYS}
    normalized: dict[str, int] = {} if strict else dict(_EMPTY_USAGE)
    valid: dict[str, bool] = {}
    for key in recognized:
        value = usage[key]
        if strict and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(
                f"Strategy token usage[{key!r}] must be a non-negative integer (got {value!r})"
            )
        normalized[key], valid[key] = _nonnegative_int(value)

    # Provider exception objects commonly expose an optional total. None means
    # no total was reported, so valid components can still establish usage.
    # Strict successful mappings reject None in the validation loop above.
    if "total_tokens" in recognized and usage["total_tokens"] is not None:
        reported_tokens = normalized["total_tokens"] if valid["total_tokens"] else None
    elif "input_tokens" in recognized or "output_tokens" in recognized:
        input_valid = valid.get("input_tokens", True)
        output_valid = valid.get("output_tokens", True)
        reported_tokens = (
            normalized.get("input_tokens", 0) + normalized.get("output_tokens", 0)
            if input_valid and output_valid
            else None
        )
    else:
        reported_tokens = None
    if reported_tokens is not None:
        normalized["total_tokens"] = reported_tokens
    return TokenUsageObservation(
        usage=cast(TokenUsage, normalized),
        known=reported_tokens is not None,
        reported_tokens=reported_tokens,
    )


def _nonnegative_int(v: Any) -> tuple[int, bool]:
    if v is None or isinstance(v, bool):
        return 0, False
    try:
        value = int(v)
    except (TypeError, ValueError, OverflowError):
        return 0, False
    if value < 0 or (not isinstance(v, str) and v != value):
        return 0, False
    return value, True


__all__ = ["TokenExtractor", "TokenUsageObservation"]
