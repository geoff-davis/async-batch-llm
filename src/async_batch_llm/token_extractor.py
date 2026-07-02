"""Centralized token-usage extraction for LLM responses and exceptions.

The framework needs to account for tokens consumed even by failed attempts
so users see accurate cost/usage telemetry. Different providers surface
usage in different ways:

1. **Custom framework attribute** — strategies attach `_failed_token_usage`
   to exceptions via `__dict__` when they know the count. Checked first:
   it's an exact per-attempt count, so it must win over the heuristics.
2. **PydanticAI-style** — exception's `__cause__` has a `.result` with a
   callable `.usage()`.
3. **Direct `.usage` attribute** on the exception (OpenAI-style wrappers).

Previously this logic lived inline on `ParallelBatchProcessor`. Extracting
it makes each path testable in isolation and keeps the processor lean.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


_EMPTY_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "cached_input_tokens": 0,
}


class TokenExtractor:
    """Best-effort token-usage extraction from LLM exceptions."""

    def extract_from_exception(self, exception: BaseException) -> dict[str, int]:
        """Return a token-usage dict for a failed LLM call.

        Tries three strategies in order and returns the first match. Returns
        zeroed dict if no extraction succeeds. Never raises for normal
        extraction failures — only `asyncio.CancelledError` propagates.
        """
        try:
            # Strategy 1: Custom _failed_token_usage attribute (set by this
            # framework). Checked first — it carries the exact per-attempt
            # count and must not be shadowed by the heuristic paths below.
            exc_dict = getattr(exception, "__dict__", None)
            if isinstance(exc_dict, dict):
                failed = exc_dict.get("_failed_token_usage")
                if isinstance(failed, dict):
                    merged = dict(_EMPTY_USAGE)
                    merged.update({k: _int(v) for k, v in failed.items()})
                    return merged

            # Strategy 2: PydanticAI-style exception with result in __cause__
            cause = getattr(exception, "__cause__", None)
            if cause is not None:
                result = getattr(cause, "result", None)
                if result is not None:
                    usage_fn = getattr(result, "usage", None)
                    if callable(usage_fn):
                        usage = usage_fn()
                        if usage is not None:
                            return _coerce_usage(usage)

            # Strategy 3: Direct .usage attribute on exception
            usage = getattr(exception, "usage", None)
            if usage is not None:
                if callable(usage):
                    usage = usage()
                if usage is not None:
                    return _coerce_usage(usage)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Extraction is best-effort; log for debugging.
            logger.debug(
                "Failed to extract token usage from %s: %s. Returning zero tokens.",
                type(exception).__name__,
                e,
            )

        return dict(_EMPTY_USAGE)

    @staticmethod
    def accumulate(cumulative: dict[str, int], attempt_tokens: dict[str, int]) -> None:
        """Add per-attempt token counts into a running cumulative total.

        Missing fields on `attempt_tokens` are treated as zero.
        """
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
            cumulative[key] = cumulative.get(key, 0) + attempt_tokens.get(key, 0)


def _first_attr(usage: Any, *names: str) -> Any:
    """Return the first non-None attribute among ``names``, else 0."""
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return value
    return 0


def _coerce_usage(usage: Any) -> dict[str, int]:
    """Convert a provider-specific usage object into our dict shape.

    Field-name aliasing covers the common providers:

    - Anthropic / pydantic-ai v1 / our normalized shape: ``input_tokens`` /
      ``output_tokens``
    - OpenAI / OpenRouter: ``prompt_tokens`` / ``completion_tokens``
    - pydantic-ai legacy: ``request_tokens`` / ``response_tokens`` — checked
      last, so modern usage objects never touch the deprecated properties
      (which emit a ``DeprecationWarning`` on access).
    """
    input_tokens = _int(_first_attr(usage, "input_tokens", "prompt_tokens", "request_tokens"))
    output_tokens = _int(
        _first_attr(usage, "output_tokens", "completion_tokens", "response_tokens")
    )
    cached = _int(getattr(usage, "cached_input_tokens", 0))
    if not cached:
        # pydantic-ai v1 surfaces cache hits as cache_read_tokens.
        cached = _int(getattr(usage, "cache_read_tokens", 0))
    if not cached:
        # OpenAI surfaces cached prompt tokens nested under prompt_tokens_details.
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = _int(getattr(details, "cached_tokens", 0))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _int(getattr(usage, "total_tokens", 0)),
        "cached_input_tokens": cached,
    }


def _int(v: Any) -> int:
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0
