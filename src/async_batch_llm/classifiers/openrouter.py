"""OpenRouter-specific error classification.

Extends :class:`OpenAIErrorClassifier` with OpenRouter-specific cases:

- ``no_provider_available`` (OpenRouter returns 502 with this body when none
  of the upstream providers can serve the request) → retryable network-style
  error rather than a hard server failure.
- :class:`ProviderResponseError` (upstream failure reported inside an
  HTTP-200 body) → retryable; embedded 429s count as rate limits.

Everything else delegates to the OpenAI parent.

Added in v0.9.0.
"""

from __future__ import annotations

from ..strategies.errors import (
    ErrorInfo,
    ProviderResponseError,
    _insufficient_quota,
    _insufficient_quota_info,
    http_status,
)
from .openai import RATE_LIMIT_PATTERNS, OpenAIErrorClassifier

# OpenRouter-specific body markers we look for on APIStatusError responses.
NO_PROVIDER_PATTERNS = (
    "no_provider_available",
    "no provider available",
    "no allowed providers",
)


class OpenRouterErrorClassifier(OpenAIErrorClassifier):
    """OpenAI-compatible classifier with OpenRouter-specific overrides."""

    def classify(self, exception: Exception) -> ErrorInfo:
        info = self._framework_prelude(exception)
        if info is not None:
            return info
        # OpenRouter reports upstream failures inside HTTP-200 bodies;
        # OpenRouterModel surfaces those as ProviderResponseError. They're
        # transient routing failures — retry, and treat embedded 429s as
        # rate limits so the coordinated cooldown engages.
        if isinstance(exception, ProviderResponseError):
            error_str = str(exception)
            # Billing exhaustion outranks routing and rate-limit signals.
            if _insufficient_quota(exception):
                return _insufficient_quota_info()
            if self._matches_any_pattern(error_str, NO_PROVIDER_PATTERNS):
                transient = "no allowed providers" not in error_str.lower() or exception.code in (
                    502,
                    503,
                )
                return ErrorInfo(
                    is_retryable=transient,
                    is_rate_limit=False,
                    is_timeout=False,
                    error_category="network_error" if transient else "client_error",
                )
            # An embedded status is authoritative: only 429, or rate-limit text
            # without any status, is a rate limit.
            status = http_status(exception)
            if status == 429 or (
                status is None and self._matches_any_pattern(error_str, RATE_LIMIT_PATTERNS)
            ):
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=True,
                    is_timeout=False,
                    error_category="rate_limit",
                )
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="upstream_error",
            )
        return super().classify(exception)

    def _classify_openai_exception(self, exception: Exception) -> ErrorInfo | None:
        try:
            from openai import APIStatusError
        except ImportError:
            return None
        if isinstance(exception, APIStatusError) and self._matches_any_pattern(
            str(exception), ("no allowed providers",)
        ):
            return self._classify_status_error(exception)
        return super()._classify_openai_exception(exception)

    def _classify_status_error(self, exception: Exception) -> ErrorInfo:
        if _insufficient_quota(exception):
            return _insufficient_quota_info()
        # OpenRouter wraps "no upstream available" as a 502 with a specific
        # error body. Treat it as transient/network rather than server_error.
        body = str(exception).lower()
        if any(pat in body for pat in NO_PROVIDER_PATTERNS):
            transient = "no allowed providers" not in body or getattr(
                exception, "status_code", None
            ) in (502, 503)
            return ErrorInfo(
                is_retryable=transient,
                is_rate_limit=False,
                is_timeout=False,
                error_category="network_error" if transient else "client_error",
            )
        return super()._classify_status_error(exception)
