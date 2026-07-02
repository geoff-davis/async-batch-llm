"""OpenRouter-specific error classification.

Extends :class:`OpenAIErrorClassifier` with OpenRouter-specific cases:

- ``no_provider_available`` (OpenRouter returns 502 with this body when none
  of the upstream providers can serve the request) → retryable network-style
  error rather than a hard server failure.

Everything else delegates to the OpenAI parent.

Added in v0.9.0.
"""

from __future__ import annotations

from ..strategies.errors import ErrorInfo, ProviderResponseError
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
        # OpenRouter reports upstream failures inside HTTP-200 bodies;
        # OpenRouterModel surfaces those as ProviderResponseError. They're
        # transient routing failures — retry, and treat embedded 429s as
        # rate limits so the coordinated cooldown engages.
        if isinstance(exception, ProviderResponseError):
            error_str = str(exception)
            if exception.code == 429 or self._matches_any_pattern(error_str, RATE_LIMIT_PATTERNS):
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=True,
                    is_timeout=False,
                    error_category="rate_limit",
                )
            if self._matches_any_pattern(error_str, NO_PROVIDER_PATTERNS):
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=False,
                    is_timeout=False,
                    error_category="network_error",
                )
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="upstream_error",
            )
        return super().classify(exception)

    def _classify_status_error(self, exception: Exception) -> ErrorInfo:
        # OpenRouter wraps "no upstream available" as a 502 with a specific
        # error body. Treat it as transient/network rather than server_error.
        body = str(exception).lower()
        if any(pat in body for pat in NO_PROVIDER_PATTERNS):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="network_error",
            )
        return super()._classify_status_error(exception)
