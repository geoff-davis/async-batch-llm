"""OpenRouter-specific error classification.

Extends :class:`OpenAIErrorClassifier` with OpenRouter-specific cases:

- ``no_provider_available`` (OpenRouter returns 502 with this body when none
  of the upstream providers can serve the request) → retryable network-style
  error rather than a hard server failure.

Everything else delegates to the OpenAI parent.

Added in v0.9.0.
"""

from __future__ import annotations

from ..strategies.errors import ErrorInfo
from .openai import OpenAIErrorClassifier

# OpenRouter-specific body markers we look for on APIStatusError responses.
NO_PROVIDER_PATTERNS = (
    "no_provider_available",
    "no provider available",
    "no allowed providers",
)


class OpenRouterErrorClassifier(OpenAIErrorClassifier):
    """OpenAI-compatible classifier with OpenRouter-specific overrides."""

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
