"""OpenAI-specific error classification.

Handles exceptions raised by the ``openai`` Python SDK (RateLimitError,
APITimeoutError, APIConnectionError, APIStatusError) plus the generic
fallbacks used across the library (FrameworkTimeoutError, validation errors,
logic bugs).

Added in v0.9.0.
"""

from __future__ import annotations

from ..strategies.errors import (
    ErrorClassifier,
    ErrorInfo,
    FrameworkTimeoutError,
    _retry_after_seconds,
)

RATE_LIMIT_PATTERNS = (
    "429",
    "rate limit",
    "rate_limit_exceeded",
    "too many requests",
    "quota",
)
TIMEOUT_PATTERNS = ("timeout", "504", "deadline", "request timed out")
NETWORK_PATTERNS = ("connection", "network", "econnreset", "broken pipe")


class OpenAIErrorClassifier(ErrorClassifier):
    """Classifier for OpenAI SDK exceptions plus generic fallbacks.

    Designed to be subclassed: provider-specific classifiers (e.g.
    :class:`OpenRouterErrorClassifier`) override :meth:`classify` to handle
    extra cases first and delegate to ``super().classify()`` for the rest.
    """

    # Status codes that should be retried (transient server-side issues).
    _RETRYABLE_STATUS = frozenset({408, 425, 500, 502, 503, 504})
    # Status codes that should NOT be retried (deterministic client errors).
    _NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 405, 409, 410, 422})

    # Knobs for the shared generic fallback chain (ErrorClassifier).
    _rate_limit_patterns = RATE_LIMIT_PATTERNS
    _timeout_patterns = TIMEOUT_PATTERNS
    _network_patterns = NETWORK_PATTERNS
    _timeout_category = "api_timeout"
    _network_category = "network_error"

    def classify(self, exception: Exception) -> ErrorInfo:
        # Framework timeout takes priority over everything else.
        if isinstance(exception, FrameworkTimeoutError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="framework_timeout",
            )

        # Try to dispatch on the openai SDK's exception types when available;
        # everything else goes through the shared generic chain (validation,
        # rate-limit/timeout/network patterns, logic bugs, unknown).
        info = self._classify_openai_exception(exception)
        if info is not None:
            return info
        return self._classify_generic(exception)

    def _classify_openai_exception(self, exception: Exception) -> ErrorInfo | None:
        """Return ErrorInfo for openai-SDK exceptions, or None to defer."""
        try:
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                RateLimitError,
            )
        except ImportError:
            return None

        if isinstance(exception, RateLimitError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=_retry_after_seconds(exception),
            )

        if isinstance(exception, APITimeoutError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="api_timeout",
            )

        if isinstance(exception, APIConnectionError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="network_error",
            )

        if isinstance(exception, APIStatusError):
            return self._classify_status_error(exception)

        return None

    def _classify_status_error(self, exception: Exception) -> ErrorInfo:
        """Branch on ``APIStatusError.status_code``."""
        status_code = getattr(exception, "status_code", None)

        if status_code == 429:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=_retry_after_seconds(exception),
            )

        if isinstance(status_code, int) and status_code in self._RETRYABLE_STATUS:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=status_code == 504,
                error_category="server_error",
            )

        if isinstance(status_code, int) and status_code in self._NON_RETRYABLE_STATUS:
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="client_error",
            )

        # Unrecognized status — be conservative and retry.
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=False,
            is_timeout=False,
            error_category="api_error",
        )
