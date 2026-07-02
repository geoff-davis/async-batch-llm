"""Google Gemini-specific error classification."""

from ..strategies.errors import (
    ErrorClassifier,
    ErrorInfo,
    _retry_after_seconds,
)

# Error pattern constants
RATE_LIMIT_PATTERNS = ("429", "resource_exhausted", "quota", "rate limit")
TIMEOUT_PATTERNS = ("timeout", "504", "deadline")


class GeminiErrorClassifier(ErrorClassifier):
    """Google Gemini-specific error classification.

    Dispatches on ``google.genai.errors.APIError.code`` (the HTTP status)
    when the SDK is installed, then falls back to generic message/type-based
    classification — so mocks and non-Gemini exceptions still classify
    sensibly even without the ``[gemini]`` extra.
    """

    # Status codes that should be retried (transient server-side issues).
    _RETRYABLE_STATUS = frozenset({408, 425, 500, 502, 503, 504})
    # Status codes that should NOT be retried (deterministic client errors
    # such as an invalid API key, malformed request, or missing model).
    _NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 405, 409, 410, 422})

    # Knobs for the shared generic fallback chain (ErrorClassifier).
    _rate_limit_patterns = RATE_LIMIT_PATTERNS
    _timeout_patterns = TIMEOUT_PATTERNS
    _timeout_category = "timeout"

    def classify(self, exception: Exception) -> ErrorInfo:
        """Classify Gemini-specific errors.

        Dispatches on the genai SDK's exception types when available; when
        the SDK isn't installed (or the exception isn't a genai one), the
        shared generic chain still recognizes rate limits, timeouts,
        validation errors, and logic bugs.
        """
        info = self._classify_genai_exception(exception)
        if info is not None:
            return info
        return self._classify_generic(exception)

    def _classify_genai_exception(self, exception: Exception) -> ErrorInfo | None:
        """Return ErrorInfo for google-genai SDK exceptions, or None to defer."""
        try:
            from google.genai.errors import APIError
        except ImportError:
            return None

        if not isinstance(exception, APIError):
            return None

        code = getattr(exception, "code", None)

        if code == 429:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=_retry_after_seconds(exception),
            )

        if isinstance(code, int) and code in self._NON_RETRYABLE_STATUS:
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="client_error",
            )

        if isinstance(code, int) and code in self._RETRYABLE_STATUS:
            # 500 INTERNAL / 503 UNAVAILABLE ("model is overloaded") are
            # Gemini's most common transient production failures.
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=code == 504,
                error_category="server_timeout" if code == 504 else "server_error",
            )

        # Unrecognized or missing status code — fall back to message patterns
        # (mirrors the SDK-less path), defaulting to conservative retry.
        error_str = str(exception)
        if self._matches_any_pattern(error_str, RATE_LIMIT_PATTERNS):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=_retry_after_seconds(exception),
            )
        is_timeout = self._matches_any_pattern(error_str, TIMEOUT_PATTERNS)
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=False,
            is_timeout=is_timeout,
            error_category="server_timeout" if is_timeout else "api_error",
        )
