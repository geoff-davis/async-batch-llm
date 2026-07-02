"""Google Gemini-specific error classification."""

from ..strategies.errors import (
    ErrorClassifier,
    ErrorInfo,
    FrameworkTimeoutError,
    _retry_after_seconds,
    matches_any_pattern,
)

# Error pattern constants
RATE_LIMIT_PATTERNS = ("429", "resource_exhausted", "quota", "rate limit")
TIMEOUT_PATTERNS = ("timeout", "504", "deadline")
# 503 Service Unavailable / model overload — a transient server-side capacity
# blip (distinct from a 429 quota rate limit). Retried with per-item exponential
# backoff like any other 5xx, NOT a coordinated cooldown (see classify()).
OVERLOAD_PATTERNS = ("503", "unavailable", "overloaded", "high demand")


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

    def _matches_any_pattern(self, error_str: str, patterns: tuple[str, ...]) -> bool:
        """Check if error string matches any of the given patterns (case-insensitive).

        Numeric codes ("429", "503", "504") match on word boundaries so an
        unrelated number (e.g. "4290 tokens") doesn't get classified as a rate
        limit / overload. Status codes off the SDK's APIError are still
        preferred and checked before this string fallback.
        """
        return matches_any_pattern(error_str, patterns)

    def classify(self, exception: Exception) -> ErrorInfo:
        """Classify Gemini-specific errors."""
        # Check for framework timeout first (highest priority)
        if isinstance(exception, FrameworkTimeoutError):
            return ErrorInfo(
                is_retryable=True,  # Retry - might succeed if LLM is faster
                is_rate_limit=False,
                is_timeout=True,
                error_category="framework_timeout",
            )

        # Dispatch on the genai SDK's status code when available. When the
        # SDK isn't installed (or the exception isn't a genai one), fall
        # through to the generic checks below so rate limits, timeouts, and
        # logic bugs are still recognized.
        info = self._classify_genai_exception(exception)
        if info is not None:
            return info

        # Check for PydanticAI validation errors
        try:
            from pydantic_ai.exceptions import UnexpectedModelBehavior

            if isinstance(exception, UnexpectedModelBehavior):
                return ErrorInfo(
                    is_retryable=True,  # Retry validation errors
                    is_rate_limit=False,
                    is_timeout=False,
                    error_category="validation_error",
                )
        except ImportError:
            pass

        # Fallback: Check error message for common patterns
        error_str = str(exception)

        # Check if it looks like a rate limit error (for mocks and other providers)
        if self._matches_any_pattern(error_str, RATE_LIMIT_PATTERNS):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
            )

        # Check if it looks like a timeout
        if self._matches_any_pattern(error_str, TIMEOUT_PATTERNS):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="timeout",
            )

        # Check for Pydantic validation errors (retryable - LLM might generate valid output on retry)
        try:
            from pydantic import ValidationError

            if isinstance(exception, ValidationError):
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=False,
                    is_timeout=False,
                    error_category="validation_error",
                )
        except ImportError:
            pass

        # Check for logic bugs (deterministic errors that won't be fixed by retrying)
        logic_bug_types = (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            NameError,
            ZeroDivisionError,
            AssertionError,
        )
        if isinstance(exception, logic_bug_types):
            return ErrorInfo(
                is_retryable=False,  # Don't retry logic bugs (deterministic failures)
                is_rate_limit=False,
                is_timeout=False,
                error_category="logic_error",
            )

        # Default: treat unknown generic exceptions as retryable
        # This allows custom transient errors and test mocks to work
        return ErrorInfo(
            is_retryable=True,  # Retry unknown exceptions (might be transient)
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown",
        )

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
            # Genuine quota exhaustion — the one case that warrants the
            # coordinated cooldown (all workers pause).
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=_retry_after_seconds(exception),
            )

        if isinstance(code, int) and code in self._NON_RETRYABLE_STATUS:
            # Deterministic client errors (invalid API key, malformed request,
            # missing model) — retrying burns the budget without ever
            # succeeding.
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="client_error",
            )

        if isinstance(code, int) and code in self._RETRYABLE_STATUS:
            # 503 Service Unavailable / overload is a *transient, server-side*
            # capacity blip — usually per-request (a retry often lands on a
            # healthy backend), not your quota. So treat it like any other 5xx:
            # per-item exponential backoff (is_rate_limit=False), matching
            # OpenAIErrorClassifier. We deliberately do NOT trigger a
            # coordinated cooldown here — that global pause is for genuine
            # quota exhaustion (429 / RESOURCE_EXHAUSTED, handled above). For
            # *sustained* overload at high concurrency, lower max_workers or
            # set ProcessorConfig.max_requests_per_minute rather than relying
            # on a reactive pause.
            if code == 503:
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=False,
                    is_timeout=False,
                    error_category="server_overload",
                )
            # Other 5xx (500, 502, 504) are one-off transient errors → per-item
            # retry/backoff. All retryable, matching OpenAIErrorClassifier's
            # 5xx handling.
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
        if self._matches_any_pattern(error_str, OVERLOAD_PATTERNS):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="server_overload",
            )
        is_timeout = self._matches_any_pattern(error_str, TIMEOUT_PATTERNS)
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=False,
            is_timeout=is_timeout,
            error_category="server_timeout" if is_timeout else "api_error",
        )
