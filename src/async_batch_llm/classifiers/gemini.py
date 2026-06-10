"""Google Gemini-specific error classification."""

from ..strategies.errors import (
    ErrorClassifier,
    ErrorInfo,
    FrameworkTimeoutError,
    matches_any_pattern,
)

# Error pattern constants
RATE_LIMIT_PATTERNS = ("429", "resource_exhausted", "quota", "rate limit")
TIMEOUT_PATTERNS = ("timeout", "504", "deadline")
# 503 Service Unavailable / model overload. Distinct from a quota rate limit,
# but it's the same kind of capacity signal — so we route it through the
# coordinated cooldown (pause all workers + slow-start) rather than per-item
# backoff, which alone can't relieve an overloaded model under high concurrency.
OVERLOAD_PATTERNS = ("503", "unavailable", "overloaded", "high demand")


class GeminiErrorClassifier(ErrorClassifier):
    """Google Gemini-specific error classification."""

    def _matches_any_pattern(self, error_str: str, patterns: tuple[str, ...]) -> bool:
        """Check if error string matches any of the given patterns (case-insensitive).

        Numeric codes ("429", "503", "504") match on word boundaries so an
        unrelated number (e.g. "4290 tokens") doesn't get classified as a rate
        limit / overload. Gemini SDK exception types (ClientError/ServerError)
        are still preferred and checked before this string fallback.
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

        try:
            from google.genai.errors import ClientError, ServerError
        except ImportError:
            # If google.genai not installed, fall back to default classification
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="unknown",
            )

        if isinstance(exception, ClientError):
            error_str = str(exception)
            is_rate_limit = self._matches_any_pattern(error_str, RATE_LIMIT_PATTERNS)
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=is_rate_limit,
                is_timeout=False,
                error_category="rate_limit" if is_rate_limit else "client_error",
            )

        if isinstance(exception, ServerError):
            error_str = str(exception)
            # 503 Service Unavailable / overload is a capacity signal — route it
            # through the coordinated cooldown (is_rate_limit=True pauses all
            # workers and slow-starts), since per-item backoff alone can't
            # relieve an overloaded model when dozens of workers keep hitting it.
            if self._matches_any_pattern(error_str, OVERLOAD_PATTERNS):
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=True,
                    is_timeout=False,
                    error_category="server_overload",
                )
            # Other 5xx (500, 502, 504) are one-off transient errors → per-item
            # retry/backoff. All retryable, matching OpenAIErrorClassifier's 5xx
            # handling (the two used to disagree: Gemini retried only timeouts).
            is_timeout = self._matches_any_pattern(error_str, TIMEOUT_PATTERNS)
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=is_timeout,
                error_category="server_timeout" if is_timeout else "server_error",
            )

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
