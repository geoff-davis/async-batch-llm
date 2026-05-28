"""OpenAI-specific error classification.

Handles exceptions raised by the ``openai`` Python SDK (RateLimitError,
APITimeoutError, APIConnectionError, APIStatusError) plus the generic
fallbacks used across the library (FrameworkTimeoutError, validation errors,
logic bugs).

Added in v0.9.0.
"""

from __future__ import annotations

from ..strategies.errors import ErrorClassifier, ErrorInfo, FrameworkTimeoutError

RATE_LIMIT_PATTERNS = (
    "429",
    "rate limit",
    "rate_limit_exceeded",
    "too many requests",
    "quota",
)
TIMEOUT_PATTERNS = ("timeout", "504", "deadline", "request timed out")
NETWORK_PATTERNS = ("connection", "network", "econnreset", "broken pipe")


def _retry_after_seconds(exception: Exception) -> float | None:
    """Parse a ``Retry-After`` header off an openai-SDK exception, if present.

    The openai SDK attaches the underlying ``httpx`` response (with headers) to
    ``RateLimitError`` / ``APIStatusError``. ``Retry-After`` may be either an
    integer number of seconds or an HTTP-date; we handle both and return the
    delay in seconds, or ``None`` when no usable header is present.
    """
    response = getattr(exception, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        pass
    # HTTP-date form: compute the delay relative to now.
    try:
        import time
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(raw)
        delay = when.timestamp() - time.time()
        return delay if delay > 0 else None
    except (TypeError, ValueError):
        return None


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

    def _matches_any_pattern(self, error_str: str, patterns: tuple[str, ...]) -> bool:
        error_lower = error_str.lower()
        return any(pattern in error_lower for pattern in patterns)

    def classify(self, exception: Exception) -> ErrorInfo:
        # Framework timeout takes priority over everything else.
        if isinstance(exception, FrameworkTimeoutError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="framework_timeout",
            )

        # Try to dispatch on the openai SDK's exception types when available.
        info = self._classify_openai_exception(exception)
        if info is not None:
            return info

        # Pydantic validation — the LLM produced output that failed schema.
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

        # Generic timeout/connection by exception type or message.
        error_str = str(exception)

        if isinstance(exception, TimeoutError) or self._matches_any_pattern(
            error_str, TIMEOUT_PATTERNS
        ):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="api_timeout",
            )

        if isinstance(exception, ConnectionError) or self._matches_any_pattern(
            error_str, NETWORK_PATTERNS
        ):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="network_error",
            )

        # String-pattern fallback for rate limits when the SDK isn't installed
        # or for mocked test exceptions. No response object to parse a
        # Retry-After from, so no server-suggested wait.
        if self._matches_any_pattern(error_str, RATE_LIMIT_PATTERNS):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
            )

        # Logic bugs — deterministic; don't retry.
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
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="logic_error",
            )

        # Default: unknown but retryable (likely transient).
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown",
        )

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
