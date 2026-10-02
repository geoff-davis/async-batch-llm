"""OpenAI-specific error classification.

Handles exceptions raised by the ``openai`` Python SDK (RateLimitError,
APITimeoutError, APIConnectionError, APIStatusError) plus the generic
fallbacks used across the library (FrameworkTimeoutError, validation errors,
logic bugs).

Added in v0.9.0.
"""

from __future__ import annotations

import asyncio

from ..strategies.errors import (
    ErrorClassifier,
    ErrorInfo,
    _insufficient_quota,
    _insufficient_quota_info,
    _retry_after_seconds,
    http_status,
    matches_any_pattern,
)

RATE_LIMIT_PATTERNS = (
    "429",
    "rate limit",
    "rate_limit_exceeded",
    "too many requests",
    "quota exceeded",
)
TIMEOUT_PATTERNS = ("timeout", "504", "deadline", "request timed out")
NETWORK_PATTERNS = ("connection", "network", "econnreset", "broken pipe")
# 402 Payment Required: balance/credits exhausted. DeepSeek in particular
# returns "402 Insufficient Balance" on a prepaid account that's run dry.
INSUFFICIENT_BALANCE_PATTERNS = ("402", "insufficient balance", "insufficient_quota")

# Operator-facing hint attached to 402 errors so an exhausted balance doesn't
# read like a generic API/code bug. Auth has already passed at this point.
_INSUFFICIENT_BALANCE_HINT = (
    "402 Payment Required — the provider account's balance or credits are "
    "exhausted. Top up with the provider (for DeepSeek, "
    "https://platform.deepseek.com/). Not retryable."
)


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
        # Numeric codes ("429", "402", "504") match on word boundaries so an
        # unrelated number (e.g. "4290 tokens") doesn't trip a pattern. SDK
        # exception types and HTTP status codes are still preferred over this
        # string sniffing — see _classify_openai_exception, which runs first.
        return matches_any_pattern(error_str, patterns)

    def classify(self, exception: Exception) -> ErrorInfo:
        info = self._framework_prelude(exception)
        if info is not None:
            return info

        # Try to dispatch on the openai SDK's exception types when available.
        info = self._classify_openai_exception(exception)
        if info is not None:
            return info

        # A non-OpenAI exception carrying HTTP 429, then validation and
        # programming-error types, all outrank message text.
        info = self._structured_rate_limit(exception) or self._typed_failure(exception)
        if info is not None:
            return info

        error_str = str(exception)

        # Insufficient balance / payment required (structured code or string
        # fallback for when the SDK isn't installed or for mocked exceptions).
        # Billing exhaustion is deterministic, so it outranks the transient
        # timeout/network/rate-limit heuristics below.
        if _insufficient_quota(exception) or self._matches_any_pattern(
            error_str, INSUFFICIENT_BALANCE_PATTERNS
        ):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="insufficient_balance",
                hint=_INSUFFICIENT_BALANCE_HINT,
            )

        # Generic timeout/connection by exception type or message.
        if isinstance(exception, (TimeoutError, asyncio.TimeoutError)) or self._matches_any_pattern(
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
        # or for mocked test exceptions. An exception with any other HTTP status
        # is not a rate limit by message. No response object to parse a
        # Retry-After from, so no server-suggested wait.
        if http_status(exception) is None and self._matches_any_pattern(
            error_str, RATE_LIMIT_PATTERNS
        ):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
            )

        return self._generic_tail(exception)

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
            return self._classify_status_error(exception)

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
        if not isinstance(status_code, int):
            # SDK-shaped test doubles may omit a concrete response status.
            # Preserve the SDK exception type without mutating the exception.
            try:
                from openai import RateLimitError
            except ImportError:
                pass
            else:
                if isinstance(exception, RateLimitError):
                    status_code = 429

        # Billing exhaustion outranks every status: retrying cannot succeed.
        if _insufficient_quota(exception):
            return _insufficient_quota_info()

        if status_code == 429:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=_retry_after_seconds(exception),
            )

        if status_code == 402:
            # Payment required / balance exhausted — deterministic, don't retry.
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="insufficient_balance",
                hint=_INSUFFICIENT_BALANCE_HINT,
            )

        if status_code == 401:
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="authentication",
            )

        if status_code == 403:
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="permission_denied",
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
