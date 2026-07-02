"""Error classification for different LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from async_batch_llm.base import TokenUsage

# Common error pattern constants
RATE_LIMIT_PATTERNS = ("429", "resource_exhausted", "quota", "rate limit")
TIMEOUT_PATTERNS = ("timeout",)
NETWORK_PATTERNS = ("connection",)

# Deterministic programming errors — retrying can't fix these.
LOGIC_BUG_TYPES = (
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    IndexError,
    NameError,
    ZeroDivisionError,
    AssertionError,
)


class TokenTrackingError(Exception):
    """
    Wrapper exception that preserves token usage from failed LLM calls.

    When an LLM call fails (e.g., validation error), we still want to track
    the tokens that were consumed. This wrapper attaches token usage to
    exceptions that don't natively support it (e.g., built-in exceptions
    without __dict__).

    Attributes:
        token_usage: Dictionary with input_tokens, output_tokens, total_tokens,
            and optionally cached_input_tokens.

    Example:
        >>> try:
        ...     # LLM call that fails validation
        ...     output = parse_response(response)
        ... except Exception as e:
        ...     wrapped = TokenTrackingError(str(e), token_usage=tokens)
        ...     wrapped.__cause__ = e
        ...     raise wrapped from e
    """

    def __init__(self, message: str, *, token_usage: TokenUsage | dict[str, int] | None = None):
        """
        Initialize TokenTrackingError.

        Args:
            message: Human-readable error message
            token_usage: Token usage dict to preserve (input_tokens, output_tokens, etc.)
        """
        super().__init__(message)
        self._failed_token_usage = token_usage or {}


class EmptyResponseError(ValueError):
    """The provider returned a billed response with no usable text.

    Raised by the built-in models when the API call succeeded (and was
    billed) but produced no content — e.g. a Gemini safety block, or an
    OpenAI response whose ``finish_reason`` is ``length``/``content_filter``
    or a tool call.

    Subclasses ``ValueError`` so existing handlers and classifiers keep
    treating it as a deterministic, non-retryable failure. The tokens the
    provider already billed are attached as ``_failed_token_usage`` so
    failed-attempt accounting (``WorkItemResult.token_usage``) reflects the
    real spend.

    Added in v0.10.0.
    """

    def __init__(self, message: str, *, token_usage: TokenUsage | dict[str, int] | None = None):
        super().__init__(message)
        if token_usage:
            self._failed_token_usage = dict(token_usage)


class ProviderResponseError(Exception):
    """Provider signaled failure inside an HTTP-200 response body.

    Some gateways (notably OpenRouter) report upstream failures — no
    provider available, upstream 5xx, upstream rate limits — as HTTP 200
    with an ``error`` object in the body and no choices, so the SDK never
    raises. These are typically transient routing failures: classifiers
    treat them as retryable, and as rate limits when the embedded code
    is 429.

    Attributes:
        code: Numeric error code embedded in the body, if any.
        provider_error: The raw error payload from the response body.

    Added in v0.10.0.
    """

    def __init__(self, message: str, *, code: int | None = None, provider_error: Any = None):
        super().__init__(message)
        self.code = code
        self.provider_error = provider_error


class FrameworkTimeoutError(TimeoutError):
    """
    Timeout enforced by the batch-llm framework (asyncio.wait_for).

    This distinguishes framework-level timeouts from API-level timeouts.
    Framework timeouts indicate the configured timeout_per_item was exceeded,
    whereas API timeouts indicate the LLM provider returned a timeout error.

    Attributes:
        item_id: ID of the work item that timed out (if available)
        elapsed: Actual time elapsed in seconds
        timeout_limit: Configured timeout limit in seconds
    """

    def __init__(
        self,
        message: str,
        *,
        item_id: str | None = None,
        elapsed: float | None = None,
        timeout_limit: float | None = None,
    ):
        """
        Initialize FrameworkTimeoutError with structured context (v0.4.0).

        Args:
            message: Human-readable error message
            item_id: ID of the work item that timed out
            elapsed: Actual time elapsed before timeout
            timeout_limit: The timeout limit that was exceeded
        """
        super().__init__(message)
        self.item_id = item_id
        self.elapsed = elapsed
        self.timeout_limit = timeout_limit


@dataclass
class ErrorInfo:
    """Structured information about an error.

    Attributes:
        is_retryable: Whether the framework should retry the call.
        is_rate_limit: Whether this is a rate-limit error (triggers a
            coordinated cooldown rather than per-item backoff).
        is_timeout: Whether this is a timeout.
        error_category: A short label for stats/logging.
        suggested_wait: For rate limits, a server-suggested minimum wait in
            seconds (e.g. parsed from a ``Retry-After`` header). The
            ``RateLimitCoordinator`` honors this as a *floor* on the cooldown:
            the backoff strategy may wait longer, but never shorter. ``None``
            means "no suggestion; use the strategy's value as-is".
    """

    is_retryable: bool
    is_rate_limit: bool
    is_timeout: bool
    error_category: str
    suggested_wait: float | None = None


def _retry_after_seconds(exception: Exception) -> float | None:
    """Parse a ``Retry-After`` header off an SDK exception, if present.

    Both the ``openai`` and ``google-genai`` SDKs attach the underlying HTTP
    response (with headers) to their exceptions as ``.response``.
    ``Retry-After`` may be either a number of seconds or an HTTP-date; we
    handle both and return the delay in seconds, or ``None`` when no usable
    header is present (including malformed or non-positive values).
    """
    response = getattr(exception, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        delay = float(raw)
        return delay if delay > 0 else None
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


class ErrorClassifier(ABC):
    """Abstract base class for classifying LLM provider errors.

    Provides a shared generic fallback chain (:meth:`_classify_generic`)
    that provider classifiers delegate to after their SDK-specific
    dispatch. Subclasses tune the pattern/category knobs below to keep
    their provider vocabulary.
    """

    # Knobs for the shared fallback chain. Subclasses override.
    _rate_limit_patterns: tuple[str, ...] = RATE_LIMIT_PATTERNS
    _timeout_patterns: tuple[str, ...] = TIMEOUT_PATTERNS
    _network_patterns: tuple[str, ...] = NETWORK_PATTERNS
    _timeout_category: str = "api_timeout"
    _network_category: str = "connection_error"

    @abstractmethod
    def classify(self, exception: Exception) -> ErrorInfo:
        """
        Classify an exception and determine handling strategy.

        Args:
            exception: The exception to classify

        Returns:
            ErrorInfo with classification details
        """
        pass

    @staticmethod
    def _matches_any_pattern(error_str: str, patterns: tuple[str, ...]) -> bool:
        """Check if error string matches any of the given patterns (case-insensitive)."""
        error_lower = error_str.lower()
        return any(pattern in error_lower for pattern in patterns)

    def _classify_generic(self, exception: Exception) -> ErrorInfo:
        """Shared message/type-based fallback used by all built-in classifiers.

        Runs after provider-specific SDK dispatch (or as the entire chain
        for :class:`DefaultErrorClassifier`). Ordering matters:

        1. Framework timeout (our own exception type — unambiguous).
        2. Validation errors (checked before the logic-bug isinstance test
           because pydantic's ``ValidationError`` subclasses ``ValueError``).
        3. Message-pattern transient categories (rate limit, timeout,
           network) — skipped for logic-bug exception types, so a
           ``ValueError("invalid connection string")`` isn't retried and a
           ``KeyError('quota')`` can't trigger a global cooldown.
        4. Logic bugs → non-retryable.
        5. Unknown → retryable (conservative; lets custom transient errors
           and test mocks work).
        """
        if isinstance(exception, FrameworkTimeoutError):
            return ErrorInfo(
                is_retryable=True,  # Retry - might succeed if LLM is faster
                is_rate_limit=False,
                is_timeout=True,
                error_category="framework_timeout",
            )

        # Validation errors: the LLM produced output that failed the schema —
        # retryable, the model may produce valid output next attempt.
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

        try:
            from pydantic_ai.exceptions import UnexpectedModelBehavior

            if isinstance(exception, UnexpectedModelBehavior):
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=False,
                    is_timeout=False,
                    error_category="validation_error",
                )
        except ImportError:
            pass

        is_logic_bug = isinstance(exception, LOGIC_BUG_TYPES)
        error_str = str(exception)

        # Message/type-based transient categories. Deterministic logic-bug
        # types are excluded: their messages merely *mentioning* "quota" or
        # "connection" doesn't make them transient.
        if not is_logic_bug:
            if self._matches_any_pattern(error_str, self._rate_limit_patterns):
                return ErrorInfo(
                    is_retryable=True,  # Framework handles the cooldown
                    is_rate_limit=True,
                    is_timeout=False,
                    error_category="rate_limit",
                )

            if isinstance(exception, TimeoutError) or self._matches_any_pattern(
                error_str, self._timeout_patterns
            ):
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=False,
                    is_timeout=True,
                    error_category=self._timeout_category,
                )

            if isinstance(exception, ConnectionError) or self._matches_any_pattern(
                error_str, self._network_patterns
            ):
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=False,
                    is_timeout=False,
                    error_category=self._network_category,
                )

        if is_logic_bug:
            return ErrorInfo(
                is_retryable=False,  # Don't retry logic bugs (deterministic failures)
                is_rate_limit=False,
                is_timeout=False,
                error_category="logic_error",
            )

        # Default: treat unknown generic exceptions as retryable
        # This allows custom transient errors and test mocks to work
        # Users with non-retryable custom errors should implement a custom ErrorClassifier
        return ErrorInfo(
            is_retryable=True,  # Retry unknown exceptions (might be transient)
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown",
        )


class DefaultErrorClassifier(ErrorClassifier):
    """Default error classifier that handles common error types."""

    def _matches_rate_limit(self, error_str: str) -> bool:
        """Return True if the error string looks like a rate limit."""
        return self._matches_any_pattern(error_str, RATE_LIMIT_PATTERNS)

    def classify(self, exception: Exception) -> ErrorInfo:
        """Classify common errors with conservative defaults."""
        return self._classify_generic(exception)
