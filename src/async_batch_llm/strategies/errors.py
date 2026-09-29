"""Error classification for different LLM providers."""

from __future__ import annotations

import asyncio
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from async_batch_llm.base import TokenUsage

# Common error pattern constants. A bare "quota" also matches billing
# exhaustion and arbitrary application text, so require the provider phrase.
RATE_LIMIT_PATTERNS = ("429", "resource_exhausted", "quota exceeded", "rate limit")
# Billing exhaustion is checked before rate limits: retrying cannot succeed.
INSUFFICIENT_QUOTA_PATTERNS = ("insufficient_quota",)
_INSUFFICIENT_QUOTA_HINT = "Provider quota or credits exhausted; check account billing and limits."

# Deterministic programming failures. Their type outranks message text: a
# user ``ValueError("429: invalid item")`` is a bug, not a provider rate limit.
LOGIC_ERROR_TYPES: tuple[type[Exception], ...] = (
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    IndexError,
    NameError,
    ZeroDivisionError,
    AssertionError,
)


@lru_cache(maxsize=64)
def _word_boundary_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(pattern)}\b")


def pattern_in(text_lower: str, pattern: str) -> bool:
    """Return True if ``pattern`` appears in already-lowercased ``text_lower``.

    Purely-numeric patterns (HTTP status codes like ``"429"``, ``"503"``,
    ``"402"``) are matched on **word boundaries** so an unrelated number such
    as ``"Expected 4290 tokens"`` doesn't get mistaken for a ``429`` rate
    limit and trigger a coordinated cooldown of every worker. Non-numeric
    patterns (``"quota"``, ``"rate limit"``) use plain substring containment.
    """
    if pattern.isdigit():
        return _word_boundary_regex(pattern).search(text_lower) is not None
    return pattern in text_lower


def matches_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    """Case-insensitively test ``text`` against ``patterns`` (see :func:`pattern_in`)."""
    lowered = text.lower()
    return any(pattern_in(lowered, pattern) for pattern in patterns)


def _http_status_value(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    return None


def _safe_attribute(target: object, name: str) -> object:
    # SDK exception properties can raise when an optional request/response is absent.
    try:
        return getattr(target, name, None)
    except Exception:
        return None


def http_status(exception: Exception) -> int | None:
    """Return an HTTP status carried by an SDK-style exception, if any.

    Recognizes ``status_code`` (OpenAI, Anthropic, PydanticAI), ``status``
    (aiohttp), ``code`` (google-genai, urllib, :class:`ProviderResponseError`)
    and ``response.status_code``/``response.status`` (httpx, requests). Only
    integers in the HTTP range count, so string error codes are ignored.
    """
    for name in ("status_code", "status", "code"):
        status = _http_status_value(_safe_attribute(exception, name))
        if status is not None:
            return status
    response = _safe_attribute(exception, "response")
    if response is not None:
        for name in ("status_code", "status"):
            status = _http_status_value(_safe_attribute(response, name))
            if status is not None:
                return status
    return None


def _insufficient_quota_info() -> ErrorInfo:
    return ErrorInfo(False, False, False, "insufficient_balance", hint=_INSUFFICIENT_QUOTA_HINT)


def _insufficient_quota(exception: Exception) -> bool:
    if _safe_attribute(exception, "code") == "insufficient_quota":
        return True
    body = _safe_attribute(exception, "body")
    error = body.get("error", body) if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("code") == "insufficient_quota":
        return True
    # Some SDKs (e.g. PydanticAI's ModelHTTPError) keep the raw body out of str().
    if isinstance(body, str) and matches_any_pattern(body, INSUFFICIENT_QUOTA_PATTERNS):
        return True
    return matches_any_pattern(str(exception), INSUFFICIENT_QUOTA_PATTERNS)


def _retry_after_seconds(exception: Exception) -> float | None:
    """Parse a ``Retry-After`` header off an SDK exception, if present.

    Both the ``openai`` and ``google-genai`` SDKs attach the underlying HTTP
    response (with headers) to their exceptions as ``.response``.
    ``Retry-After`` may be either a number of seconds or an HTTP-date; we
    handle both and return the delay in seconds, or ``None`` when no usable
    header is present (including malformed or non-positive values).
    """
    # The header is an optional hint: a missing or failing response must not
    # turn a recognized rate limit into a classifier failure.
    headers: Any = _safe_attribute(_safe_attribute(exception, "response"), "headers")
    try:
        if not headers:
            return None
        milliseconds = headers.get("retry-after-ms") or headers.get("Retry-After-Ms")
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if milliseconds is not None:
        try:
            delay = float(milliseconds) / 1000
            return delay if math.isfinite(delay) and delay >= 0 else None
        except (TypeError, ValueError, OverflowError):
            return None
    if raw is None:
        return None
    try:
        delay = float(raw)
        return delay if math.isfinite(delay) and delay >= 0 else None
    except (TypeError, ValueError):
        pass
    # HTTP-date form: compute the delay relative to now.
    try:
        import time
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(raw)
        delay = when.timestamp() - time.time()
        return delay if math.isfinite(delay) and delay >= 0 else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


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


class MiddlewareContractError(ValueError):
    """A preprocessing result violated accepted work-item identity or invariants."""

    error_category = "middleware_contract_error"


class QuotaScopeError(ValueError):
    """A strategy quota scope could not be resolved safely before admission."""

    error_category = "quota_scope_error"


class TokenEstimationError(Exception):
    """Framework-owned, non-retryable token-estimation failure."""

    error_category = "token_estimation_error"


class TokenEstimatorRequired(TokenEstimationError):
    """TPM admission was enabled but no estimator resolved for an item."""

    error_category = "token_estimator_required"


class TokenEstimateExceedsLimit(TokenEstimationError):
    """One attempt's estimate cannot fit in the configured TPM bucket."""

    error_category = "token_estimate_exceeds_limit"


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

    Added in v0.16.0.
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

    Added in v0.16.0.
    """

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        provider_error: Any = None,
        token_usage: TokenUsage | dict[str, int] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.provider_error = provider_error
        if token_usage:
            self._failed_token_usage = dict(token_usage)


class StructuredOutputSchemaError(ValueError):
    """A provider rejected a requested structured-output schema.

    This is a deterministic request/schema compatibility failure, distinct
    from a model response that was accepted by the provider but failed local
    output validation. Classifiers do not retry it.
    """

    def __init__(
        self,
        message: str,
        *,
        token_usage: TokenUsage | dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        if token_usage:
            self._failed_token_usage = dict(token_usage)


class StructuredOutputValidationError(ValueError):
    """A provider returned output that failed local structured validation.

    Provider-enforced output should normally make this impossible, but a
    malformed or schema-invalid successful response is retryable because a
    subsequent model attempt may be usable.
    """


class FrameworkTimeoutError(TimeoutError):
    """
    Timeout enforced by the batch-llm framework (asyncio.wait_for).

    This distinguishes framework-level timeouts from API-level timeouts.
    Framework timeouts indicate the configured attempt_timeout was exceeded,
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


class ItemDeadlineExceeded(TimeoutError):
    """The end-to-end monotonic deadline for one logical item expired."""

    def __init__(self, message: str, *, item_id: str | None = None) -> None:
        super().__init__(message)
        self.item_id = item_id


class BatchDeadlineExceeded(TimeoutError):
    """The batch deadline stopped an accepted item before completion."""

    def __init__(self, message: str, *, item_id: str | None = None) -> None:
        super().__init__(message)
        self.item_id = item_id


class BatchAdmissionClosedError(RuntimeError):
    """The processor no longer accepts work; create a new processor to submit more."""


class BatchAbortedError(RuntimeError):
    """An accepted collateral item was stopped by configured fail-fast."""

    def __init__(self, message: str, *, item_id: str | None = None) -> None:
        super().__init__(message)
        self.item_id = item_id


class BatchBudgetExceeded(BatchAbortedError):
    """The run's token or cost budget stopped an accepted item before completion."""


# Terminal categories of items stopped by a controlled batch abort. The single
# source for stats, metrics, and artifact audit policy.
ABORT_RESULT_CATEGORIES = (
    "batch_aborted",
    "batch_deadline_exceeded",
    "batch_budget_exceeded",
)


class RateLimitRetriesExceeded(Exception):
    """A work item was retried after rate limits more than ``max_rate_limit_retries``.

    Rate-limit errors don't consume the ``max_attempts`` budget (they're a "wait
    and try again", not a failed attempt), but they ARE bounded separately by
    ``RetryConfig.max_rate_limit_retries`` so an endpoint that throttles forever
    can't hang the batch indefinitely. When that bound is exceeded the framework
    fails the item with this exception.

    Attributes:
        item_id: ID of the work item that exhausted its rate-limit retries.
        rate_limit_retries: How many rate-limit retries were attempted.
    """

    def __init__(
        self,
        message: str,
        *,
        item_id: str | None = None,
        rate_limit_retries: int | None = None,
    ):
        super().__init__(message)
        self.item_id = item_id
        self.rate_limit_retries = rate_limit_retries


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
            the backoff strategy may wait longer, subject to the configured maximum. ``None``
            means "no suggestion; use the strategy's value as-is".
        hint: Optional human-readable remediation hint for the operator
            (e.g. "top up your prepaid balance" for a 402). Surfaced in the
            logs when the error is non-retryable so a misconfiguration doesn't
            look like a generic API bug. ``None`` means no extra guidance.
    """

    is_retryable: bool
    is_rate_limit: bool
    is_timeout: bool
    error_category: str
    suggested_wait: float | None = None
    hint: str | None = None

    def __post_init__(self) -> None:
        if self.suggested_wait is not None:
            if not math.isfinite(self.suggested_wait):
                self.suggested_wait = None
            else:
                self.suggested_wait = max(0.0, self.suggested_wait)


class ErrorClassifier(ABC):
    """Abstract base class for classifying LLM provider errors."""

    def _framework_prelude(self, exception: Exception) -> ErrorInfo | None:
        """Classify framework-owned failures before provider/message dispatch."""
        if isinstance(exception, (MiddlewareContractError, QuotaScopeError)):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category=exception.error_category,
            )
        if isinstance(exception, ItemDeadlineExceeded):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=True,
                error_category="framework_total_item_timeout",
            )
        if isinstance(exception, BatchDeadlineExceeded):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=True,
                error_category="batch_deadline_exceeded",
            )
        if isinstance(exception, BatchBudgetExceeded):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="batch_budget_exceeded",
            )
        if isinstance(exception, BatchAbortedError):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="batch_aborted",
            )

        if isinstance(exception, StructuredOutputSchemaError):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="structured_output_schema_rejected",
            )

        if isinstance(exception, StructuredOutputValidationError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="structured_output_validation_error",
            )

        if isinstance(exception, FrameworkTimeoutError):
            return ErrorInfo(True, False, True, "framework_timeout")
        if isinstance(exception, RateLimitRetriesExceeded):
            return ErrorInfo(False, False, False, "rate_limit_retries_exceeded")
        if isinstance(exception, EmptyResponseError):
            return ErrorInfo(False, False, False, "empty_response")
        return None

    def _structured_rate_limit(self, exception: Exception) -> ErrorInfo | None:
        """Classify an explicit HTTP 429 carried by a non-dispatched exception."""
        if http_status(exception) != 429:
            return None
        if _insufficient_quota(exception):
            return _insufficient_quota_info()
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=True,
            is_timeout=False,
            error_category="rate_limit",
            suggested_wait=_retry_after_seconds(exception),
        )

    def _typed_failure(self, exception: Exception) -> ErrorInfo | None:
        """Classify validation and programming failures by type, before message text."""
        # Pydantic's ValidationError subclasses ValueError; the LLM may produce
        # valid output on retry, so it is checked before logic errors.
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

        # Deterministic errors that won't be fixed by retrying.
        if isinstance(exception, LOGIC_ERROR_TYPES):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="logic_error",
            )
        return None

    def _message_rate_limit(
        self, exception: Exception, patterns: tuple[str, ...] = RATE_LIMIT_PATTERNS
    ) -> ErrorInfo | None:
        """Heuristic for untyped exceptions: billing exhaustion, then rate-limit text.

        An exception that carries an HTTP status is never a rate limit by message:
        its status was already classified, so only status-less exceptions reach
        the rate-limit patterns.
        """
        error_str = str(exception)
        if _insufficient_quota(exception):
            return _insufficient_quota_info()
        if http_status(exception) is None and matches_any_pattern(error_str, patterns):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
            )
        return None

    def _generic_tail(self, exception: Exception) -> ErrorInfo:
        """Classify validation and deterministic programming failures."""
        info = self._typed_failure(exception)
        if info is not None:
            return info

        # Default: treat unknown generic exceptions as retryable
        # This allows custom transient errors and test mocks to work
        # Users with non-retryable custom errors should implement a custom ErrorClassifier
        return ErrorInfo(
            is_retryable=True,  # Retry unknown exceptions (might be transient)
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown",
        )

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


class DefaultErrorClassifier(ErrorClassifier):
    """Default error classifier that handles common error types."""

    def _matches_rate_limit(self, error_str: str) -> bool:
        """Return True if the error string looks like a rate limit.

        Numeric codes ("429") match on word boundaries via
        :func:`matches_any_pattern`, so "Expected 4290 tokens" is not
        misread as a rate limit.
        """
        return matches_any_pattern(error_str, RATE_LIMIT_PATTERNS)

    def classify(self, exception: Exception) -> ErrorInfo:
        """Classify common errors with conservative defaults.

        Precedence: framework errors, an HTTP 429 carried by the exception,
        validation and programming-error types, then message heuristics for
        untyped exceptions.
        """
        error_str = str(exception).lower()

        info = self._framework_prelude(exception)
        if info is not None:
            return info

        # PydanticAIStrategy uses this default classifier. Preserve immediate
        # validation retries by the exact optional SDK type, never its name.
        # ContentFilterError and IncompleteToolCall subclasses keep backoff.
        try:
            from pydantic_ai.exceptions import UnexpectedModelBehavior

            if type(exception) is UnexpectedModelBehavior:
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=False,
                    is_timeout=False,
                    error_category="validation_error",
                )
        except ImportError:
            pass

        info = self._structured_rate_limit(exception) or self._typed_failure(exception)
        if info is not None:
            return info

        # Message heuristics apply only to exceptions not classified by type or
        # status above (e.g. SDKs without a built-in classifier, test doubles).
        if _insufficient_quota(exception):
            return _insufficient_quota_info()
        if http_status(exception) is None and self._matches_rate_limit(error_str):
            return ErrorInfo(
                is_retryable=True,  # Rate limits are retryable - framework handles cooldown
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
            )

        # Check for API timeout (retryable - might be transient)
        if isinstance(exception, (TimeoutError, asyncio.TimeoutError)) or "timeout" in error_str:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="api_timeout",
            )

        # Check for connection errors
        if isinstance(exception, ConnectionError) or "connection" in error_str:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="connection_error",
            )

        return self._generic_tail(exception)
