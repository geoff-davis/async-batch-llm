"""The documented error and timeout category vocabulary.

``WorkItemResult.error_category`` and ``AttemptTiming.timeout_category`` stay
plain strings. These enums name the values the library itself produces, so you
can refer to them without string literals, for example in
``GuardrailConfig(abort_on_error_categories=...)``. Members are ``str``
subclasses: they compare and hash equal to their values, and ``str()``,
f-strings and JSON give the plain value on every supported Python version.

A custom :class:`ErrorClassifier` may return categories that aren't listed here;
they work everywhere a category is accepted.
"""

from __future__ import annotations

from enum import Enum


class _StrValueEnum(str, Enum):
    # Format as the plain value on every Python version (3.11+ StrEnum behavior).
    def __str__(self) -> str:
        return str.__str__(self)

    def __format__(self, format_spec: str) -> str:
        return str.__format__(self, format_spec)


class ErrorCategory(_StrValueEnum):
    """Values of ``WorkItemResult.error_category`` produced by the library."""

    # Provider and transport (built-in classifiers)
    RATE_LIMIT = "rate_limit"
    """The provider rate-limited the request (HTTP 429 or equivalent). Retried after a
    coordinated cooldown."""
    INSUFFICIENT_BALANCE = "insufficient_balance"
    """Billing or credit exhaustion (for example OpenAI ``insufficient_quota``). Not retried."""
    QUOTA_EXHAUSTED = "quota_exhausted"
    """A Gemini 429 whose quota violations are all daily limits, so waiting minutes
    won't help. Not retried."""
    USAGE_LIMIT_EXCEEDED = "usage_limit_exceeded"
    """PydanticAI stopped the run at a configured usage limit. Not retried."""
    AUTHENTICATION = "authentication"
    """The provider rejected the credentials (HTTP 401). Not retried."""
    PERMISSION_DENIED = "permission_denied"
    """The credentials lack access to the model or resource (HTTP 403). Not retried."""
    CLIENT_ERROR = "client_error"
    """Another non-retryable 4xx rejection, such as a bad request or unknown model."""
    SERVER_ERROR = "server_error"
    """A retryable 5xx response from the provider."""
    SERVER_OVERLOAD = "server_overload"
    """The provider reported it is overloaded (Gemini 503). Retried."""
    SERVER_TIMEOUT = "server_timeout"
    """The provider timed out upstream (Gemini 504). Retried."""
    UPSTREAM_ERROR = "upstream_error"
    """OpenRouter could not reach or use an upstream provider. Retried."""
    API_ERROR = "api_error"
    """A provider error with no recognized status. Retried conservatively."""
    API_TIMEOUT = "api_timeout"
    """The SDK or transport timed out waiting for the provider. Retried."""
    TIMEOUT = "timeout"
    """A timeout reported by the Gemini SDK. Retried."""
    NETWORK_ERROR = "network_error"
    """A connection failure reported by the OpenAI-compatible SDK. Retried."""
    CONNECTION_ERROR = "connection_error"
    """A connection failure seen by the default classifier. Retried."""

    # Output and application errors
    VALIDATION_ERROR = "validation_error"
    """The response failed validation (for example a Pydantic ``ValidationError``).
    Retried, since the model may do better on another attempt."""
    STRUCTURED_OUTPUT_VALIDATION_ERROR = "structured_output_validation_error"
    """Structured output did not match the requested schema. Retried."""
    STRUCTURED_OUTPUT_SCHEMA_REJECTED = "structured_output_schema_rejected"
    """The provider rejected the requested schema itself. Not retried."""
    LOGIC_ERROR = "logic_error"
    """A programming error in application or strategy code (``ValueError``,
    ``TypeError``, ``KeyError`` and similar). Not retried."""
    EMPTY_RESPONSE = "empty_response"
    """The provider returned a billed response with no usable text. Not retried."""
    UNKNOWN = "unknown"
    """An exception no classifier recognized. Retried, in case it is transient."""
    CLASSIFIER_ERROR = "classifier_error"
    """The error classifier itself raised. Not retried."""

    # Framework guardrails and stops
    FRAMEWORK_TIMEOUT = "framework_timeout"
    """One attempt exceeded ``attempt_timeout``. Retried."""
    RATE_LIMIT_RETRIES_EXCEEDED = "rate_limit_retries_exceeded"
    """The item hit more rate limits than ``RetryConfig.max_rate_limit_retries``
    allows. Not retried."""
    FRAMEWORK_TOTAL_ITEM_TIMEOUT = "framework_total_item_timeout"
    """The item's end-to-end deadline (``total_timeout_per_item``) expired."""
    BATCH_DEADLINE_EXCEEDED = "batch_deadline_exceeded"
    """The run's ``batch_timeout`` expired before this item finished."""
    BATCH_ABORTED = "batch_aborted"
    """The run stopped (fail-fast or an explicit abort) before this item finished."""
    BATCH_BUDGET_EXCEEDED = "batch_budget_exceeded"
    """The run reached its token or cost budget before this item finished."""

    # Configuration failures (never replayed from artifacts)
    TOKEN_ESTIMATOR_REQUIRED = "token_estimator_required"
    """TPM admission is enabled but no token estimator applies to the item."""
    TOKEN_ESTIMATION_ERROR = "token_estimation_error"
    """The token estimator raised or returned an invalid estimate."""
    TOKEN_ESTIMATE_EXCEEDS_LIMIT = "token_estimate_exceeds_limit"
    """One attempt's estimate can never fit in the TPM limit."""
    QUOTA_SCOPE_ERROR = "quota_scope_error"
    """A strategy's ``quota_scope`` could not be resolved."""

    # Middleware
    MIDDLEWARE_CONTRACT_ERROR = "middleware_contract_error"
    """Middleware returned a value that breaks its contract."""
    MIDDLEWARE_FILTERED = "middleware_filtered"
    """Middleware filtered the item out, so it was not executed. Never replayed."""

    # Artifacts
    ARTIFACT_PREPARATION_ERROR = "artifact_preparation_error"
    """The item's input could not be prepared for the artifact store."""
    ARTIFACT_SERIALIZATION_ERROR = "artifact_serialization_error"
    """The item's result could not be serialized for the artifact store."""


class TimeoutCategory(_StrValueEnum):
    """Values of ``AttemptTiming.timeout_category`` and ``WorkItemTiming.timeout_category``."""

    FRAMEWORK_EXECUTION_TIMEOUT = "framework_execution_timeout"
    """One provider attempt exceeded ``attempt_timeout``."""
    FRAMEWORK_TOTAL_ITEM_TIMEOUT = "framework_total_item_timeout"
    """The item deadline expired during a provider call."""
    PROVIDER_OR_TRANSPORT_TIMEOUT = "provider_or_transport_timeout"
    """The provider SDK or transport raised its own timeout."""
    ADMISSION_TIMEOUT = "admission_timeout"
    """An item or batch deadline expired while the item waited for provider capacity."""


__all__ = ["ErrorCategory", "TimeoutCategory"]
