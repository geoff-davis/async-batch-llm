"""Rich error formatting for retryable / validation errors.

Extracted from ``parallel.py`` in v0.7.0. The logic here walks exception
chains to find Pydantic ``ValidationError`` instances and formats
field-level details (what the LLM got wrong, the raw response) so users
debugging flaky structured-output calls have something to read.

Pure — no ``self`` state. Callers inject the exception, item id, and
attempt number.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import TracebackType

    from ..strategies import ErrorInfo

logger = logging.getLogger(__name__)


# Truncation for error messages embedded in the multi-line validation log.
# Kept in sync with parallel.ERROR_MESSAGE_MAX_LENGTH.
_ERROR_MESSAGE_MAX_LENGTH = 200
_MAX_TRACKED_TERMINAL_ERRORS = 1000


def log_retryable_error(
    exception: Exception,
    item_id: str,
    attempt_number: int,
    failed_token_usage: dict[str, int],
) -> None:
    """Log a retryable error, escalating to full validation detail when applicable."""
    error_name = type(exception).__name__
    error_msg = str(exception)

    token_msg = ""
    if failed_token_usage:
        token_msg = f" ({failed_token_usage.get('total_tokens', 0)} tokens consumed)"

    is_validation_type = (
        "validation" in error_name.lower()
        or "unexpectedmodelbehavior" in error_name.lower()
        or "result validation" in error_msg.lower()
    )

    if is_validation_type:
        log_validation_error(exception, item_id, attempt_number, token_msg)
    else:
        logger.debug(
            f"Retryable {error_name} on attempt {attempt_number} for {item_id}: {error_msg[:300]}"
        )


def log_validation_error(
    exception: Exception,
    item_id: str,
    attempt_number: int,
    token_msg: str,
) -> None:
    """Log field-level validation errors and the raw LLM response."""
    error_name = type(exception).__name__
    error_msg = str(exception)

    raw_response, underlying_validation_error = _walk_chain_for_validation(exception)

    try:
        from pydantic import ValidationError

        if underlying_validation_error or isinstance(exception, ValidationError):
            validation_err = cast(ValidationError, underlying_validation_error or exception)
            error_details = []
            for err in validation_err.errors():
                field_path = " -> ".join(str(loc) for loc in err["loc"])
                error_details.append(
                    f"    Field: {field_path}\n"
                    f"      Type: {err['type']}\n"
                    f"      Message: {err['msg']}\n"
                    f"      Input: {str(err.get('input', 'N/A'))[:100]}"
                )

            log_msg = (
                f"[FAIL]Validation error on attempt {attempt_number} for {item_id}"
                f"{token_msg}:\n"
                f"  Error type: {error_name}\n"
                f"  Field-level errors:\n" + "\n".join(error_details)
            )
            if raw_response:
                log_msg += f"\n  Raw LLM response (first 1000 chars):\n{raw_response}"
            logger.error(log_msg)
            return

        # Not a Pydantic ValidationError — dump the exception chain.
        log_msg = (
            f"[FAIL]Validation error on attempt {attempt_number} for {item_id}"
            f"{token_msg}:\n"
            f"  Error type: {error_name}\n"
            f"  Full error message: {error_msg}\n"
            f"  Exception chain:"
        )
        current: BaseException | None = exception
        depth = 0
        while current and depth < 5:
            log_msg += (
                f"\n    {depth}: {type(current).__name__}: "
                f"{str(current)[:_ERROR_MESSAGE_MAX_LENGTH]}"
            )
            next_cause = getattr(current, "__cause__", None)
            if next_cause is None or not isinstance(next_cause, BaseException):
                break
            current = next_cause
            depth += 1
        if raw_response:
            log_msg += f"\n  Raw LLM response (first 1000 chars):\n{raw_response}"
        logger.error(log_msg)

    except Exception as parse_error:
        # Fallback if we can't parse the error at all.
        log_msg = (
            f"[FAIL]Validation error on attempt {attempt_number} for {item_id}"
            f"{token_msg}:\n"
            f"  Error type: {error_name}\n"
            f"  Full error: {error_msg}\n"
            f"  (Failed to parse error details: {parse_error})"
        )
        if raw_response:
            log_msg += f"\n  Raw LLM response (first 1000 chars):\n{raw_response}"
        logger.error(log_msg)


def _walk_chain_for_validation(
    exception: BaseException,
) -> tuple[str | None, object | None]:
    """Walk ``__cause__`` looking for a Pydantic ``ValidationError`` and the
    raw LLM response. Returns ``(raw_response, validation_error)``; either
    may be ``None``."""
    from pydantic import ValidationError

    raw_response: str | None = None
    found_error: object | None = None

    exc: BaseException | None = exception
    depth = 0
    try:
        while exc and depth < 10:
            if hasattr(exc, "response"):
                raw_response = str(exc.response)[:1000]
            if hasattr(exc, "messages"):
                try:
                    raw_response = str(exc.messages)[:1000]
                except Exception:
                    pass

            if isinstance(exc, ValidationError):
                found_error = exc
                break

            next_cause = getattr(exc, "__cause__", None)
            if next_cause is None or not isinstance(next_cause, BaseException):
                break
            exc = next_cause
            depth += 1
    except Exception:
        pass

    return raw_response, found_error


def terminal_traceback(
    exception: Exception, error_info: ErrorInfo
) -> tuple[type[Exception], Exception, TracebackType | None] | None:
    """Select diagnostic stacks without making routine provider failures noisy."""
    category = error_info.error_category or "unknown"
    diagnostic = category in {"logic_error", "classifier_error", "unknown"}
    sdk_error = any(
        cls.__module__.split(".")[0] in {"openai", "google", "pydantic_ai", "anthropic"}
        for cls in type(exception).__mro__
    )
    framework_error = type(exception).__module__.startswith("async_batch_llm.")
    if diagnostic or (not error_info.is_retryable and not sdk_error and not framework_error):
        return type(exception), exception, exception.__traceback__
    return None


class TerminalFailureLogs:
    """Bounded processor-local counts; retain no exception instances or frames."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str | None, type[Exception], str], int] = {}
        self._summarized = False

    def log(
        self, target: logging.Logger, exception: Exception, info: ErrorInfo, message: str
    ) -> None:
        # Match the short result-message limit, rather than retaining arbitrary
        # provider payloads. Once full, keep new failures visible without growing
        # state; already tracked keys can still be counted and deduplicated.
        key = (info.error_category, type(exception), str(exception)[:_ERROR_MESSAGE_MAX_LENGTH])
        previous = self._counts.get(key)
        if previous is None and len(self._counts) >= _MAX_TRACKED_TERMINAL_ERRORS:
            target.error(message, exc_info=terminal_traceback(exception, info))
            return
        count = (previous or 0) + 1
        self._counts[key] = count
        if count == 1:
            target.error(message, exc_info=terminal_traceback(exception, info))
        else:
            target.debug(message)

    def summarize(self, target: logging.Logger) -> None:
        if self._summarized:
            return
        self._summarized = True
        for (category, error_type, message), count in self._counts.items():
            if count > 1:
                target.error(
                    "%d items failed with the same error: %s: %s [%s]",
                    count,
                    error_type.__name__,
                    message,
                    category,
                )
