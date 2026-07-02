"""Tests for the rich error-formatting helpers in ``_internal/error_logging.py``.

These are the diagnostics users read when structured-output calls fail, so
the tests assert on the substance of the log records (error type names,
field-level details, raw-response snippets) and — critically — that a
malformed or hostile exception never raises out of the logging helpers.
"""

import logging

import pytest
from pydantic import BaseModel, ValidationError

from async_batch_llm._internal import error_logging
from async_batch_llm._internal.error_logging import (
    _walk_chain_for_validation,
    log_retryable_error,
    log_validation_error,
)

LOGGER_NAME = "async_batch_llm._internal.error_logging"


class _Schema(BaseModel):
    """Schema used to manufacture genuine pydantic ValidationErrors."""

    name: str
    count: int


def _make_validation_error() -> ValidationError:
    """Produce a real ValidationError with two field-level errors."""
    with pytest.raises(ValidationError) as exc_info:
        _Schema.model_validate({"count": "not-a-number"})
    return exc_info.value


class _ResponseCarryingError(Exception):
    """Mimics SDK errors that attach the raw HTTP/LLM response."""

    def __init__(self, message: str, response: object):
        super().__init__(message)
        self.response = response


class _MessagesCarryingError(Exception):
    """Mimics pydantic-ai style errors that attach the message history."""

    def __init__(self, message: str, messages: object):
        super().__init__(message)
        self.messages = messages


# =============================================================================
# log_retryable_error
# =============================================================================


def test_retryable_error_non_validation_logs_debug(caplog):
    """Plain transient errors get a one-line DEBUG record with type + message."""
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        log_retryable_error(
            RuntimeError("connection reset by peer"),
            item_id="item_42",
            attempt_number=2,
            failed_token_usage={},
        )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.DEBUG
    message = record.getMessage()
    assert "Retryable RuntimeError" in message
    assert "attempt 2" in message
    assert "item_42" in message
    assert "connection reset by peer" in message


def test_retryable_error_escalates_pydantic_validation_error(caplog):
    """A pydantic ValidationError escalates to the rich field-level ERROR dump,
    including the tokens consumed by the failed attempt."""
    exc = _make_validation_error()

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        log_retryable_error(
            exc,
            item_id="item_val",
            attempt_number=1,
            failed_token_usage={"total_tokens": 123},
        )

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "item_val" in message
    assert "attempt 1" in message
    assert "(123 tokens consumed)" in message
    assert "Error type: ValidationError" in message
    assert "Field-level errors:" in message
    # Both failing fields surface with their pydantic error types.
    assert "Field: name" in message
    assert "Field: count" in message
    assert "missing" in message
    assert "int_parsing" in message
    assert "not-a-number" in message


def test_retryable_error_escalates_on_message_pattern(caplog):
    """'result validation' in the message escalates even for generic types."""
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        log_retryable_error(
            RuntimeError("Result validation failed for output"),
            item_id="item_msg",
            attempt_number=3,
            failed_token_usage={},
        )

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "Error type: RuntimeError" in errors[0].getMessage()


# =============================================================================
# log_validation_error — non-Pydantic exception-chain dump
# =============================================================================


def test_validation_error_dumps_exception_chain(caplog):
    """Non-Pydantic validation-ish errors dump the full __cause__ chain so
    users can see the root cause, not just the wrapper."""
    root = ConnectionError("socket closed by remote host")
    mid = RuntimeError("failed to parse model output")
    mid.__cause__ = root
    top = Exception("result validation wrapper")
    top.__cause__ = mid

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        log_validation_error(top, item_id="item_chain", attempt_number=2, token_msg="")

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "item_chain" in message
    assert "Exception chain:" in message
    # Every link in the chain appears with its type name and message.
    assert "0: Exception: result validation wrapper" in message
    assert "1: RuntimeError: failed to parse model output" in message
    assert "2: ConnectionError: socket closed by remote host" in message


def test_validation_error_chain_includes_raw_response(caplog):
    """When a link in the chain carries `.response`, the raw LLM response
    snippet is appended to the chain dump."""
    cause = _ResponseCarryingError("upstream said no", response='{"value": "truncated json')
    top = Exception("validation wrapper")
    top.__cause__ = cause

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        log_validation_error(top, item_id="item_raw", attempt_number=1, token_msg="")

    message = caplog.records[0].getMessage()
    assert "Raw LLM response (first 1000 chars):" in message
    assert '{"value": "truncated json' in message


# =============================================================================
# log_validation_error — parse-failure fallback
# =============================================================================


def test_parse_failure_fallback_when_errors_call_raises(monkeypatch, caplog):
    """If the found validation error's .errors() exists but raises, the
    formatter must not blow up — the fallback record fires and still carries
    the error type, message, and raw response."""

    class _BrokenValidationError:
        def errors(self):
            raise RuntimeError("errors() exploded")

    monkeypatch.setattr(
        error_logging,
        "_walk_chain_for_validation",
        lambda exception: ("raw response snippet", _BrokenValidationError()),
    )

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        # Must not raise.
        log_validation_error(
            Exception("output failed validation"),
            item_id="item_broken",
            attempt_number=4,
            token_msg=" (55 tokens consumed)",
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "item_broken" in message
    assert "attempt 4" in message
    assert "(55 tokens consumed)" in message
    assert "Failed to parse error details" in message
    assert "errors() exploded" in message
    assert "output failed validation" in message
    assert "raw response snippet" in message


def test_hostile_cause_with_raising_str_does_not_escape(caplog):
    """A __cause__ whose __str__ raises must not propagate out of the chain
    dump; the fallback path logs instead."""

    class _HostileStr(Exception):
        def __str__(self):
            raise RuntimeError("__str__ exploded")

    top = Exception("result validation failed")
    top.__cause__ = _HostileStr()

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        # Must not raise.
        log_validation_error(top, item_id="item_hostile", attempt_number=1, token_msg="")

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "item_hostile" in message
    assert "Failed to parse error details" in message


# =============================================================================
# _walk_chain_for_validation
# =============================================================================


def test_walk_chain_extracts_response_from_cause():
    """A `.response` attribute on a chained cause is extracted as the raw
    response (stringified, truncated to 1000 chars)."""
    cause = _ResponseCarryingError("bad output", response="R" * 5000)
    top = Exception("wrapper")
    top.__cause__ = cause

    raw_response, validation_error = _walk_chain_for_validation(top)

    assert raw_response == "R" * 1000
    assert validation_error is None


def test_walk_chain_extracts_messages_from_cause():
    """A `.messages` attribute (pydantic-ai style) is also usable as the raw
    response source."""
    cause = _MessagesCarryingError("bad output", messages=["msg-one", "msg-two"])
    top = Exception("wrapper")
    top.__cause__ = cause

    raw_response, validation_error = _walk_chain_for_validation(top)

    assert raw_response is not None
    assert "msg-one" in raw_response
    assert validation_error is None


def test_walk_chain_finds_deeply_nested_validation_error():
    """The walk keeps following __cause__ until it hits the ValidationError,
    picking up the raw response from an intermediate link on the way."""
    validation_error = _make_validation_error()
    mid = _ResponseCarryingError("parse failed", response='{"partial": ')
    mid.__cause__ = validation_error
    top = Exception("wrapper")
    top.__cause__ = mid

    raw_response, found = _walk_chain_for_validation(top)

    assert found is validation_error
    assert raw_response == '{"partial": '


def test_walk_chain_survives_raising_attributes():
    """Hostile attributes (a `response` property that raises, a `messages`
    whose __str__ raises) must never escape the walk."""

    class _RaisingResponseError(Exception):
        @property
        def response(self):
            raise RuntimeError("response property exploded")

    raw_response, found = _walk_chain_for_validation(_RaisingResponseError("hostile"))
    assert raw_response is None
    assert found is None

    class _UnprintableMessages:
        def __str__(self):
            raise RuntimeError("unprintable")

    exc = _MessagesCarryingError("bad", messages=_UnprintableMessages())
    raw_response, found = _walk_chain_for_validation(exc)
    assert raw_response is None
    assert found is None


def test_walk_chain_nested_validation_error_reaches_rich_log(caplog):
    """End-to-end: a ValidationError buried under two wrappers still produces
    the field-level dump plus the raw response from the chain."""
    validation_error = _make_validation_error()
    mid = _ResponseCarryingError("parse failed", response="the raw model output")
    mid.__cause__ = validation_error
    top = Exception("result validation failed")
    top.__cause__ = mid

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        log_validation_error(top, item_id="item_nested", attempt_number=1, token_msg="")

    message = caplog.records[0].getMessage()
    assert "Field-level errors:" in message
    assert "Field: name" in message
    assert "Field: count" in message
    assert "Raw LLM response (first 1000 chars):" in message
    assert "the raw model output" in message
