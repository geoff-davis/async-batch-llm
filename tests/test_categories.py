"""The documented category vocabulary matches what the library produces."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from async_batch_llm import (
    ErrorCategory,
    GuardrailConfig,
    LLMCallStrategy,
    ProcessorConfig,
    TimeoutCategory,
    process_prompts,
)

SRC = Path(__file__).parents[1] / "src" / "async_batch_llm"

# Categories the library assigns through a variable rather than a literal at the
# point of use (_internal/guardrails.py, item_executor.py, artifact_codec.py).
DYNAMIC_ERROR_CATEGORIES = {
    "batch_deadline_exceeded",
    "batch_budget_exceeded",
    "batch_aborted",
    "middleware_filtered",
    "artifact_serialization_error",
}


def _strings(node: ast.AST | None) -> set[str]:
    """String literals a category expression can produce (both branches of `a if c else b`)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _strings(node.body) | _strings(node.orelse)
    return set()


def _produced(attribute: str) -> set[str]:
    """Every literal the library assigns to ``attribute``.

    Covers ``ErrorInfo(...)`` (keyword or 4th positional argument), any
    ``attribute=...`` keyword argument, and assignments to a name or attribute
    called ``attribute`` (including class attributes on exceptions).
    """
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        if path.name == "categories.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == attribute:
                        found |= _strings(keyword.value)
                name = getattr(node.func, "id", getattr(node.func, "attr", None))
                if attribute == "error_category" and name == "ErrorInfo" and len(node.args) >= 4:
                    found |= _strings(node.args[3])
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(
                    getattr(target, "id", getattr(target, "attr", None)) == attribute
                    for target in targets
                ):
                    found |= _strings(node.value)
    return found


def _source_strings() -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        if path.name != "categories.py":
            found |= {
                node.value
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
    return found


def test_every_produced_error_category_is_documented_and_vice_versa():
    produced = _produced("error_category") | DYNAMIC_ERROR_CATEGORIES
    assert produced == {member.value for member in ErrorCategory}


def test_dynamic_categories_still_exist_in_the_source():
    assert DYNAMIC_ERROR_CATEGORIES <= _source_strings()


def test_every_produced_timeout_category_is_documented():
    assert _produced("timeout_category") == {member.value for member in TimeoutCategory}


def test_positional_error_info_categories_are_seen():
    # Regression guard for the scan itself: these are only ever passed positionally.
    assert {
        "framework_timeout",
        "rate_limit_retries_exceeded",
        "empty_response",
        "quota_exhausted",
        "usage_limit_exceeded",
    } <= _produced("error_category")


@pytest.mark.parametrize("member", [ErrorCategory.RATE_LIMIT, TimeoutCategory.ADMISSION_TIMEOUT])
def test_members_behave_as_their_string_values(member):
    value = member.value
    assert member == value and hash(member) == hash(value)
    assert value in frozenset({member}) and member in frozenset({value})
    assert str(member) == f"{member}" == value
    assert json.dumps(member) == json.dumps(value)
    assert type(member)(value) is member


@pytest.mark.asyncio
async def test_enum_members_drive_fail_fast():
    class Broken(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            raise TypeError("bug")

    batch = await process_prompts(
        Broken(),
        ["a"],
        config=ProcessorConfig(
            max_workers=1,
            guardrails=GuardrailConfig(abort_on_error_categories={ErrorCategory.LOGIC_ERROR}),
        ),
    )
    assert batch.results[0].error_category == ErrorCategory.LOGIC_ERROR
    assert batch.termination.kind == "fail_fast"
    assert batch.termination.error_category == "logic_error"


def _classified():
    from async_batch_llm import (
        DefaultErrorClassifier,
        EmptyResponseError,
        FrameworkTimeoutError,
        GeminiErrorClassifier,
        RateLimitRetriesExceeded,
    )

    cases = [
        (DefaultErrorClassifier(), FrameworkTimeoutError("x"), "framework_timeout"),
        (DefaultErrorClassifier(), RateLimitRetriesExceeded("x"), "rate_limit_retries_exceeded"),
        (DefaultErrorClassifier(), EmptyResponseError("x"), "empty_response"),
    ]
    genai_errors = pytest.importorskip("google.genai.errors")
    daily = {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaId": "GenerateRequestsPerDayPerProject"}],
    }
    cases.append(
        (
            GeminiErrorClassifier(),
            genai_errors.ClientError(429, {"error": {"message": "quota", "details": [daily]}}),
            "quota_exhausted",
        )
    )
    return cases


def test_classifier_outputs_are_enum_members():
    for classifier, exception, expected in _classified():
        category = classifier.classify(exception).error_category
        assert category == expected
        assert ErrorCategory(category).value == expected


def test_pydantic_ai_usage_limit_is_an_enum_member():
    exceptions = pytest.importorskip("pydantic_ai.exceptions")
    from async_batch_llm import PydanticAIErrorClassifier

    info = PydanticAIErrorClassifier().classify(exceptions.UsageLimitExceeded("limit"))
    assert info.error_category == ErrorCategory.USAGE_LIMIT_EXCEEDED
    assert not info.is_retryable
