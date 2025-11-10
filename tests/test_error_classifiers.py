"""Tests for error classifiers."""

import pytest

from batch_llm.classifiers.gemini import GeminiErrorClassifier
from batch_llm.strategies.errors import DefaultErrorClassifier, FrameworkTimeoutError


@pytest.mark.asyncio
async def test_gemini_classifier_framework_timeout():
    """Test GeminiErrorClassifier detects framework timeouts."""
    classifier = GeminiErrorClassifier()

    error = FrameworkTimeoutError("Framework timeout after 120s")
    info = classifier.classify(error)

    assert info.is_timeout is True
    assert info.is_rate_limit is False
    assert info.is_retryable is True
    assert info.error_category == "framework_timeout"


@pytest.mark.asyncio
async def test_gemini_classifier_logic_bugs_not_retryable():
    """Test GeminiErrorClassifier marks logic bugs as non-retryable."""
    classifier = GeminiErrorClassifier()

    # Test ValueError
    error = ValueError("Invalid input format")
    info = classifier.classify(error)
    assert info.is_retryable is False
    assert info.error_category == "logic_error"

    # Test TypeError
    error = TypeError("Expected str, got int")
    info = classifier.classify(error)
    assert info.is_retryable is False
    assert info.error_category == "logic_error"

    # Test AttributeError
    error = AttributeError("'NoneType' object has no attribute 'foo'")
    info = classifier.classify(error)
    assert info.is_retryable is False
    assert info.error_category == "logic_error"

    # Test KeyError
    error = KeyError("missing_key")
    info = classifier.classify(error)
    assert info.is_retryable is False
    assert info.error_category == "logic_error"

    # Test IndexError
    error = IndexError("list index out of range")
    info = classifier.classify(error)
    assert info.is_retryable is False
    assert info.error_category == "logic_error"

    # Test NameError
    error = NameError("name 'undefined_var' is not defined")
    info = classifier.classify(error)
    assert info.is_retryable is False
    assert info.error_category == "logic_error"

    # Test ZeroDivisionError
    error = ZeroDivisionError("division by zero")
    info = classifier.classify(error)
    assert info.is_retryable is False
    assert info.error_category == "logic_error"

    # Test AssertionError
    error = AssertionError("Assertion failed")
    info = classifier.classify(error)
    assert info.is_retryable is False
    assert info.error_category == "logic_error"


@pytest.mark.asyncio
async def test_gemini_classifier_rate_limit_patterns():
    """Test GeminiErrorClassifier detects rate limit patterns."""
    classifier = GeminiErrorClassifier()

    # Test "429" pattern
    error = Exception("429 RESOURCE_EXHAUSTED")
    info = classifier.classify(error)
    assert info.is_rate_limit is True
    assert info.is_retryable is False
    assert info.error_category == "rate_limit"
    assert info.suggested_wait == 300.0

    # Test "resource_exhausted" pattern
    error = Exception("API quota exceeded: RESOURCE_EXHAUSTED")
    info = classifier.classify(error)
    assert info.is_rate_limit is True
    assert info.error_category == "rate_limit"

    # Test "quota" pattern
    error = Exception("Quota exceeded for this request")
    info = classifier.classify(error)
    assert info.is_rate_limit is True
    assert info.error_category == "rate_limit"

    # Test "rate limit" pattern
    error = Exception("Rate limit exceeded")
    info = classifier.classify(error)
    assert info.is_rate_limit is True
    assert info.error_category == "rate_limit"


@pytest.mark.asyncio
async def test_gemini_classifier_timeout_patterns():
    """Test GeminiErrorClassifier detects timeout patterns."""
    classifier = GeminiErrorClassifier()

    # Test "timeout" pattern
    error = Exception("Request timeout after 30s")
    info = classifier.classify(error)
    assert info.is_timeout is True
    assert info.is_retryable is True
    assert info.error_category == "timeout"

    # Test "504" pattern
    error = Exception("504 Gateway Timeout")
    info = classifier.classify(error)
    assert info.is_timeout is True
    assert info.is_retryable is True
    assert info.error_category == "timeout"

    # Test "deadline" pattern
    error = Exception("Deadline exceeded")
    info = classifier.classify(error)
    assert info.is_timeout is True
    assert info.is_retryable is True
    assert info.error_category == "timeout"


@pytest.mark.asyncio
async def test_gemini_classifier_pydantic_validation_error():
    """Test GeminiErrorClassifier marks Pydantic ValidationError as retryable."""
    classifier = GeminiErrorClassifier()

    try:
        from pydantic import BaseModel, ValidationError

        class TestModel(BaseModel):
            value: int

        try:
            TestModel(value="not_an_int")
        except ValidationError as e:
            info = classifier.classify(e)
            assert info.is_retryable is True
            assert info.error_category == "validation_error"
            assert info.is_rate_limit is False
            assert info.is_timeout is False
    except ImportError:
        pytest.skip("Pydantic not installed")


@pytest.mark.asyncio
async def test_gemini_classifier_pydantic_ai_validation_error():
    """Test GeminiErrorClassifier marks PydanticAI UnexpectedModelBehavior as retryable."""
    classifier = GeminiErrorClassifier()

    try:
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        error = UnexpectedModelBehavior("Model output validation failed")
        info = classifier.classify(error)
        assert info.is_retryable is True
        assert info.error_category == "validation_error"
        assert info.is_rate_limit is False
        assert info.is_timeout is False
    except ImportError:
        pytest.skip("PydanticAI not installed")


@pytest.mark.asyncio
async def test_gemini_classifier_unknown_exception_retryable():
    """Test GeminiErrorClassifier marks unknown exceptions as retryable."""
    classifier = GeminiErrorClassifier()

    # Custom exception should be treated as retryable
    class CustomException(Exception):
        pass

    error = CustomException("Something went wrong")
    info = classifier.classify(error)
    assert info.is_retryable is True
    assert info.error_category == "unknown"
    assert info.is_rate_limit is False
    assert info.is_timeout is False


@pytest.mark.asyncio
async def test_gemini_classifier_without_google_genai():
    """Test GeminiErrorClassifier falls back gracefully without google-genai."""
    classifier = GeminiErrorClassifier()

    # Generic exception when google.genai not available
    # (The classifier handles ImportError internally)
    error = Exception("Some error")
    info = classifier.classify(error)
    # Should fall back to checking error message patterns
    assert info.error_category in ["unknown", "rate_limit", "timeout"]


@pytest.mark.asyncio
async def test_default_classifier_logic_bugs():
    """Test DefaultErrorClassifier marks logic bugs as non-retryable."""
    classifier = DefaultErrorClassifier()

    # Logic bugs should not be retryable
    logic_bugs = [
        ValueError("test"),
        KeyError("test"),
        TypeError("test"),
    ]

    for error in logic_bugs:
        info = classifier.classify(error)
        assert info.is_retryable is False
        assert info.error_category == "logic_error"

    # But other exceptions should be retryable
    retryable_errors = [
        RuntimeError("test"),
        Exception("test"),
        ConnectionError("test"),
    ]

    for error in retryable_errors:
        info = classifier.classify(error)
        assert info.is_retryable is True


@pytest.mark.asyncio
async def test_gemini_classifier_case_insensitive():
    """Test GeminiErrorClassifier pattern matching is case-insensitive."""
    classifier = GeminiErrorClassifier()

    # Test uppercase rate limit pattern
    error = Exception("RATE LIMIT EXCEEDED")
    info = classifier.classify(error)
    assert info.is_rate_limit is True

    # Test mixed case timeout pattern
    error = Exception("Request TIMEOUT")
    info = classifier.classify(error)
    assert info.is_timeout is True

    # Test lowercase quota pattern
    error = Exception("quota exceeded")
    info = classifier.classify(error)
    assert info.is_rate_limit is True
