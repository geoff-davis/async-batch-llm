"""Tests for error classifiers."""

import pytest
from pydantic import BaseModel

from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, PydanticAIStrategy
from async_batch_llm.classifiers.gemini import GeminiErrorClassifier
from async_batch_llm.core import RateLimitConfig, RetryConfig
from async_batch_llm.strategies.errors import DefaultErrorClassifier, FrameworkTimeoutError
from async_batch_llm.testing import MockAgent


class ClassifierTestOutput(BaseModel):
    """Structured output for classifier integration tests."""

    value: str


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
    assert info.is_retryable is True
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
async def test_gemini_classifier_rate_limit_retries_after_processor_cooldown():
    """Gemini rate limits should pause workers and then retry the item."""

    def mock_response(prompt: str) -> ClassifierTestOutput:
        return ClassifierTestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(
        response_factory=mock_response,
        latency=0.001,
        rate_limit_on_call=1,
    )
    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=1.0,
        retry=RetryConfig(max_attempts=2, initial_wait=0.001, max_wait=0.001, jitter=False),
        rate_limit=RateLimitConfig(
            cooldown_seconds=0.001,
            slow_start_items=0,
            slow_start_initial_delay=0.0,
            slow_start_final_delay=0.0,
            backoff_multiplier=1.0,
        ),
    )
    processor = ParallelBatchProcessor[str, ClassifierTestOutput, None](
        config=config,
        error_classifier=GeminiErrorClassifier(),
    )

    await processor.add_work(
        LLMWorkItem(
            item_id="gemini_rate_limit",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt="Test",
        )
    )

    result = await processor.process_all()

    assert result.succeeded == 1
    assert result.failed == 0
    assert mock_agent.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_retry_does_not_add_exponential_backoff():
    """Rate-limit retries should not add retry-loop exponential backoff on top of
    the coordinated cooldown already applied by _handle_rate_limit().
    """
    import time

    def mock_response(prompt: str) -> ClassifierTestOutput:
        return ClassifierTestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(
        response_factory=mock_response,
        latency=0.001,
        rate_limit_on_call=1,
    )

    # Cooldown is small; initial_wait is large. If the retry loop applies
    # exponential backoff on top of the cooldown, elapsed time approaches
    # initial_wait (0.5s). With the fix, total elapsed time stays close to
    # the cooldown itself.
    cooldown = 0.05
    initial_wait = 0.5
    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=2.0,
        retry=RetryConfig(
            max_attempts=2,
            initial_wait=initial_wait,
            max_wait=initial_wait,
            jitter=False,
        ),
        rate_limit=RateLimitConfig(
            cooldown_seconds=cooldown,
            slow_start_items=0,
            slow_start_initial_delay=0.0,
            slow_start_final_delay=0.0,
            backoff_multiplier=1.0,
        ),
    )
    processor = ParallelBatchProcessor[str, ClassifierTestOutput, None](
        config=config,
        error_classifier=GeminiErrorClassifier(),
    )

    await processor.add_work(
        LLMWorkItem(
            item_id="no_double_wait",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt="Test",
        )
    )

    start = time.monotonic()
    result = await processor.process_all()
    elapsed = time.monotonic() - start

    assert result.succeeded == 1
    assert result.failed == 0
    assert mock_agent.call_count == 2
    assert elapsed < initial_wait, (
        f"Rate-limit retry took {elapsed:.3f}s; expected well under {initial_wait}s "
        f"(retry loop is adding exponential backoff on top of coordinated cooldown)."
    )


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
