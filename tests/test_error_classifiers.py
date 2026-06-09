"""Tests for error classifiers."""

import pytest
from pydantic import BaseModel

from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, PydanticAIStrategy
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.classifiers.gemini import GeminiErrorClassifier
from async_batch_llm.core import RateLimitConfig, RetryConfig
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.strategies.errors import DefaultErrorClassifier, FrameworkTimeoutError
from async_batch_llm.testing import MockAgent


class ClassifierTestOutput(BaseModel):
    """Structured output for classifier integration tests."""

    value: str


class _PersistentRateLimitStrategy(LLMCallStrategy[ClassifierTestOutput]):
    """Test helper: always raises a Gemini-shaped rate-limit error."""

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[ClassifierTestOutput, TokenUsage]:
        self.call_count += 1

        # Mirror the error shape MockAgent produces so GeminiErrorClassifier
        # recognizes it as a rate limit (matches on class name + message).
        class MockRateLimitError(Exception):
            pass

        error = MockRateLimitError("429 RESOURCE_EXHAUSTED")
        error.__class__.__name__ = "ClientError"
        raise error


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
async def test_gemini_classifier_server_errors_retryable():
    """5xx ServerErrors are transient and retryable (aligns with OpenAI 5xx)."""
    pytest.importorskip("google.genai.errors")
    from google.genai.errors import ServerError

    classifier = GeminiErrorClassifier()

    # 503 overload / UNAVAILABLE — a capacity signal, routed through the
    # coordinated cooldown (is_rate_limit=True) rather than per-item backoff,
    # so all workers pause + slow-start instead of hammering an overloaded model.
    err = ServerError(
        503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
    )
    info = classifier.classify(err)
    assert info.is_retryable is True
    assert info.is_rate_limit is True
    assert info.error_category == "server_overload"

    # 500 internal error — transient one-off, per-item retry (not a cooldown).
    info = classifier.classify(ServerError(500, {"error": {"code": 500, "message": "internal"}}))
    assert info.is_retryable is True
    assert info.is_rate_limit is False
    assert info.error_category == "server_error"

    # 504 deadline — retryable and flagged as a timeout.
    info = classifier.classify(
        ServerError(504, {"error": {"code": 504, "message": "deadline exceeded"}})
    )
    assert info.is_retryable is True
    assert info.is_timeout is True
    assert info.error_category == "server_timeout"


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
    # No server signal (Retry-After) available from a bare string match, so
    # suggested_wait stays None; the RateLimitStrategy owns the cooldown.
    assert info.suggested_wait is None

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
async def test_persistent_rate_limit_exhausts_max_attempts():
    """A rate limit that never clears must respect retry.max_attempts and
    record a permanent failure instead of looping forever.

    Regression test for the is_retryable: False -> True change: before the
    classifier flip, rate-limited items failed fast after one attempt; now
    they retry, so we need to ensure the retry loop still terminates.
    """
    max_attempts = 3
    strategy = _PersistentRateLimitStrategy()
    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=1.0,
        retry=RetryConfig(
            max_attempts=max_attempts,
            initial_wait=0.001,
            max_wait=0.001,
            jitter=False,
        ),
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
            item_id="persistent_rl",
            strategy=strategy,
            prompt="Test",
        )
    )

    result = await processor.process_all()

    assert result.succeeded == 0
    assert result.failed == 1
    assert strategy.call_count == max_attempts, (
        f"Expected exactly {max_attempts} attempts before giving up, got {strategy.call_count}."
    )
    failure = result.results[0]
    assert failure.success is False
    assert "429" in (failure.error or "") or "RESOURCE_EXHAUSTED" in (failure.error or "")


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


# =============================================================================
# OpenAI / OpenRouter classifier tests
# =============================================================================


def _make_openai_status_error(status_code: int, message: str = "boom"):
    """Construct an openai.APIStatusError with the given status code."""
    from openai import APIStatusError

    request = httpx_request_or_none()
    response = httpx_response_or_none(status_code, message)

    if request is None or response is None:
        # Fallback: build a minimal mock that satisfies isinstance() checks.
        from unittest.mock import MagicMock

        err = APIStatusError.__new__(APIStatusError)
        err.status_code = status_code
        err.response = MagicMock(status_code=status_code, text=message)
        err.message = message
        err.body = {}
        err.request_id = None
        Exception.__init__(err, message)
        return err

    return APIStatusError(message, response=response, body={})


def httpx_request_or_none():
    """Build an httpx.Request if httpx is available; else return None."""
    try:
        import httpx

        return httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    except Exception:
        return None


def httpx_response_or_none(status_code: int, text: str):
    """Build an httpx.Response if httpx is available; else return None."""
    try:
        import httpx

        return httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
            text=text,
        )
    except Exception:
        return None


class TestOpenAIErrorClassifier:
    """Tests for OpenAIErrorClassifier across all branches."""

    def test_framework_timeout(self):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()
        info = classifier.classify(FrameworkTimeoutError("framework timeout"))
        assert info.is_retryable is True
        assert info.is_timeout is True
        assert info.error_category == "framework_timeout"

    def test_rate_limit_error(self):
        from openai import RateLimitError

        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()

        # Construct without network requirements — RateLimitError has a strict
        # constructor; use __new__ to dodge it.
        err = RateLimitError.__new__(RateLimitError)
        err.status_code = 429
        err.message = "rate limited"
        Exception.__init__(err, "rate limited")

        info = classifier.classify(err)
        assert info.is_rate_limit is True
        assert info.is_retryable is True
        assert info.error_category == "rate_limit"
        # No Retry-After header on this hand-built error, so no server signal.
        assert info.suggested_wait is None

    def test_api_timeout_error(self):
        from openai import APITimeoutError

        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()

        err = APITimeoutError.__new__(APITimeoutError)
        Exception.__init__(err, "timed out")

        info = classifier.classify(err)
        assert info.is_timeout is True
        assert info.is_retryable is True
        assert info.error_category == "api_timeout"

    def test_api_connection_error(self):
        from openai import APIConnectionError

        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()

        err = APIConnectionError.__new__(APIConnectionError)
        Exception.__init__(err, "connection refused")

        info = classifier.classify(err)
        assert info.is_retryable is True
        assert info.is_rate_limit is False
        assert info.is_timeout is False
        assert info.error_category == "network_error"

    def test_status_429_is_rate_limit(self):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()
        info = classifier.classify(_make_openai_status_error(429, "too many"))
        assert info.is_rate_limit is True
        assert info.is_retryable is True

    def test_429_retry_after_header_seconds_used_as_suggested_wait(self):
        import httpx
        from openai import APIStatusError

        from async_batch_llm.classifiers import OpenAIErrorClassifier

        request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        response = httpx.Response(
            status_code=429, request=request, text="slow down", headers={"retry-after": "12"}
        )
        err = APIStatusError("slow down", response=response, body={})

        info = OpenAIErrorClassifier().classify(err)
        assert info.is_rate_limit is True
        # The server's Retry-After becomes the suggested_wait floor.
        assert info.suggested_wait == 12.0

    def test_429_without_retry_after_has_no_suggested_wait(self):
        import httpx
        from openai import APIStatusError

        from async_batch_llm.classifiers import OpenAIErrorClassifier

        request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        response = httpx.Response(status_code=429, request=request, text="slow down")
        err = APIStatusError("slow down", response=response, body={})

        info = OpenAIErrorClassifier().classify(err)
        # No Retry-After header → no server signal; cooldown left to the strategy.
        assert info.suggested_wait is None

    @pytest.mark.parametrize("status", [408, 425, 500, 502, 503, 504])
    def test_retryable_5xx_codes(self, status):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()
        info = classifier.classify(_make_openai_status_error(status))
        assert info.is_retryable is True
        assert info.error_category == "server_error"
        assert info.is_timeout is (status == 504)

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_non_retryable_4xx_codes(self, status):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()
        info = classifier.classify(_make_openai_status_error(status))
        assert info.is_retryable is False
        assert info.error_category == "client_error"

    def test_pydantic_validation_error_retryable(self):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()
        try:
            from pydantic import BaseModel as PBM
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("pydantic not installed")

        class _M(PBM):
            v: int

        try:
            _M(v="not int")
        except ValidationError as e:
            info = classifier.classify(e)
            assert info.is_retryable is True
            assert info.error_category == "validation_error"

    def test_logic_bugs_not_retryable(self):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()
        for err in [
            ValueError("x"),
            TypeError("y"),
            KeyError("k"),
            IndexError("i"),
        ]:
            info = classifier.classify(err)
            assert info.is_retryable is False
            assert info.error_category == "logic_error"

    def test_unknown_exception_retryable(self):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        class _Custom(Exception):
            pass

        classifier = OpenAIErrorClassifier()
        info = classifier.classify(_Custom("transient"))
        assert info.is_retryable is True
        assert info.error_category == "unknown"

    def test_string_pattern_rate_limit_fallback(self):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()
        info = classifier.classify(Exception("Too Many Requests"))
        assert info.is_rate_limit is True
        assert info.error_category == "rate_limit"

    def test_402_insufficient_balance_not_retryable(self):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()
        info = classifier.classify(_make_openai_status_error(402, "Insufficient Balance"))
        assert info.is_retryable is False
        assert info.is_rate_limit is False
        assert info.error_category == "insufficient_balance"
        assert info.hint is not None
        assert "balance" in info.hint.lower()

    def test_402_string_fallback_not_retryable(self):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()
        # No openai SDK exception type — message-only path (e.g. mocked errors).
        info = classifier.classify(Exception("Error code: 402 - Insufficient Balance"))
        assert info.is_retryable is False
        assert info.error_category == "insufficient_balance"
        assert info.hint is not None


class TestOpenRouterErrorClassifier:
    """OpenRouter-specific overrides; everything else inherits from OpenAI."""

    def test_no_provider_available_is_network_error(self):
        from async_batch_llm.classifiers import OpenRouterErrorClassifier

        classifier = OpenRouterErrorClassifier()
        err = _make_openai_status_error(
            502, "no_provider_available: every provider returned errors"
        )
        info = classifier.classify(err)
        assert info.is_retryable is True
        assert info.error_category == "network_error"

    def test_falls_back_to_openai_for_normal_502(self):
        from async_batch_llm.classifiers import OpenRouterErrorClassifier

        classifier = OpenRouterErrorClassifier()
        err = _make_openai_status_error(502, "Bad Gateway")
        info = classifier.classify(err)
        assert info.is_retryable is True
        # Without the no_provider_available marker, parent behavior applies.
        assert info.error_category == "server_error"

    def test_inherits_429_handling(self):
        from async_batch_llm.classifiers import OpenRouterErrorClassifier

        classifier = OpenRouterErrorClassifier()
        info = classifier.classify(_make_openai_status_error(429))
        assert info.is_rate_limit is True
        assert info.is_retryable is True

    def test_inherits_logic_bug_handling(self):
        from async_batch_llm.classifiers import OpenRouterErrorClassifier

        classifier = OpenRouterErrorClassifier()
        info = classifier.classify(ValueError("bad input"))
        assert info.is_retryable is False
        assert info.error_category == "logic_error"
