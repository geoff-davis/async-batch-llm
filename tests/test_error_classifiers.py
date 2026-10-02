"""Tests for error classifiers."""

import pytest
from pydantic import BaseModel

from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, PydanticAIStrategy
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.classifiers.gemini import GeminiErrorClassifier
from async_batch_llm.classifiers.openai import OpenAIErrorClassifier
from async_batch_llm.classifiers.openrouter import OpenRouterErrorClassifier
from async_batch_llm.classifiers.pydantic_ai import PydanticAIErrorClassifier
from async_batch_llm.core import RateLimitConfig, RetryConfig
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.strategies.errors import (
    DefaultErrorClassifier,
    FrameworkTimeoutError,
    ProviderResponseError,
)
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

    # 503 overload / UNAVAILABLE — a transient server-side capacity blip, retried
    # with per-item exponential backoff like any other 5xx (NOT a coordinated
    # cooldown, which is reserved for 429/quota). Matches OpenAIErrorClassifier.
    err = ServerError(
        503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
    )
    info = classifier.classify(err)
    assert info.is_retryable is True
    assert info.is_rate_limit is False
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
        attempt_timeout=1.0,
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
async def test_persistent_rate_limit_exhausts_rate_limit_budget():
    """A rate limit that never clears terminates via max_rate_limit_retries
    (NOT max_attempts) and records a permanent failure instead of looping forever.

    Rate-limit errors are exempt from the max_attempts budget — they're retried at
    the same logical attempt number — so a persistent throttle is bounded by the
    separate retry.max_rate_limit_retries backstop.
    """
    max_rate_limit_retries = 3
    strategy = _PersistentRateLimitStrategy()
    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=1.0,
        retry=RetryConfig(
            max_attempts=3,  # irrelevant for pure rate limits; budget never consumed
            initial_wait=0.001,
            max_wait=0.001,
            jitter=False,
            max_rate_limit_retries=max_rate_limit_retries,
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
    # max_rate_limit_retries retries + the call that trips the cap.
    assert strategy.call_count == max_rate_limit_retries + 1, (
        f"Expected {max_rate_limit_retries + 1} calls before giving up, got {strategy.call_count}."
    )
    failure = result.results[0]
    assert failure.success is False
    assert "RateLimitRetriesExceeded" in (failure.error or "")


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
        attempt_timeout=2.0,
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

    @pytest.mark.parametrize(
        ("status", "category"),
        [
            (400, "client_error"),
            (401, "authentication"),
            (403, "permission_denied"),
            (404, "client_error"),
            (422, "client_error"),
        ],
    )
    def test_non_retryable_4xx_codes(self, status, category):
        from async_batch_llm.classifiers import OpenAIErrorClassifier

        classifier = OpenAIErrorClassifier()
        info = classifier.classify(_make_openai_status_error(status))
        assert info.is_retryable is False
        assert info.error_category == category

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
        # The hint applies to every OpenAI-compatible provider, not only DeepSeek.
        assert info.hint.startswith("402 Payment Required — the provider account's balance")

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


class TestGeminiSDKErrorClassification:
    """GeminiErrorClassifier against real google.genai error instances.

    Regression tests for the status-code rewrite: classification dispatches
    on APIError.code instead of string-matching str(exception), so transient
    500/502 errors retry and deterministic 4xx errors fail fast.
    """

    @pytest.fixture(autouse=True)
    def _require_genai(self):
        pytest.importorskip("google.genai.errors")

    @staticmethod
    def _client_error(code: int, message: str, response=None):
        from google.genai.errors import ClientError

        return ClientError(code, {"error": {"message": message}}, response)

    @staticmethod
    def _server_error(code: int, message: str):
        from google.genai.errors import ServerError

        return ServerError(code, {"error": {"message": message}})

    def test_429_is_rate_limit(self):
        classifier = GeminiErrorClassifier()
        info = classifier.classify(self._client_error(429, "Resource has been exhausted"))
        assert info.is_rate_limit is True
        assert info.is_retryable is True
        assert info.error_category == "rate_limit"

    def test_429_parses_retry_after_as_suggested_wait(self):
        class FakeResponse:
            headers = {"retry-after": "7"}

        classifier = GeminiErrorClassifier()
        info = classifier.classify(
            self._client_error(429, "Resource has been exhausted", FakeResponse())
        )
        assert info.is_rate_limit is True
        assert info.suggested_wait == 7.0

    @pytest.mark.parametrize(
        ("code", "message", "category"),
        [
            (400, "Invalid request", "client_error"),
            (401, "API key not valid", "authentication"),
            (403, "Permission denied", "permission_denied"),
            (404, "Model not found", "client_error"),
        ],
    )
    def test_deterministic_client_errors_fail_fast(self, code, message, category):
        """4xx errors are deterministic; retrying an invalid API key on every
        item in the batch just multiplies latency."""
        classifier = GeminiErrorClassifier()
        info = classifier.classify(self._client_error(code, message))
        assert info.is_retryable is False
        assert info.error_category == category

    @pytest.mark.parametrize(
        ("code", "message"),
        [
            (500, "Internal error encountered"),
            (502, "Bad gateway"),
        ],
    )
    def test_transient_server_errors_retry(self, code, message):
        """500/502 are transient one-offs; they must retry even though their
        messages match no timeout pattern."""
        classifier = GeminiErrorClassifier()
        info = classifier.classify(self._server_error(code, message))
        assert info.is_retryable is True
        assert info.is_rate_limit is False
        assert info.error_category == "server_error"

    def test_503_is_server_overload(self):
        """503 keeps its dedicated category: per-item backoff, no coordinated
        cooldown (which is reserved for 429/quota)."""
        classifier = GeminiErrorClassifier()
        info = classifier.classify(
            self._server_error(503, "The model is overloaded. Please try again later.")
        )
        assert info.is_retryable is True
        assert info.is_rate_limit is False
        assert info.error_category == "server_overload"

    def test_504_is_retryable_timeout(self):
        classifier = GeminiErrorClassifier()
        info = classifier.classify(self._server_error(504, "Deadline exceeded"))
        assert info.is_retryable is True
        assert info.is_timeout is True
        assert info.error_category == "server_timeout"

    def test_unrecognized_status_retries_conservatively(self):
        classifier = GeminiErrorClassifier()
        info = classifier.classify(self._client_error(418, "I'm a teapot"))
        assert info.is_retryable is True
        assert info.error_category == "api_error"


def test_gemini_generic_fallback_still_runs_without_genai_sdk(monkeypatch):
    """Without the [gemini] extra, rate limits and logic bugs must still
    classify via the generic chain (previously the ImportError path
    returned unknown/retryable immediately, so cooldowns never engaged)."""
    import sys

    monkeypatch.setitem(sys.modules, "google.genai.errors", None)
    classifier = GeminiErrorClassifier()

    rate_info = classifier.classify(Exception("429 RESOURCE_EXHAUSTED: rate limit"))
    assert rate_info.is_rate_limit is True
    assert rate_info.is_retryable is True

    bug_info = classifier.classify(ValueError("deterministic parse bug"))
    assert bug_info.is_retryable is False
    assert bug_info.error_category == "logic_error"


def test_default_classifier_without_optional_pydantic_ai(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_pydantic_ai(name, *args, **kwargs):
        if name == "pydantic_ai.exceptions":
            raise ImportError("optional pydantic-ai is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pydantic_ai)
    info = DefaultErrorClassifier().classify(ValueError("invalid argument"))
    assert not info.is_retryable
    assert info.error_category == "logic_error"


@pytest.mark.parametrize("kind", ["default", "openai", "openrouter", "gemini", "pydantic_ai"])
@pytest.mark.parametrize(
    "case,category,retryable,timeout",
    [
        ("rate_budget", "rate_limit_retries_exceeded", False, False),
        ("empty", "empty_response", False, False),
        ("schema", "structured_output_schema_rejected", False, False),
        ("validation", "structured_output_validation_error", True, False),
        ("framework_timeout", "framework_timeout", True, True),
        ("bare_timeout", None, True, True),
    ],
)
def test_prov9_shared_classification(kind, case, category, retryable, timeout):
    from async_batch_llm.classifiers import OpenAIErrorClassifier, OpenRouterErrorClassifier
    from async_batch_llm.strategies.errors import (
        EmptyResponseError,
        RateLimitRetriesExceeded,
        StructuredOutputSchemaError,
        StructuredOutputValidationError,
    )

    classifiers = {
        "default": DefaultErrorClassifier(),
        "openai": OpenAIErrorClassifier(),
        "openrouter": OpenRouterErrorClassifier(),
        "gemini": GeminiErrorClassifier(),
        "pydantic_ai": PydanticAIStrategy(MockAgent()).recommended_error_classifier()
        or DefaultErrorClassifier(),
    }
    errors = {
        "rate_budget": RateLimitRetriesExceeded("quota retry budget exhausted"),
        "empty": EmptyResponseError("MAX_TOKENS"),
        "schema": StructuredOutputSchemaError("bad schema"),
        "validation": StructuredOutputValidationError("invalid output"),
        "framework_timeout": FrameworkTimeoutError("quota timeout"),
        "bare_timeout": TimeoutError(),
    }
    info = classifiers[kind].classify(errors[case])
    assert info.error_category == (category or ("timeout" if kind == "gemini" else "api_timeout"))
    assert info.is_retryable is retryable
    assert info.is_timeout is timeout
    assert not info.is_rate_limit


@pytest.mark.parametrize(
    "status,category,retryable",
    [
        (401, "authentication", False),
        (403, "permission_denied", False),
        (404, "client_error", False),
        (429, "rate_limit", True),
        (500, "server_error", True),
        (503, "server_error", True),
    ],
)
def test_prov8_pydantic_status_dispatch(status, category, retryable):
    from pydantic_ai.exceptions import ModelHTTPError

    classifier = (
        PydanticAIStrategy(MockAgent()).recommended_error_classifier() or DefaultErrorClassifier()
    )
    info = classifier.classify(ModelHTTPError(status, "test"))
    assert info.error_category == category
    assert info.is_retryable is retryable
    assert info.is_rate_limit is (status == 429)


def test_prov8_usage_limit_is_terminal():
    from pydantic_ai.exceptions import UsageLimitExceeded

    classifier = (
        PydanticAIStrategy(MockAgent()).recommended_error_classifier() or DefaultErrorClassifier()
    )
    info = classifier.classify(UsageLimitExceeded("limit"))
    assert not info.is_retryable
    assert info.error_category == "usage_limit_exceeded"


@pytest.mark.parametrize(
    "body",
    [
        {"code": "insufficient_quota"},
        {"error": {"code": "insufficient_quota"}},
    ],
)
def test_prov4_openai_quota_is_not_cooldown(body):
    import httpx2
    from openai import RateLimitError

    from async_batch_llm.classifiers import OpenAIErrorClassifier

    response = httpx2.Response(429, request=httpx2.Request("POST", "https://test.invalid"))
    info = OpenAIErrorClassifier().classify(RateLimitError("quota", response=response, body=body))
    assert info.error_category == "insufficient_balance"
    assert not info.is_retryable
    assert not info.is_rate_limit


@pytest.mark.parametrize(
    "ids,expected",
    [
        (["GenerateRequestsPerDayPerProject"], "quota_exhausted"),
        (["GenerateRequestsPerDayPerProject", "GenerateRequestsPerMinute"], "rate_limit"),
        (["unknown"], "rate_limit"),
    ],
)
def test_prov4_gemini_daily_quota(ids, expected):
    from google.genai.errors import ClientError

    exc = ClientError(
        429,
        {
            "error": {
                "message": "quota",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": value} for value in ids],
                    }
                ],
            }
        },
    )
    info = GeminiErrorClassifier().classify(exc)
    assert info.error_category == expected
    assert info.is_retryable is (expected == "rate_limit")


def test_prov5_gemini_retry_delay():
    from google.genai.errors import ClientError

    exc = ClientError(
        429,
        {
            "error": {
                "message": "rate limit",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "37s",
                    }
                ],
            }
        },
    )
    assert GeminiErrorClassifier().classify(exc).suggested_wait == 37


@pytest.mark.parametrize("delay,expected", [(float("inf"), None), (float("nan"), None), (-1, 0)])
def test_prov5_error_info_sanitizes_delay(delay, expected):
    from async_batch_llm.strategies.errors import ErrorInfo

    assert ErrorInfo(True, True, False, "rate_limit", delay).suggested_wait == expected


def test_prov5_retry_after_milliseconds_precedes_seconds():
    from types import SimpleNamespace

    from async_batch_llm.strategies.errors import _retry_after_seconds

    exc = Exception()
    exc.response = SimpleNamespace(headers={"retry-after-ms": "1250", "retry-after": "99"})
    assert _retry_after_seconds(exc) == 1.25


@pytest.mark.parametrize("status", [400, 403, 404, 429, 502, 503])
def test_prov9_openrouter_no_provider_only_transient_statuses(status):
    from async_batch_llm.classifiers import OpenRouterErrorClassifier
    from async_batch_llm.strategies.errors import ProviderResponseError

    info = OpenRouterErrorClassifier().classify(
        ProviderResponseError("No allowed providers", code=status)
    )
    assert info.is_retryable is (status in (502, 503))
    assert info.error_category == ("network_error" if status in (502, 503) else "client_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["batch", "stream", "single", "pool"])
@pytest.mark.parametrize("provider", ["openai", "pydantic_flat", "pydantic_nested"])
async def test_prov4_quota_stops_after_one_provider_call(surface, monkeypatch, provider):
    import httpx2
    from openai import RateLimitError

    from async_batch_llm import LLMCallPool, call_result, process_prompts, process_stream
    from async_batch_llm._internal.rate_limit_coordinator import RateLimitCoordinator
    from async_batch_llm.classifiers import OpenAIErrorClassifier

    cooldown_calls = 0
    original = RateLimitCoordinator.handle_rate_limit

    async def record_cooldown(self, *args, **kwargs):
        nonlocal cooldown_calls
        cooldown_calls += 1
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(RateLimitCoordinator, "handle_rate_limit", record_cooldown)

    class QuotaFailure(LLMCallStrategy[str]):
        calls = 0

        def recommended_error_classifier(self):
            if provider != "openai":
                from async_batch_llm.classifiers import PydanticAIErrorClassifier

                return PydanticAIErrorClassifier()
            return OpenAIErrorClassifier()

        async def execute(self, prompt, attempt, timeout, state=None):
            self.calls += 1
            if provider != "openai":
                from pydantic_ai.exceptions import ModelHTTPError

                body = {"code": "insufficient_quota"}
                if provider == "pydantic_nested":
                    body = {"error": body}
                raise ModelHTTPError(429, "test", body=body)
            response = httpx2.Response(429, request=httpx2.Request("POST", "https://test.invalid"))
            raise RateLimitError("quota", response=response, body={"code": "insufficient_quota"})

    strategy = QuotaFailure()
    config = ProcessorConfig(
        retry=RetryConfig(max_attempts=2, max_rate_limit_retries=1, initial_wait=0.001),
        rate_limit=RateLimitConfig(
            cooldown_seconds=0.001, max_cooldown_seconds=0.01, slow_start_items=0
        ),
    )
    if surface == "batch":
        result = (await process_prompts(strategy, ["q"], config=config)).results[0]
    elif surface == "stream":
        result = [r async for r in process_stream(strategy, ["q"], config=config)][0]
    elif surface == "single":
        result = await call_result(strategy, "q", config=config)
    else:
        async with LLMCallPool(strategy, config=config) as pool:
            result = await pool.submit_result("q")
    assert strategy.calls == 1
    assert result.error_category == "insufficient_balance"
    assert cooldown_calls == 0


@pytest.mark.parametrize("nested", [False, True])
def test_sc4_pydantic_quota_exhaustion(nested):
    from pydantic_ai.exceptions import ModelHTTPError

    from async_batch_llm.classifiers import PydanticAIErrorClassifier

    body = {"code": "insufficient_quota"}
    if nested:
        body = {"error": body}
    info = PydanticAIErrorClassifier().classify(ModelHTTPError(429, "test", body=body))
    assert info.error_category == "insufficient_balance"
    assert not info.is_retryable
    assert not info.is_rate_limit


@pytest.mark.parametrize("classifier_name", ["openai", "openrouter"])
@pytest.mark.parametrize("status", [None, "429", "mock", 429])
@pytest.mark.parametrize("quota", [False, True])
def test_sc5_rate_limit_type_fallback(classifier_name, status, quota):
    from unittest.mock import MagicMock

    from openai import RateLimitError

    from async_batch_llm.classifiers.openai import OpenAIErrorClassifier
    from async_batch_llm.classifiers.openrouter import OpenRouterErrorClassifier

    response = MagicMock()
    response.headers = {"retry-after": "2"}
    if status != "mock":
        response.status_code = status
    error = RateLimitError(
        "limited",
        response=response,
        body={"error": {"code": "insufficient_quota"}} if quota else None,
    )
    original_status = error.status_code
    classifier = (
        OpenAIErrorClassifier() if classifier_name == "openai" else OpenRouterErrorClassifier()
    )
    info = classifier.classify(error)
    assert info.error_category == ("insufficient_balance" if quota else "rate_limit")
    assert info.is_rate_limit is (not quota)
    assert info.is_retryable is (not quota)
    assert info.suggested_wait == (None if quota else 2)
    assert error.status_code is original_status


def test_sc5_unknown_status_does_not_imply_rate_limit():
    from unittest.mock import MagicMock

    from openai import APIStatusError

    from async_batch_llm.classifiers.openai import OpenAIErrorClassifier

    error = APIStatusError("unknown", response=MagicMock(), body=None)
    info = OpenAIErrorClassifier().classify(error)
    assert info.error_category == "api_error"
    assert not info.is_rate_limit


# --- Issue #177: typed and structured signals outrank message text ---------

_ISSUE_177_CLASSIFIERS = [
    pytest.param(DefaultErrorClassifier, id="default"),
    pytest.param(OpenAIErrorClassifier, id="openai"),
    pytest.param(OpenRouterErrorClassifier, id="openrouter"),
    pytest.param(GeminiErrorClassifier, id="gemini"),
    pytest.param(PydanticAIErrorClassifier, id="pydantic_ai"),
]


class _Response:
    def __init__(self, *, status_code=None, status=None, headers=None):
        if status_code is not None:
            self.status_code = status_code
        if status is not None:
            self.status = status
        self.headers = headers or {}


class _SdkStyleError(Exception):
    """An exception from an SDK without a built-in classifier (e.g. Anthropic)."""

    def __init__(self, message, **attributes):
        super().__init__(message)
        for name, value in attributes.items():
            setattr(self, name, value)


@pytest.mark.parametrize("classifier_type", _ISSUE_177_CLASSIFIERS)
@pytest.mark.parametrize(
    "error",
    [
        ValueError("429: invalid item"),
        ValueError("quota field missing"),
        ValueError("insufficient_quota"),
        TypeError("timeout must be float"),
        AssertionError("rate limit reached"),
        KeyError("RESOURCE_EXHAUSTED"),
    ],
    ids=repr,
)
def test_issue177_programming_errors_outrank_message_patterns(classifier_type, error):
    info = classifier_type().classify(error)
    assert info.error_category == "logic_error"
    assert not info.is_retryable
    assert not info.is_rate_limit


@pytest.mark.parametrize("classifier_type", _ISSUE_177_CLASSIFIERS)
@pytest.mark.parametrize(
    "attributes",
    [
        {"status_code": 429},
        {"status": 429},
        {"code": 429},
        {"response": _Response(status_code=429, headers={"Retry-After": "3"})},
        {"response": _Response(status=429, headers={"Retry-After": "3"})},
    ],
    ids=["status_code", "status", "code", "response.status_code", "response.status"],
)
def test_issue177_structured_429_is_rate_limit_without_message(classifier_type, attributes):
    attributes.setdefault("response", _Response(headers={"Retry-After": "3"}))
    info = classifier_type().classify(_SdkStyleError("request failed", **attributes))
    assert info.error_category == "rate_limit"
    assert info.is_rate_limit and info.is_retryable
    assert info.suggested_wait == 3


@pytest.mark.parametrize("classifier_type", _ISSUE_177_CLASSIFIERS)
def test_issue177_other_status_is_not_rate_limit_by_message(classifier_type):
    error = _SdkStyleError("Error code: 400 - rate limit fields invalid", status_code=400)
    assert not classifier_type().classify(error).is_rate_limit


@pytest.mark.parametrize("classifier_type", _ISSUE_177_CLASSIFIERS)
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Error code: 429 - {'code': 'insufficient_quota'}"),
        _SdkStyleError(
            "You exceeded your current quota", status_code=429, code="insufficient_quota"
        ),
        _SdkStyleError("exceeded", status_code=429, body={"error": {"code": "insufficient_quota"}}),
    ],
    ids=["untyped-text", "status+code", "status+body"],
)
def test_issue177_insufficient_quota_is_not_a_retried_rate_limit(classifier_type, error):
    info = classifier_type().classify(error)
    assert info.error_category == "insufficient_balance"
    assert not info.is_retryable
    assert not info.is_rate_limit


@pytest.mark.parametrize("classifier_type", _ISSUE_177_CLASSIFIERS)
@pytest.mark.parametrize(
    ("message", "rate_limited"),
    [
        ("quota field missing", False),
        ("Quota exceeded for metric generate_requests", True),
        ("Error code: 429 - Too Many Requests", True),
    ],
)
def test_issue177_untyped_message_fallback_is_preserved_without_bare_quota(
    classifier_type, message, rate_limited
):
    # Untyped exceptions (custom clients, test doubles) keep the heuristic.
    assert classifier_type().classify(RuntimeError(message)).is_rate_limit is rate_limited


@pytest.mark.asyncio
async def test_issue177_mock_agent_rate_limit_is_structured():
    agent = MockAgent(response_factory=lambda prompt: "ok", latency=0, rate_limit_on_call=1)
    with pytest.raises(Exception) as caught:
        await agent.run("x")
    assert getattr(caught.value, "code", None) == 429
    for classifier_type in (DefaultErrorClassifier, GeminiErrorClassifier):
        assert classifier_type().classify(caught.value).is_rate_limit


class _ResponseUnavailable(Exception):
    status_code = 429

    @property
    def response(self):
        raise RuntimeError("response unavailable")


def _structured_billing_error() -> Exception:
    error = _SdkStyleError("request failed")
    error.code = "insufficient_quota"
    return error


@pytest.mark.parametrize("classifier_type", _ISSUE_177_CLASSIFIERS)
@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: RuntimeError("insufficient_quota: connection rejected"),
        lambda: RuntimeError("insufficient_quota: request timeout"),
        _structured_billing_error,
        lambda: ProviderResponseError("insufficient_quota", code=429),
    ],
    ids=["billing+connection-text", "billing+timeout-text", "code-only", "provider-response-429"],
)
def test_issue177_billing_exhaustion_outranks_transient_heuristics(classifier_type, error_factory):
    info = classifier_type().classify(error_factory())
    assert info.error_category == "insufficient_balance"
    assert not info.is_retryable
    assert not info.is_rate_limit


@pytest.mark.parametrize("classifier_type", _ISSUE_177_CLASSIFIERS)
def test_issue177_provider_response_status_outranks_rate_limit_text(classifier_type):
    error = ProviderResponseError("rate limit field invalid", code=400)
    assert not classifier_type().classify(error).is_rate_limit
    assert classifier_type().classify(ProviderResponseError("rate limited", code=429)).is_rate_limit


def test_issue177_gemini_unrecognized_status_outranks_rate_limit_text():
    from google.genai.errors import APIError

    error = APIError(418, {"error": {"message": "rate limit field invalid", "code": 418}})
    info = GeminiErrorClassifier().classify(error)
    assert not info.is_rate_limit
    assert info.error_category == "api_error"


@pytest.mark.parametrize("classifier_type", _ISSUE_177_CLASSIFIERS)
def test_issue177_failing_response_property_keeps_rate_limit_without_hint(classifier_type):
    info = classifier_type().classify(_ResponseUnavailable("request failed"))
    assert info.error_category == "rate_limit"
    assert info.is_rate_limit
    assert info.suggested_wait is None


def _sdk_billing_cases():
    import httpx
    from google.genai.errors import APIError
    from openai import APIStatusError, RateLimitError
    from pydantic_ai.exceptions import ModelHTTPError

    request = httpx.Request("POST", "https://example.invalid")

    def response(status: int) -> httpx.Response:
        return httpx.Response(status, request=request)

    return [
        (
            "openai-429-text",
            OpenAIErrorClassifier,
            lambda: RateLimitError("insufficient_quota", response=response(429), body=None),
        ),
        (
            "openai-500-body",
            OpenAIErrorClassifier,
            lambda: APIStatusError(
                "request failed", response=response(500), body={"code": "insufficient_quota"}
            ),
        ),
        (
            "pydantic-ai-429-string-body",
            PydanticAIErrorClassifier,
            lambda: ModelHTTPError(429, "test", body="insufficient_quota"),
        ),
        (
            "openrouter-provider-route",
            OpenRouterErrorClassifier,
            lambda: ProviderResponseError("no provider available: insufficient_quota", code=429),
        ),
        (
            "openrouter-status-route",
            OpenRouterErrorClassifier,
            lambda: APIStatusError(
                "no allowed providers: insufficient_quota", response=response(503), body=None
            ),
        ),
        (
            "gemini-apierror-429",
            GeminiErrorClassifier,
            lambda: APIError(429, {"error": {"message": "insufficient_quota", "code": 429}}),
        ),
    ]


@pytest.mark.parametrize(
    ("classifier_type", "error_factory"),
    [pytest.param(c, f, id=label) for label, c, f in _sdk_billing_cases()],
)
def test_issue177_sdk_billing_exhaustion_outranks_status_and_routing(
    classifier_type, error_factory
):
    info = classifier_type().classify(error_factory())
    assert info.error_category == "insufficient_balance"
    assert not info.is_retryable
    assert not info.is_rate_limit


def test_issue177_sdk_rate_limit_without_billing_is_unchanged():
    import httpx
    from openai import RateLimitError

    response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
    info = OpenAIErrorClassifier().classify(
        RateLimitError("Rate limit reached", response=response, body=None)
    )
    assert info.error_category == "rate_limit"
    assert info.is_rate_limit
