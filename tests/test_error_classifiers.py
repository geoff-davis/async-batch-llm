"""Tests for error classifiers."""

import asyncio

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
    """With count_rate_limits=True, a rate limit that never clears must
    respect retry.max_attempts and record a permanent failure instead of
    looping forever.

    Regression test for the is_retryable: False -> True change: before the
    classifier flip, rate-limited items failed fast after one attempt; now
    they retry, so we need to ensure the retry loop still terminates.
    (Since v0.10.x the default exempts rate limits from the budget — see
    test_persistent_rate_limit_capped_by_max_rate_limit_retries for the
    default-path termination guarantee.)
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
            count_rate_limits=True,
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


class _RateLimitThenSuccessStrategy(LLMCallStrategy[ClassifierTestOutput]):
    """Test helper: raises rate-limit errors for the first N calls, then succeeds."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.call_count = 0

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[ClassifierTestOutput, TokenUsage]:
        self.call_count += 1
        if self.call_count <= self.failures:
            raise Exception("429 RESOURCE_EXHAUSTED")
        return ClassifierTestOutput(value="ok"), {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        }


def _fast_rate_limit_config(retry: RetryConfig) -> ProcessorConfig:
    return ProcessorConfig(
        max_workers=1,
        timeout_per_item=1.0,
        retry=retry,
        rate_limit=RateLimitConfig(
            cooldown_seconds=0.001,
            slow_start_items=0,
            slow_start_initial_delay=0.0,
            slow_start_final_delay=0.0,
            backoff_multiplier=1.0,
        ),
    )


@pytest.mark.asyncio
async def test_rate_limits_do_not_consume_retry_budget_by_default():
    """An item unlucky enough to be in flight during several 429 bursts must
    not permanently fail: the framework cooled down each time, so the item
    never got a clean attempt. (v0.10.x: count_rate_limits defaults False.)"""
    strategy = _RateLimitThenSuccessStrategy(failures=3)
    config = _fast_rate_limit_config(
        RetryConfig(max_attempts=2, initial_wait=0.001, max_wait=0.001, jitter=False)
    )
    processor = ParallelBatchProcessor[str, ClassifierTestOutput, None](config=config)

    await processor.add_work(LLMWorkItem(item_id="bursty_rl", strategy=strategy, prompt="Test"))
    result = await processor.process_all()

    # 3 rate-limited attempts (exempt from the 2-attempt budget) + 1 success.
    assert result.succeeded == 1
    assert result.failed == 0
    assert strategy.call_count == 4


@pytest.mark.asyncio
async def test_persistent_rate_limit_capped_by_max_rate_limit_retries():
    """With the default budget exemption, a persistently-throttled item still
    terminates: max_rate_limit_retries caps the exempted attempts."""
    strategy = _PersistentRateLimitStrategy()
    config = _fast_rate_limit_config(
        RetryConfig(
            max_attempts=2,
            initial_wait=0.001,
            max_wait=0.001,
            jitter=False,
            max_rate_limit_retries=4,
        )
    )
    processor = ParallelBatchProcessor[str, ClassifierTestOutput, None](config=config)

    await processor.add_work(
        LLMWorkItem(item_id="persistent_rl_capped", strategy=strategy, prompt="Test")
    )
    result = await processor.process_all()

    assert result.succeeded == 0
    assert result.failed == 1
    assert strategy.call_count == 4


@pytest.mark.asyncio
async def test_retry_waits_for_cooldown_started_mid_retry():
    """A retry must re-check the cooldown gate: if another worker started a
    cooldown while this item was between attempts, the retry must not fire
    mid-cooldown and burn quota."""

    class NetworkErrorThenSuccess(LLMCallStrategy[ClassifierTestOutput]):
        def __init__(self, pause_event: asyncio.Event) -> None:
            self.pause_event = pause_event
            self.call_count = 0

        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> tuple[ClassifierTestOutput, TokenUsage]:
            self.call_count += 1
            if self.call_count == 1:
                # Simulate another worker triggering a cooldown while this
                # item's first attempt fails with a transient network error.
                self.pause_event.clear()
                raise Exception("connection reset by peer")
            return ClassifierTestOutput(value="ok"), {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            }

    config = _fast_rate_limit_config(
        RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.01, jitter=False)
    )
    processor = ParallelBatchProcessor[str, ClassifierTestOutput, None](config=config)
    pause_event = processor._rate_limit_coord._rate_limit_event
    strategy = NetworkErrorThenSuccess(pause_event)

    work_item = LLMWorkItem(item_id="paused_retry", strategy=strategy, prompt="Test")
    task = asyncio.create_task(processor._process_item_with_retries(work_item, worker_id=0))

    # Give the first attempt and its (0.01s) retry wait plenty of time to
    # elapse; the retry must still be gated on the cleared pause event.
    await asyncio.sleep(0.2)
    assert strategy.call_count == 1, "retry fired while workers were paused"

    pause_event.set()
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result.success is True
    assert strategy.call_count == 2


@pytest.mark.asyncio
async def test_stale_rate_limit_does_not_start_second_cooldown():
    """A 429 from a request issued before a cooldown that has since completed
    must not coordinate a redundant new cooldown (generation snapshot is
    taken before the request goes out, not in the error handler)."""

    processor_config = _fast_rate_limit_config(
        RetryConfig(max_attempts=3, initial_wait=0.001, max_wait=0.001, jitter=False)
    )
    processor = ParallelBatchProcessor[str, ClassifierTestOutput, None](config=processor_config)
    coord = processor._rate_limit_coord

    class StaleRateLimitStrategy(LLMCallStrategy[ClassifierTestOutput]):
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
            if self.call_count == 1:
                # While our request is "in flight", another worker triggers a
                # full cooldown cycle (start + finish)...
                await coord.handle_rate_limit(
                    worker_id=99, observed_generation=coord.current_generation
                )
                # ...then our own (older) request surfaces its 429.
                raise Exception("429 RESOURCE_EXHAUSTED")
            return ClassifierTestOutput(value="ok"), {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            }

    strategy = StaleRateLimitStrategy()
    await processor.add_work(LLMWorkItem(item_id="stale_429", strategy=strategy, prompt="Test"))
    result = await processor.process_all()

    assert result.succeeded == 1
    # Only the simulated cooldown cycle ran; the stale 429 must not have
    # incremented the generation with a redundant second cooldown.
    assert processor._cooldown_generation == 1
    assert strategy.call_count == 2


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


def test_logic_bug_messages_do_not_look_transient():
    """Regression for the shared fallback chain's ordering: substring checks
    ran before the isinstance logic-bug test, so a ValueError mentioning
    "connection" was retried pointlessly and a KeyError('quota') triggered a
    *global* rate-limit cooldown."""
    from async_batch_llm.classifiers import OpenAIErrorClassifier

    for classifier in (DefaultErrorClassifier(), GeminiErrorClassifier(), OpenAIErrorClassifier()):
        info = classifier.classify(ValueError("invalid connection string"))
        assert info.is_retryable is False, type(classifier).__name__
        assert info.error_category == "logic_error"

        info = classifier.classify(KeyError("quota"))
        assert info.is_rate_limit is False, type(classifier).__name__
        assert info.error_category == "logic_error"

        info = classifier.classify(TypeError("timeout must be a float"))
        assert info.is_retryable is False, type(classifier).__name__
        assert info.error_category == "logic_error"


@pytest.mark.asyncio
async def test_current_generation_event_alias_tracks_coordinator():
    """Regression: the processor's _current_generation_event was a plain
    alias snapshotted at construction, but the coordinator replaces the
    event on every cooldown — the alias pointed at the first generation's
    (permanently set) event forever after."""
    config = _fast_rate_limit_config(
        RetryConfig(max_attempts=2, initial_wait=0.001, max_wait=0.001, jitter=False)
    )
    processor = ParallelBatchProcessor[str, ClassifierTestOutput, None](config=config)
    coord = processor._rate_limit_coord

    assert processor._current_generation_event is coord._current_generation_event

    # Run one full cooldown cycle — the coordinator replaces the event.
    await coord.handle_rate_limit(worker_id=0, observed_generation=0)

    assert processor._current_generation_event is coord._current_generation_event


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

    Regression tests for the v0.10.x rewrite: classification dispatches on
    APIError.code instead of string-matching str(exception), so transient
    500/503 errors retry and deterministic 4xx errors fail fast.
    """

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
        ("code", "message"),
        [
            (400, "Invalid request"),
            (401, "API key not valid"),
            (403, "Permission denied"),
            (404, "Model not found"),
        ],
    )
    def test_deterministic_client_errors_fail_fast(self, code, message):
        """4xx errors are deterministic; retrying an invalid API key on every
        item in the batch just multiplies latency."""
        classifier = GeminiErrorClassifier()
        info = classifier.classify(self._client_error(code, message))
        assert info.is_retryable is False
        assert info.error_category == "client_error"

    @pytest.mark.parametrize(
        ("code", "message"),
        [
            (500, "Internal error encountered"),
            (503, "The model is overloaded. Please try again later."),
            (502, "Bad gateway"),
        ],
    )
    def test_transient_server_errors_retry(self, code, message):
        """500/503 are Gemini's most common transient failures; they must
        retry even though their messages match no timeout pattern."""
        classifier = GeminiErrorClassifier()
        info = classifier.classify(self._server_error(code, message))
        assert info.is_retryable is True
        assert info.is_rate_limit is False
        assert info.error_category == "server_error"

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

    def test_generic_fallback_still_runs_without_genai_sdk(self, monkeypatch):
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
