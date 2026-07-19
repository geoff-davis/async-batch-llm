"""Tests for LLM call strategies."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from async_batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryConfig,
    RetryState,
)
from async_batch_llm.base import LLMResponse
from async_batch_llm.llm_strategies import (
    GeminiStrategy,
    LLMCallStrategy,
    ModelStrategy,
    PydanticAIStrategy,
)
from async_batch_llm.models import GeminiCachedModel
from async_batch_llm.testing import MockAgent
from tests.test_gemini_strategies import AsyncIterList


class TestOutput(BaseModel):
    """Test output model."""

    text: str


# Test LLMCallStrategy base class


class MockStrategy(LLMCallStrategy[TestOutput]):
    """Mock strategy for testing base class behavior."""

    def __init__(self):
        self.prepare_called = False
        self.execute_calls = []
        self.cleanup_called = False
        self.on_error_calls = []

    async def prepare(self):
        self.prepare_called = True

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[TestOutput, dict[str, int]]:
        self.execute_calls.append((prompt, attempt, timeout, state))
        return TestOutput(text=f"Response for {prompt}"), {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ) -> None:
        self.on_error_calls.append((exception, attempt, state))

    async def cleanup(self):
        self.cleanup_called = False


@pytest.mark.asyncio
async def test_strategy_lifecycle():
    """Test that strategy prepare/execute/cleanup are called correctly."""
    strategy = MockStrategy()
    config = ProcessorConfig(max_workers=1, attempt_timeout=10.0)

    async with ParallelBatchProcessor[None, TestOutput, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Test prompt",
            )
        )

        result = await processor.process_all()

    assert result.succeeded == 1
    assert strategy.prepare_called
    assert len(strategy.execute_calls) == 1
    assert strategy.execute_calls[0][0] == "Test prompt"
    assert strategy.execute_calls[0][1] == 1  # First attempt
    # Note: cleanup_called will be False because we set it to False in cleanup()
    # This tests that cleanup was actually called


@pytest.mark.asyncio
async def test_strategy_with_retries():
    """Test that strategy execute is called for each retry attempt."""

    class FailingStrategy(LLMCallStrategy[TestOutput]):
        def __init__(self, fail_count=2):
            self.fail_count = fail_count
            self.attempt_count = 0

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[TestOutput, dict[str, int]]:
            self.attempt_count += 1
            if self.attempt_count < self.fail_count:
                raise Exception("Simulated transient failure")
            return TestOutput(text="Success"), {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }

    strategy = FailingStrategy(fail_count=3)
    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=10.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.05, jitter=False),
    )

    async with ParallelBatchProcessor[None, TestOutput, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Test prompt",
            )
        )

        result = await processor.process_all()

    assert result.succeeded == 1
    assert strategy.attempt_count == 3  # Should have tried 3 times


# Test PydanticAIStrategy


@pytest.mark.asyncio
async def test_pydantic_ai_strategy():
    """Test PydanticAIStrategy with a mock agent."""
    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(text=f"Response: {p}"),
        latency=0.001,
    )

    strategy = PydanticAIStrategy(agent=mock_agent)
    config = ProcessorConfig(max_workers=1, attempt_timeout=10.0)

    async with ParallelBatchProcessor[None, TestOutput, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Hello",
            )
        )

        result = await processor.process_all()

    assert result.succeeded == 1
    assert result.results[0].output.text == "Response: Hello"
    assert result.results[0].token_usage["total_tokens"] > 0


# Test Gemini strategies (mock-based since we don't want to make real API calls)


@pytest.mark.asyncio
async def test_gemini_strategy_mock():
    """Test GeminiStrategy with mocked model."""
    # Create a mock LLMModel
    mock_model = AsyncMock()
    mock_model.generate = AsyncMock(
        return_value=LLMResponse(
            text="Gemini response",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )
    )

    # Create strategy
    strategy = GeminiStrategy(
        model=mock_model,
        response_parser=lambda r: TestOutput(text=r.text),
    )

    # Test execute (3-tuple shape from v0.10.0; metadata not asserted here).
    output, tokens, _metadata = await strategy.execute("Test prompt", 1, 10.0)

    assert output.text == "Gemini response"
    assert tokens["input_tokens"] == 10
    assert tokens["output_tokens"] == 20
    assert tokens["total_tokens"] == 30


@pytest.mark.asyncio
async def test_gemini_cached_model_lifecycle():
    """Test GeminiCachedModel prepare/generate/cleanup/delete lifecycle (v0.6.0)."""

    # Create mock cache
    mock_cache = MagicMock()
    mock_cache.name = "test-cache"

    # Create mock response
    mock_response = MagicMock()
    mock_response.text = "Cached response"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_response.usage_metadata.total_token_count = 30
    mock_response.usage_metadata.cached_content_token_count = 0
    mock_response.candidates = []

    # Create mock client
    mock_client = MagicMock()
    mock_client.aio.caches.create = AsyncMock(return_value=mock_cache)
    mock_client.aio.caches.list = AsyncMock(return_value=AsyncIterList([]))  # No existing caches
    mock_client.aio.caches.update = AsyncMock(return_value=mock_cache)
    mock_client.aio.caches.delete = AsyncMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    # Create cached model
    cached_model = GeminiCachedModel(
        model="gemini-test",
        client=mock_client,
        cached_content=[],
        cache_ttl_seconds=3600,
    )

    # Test prepare
    await cached_model.prepare()
    mock_client.aio.caches.create.assert_called_once()

    # Test generate
    llm_response = await cached_model.generate("Test prompt")
    assert llm_response.text == "Cached response"
    assert llm_response.total_tokens == 30

    # Test cleanup (v0.6.0: preserves cache by default)
    await cached_model.cleanup()
    mock_client.aio.caches.delete.assert_not_called()  # Should NOT delete

    # Test explicit deletion
    await cached_model.delete_cache()
    mock_client.aio.caches.delete.assert_called_once_with(name="test-cache")


@pytest.mark.asyncio
async def test_gemini_cached_model_auto_renewal():
    """Test automatic cache renewal when close to expiring (v0.6.0)."""

    # Create mock caches
    mock_cache1 = MagicMock()
    mock_cache1.name = "test-cache-1"

    mock_cache2 = MagicMock()
    mock_cache2.name = "test-cache-2"

    # Create mock response
    mock_response = MagicMock()
    mock_response.text = "Cached response"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_response.usage_metadata.total_token_count = 30
    mock_response.usage_metadata.cached_content_token_count = 0
    mock_response.candidates = []

    # Create mock client - returns different caches on subsequent creates
    mock_client = MagicMock()
    create_calls = [mock_cache1, mock_cache2]
    mock_client.aio.caches.create = AsyncMock(side_effect=create_calls)
    mock_client.aio.caches.list = AsyncMock(return_value=AsyncIterList([]))  # No existing caches
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    # Create cached model with short TTL and renewal buffer
    cached_model = GeminiCachedModel(
        model="gemini-test",
        client=mock_client,
        cached_content=[],
        cache_ttl_seconds=2,  # 2 second TTL
        cache_renewal_buffer_seconds=1,  # Renew when 1 second remains
        auto_renew=True,
    )

    # Test prepare
    await cached_model.prepare()
    assert mock_client.aio.caches.create.call_count == 1

    # First generate - no renewal needed (just created, 2s remaining > 1s buffer)
    await cached_model.generate("Test prompt 1")
    assert mock_client.aio.caches.create.call_count == 1  # Still 1

    # Wait until within renewal buffer (>1s elapsed, <1s remaining)
    await asyncio.sleep(1.2)

    # Second generate - should trigger auto-renewal (only 0.8s remaining < 1s buffer)
    await cached_model.generate("Test prompt 2")
    assert mock_client.aio.caches.create.call_count == 2  # New cache created

    # Cleanup
    await cached_model.cleanup()


@pytest.mark.asyncio
async def test_strategy_error_handling():
    """Test that strategy errors are handled correctly."""

    class ErrorStrategy(LLMCallStrategy[TestOutput]):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[TestOutput, dict[str, int]]:
            raise ValueError("Strategy error")

    strategy = ErrorStrategy()
    config = ProcessorConfig(max_workers=1, attempt_timeout=10.0, retry=RetryConfig(max_attempts=1))

    async with ParallelBatchProcessor[None, TestOutput, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Test",
            )
        )

        result = await processor.process_all()

    assert result.failed == 1
    assert "ValueError" in result.results[0].error


@pytest.mark.asyncio
async def test_work_item_validation():
    """Test that work item validation works correctly."""

    # Should accept strategy
    item = LLMWorkItem(
        item_id="test1",
        strategy=MockStrategy(),
        prompt="Test",
    )
    assert item.strategy is not None


# Test on_error callback


@pytest.mark.asyncio
async def test_on_error_callback_called():
    """Test that on_error callback is called when execute raises exception."""

    class ErrorTrackingStrategy(LLMCallStrategy[TestOutput]):
        def __init__(self):
            self.errors_received = []
            self.attempts_received = []

        async def on_error(
            self, exception: Exception, attempt: int, state: RetryState | None = None
        ) -> None:
            self.errors_received.append(exception)
            self.attempts_received.append(attempt)

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[TestOutput, dict[str, int]]:
            if attempt < 3:
                raise Exception(f"Transient error on attempt {attempt}")
            return TestOutput(text="Success"), {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }

    strategy = ErrorTrackingStrategy()
    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=10.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.05, jitter=False),
    )

    async with ParallelBatchProcessor[None, TestOutput, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Test",
            )
        )

        result = await processor.process_all()

    # Should succeed on 3rd attempt
    assert result.succeeded == 1

    # on_error should have been called twice (for attempts 1 and 2)
    assert len(strategy.errors_received) == 2
    assert len(strategy.attempts_received) == 2

    # Check that correct attempt numbers were passed
    assert strategy.attempts_received[0] == 1
    assert strategy.attempts_received[1] == 2

    # Check that errors were passed correctly
    assert all(isinstance(e, Exception) for e in strategy.errors_received)
    assert "Transient error on attempt 1" in str(strategy.errors_received[0])
    assert "Transient error on attempt 2" in str(strategy.errors_received[1])


@pytest.mark.asyncio
async def test_on_error_callback_with_state():
    """Test using on_error to track state for smart retry logic."""

    class SmartRetryStrategy(LLMCallStrategy[TestOutput]):
        def __init__(self):
            self.validation_errors = 0
            self.network_errors = 0
            self.last_error = None

        async def on_error(
            self, exception: Exception, attempt: int, state: RetryState | None = None
        ) -> None:
            self.last_error = exception
            # Track different error types
            if "validation" in str(exception).lower():
                self.validation_errors += 1
            elif isinstance(exception, ConnectionError):
                self.network_errors += 1

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[TestOutput, dict[str, int]]:
            if attempt == 1:
                raise Exception("Validation error")  # Generic exception (retryable)
            elif attempt == 2:
                raise ConnectionError("Network error")
            else:
                # On 3rd attempt, use state to create custom response
                return TestOutput(
                    text=f"Recovered after {self.validation_errors} validation "
                    f"and {self.network_errors} network errors"
                ), {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                }

    strategy = SmartRetryStrategy()
    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=10.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.05, jitter=False),
    )

    async with ParallelBatchProcessor[None, TestOutput, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Test",
            )
        )

        result = await processor.process_all()

    assert result.succeeded == 1
    assert strategy.validation_errors == 1
    assert strategy.network_errors == 1
    assert "Recovered after 1 validation and 1 network errors" in result.results[0].output.text


@pytest.mark.asyncio
async def test_on_error_callback_exception_handling():
    """Test that exceptions in on_error callback don't crash the processor."""

    class BuggyCallbackStrategy(LLMCallStrategy[TestOutput]):
        def __init__(self):
            self.execute_count = 0

        async def on_error(
            self, exception: Exception, attempt: int, state: RetryState | None = None
        ) -> None:
            # Intentionally buggy callback
            raise RuntimeError("Buggy on_error callback")

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[TestOutput, dict[str, int]]:
            self.execute_count += 1
            if attempt < 2:
                raise Exception("First attempt fails")  # Generic exception (retryable)
            return TestOutput(text="Success"), {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }

    strategy = BuggyCallbackStrategy()
    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=10.0,
        retry=RetryConfig(max_attempts=2, initial_wait=0.01, max_wait=0.05, jitter=False),
    )

    async with ParallelBatchProcessor[None, TestOutput, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Test",
            )
        )

        result = await processor.process_all()

    # Should still succeed despite buggy callback
    assert result.succeeded == 1
    assert strategy.execute_count == 2


@pytest.mark.asyncio
async def test_on_error_not_called_on_success():
    """Test that on_error is not called when execute succeeds."""

    class CallbackTrackingStrategy(LLMCallStrategy[TestOutput]):
        def __init__(self):
            self.on_error_called = False

        async def on_error(
            self, exception: Exception, attempt: int, state: RetryState | None = None
        ) -> None:
            self.on_error_called = True

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[TestOutput, dict[str, int]]:
            # Always succeed
            return TestOutput(text="Success"), {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }

    strategy = CallbackTrackingStrategy()
    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=10.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.05, jitter=False),
    )

    async with ParallelBatchProcessor[None, TestOutput, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Test",
            )
        )

        result = await processor.process_all()

    assert result.succeeded == 1
    assert not strategy.on_error_called


def _recording_model() -> AsyncMock:
    m = AsyncMock()
    m.generate = AsyncMock(
        return_value=LLMResponse(text="ok", input_tokens=1, output_tokens=1, total_tokens=2)
    )
    return m


@pytest.mark.asyncio
async def test_generation_config_forwarded_to_model():
    """ModelStrategy(generation_config=...) forwards it to model.generate(config=...)
    on every call — lets a built-in strategy carry response_schema/tools/etc.
    without overriding execute()."""
    model = _recording_model()
    cfg = {"response_mime_type": "application/json", "response_schema": {"x": 1}}
    strategy = ModelStrategy(model, generation_config=cfg)

    await strategy.execute("prompt", 1, 10.0)

    assert model.generate.call_args.kwargs["config"] == cfg


@pytest.mark.asyncio
async def test_generation_config_omitted_when_unset():
    """Omitting generation_config calls generate() WITHOUT a config kwarg, so a
    custom LLMModel whose generate() doesn't accept config keeps working (the
    pre-generation_config call shape is preserved)."""
    model = _recording_model()
    strategy = ModelStrategy(model)

    await strategy.execute("prompt", 1, 10.0)

    assert "config" not in model.generate.call_args.kwargs
