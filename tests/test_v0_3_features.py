"""Tests for v0.3.0 features: RetryState, GeminiResponse, and cache tagging."""

import asyncio

import pytest

from batch_llm import (
    GeminiResponse,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryState,
)
from batch_llm.base import TokenUsage
from batch_llm.llm_strategies import LLMCallStrategy


# Custom exception that will be retried
class RetryableTestError(Exception):
    """Test exception that should be retried."""

    pass


# =============================================================================
# Issue #8: RetryState Tests
# =============================================================================


class RetryStateStrategy(LLMCallStrategy[str]):
    """Strategy that uses RetryState to track retry attempts."""

    def __init__(self, fail_until_attempt: int = 3):
        self.fail_until_attempt = fail_until_attempt
        self.execute_calls = []

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        """Track calls and use state to coordinate behavior."""
        self.execute_calls.append((prompt, attempt, state))

        if state is not None:
            # Increment counter in state
            count = state.get("attempt_count", 0)
            state.set("attempt_count", count + 1)
            state.set("last_prompt", prompt)

            # Track all attempts
            attempts = state.get("attempts", [])
            attempts.append(attempt)
            state.set("attempts", attempts)

        # Fail until we reach the target attempt
        if attempt < self.fail_until_attempt:
            raise RetryableTestError(f"Simulated failure on attempt {attempt}")

        return f"Success after {attempt} attempts", {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ) -> None:
        """Track errors in state."""
        if state is not None:
            errors = state.get("errors", [])
            errors.append(str(exception))
            state.set("errors", errors)


@pytest.mark.asyncio
async def test_retry_state_persistence():
    """Test that RetryState persists across retry attempts."""
    strategy = RetryStateStrategy(fail_until_attempt=3)
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test prompt")
        )

        result = await processor.process_all()

    # Should succeed on attempt 3
    assert result.succeeded == 1
    assert result.failed == 0

    # Check state was passed to all calls
    assert len(strategy.execute_calls) == 3
    for _prompt, _attempt, state in strategy.execute_calls:
        assert state is not None
        assert isinstance(state, RetryState)

    # All calls should have received the SAME state instance
    state_ids = [id(state) for _, _, state in strategy.execute_calls]
    assert len(set(state_ids)) == 1, "All attempts should share the same state instance"

    # Check state accumulated data correctly
    final_state = strategy.execute_calls[-1][2]
    assert final_state.get("attempt_count") == 3
    assert final_state.get("last_prompt") == "Test prompt"
    assert final_state.get("attempts") == [1, 2, 3]
    assert len(final_state.get("errors", [])) == 2  # 2 failures before success


@pytest.mark.asyncio
async def test_retry_state_isolation():
    """Test that each work item gets its own isolated RetryState."""
    strategy = RetryStateStrategy(fail_until_attempt=2)
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 5 work items that will all retry
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"item_{i}", strategy=strategy, prompt=f"Prompt {i}"
                )
            )

        result = await processor.process_all()

    assert result.succeeded == 5
    assert result.failed == 0

    # Check that we saw 5 different state instances (one per work item)
    state_ids = {id(state) for _, _, state in strategy.execute_calls if state}
    assert len(state_ids) == 5, "Each work item should have its own state instance"


@pytest.mark.asyncio
async def test_retry_state_operations():
    """Test RetryState dictionary operations."""
    state = RetryState()

    # Test basic set/get
    state.set("key1", "value1")
    assert state.get("key1") == "value1"

    # Test default value
    assert state.get("nonexistent", "default") == "default"

    # Test contains
    assert "key1" in state
    assert "nonexistent" not in state

    # Test delete
    state.delete("key1")
    assert "key1" not in state

    # Test clear
    state.set("key2", "value2")
    state.set("key3", "value3")
    state.clear()
    assert "key2" not in state
    assert "key3" not in state


@pytest.mark.asyncio
async def test_retry_state_none_backward_compatibility():
    """Test that strategies work when RetryState is None (backward compatibility)."""

    class LegacyStrategy(LLMCallStrategy[str]):
        """Strategy that ignores state parameter."""

        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> tuple[str, TokenUsage]:
            # Ignore state parameter completely (legacy behavior)
            return "Success", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    strategy = LegacyStrategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test")
        )

        result = await processor.process_all()

    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_on_error_receives_state():
    """Test that on_error() receives the same RetryState as execute()."""

    class ErrorTrackingStrategy(LLMCallStrategy[str]):
        def __init__(self):
            self.on_error_calls = []

        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> tuple[str, TokenUsage]:
            if attempt == 1 and state is not None:
                state.set("execute_called", True)
            if attempt < 2:
                raise RetryableTestError("Fail first attempt")
            return "Success", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        async def on_error(
            self, exception: Exception, attempt: int, state: RetryState | None = None
        ) -> None:
            self.on_error_calls.append((exception, attempt, state))

    strategy = ErrorTrackingStrategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test")
        )

        result = await processor.process_all()

    assert result.succeeded == 1

    # Check on_error was called
    assert len(strategy.on_error_calls) == 1
    exception, attempt, state = strategy.on_error_calls[0]
    assert isinstance(exception, RetryableTestError)
    assert attempt == 1
    assert state is not None

    # Check that on_error received the same state as execute
    assert state.get("execute_called") is True


# =============================================================================
# Issue #3: GeminiResponse Tests
# =============================================================================


class MockGeminiStrategy(LLMCallStrategy[str]):
    """Mock strategy that returns GeminiResponse with safety ratings."""

    def __init__(self, include_metadata: bool = False):
        self.include_metadata = include_metadata

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str | GeminiResponse[str], TokenUsage]:
        """Return either raw output or GeminiResponse based on include_metadata."""
        output = f"Response: {prompt}"
        token_usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

        if not self.include_metadata:
            return output, token_usage

        # Create mock safety ratings
        safety_ratings = {
            "HARM_CATEGORY_HATE_SPEECH": "NEGLIGIBLE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "LOW",
            "HARM_CATEGORY_HARASSMENT": "NEGLIGIBLE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "NEGLIGIBLE",
        }

        return (
            GeminiResponse(
                output=output,
                safety_ratings=safety_ratings,
                finish_reason="STOP",
                token_usage=token_usage,
                raw_response={"mock": "response"},
            ),
            token_usage,
        )


@pytest.mark.asyncio
async def test_gemini_response_with_metadata():
    """Test that GeminiResponse correctly wraps output with safety ratings."""
    strategy = MockGeminiStrategy(include_metadata=True)
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str | GeminiResponse[str], None](
        config=config
    ) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test prompt")
        )

        result = await processor.process_all()

    assert result.succeeded == 1

    # Check output is GeminiResponse
    work_result = result.results[0]
    assert isinstance(work_result.output, GeminiResponse)

    # Check safety ratings
    assert work_result.output.safety_ratings is not None
    assert "HARM_CATEGORY_HATE_SPEECH" in work_result.output.safety_ratings
    assert work_result.output.safety_ratings["HARM_CATEGORY_HATE_SPEECH"] == "NEGLIGIBLE"

    # Check finish reason
    assert work_result.output.finish_reason == "STOP"

    # Check actual output
    assert work_result.output.output == "Response: Test prompt"

    # Check raw response is preserved
    assert work_result.output.raw_response == {"mock": "response"}


@pytest.mark.asyncio
async def test_gemini_response_without_metadata():
    """Test backward compatibility: include_metadata=False returns raw output."""
    strategy = MockGeminiStrategy(include_metadata=False)
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test prompt")
        )

        result = await processor.process_all()

    assert result.succeeded == 1

    # Check output is plain string (not wrapped)
    work_result = result.results[0]
    assert isinstance(work_result.output, str)
    assert work_result.output == "Response: Test prompt"


@pytest.mark.asyncio
async def test_gemini_response_generic_type():
    """Test that GeminiResponse works with different output types."""
    from pydantic import BaseModel

    class TestOutput(BaseModel):
        value: str
        count: int

    class GenericGeminiStrategy(LLMCallStrategy[TestOutput]):
        def __init__(self, include_metadata: bool = False):
            self.include_metadata = include_metadata

        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> tuple[TestOutput | GeminiResponse[TestOutput], TokenUsage]:
            output = TestOutput(value=prompt, count=len(prompt))
            token_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

            if not self.include_metadata:
                return output, token_usage

            return (
                GeminiResponse(
                    output=output,
                    safety_ratings={"HARM_CATEGORY_HATE_SPEECH": "NEGLIGIBLE"},
                    finish_reason="STOP",
                    token_usage=token_usage,
                    raw_response={},
                ),
                token_usage,
            )

    strategy = GenericGeminiStrategy(include_metadata=True)
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[
        str, TestOutput | GeminiResponse[TestOutput], None
    ](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test")
        )

        result = await processor.process_all()

    assert result.succeeded == 1
    work_result = result.results[0]
    assert isinstance(work_result.output, GeminiResponse)
    assert isinstance(work_result.output.output, TestOutput)
    assert work_result.output.output.value == "Test"
    assert work_result.output.output.count == 4


# =============================================================================
# Issue #6: Cache Tagging Tests (Unit tests for tag matching logic)
# =============================================================================


def test_cache_tags_isolation():
    """Test that different cache tags result in isolated state tracking."""
    # Note: This is a unit test - actual cache creation requires Google API
    # We're testing that the cache_tags parameter is properly stored and used

    try:
        import google.genai as genai
        from google.genai.types import Content

        from batch_llm.llm_strategies import GeminiCachedStrategy
    except ImportError:
        pytest.skip("google-genai not installed")

    # Create mock client and content (won't actually be used)
    try:
        client = genai.Client(api_key="fake-key-for-testing")
    except Exception:
        pytest.skip("Cannot create mock client")

    cached_content = [Content(role="user", parts=[{"text": "test"}])]

    # Create two strategies with different tags
    strategy_a = GeminiCachedStrategy(
        model="gemini-2.0-flash-001",
        client=client,
        response_parser=lambda x: str(x),
        cached_content=cached_content,
        cache_tags={"version": "v1", "experiment": "A"},
    )

    strategy_b = GeminiCachedStrategy(
        model="gemini-2.0-flash-001",
        client=client,
        response_parser=lambda x: str(x),
        cached_content=cached_content,
        cache_tags={"version": "v1", "experiment": "B"},
    )

    # Verify tags are stored
    assert strategy_a.cache_tags == {"version": "v1", "experiment": "A"}
    assert strategy_b.cache_tags == {"version": "v1", "experiment": "B"}

    # Verify they have different cache identity
    assert strategy_a.cache_tags != strategy_b.cache_tags


def test_cache_tags_none_default():
    """Test that cache_tags defaults to empty dict when not provided."""
    try:
        import google.genai as genai
        from google.genai.types import Content

        from batch_llm.llm_strategies import GeminiCachedStrategy
    except ImportError:
        pytest.skip("google-genai not installed")

    # Create mock client and content (won't actually be used)
    try:
        client = genai.Client(api_key="fake-key-for-testing")
    except Exception:
        pytest.skip("Cannot create mock client")

    cached_content = [Content(role="user", parts=[{"text": "test"}])]

    strategy = GeminiCachedStrategy(
        model="gemini-2.0-flash-001",
        client=client,
        response_parser=lambda x: str(x),
        cached_content=cached_content,
        # cache_tags not provided
    )

    # Should default to empty dict
    assert strategy.cache_tags == {}


# =============================================================================
# Integration Tests: Multiple features together
# =============================================================================


@pytest.mark.asyncio
async def test_retry_state_with_gemini_response():
    """Test that RetryState and GeminiResponse work together."""

    class CombinedStrategy(LLMCallStrategy[str]):
        """Strategy using both RetryState and GeminiResponse."""

        def __init__(self):
            self.include_metadata = True

        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> tuple[GeminiResponse[str], TokenUsage]:
            if state is not None:
                state.set("attempt", attempt)

            # Fail first attempt to test retry with state
            if attempt == 1:
                raise RetryableTestError("First attempt fails")

            output = "Success"
            token_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

            return (
                GeminiResponse(
                    output=output,
                    safety_ratings={"HARM_CATEGORY_HATE_SPEECH": "NEGLIGIBLE"},
                    finish_reason="STOP",
                    token_usage=token_usage,
                    raw_response={},
                ),
                token_usage,
            )

        async def on_error(
            self, exception: Exception, attempt: int, state: RetryState | None = None
        ) -> None:
            if state is not None:
                state.set("error_seen", True)

    strategy = CombinedStrategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, GeminiResponse[str], None](
        config=config
    ) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test")
        )

        result = await processor.process_all()

    assert result.succeeded == 1

    # Check we got GeminiResponse
    work_result = result.results[0]
    assert isinstance(work_result.output, GeminiResponse)
    assert work_result.output.output == "Success"
    assert work_result.output.safety_ratings["HARM_CATEGORY_HATE_SPEECH"] == "NEGLIGIBLE"


@pytest.mark.asyncio
async def test_shared_strategy_with_retry_state():
    """Test that shared strategy instances work correctly with per-item RetryState."""

    class SharedStrategy(LLMCallStrategy[str]):
        """Shared strategy that uses state to track per-item data."""

        def __init__(self):
            self.total_executions = 0
            self.lock = asyncio.Lock()

        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> tuple[str, TokenUsage]:
            async with self.lock:
                self.total_executions += 1

            if state is not None:
                # Track per-item attempts
                attempts = state.get("attempts", 0)
                state.set("attempts", attempts + 1)

                # Fail first attempt for each item
                if attempts == 0:
                    raise RetryableTestError("First attempt")

            return "Success", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    # Use SAME strategy instance for all items
    strategy = SharedStrategy()
    config = ProcessorConfig(max_workers=3, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Prompt {i}")
            )

        result = await processor.process_all()

    # All should succeed (after retry)
    assert result.succeeded == 5
    assert result.failed == 0

    # Shared strategy should see all executions
    # 5 items * 2 attempts = 10 total executions
    assert strategy.total_executions == 10
