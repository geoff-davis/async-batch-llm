"""Tests for v0.3.0 features: RetryState, and cache tagging."""

import asyncio

import pytest

from async_batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryConfig,
    RetryState,
)
from async_batch_llm.base import TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy


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
    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=10.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.05, jitter=False),
    )

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
    config = ProcessorConfig(
        max_workers=2,
        attempt_timeout=10.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.05, jitter=False),
    )

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 5 work items that will all retry
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Prompt {i}")
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
    config = ProcessorConfig(max_workers=1, attempt_timeout=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))

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
    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=10.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.05, jitter=False),
    )

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))

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
# Issue #6: Cache Tagging Tests (Unit tests for tag matching logic)
# =============================================================================


def test_cache_tags_isolation():
    """Test that different cache tags result in isolated state tracking."""
    # Note: This is a unit test - actual cache creation requires Google API
    # We're testing that the cache_tags parameter is properly stored and used

    try:
        import google.genai as genai
        from google.genai.types import Content

        from async_batch_llm.models import GeminiCachedModel
    except ImportError:
        pytest.skip("google-genai not installed")

    # Create mock client and content (won't actually be used)
    try:
        client = genai.Client(api_key="fake-key-for-testing")
    except Exception:
        pytest.skip("Cannot create mock client")

    cached_content = [Content(role="user", parts=[{"text": "test"}])]

    # Create two models with different tags
    model_a = GeminiCachedModel(
        model="gemini-2.0-flash-001",
        client=client,
        cached_content=cached_content,
        cache_tags={"version": "v1", "experiment": "A"},
    )

    model_b = GeminiCachedModel(
        model="gemini-2.0-flash-001",
        client=client,
        cached_content=cached_content,
        cache_tags={"version": "v1", "experiment": "B"},
    )

    # Verify tags are stored
    assert model_a._cache_tags == {"version": "v1", "experiment": "A"}
    assert model_b._cache_tags == {"version": "v1", "experiment": "B"}

    # Verify they have different cache identity
    assert model_a._cache_tags != model_b._cache_tags


def test_cache_tags_none_default():
    """Test that cache_tags defaults to empty dict when not provided."""
    try:
        import google.genai as genai
        from google.genai.types import Content

        from async_batch_llm.models import GeminiCachedModel
    except ImportError:
        pytest.skip("google-genai not installed")

    # Create mock client and content (won't actually be used)
    try:
        client = genai.Client(api_key="fake-key-for-testing")
    except Exception:
        pytest.skip("Cannot create mock client")

    cached_content = [Content(role="user", parts=[{"text": "test"}])]

    model = GeminiCachedModel(
        model="gemini-2.0-flash-001",
        client=client,
        cached_content=cached_content,
        # cache_tags not provided
    )

    # Should default to empty dict
    assert model._cache_tags == {}


# =============================================================================
# Integration Tests: Multiple features together
# =============================================================================


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
    config = ProcessorConfig(
        max_workers=3,
        attempt_timeout=10.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.05, jitter=False),
    )

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
