"""Tests for token tracking when items fail validation.

This tests the fix for the bug where failed items don't report token usage.
"""

import pytest
from pydantic import BaseModel

from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig


class StrictOutput(BaseModel):
    """Output model with required field for testing validation failures."""

    required_field: str
    optional_field: str | None = None


@pytest.mark.asyncio
async def test_tokens_tracked_on_validation_failure():
    """Test that tokens are tracked even when validation fails."""
    # Use a custom strategy that simulates Gemini API + parser validation
    from async_batch_llm.base import RetryState, TokenUsage
    from async_batch_llm.llm_strategies import LLMCallStrategy

    class FailingParserStrategy(LLMCallStrategy[StrictOutput]):
        """Strategy that returns tokens but fails validation."""

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[StrictOutput, TokenUsage]:
            # Simulate successful API call with tokens
            tokens: TokenUsage = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }

            # Simulate validation failure (parser raises ValidationError)
            # This is what happens with Gemini response_parser
            try:
                invalid_data = {"optional_field": "present"}  # missing required_field
                output = StrictOutput.model_validate(invalid_data)  # Will raise ValidationError
            except Exception as e:
                # Attach token usage to exception so framework can track it
                if not hasattr(e, "__dict__"):
                    # For built-in exceptions without __dict__, wrap in custom exception
                    class TokenTrackingError(Exception):
                        """Wrapper that adds token tracking to exception."""

                        pass

                    wrapped = TokenTrackingError(str(e))
                    wrapped.__cause__ = e
                    wrapped.__dict__["_failed_token_usage"] = tokens
                    raise wrapped from e
                else:
                    e.__dict__["_failed_token_usage"] = tokens
                    raise

            return output, tokens

    strategy = FailingParserStrategy()
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, StrictOutput, None](config=config) as processor:
        # Add 5 items that will all fail validation
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

    # Verify all items failed
    assert result.succeeded == 0
    assert result.failed == 5
    assert result.total_items == 5

    # BUG FIX VERIFICATION: Tokens should still be tracked despite failures
    assert result.total_input_tokens > 0, "Should track input tokens even on validation failure"
    assert result.total_output_tokens > 0, "Should track output tokens even on validation failure"

    # Verify token counts are correct (5 items * 3 attempts each * tokens per call)
    # Each item retries 3 times, so 5 * 3 = 15 API calls total
    assert result.total_input_tokens == 5 * 3 * 100
    assert result.total_output_tokens == 5 * 3 * 50

    # Verify errors are ValidationError (stored as string in error field)
    for item_result in result.results:
        assert not item_result.success
        assert item_result.error is not None
        # Check that error message contains ValidationError
        assert (
            "ValidationError" in item_result.error
        ), f"Expected ValidationError in error: {item_result.error}"
        assert (
            "required_field" in item_result.error
        ), f"Expected 'required_field' in error: {item_result.error}"


@pytest.mark.asyncio
async def test_cached_tokens_tracked_on_failure():
    """Ensure cached token usage is aggregated when retries all fail."""

    from async_batch_llm.base import RetryState, TokenUsage
    from async_batch_llm.llm_strategies import LLMCallStrategy

    class TokenTimeoutError(TimeoutError):
        """Custom timeout that allows attaching token usage."""

        pass

    class CachedFailStrategy(LLMCallStrategy[str]):
        def __init__(self):
            self.calls = 0

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            self.calls += 1
            tokens: TokenUsage = {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cached_input_tokens": 7,
            }
            exc = TokenTimeoutError("Simulated timeout")
            exc.__dict__["_failed_token_usage"] = tokens
            raise exc

    strategy = CachedFailStrategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="cached_failure", strategy=strategy, prompt="Test cached tokens")
        )
        result = await processor.process_all()

    assert result.failed == 1
    # Default retry config is 3 attempts; ensure cached tokens are counted for every attempt
    expected_cached_tokens = strategy.calls * 7
    assert (
        result.total_cached_tokens == expected_cached_tokens
    ), f"Expected cached tokens {expected_cached_tokens}, got {result.total_cached_tokens}"
