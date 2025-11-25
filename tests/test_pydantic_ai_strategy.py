"""Tests for PydanticAIStrategy to improve coverage."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from async_batch_llm import RetryState
from async_batch_llm.llm_strategies import PydanticAIStrategy
from async_batch_llm.testing import MockAgent


class SimpleOutput(BaseModel):
    """Simple output model for testing."""

    value: str


class ComplexOutput(BaseModel):
    """Complex output model with multiple fields."""

    name: str
    count: int
    active: bool = True


class TestPydanticAIStrategy:
    """Tests for PydanticAIStrategy."""

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execute functionality."""
        mock_agent = MockAgent(
            response_factory=lambda p: SimpleOutput(value=f"Response: {p}"),
            latency=0.001,
        )

        strategy = PydanticAIStrategy(agent=mock_agent)
        output, tokens = await strategy.execute("test prompt", 1, 10.0)

        assert output.value == "Response: test prompt"
        assert tokens["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_execute_with_state(self):
        """Test execute passes state parameter correctly."""
        mock_agent = MockAgent(
            response_factory=lambda p: SimpleOutput(value="result"),
            latency=0.001,
        )

        strategy = PydanticAIStrategy(agent=mock_agent)
        state = RetryState()
        state.set("key", "value")

        output, tokens = await strategy.execute("test", 1, 10.0, state=state)
        assert output.value == "result"

    @pytest.mark.asyncio
    async def test_dry_run_with_pydantic_model(self):
        """Test dry_run creates mock instance of Pydantic model."""
        mock_agent = MagicMock()
        mock_agent.result_type = SimpleOutput

        strategy = PydanticAIStrategy(agent=mock_agent)
        output, tokens = await strategy.dry_run("test prompt")

        # Should create a model_construct() instance
        assert isinstance(output, SimpleOutput)
        # Token usage should be based on prompt length
        assert tokens["input_tokens"] == len("test prompt".split())
        assert tokens["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_dry_run_with_complex_model(self):
        """Test dry_run with complex Pydantic model."""
        mock_agent = MagicMock()
        mock_agent.result_type = ComplexOutput

        strategy = PydanticAIStrategy(agent=mock_agent)
        output, tokens = await strategy.dry_run("longer test prompt here")

        assert isinstance(output, ComplexOutput)
        # model_construct creates instance without validation
        assert tokens["input_tokens"] == 4  # "longer test prompt here" = 4 words

    @pytest.mark.asyncio
    async def test_dry_run_non_pydantic_type(self):
        """Test dry_run falls back to base class for non-Pydantic types."""
        mock_agent = MagicMock()
        mock_agent.result_type = str  # Not a Pydantic model

        strategy = PydanticAIStrategy(agent=mock_agent)
        output, tokens = await strategy.dry_run("test prompt")

        # Should use base class default
        assert "[DRY-RUN]" in output
        assert tokens["input_tokens"] == 100

    @pytest.mark.asyncio
    async def test_dry_run_exception_fallback(self):
        """Test dry_run falls back to base class on exception."""
        mock_agent = MagicMock()
        # Make result_type access raise
        type(mock_agent).result_type = property(
            lambda self: (_ for _ in ()).throw(AttributeError())
        )

        strategy = PydanticAIStrategy(agent=mock_agent)
        output, tokens = await strategy.dry_run("test prompt")

        # Should use base class default
        assert "[DRY-RUN]" in output

    def test_import_error_when_pydantic_ai_not_available(self):
        """Test that ImportError is raised when pydantic-ai is not available."""
        from typing import Any

        with patch("async_batch_llm.llm_strategies.Agent", Any):
            with pytest.raises(ImportError) as exc_info:
                PydanticAIStrategy(agent=MagicMock())
            assert "pydantic-ai is required" in str(exc_info.value)


class TestPydanticAIStrategyTokenTracking:
    """Tests for token tracking in PydanticAIStrategy."""

    @pytest.mark.asyncio
    async def test_token_tracking_on_validation_error(self):
        """Test tokens are tracked even when output access fails."""
        # Create a mock result that raises on output access
        mock_result = MagicMock()
        mock_usage = MagicMock()
        mock_usage.request_tokens = 100
        mock_usage.response_tokens = 50
        mock_usage.total_tokens = 150
        mock_result.usage.return_value = mock_usage

        # Make output access raise a custom error
        class OutputAccessError(Exception):
            pass

        type(mock_result).output = property(
            lambda self: (_ for _ in ()).throw(OutputAccessError("Validation failed"))
        )

        # Create mock agent that returns this result
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        strategy = PydanticAIStrategy(agent=mock_agent)

        with pytest.raises(OutputAccessError) as exc_info:
            await strategy.execute("test", 1, 10.0)

        # Token usage should be attached
        assert exc_info.value.__dict__["_failed_token_usage"]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_token_tracking_no_usage(self):
        """Test handling when usage() returns None."""
        mock_result = MagicMock()
        mock_result.usage.return_value = None
        mock_result.output = SimpleOutput(value="result")

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        strategy = PydanticAIStrategy(agent=mock_agent)
        output, tokens = await strategy.execute("test", 1, 10.0)

        assert tokens["input_tokens"] == 0
        assert tokens["output_tokens"] == 0
        assert tokens["total_tokens"] == 0
