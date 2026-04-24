"""LLM call strategies for flexible model configuration and execution.

This module provides strategy classes that encapsulate how LLM calls are made,
including model selection, response parsing, and retry behavior.

v0.6.0: Strategies now accept an LLMModel instead of raw client + model name.
        GeminiCachedStrategy and GeminiResponse removed — use
        GeminiStrategy(model=GeminiCachedModel(...)) instead.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload

from .base import LLMResponse, RetryState, TokenUsage
from .core.protocols import ManagedLLMModel
from .strategies.errors import TokenTrackingError

# Conditional imports for optional dependencies
if TYPE_CHECKING:
    from pydantic_ai import Agent

    from .core.protocols import LLMModel
else:
    try:
        from pydantic_ai import Agent
    except ImportError:
        Agent = Any  # type: ignore[misc,assignment]

# Module-level logger
logger = logging.getLogger(__name__)

TOutput = TypeVar("TOutput")


class LLMCallStrategy(ABC, Generic[TOutput]):
    """
    Abstract base class for LLM call strategies.

    A strategy encapsulates how LLM calls are made, including:
    - Resource initialization (caches, clients)
    - Call execution with retries
    - Resource cleanup

    The framework calls:
    1. prepare() once per unique strategy instance before its first execution
    2. execute() for each attempt (including retries)
    3. cleanup() once per prepared strategy when the processor exits or shuts down
    """

    async def prepare(self) -> None:
        """
        Initialize resources before making any LLM calls.

        Called once per unique strategy instance before the first work item using
        that instance executes. Use this to set up shared caches, clients, etc.
        Per-item retry state belongs in execute()/on_error() via RetryState.

        Default: no-op
        """
        pass

    @abstractmethod
    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: "RetryState | None" = None
    ) -> tuple[TOutput, TokenUsage]:
        """
        Execute an LLM call for the given attempt.

        Args:
            prompt: The prompt to send to the LLM
            attempt: Which retry attempt this is (1, 2, 3, ...)
            timeout: Maximum time to wait for response (seconds)
            state: Optional retry state that persists across attempts (v0.3.0)

        Returns:
            Tuple of (output, token_usage)
            where token_usage is a TokenUsage dict with optional keys:
            input_tokens, output_tokens, total_tokens, cached_input_tokens

        Raises:
            Any exception to trigger retry (if retryable) or failure

        Note (v0.3.0):
            The state parameter allows strategies to maintain state across retry
            attempts for multi-stage retry patterns. See RetryState documentation
            for examples.
        """
        pass

    async def cleanup(self) -> None:
        """
        Clean up resources when the processor exits or shuts down.

        Called once per prepared strategy instance, not once per work item.

        **Use this for:**
        - Closing connections/sessions
        - Releasing locks
        - Logging final metrics
        - Deleting temporary files

        **Do NOT use this for:**
        - Deleting caches intended for reuse across runs
        - Destructive cleanup that prevents resource reuse

        **Note on Caches (v0.2.0):**
        For reusable resources like Gemini caches with TTLs, consider letting
        them expire naturally to enable cost savings across multiple pipeline
        runs. See `GeminiCachedModel` for an example.

        Default: no-op
        """
        pass

    async def on_error(
        self, exception: Exception, attempt: int, state: "RetryState | None" = None
    ) -> None:
        """
        Handle errors that occur during execute().

        Called by the framework when execute() raises an exception, before
        deciding whether to retry. This allows strategies to:
        - Inspect the error type to adjust retry behavior
        - Store error information for use in next attempt
        - Modify prompts based on validation errors
        - Track error patterns across attempts

        Args:
            exception: The exception that was raised during execute()
            attempt: Which attempt number failed (1, 2, 3, ...)
            state: Optional retry state that persists across attempts (v0.3.0)

        Default: no-op

        Example (v0.2.0):
            async def on_error(self, exception: Exception, attempt: int) -> None:
                # Store last error for smart retry logic
                self.last_error = exception

                # Track validation errors vs network errors
                if isinstance(exception, ValidationError):
                    self.should_escalate_model = True

        Example (v0.3.0 with retry state):
            async def on_error(
                self, exception: Exception, attempt: int, state: RetryState | None = None
            ) -> None:
                if state:
                    # Track validation errors separately from other errors
                    if isinstance(exception, ValidationError):
                        count = state.get('validation_failures', 0) + 1
                        state.set('validation_failures', count)
                        # Save partial results for recovery
                        if hasattr(exception, 'partial_data'):
                            state.set('partial_data', exception.partial_data)
        """
        pass

    async def dry_run(self, prompt: str) -> tuple[TOutput, TokenUsage]:
        """
        Return mock output for dry-run mode (testing without API calls).

        Override this method to provide realistic mock data for testing.
        Default implementation returns placeholder values that may not match
        your output type.

        Args:
            prompt: The prompt that would have been sent to the LLM

        Returns:
            Tuple of (mock_output, mock_token_usage)

        Default behavior:
        - Returns string "[DRY-RUN] Mock output" as output
        - Returns mock token usage: 100 input, 50 output, 150 total
        """
        mock_output: TOutput = f"[DRY-RUN] Mock output for prompt: {prompt[:50]}..."  # ty:ignore[invalid-assignment]
        mock_tokens: TokenUsage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        return mock_output, mock_tokens


class GeminiStrategy(LLMCallStrategy[TOutput]):
    """
    Strategy for calling an LLM model and parsing the response.

    Accepts an LLMModel (e.g., GeminiModel or GeminiCachedModel) and a
    response parser. The model handles the API call and token extraction;
    the strategy handles response parsing and lifecycle delegation.

    For caching, use GeminiStrategy(model=GeminiCachedModel(...)).

    v0.6.0: Accepts LLMModel instead of raw client + model string.

    Example:
        >>> model = GeminiModel("gemini-2.5-flash", client)
        >>> strategy = GeminiStrategy(model, response_parser=lambda r: r.text)
        >>>
        >>> # With caching:
        >>> cached_model = GeminiCachedModel("gemini-2.5-flash", client, cached_content=[...])
        >>> strategy = GeminiStrategy(cached_model, response_parser=lambda r: r.text)
    """

    @overload
    def __init__(
        self: "GeminiStrategy[str]",
        model: "LLMModel",
        response_parser: None = None,
        *,
        temperature: float = 0.0,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: "LLMModel",
        response_parser: Callable[[LLMResponse], TOutput],
        *,
        temperature: float = 0.0,
    ) -> None: ...

    def __init__(
        self,
        model: "LLMModel",
        response_parser: Callable[[LLMResponse], TOutput] | None = None,
        *,
        temperature: float = 0.0,
    ) -> None:
        """
        Initialize strategy.

        Args:
            model: An LLMModel instance (e.g., GeminiModel, GeminiCachedModel).
            response_parser: Function to parse LLMResponse into TOutput. Defaults to
                returning response.text — only valid when TOutput is str. When
                TOutput is any other type, pass a response_parser (enforced by
                @overload signatures).
            temperature: Default sampling temperature (overridable by subclasses).
        """
        self.model = model
        # The overloads restrict the None-parser path to TOutput=str, so the cast
        # below is sound at static-analysis time.
        self.response_parser = response_parser or (lambda response: cast(TOutput, response.text))
        self.temperature = temperature

    async def prepare(self) -> None:
        """Delegate to model.prepare() if model has lifecycle."""
        if isinstance(self.model, ManagedLLMModel):
            await self.model.prepare()

    async def cleanup(self) -> None:
        """Delegate to model.cleanup() if model has lifecycle."""
        if isinstance(self.model, ManagedLLMModel):
            await self.model.cleanup()

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[TOutput, TokenUsage]:
        """Execute LLM call via the model and parse the response.

        Args:
            prompt: The prompt to send to the LLM.
            attempt: Which retry attempt this is (1, 2, 3, ...).
            timeout: Maximum time for response (enforced by framework).
            state: Optional retry state for cross-attempt persistence.

        Returns:
            Tuple of (parsed_output, token_usage).
        """
        llm_response = await self.model.generate(prompt, temperature=self.temperature)

        try:
            output = self.response_parser(llm_response)
        except Exception as e:
            # Attach token usage to exception so framework can track it
            tokens = llm_response.token_usage
            if not hasattr(e, "__dict__"):
                wrapped = TokenTrackingError(str(e), token_usage=tokens)
                wrapped.__cause__ = e
                raise wrapped from e
            else:
                e.__dict__["_failed_token_usage"] = tokens
                raise

        return output, llm_response.token_usage


class PydanticAIStrategy(LLMCallStrategy[TOutput]):
    """
    Strategy for using PydanticAI agents.

    This strategy wraps a PydanticAI agent, providing a clean interface
    for batch processing. The agent handles all model interaction, validation,
    and parsing.

    Best for: Structured output with Pydantic models, using PydanticAI's features.
    """

    def __init__(self, agent: "Agent[None, TOutput]"):
        """
        Initialize PydanticAI strategy.

        Args:
            agent: Configured PydanticAI agent
        """
        if Agent is Any:
            raise ImportError(
                "pydantic-ai is required for PydanticAIStrategy. "
                "Install with: pip install 'async-batch-llm[pydantic-ai]'"
            )

        self.agent = agent

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[TOutput, TokenUsage]:
        """Execute PydanticAI agent call.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().

        Args:
            prompt: The prompt to send to the LLM
            attempt: Which retry attempt this is (1, 2, 3, ...)
            timeout: Maximum time to wait for response (seconds)
            state: Optional retry state (v0.3.0, unused by this strategy)
        """
        result = await self.agent.run(prompt)

        # Extract token usage FIRST (before accessing result.output which may fail validation)
        usage = result.usage()
        tokens: TokenUsage = {
            "input_tokens": usage.request_tokens if usage else 0,
            "output_tokens": usage.response_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

        # Access result.output (may raise validation errors)
        try:
            output = result.output
        except Exception as e:
            # Attach token usage to exception so framework can track it
            if not hasattr(e, "__dict__"):
                # For built-in exceptions without __dict__, wrap in TokenTrackingError
                wrapped = TokenTrackingError(str(e), token_usage=tokens)
                wrapped.__cause__ = e
                raise wrapped from e
            else:
                e.__dict__["_failed_token_usage"] = tokens
                raise

        return output, tokens

    async def dry_run(self, prompt: str) -> tuple[TOutput, TokenUsage]:
        """Return mock output based on agent's result_type for dry-run mode."""
        # Try to create a mock instance of the expected output type
        try:
            from pydantic import BaseModel

            result_type = self.agent.result_type  # ty:ignore[unresolved-attribute]

            # If result_type is a Pydantic model, try to create an instance
            if isinstance(result_type, type) and issubclass(result_type, BaseModel):
                # Use model_construct to create instance without validation
                # This allows creating instances even with required fields
                mock_output: TOutput = result_type.model_construct()
            else:
                # For non-Pydantic types, use base class default
                return await super().dry_run(prompt)

        except Exception:
            # If anything fails, fall back to base class default
            return await super().dry_run(prompt)

        # Return mock output with realistic token usage
        mock_tokens: TokenUsage = {
            "input_tokens": len(prompt.split()),  # Rough estimate
            "output_tokens": 50,
            "total_tokens": len(prompt.split()) + 50,
        }

        return mock_output, mock_tokens
