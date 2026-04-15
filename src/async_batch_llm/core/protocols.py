"""Type protocols for batch LLM processing framework."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from async_batch_llm.base import LLMResponse

# Constrain output to Pydantic models for validation
# Use covariant for protocols (output positions only)
TOutput = TypeVar("TOutput", bound=BaseModel, covariant=True)


class AgentLike(Protocol[TOutput]):
    """Protocol that any agent must satisfy."""

    async def run(self, prompt: str, **kwargs) -> ResultLike[TOutput]:
        """Run the agent with the given prompt."""
        ...


class ResultLike(Protocol[TOutput]):
    """Protocol for agent results."""

    @property
    def output(self) -> TOutput:
        """Get the agent output."""
        ...

    def usage(self) -> UsageLike | None:
        """Get token usage information."""
        ...

    def all_messages(self) -> list[Any]:
        """Get all messages in the conversation."""
        ...


class UsageLike(Protocol):
    """Protocol for token usage."""

    @property
    def request_tokens(self) -> int:
        """Number of tokens in the request."""
        ...

    @property
    def response_tokens(self) -> int:
        """Number of tokens in the response."""
        ...

    @property
    def total_tokens(self) -> int:
        """Total number of tokens."""
        ...


@runtime_checkable
class LLMModel(Protocol):
    """
    Protocol for LLM model instances that can generate responses.

    Implementations wrap a specific provider's client and model configuration,
    handling API calls and response normalization. Strategies call generate()
    without needing to know about provider-specific details.

    Added in v0.6.0.
    """

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float = 0.0,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: Text prompt, or list of content parts for multimodal input.
            temperature: Sampling temperature (0.0 = deterministic).
            system_instruction: System instruction override (None = use default).
            config: Provider-specific configuration (e.g., response_mime_type).

        Returns:
            Normalized LLMResponse with text, token counts, and metadata.
        """
        ...


@runtime_checkable
class ManagedLLMModel(LLMModel, Protocol):
    """
    LLMModel with lifecycle management (e.g., caching).

    Models that need one-time setup (creating a cache) or cleanup
    implement this protocol. The strategy delegates prepare/cleanup
    calls to the model.

    Added in v0.6.0.
    """

    async def prepare(self) -> None:
        """Initialize resources (e.g., find or create a cache). Must be idempotent."""
        ...

    async def cleanup(self) -> None:
        """Release resources. Must be idempotent."""
        ...
