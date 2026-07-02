"""Type protocols for batch LLM processing framework."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from async_batch_llm.base import LLMResponse


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
        temperature: float | None = 0.0,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: Text prompt, or list of content parts for multimodal input.
            temperature: Sampling temperature (0.0 = deterministic). Pass ``None``
                to omit the parameter entirely and use the provider default —
                required for models that reject an explicit temperature (e.g.
                OpenAI reasoning models like o1/o3).
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
