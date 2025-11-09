"""LLM call strategies for flexible model configuration and execution.

This module provides strategy classes that encapsulate how LLM calls are made,
including caching, model selection, and retry behavior.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .base import TokenUsage

# Conditional imports for optional dependencies
if TYPE_CHECKING:
    from google import genai
    from google.genai.types import Content, GenerateContentConfig
    from pydantic_ai import Agent
else:
    try:
        from google import genai
        from google.genai.types import Content, GenerateContentConfig
    except ImportError:
        genai = None  # type: ignore[assignment]
        Content = Any  # type: ignore[misc,assignment]
        GenerateContentConfig = Any  # type: ignore[misc,assignment]

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
    1. prepare() once before any retries
    2. execute() for each attempt (including retries)
    3. cleanup() once after all attempts complete or fail
    """

    async def prepare(self) -> None:
        """
        Initialize resources before making any LLM calls.

        Called once per work item before any retry attempts.
        Use this to set up caches, initialize clients, etc.

        Default: no-op
        """
        pass

    @abstractmethod
    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, TokenUsage]:
        """
        Execute an LLM call for the given attempt.

        Args:
            prompt: The prompt to send to the LLM
            attempt: Which retry attempt this is (1, 2, 3, ...)
            timeout: Maximum time to wait for response (seconds)

        Returns:
            Tuple of (output, token_usage)
            where token_usage is a TokenUsage dict with optional keys:
            input_tokens, output_tokens, total_tokens, cached_input_tokens

        Raises:
            Any exception to trigger retry (if retryable) or failure
        """
        pass

    async def cleanup(self) -> None:
        """
        Clean up resources after all retry attempts complete.

        Called once per work item after processing finishes (success or failure).

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
        runs. See `GeminiCachedStrategy` for an example.

        Default: no-op
        """
        pass

    async def on_error(self, exception: Exception, attempt: int) -> None:
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

        Default: no-op

        Example:
            async def on_error(self, exception: Exception, attempt: int) -> None:
                # Store last error for smart retry logic
                self.last_error = exception

                # Track validation errors vs network errors
                if isinstance(exception, ValidationError):
                    self.should_escalate_model = True
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
        mock_output: TOutput = f"[DRY-RUN] Mock output for prompt: {prompt[:50]}..."  # type: ignore[assignment]
        mock_tokens: TokenUsage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        return mock_output, mock_tokens


class GeminiStrategy(LLMCallStrategy[TOutput]):
    """
    Strategy for calling Google Gemini API directly.

    This strategy uses the google-genai SDK to make direct API calls
    without caching. Best for one-off calls or when caching isn't needed.
    """

    def __init__(
        self,
        model: str,
        client: "genai.Client",
        response_parser: Callable[[Any], TOutput],
        config: "GenerateContentConfig | None" = None,
    ):
        """
        Initialize Gemini strategy.

        Args:
            model: Model name (e.g., "gemini-2.5-flash")
            client: Initialized Gemini client
            response_parser: Function to parse response into TOutput
            config: Optional generation config (temperature, etc.)
        """
        if genai is None:
            raise ImportError(
                "google-genai is required for GeminiStrategy. "
                "Install with: pip install 'batch-llm[gemini]'"
            )

        self.model = model
        self.client = client
        self.response_parser = response_parser
        self.config = config

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, TokenUsage]:
        """Execute Gemini API call.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        # Make the call
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.config,
        )

        # Parse output
        output = self.response_parser(response)

        # Extract token usage
        usage = response.usage_metadata
        tokens: TokenUsage = {
            "input_tokens": usage.prompt_token_count or 0 if usage else 0,
            "output_tokens": usage.candidates_token_count or 0 if usage else 0,
            "total_tokens": usage.total_token_count or 0 if usage else 0,
        }

        return output, tokens


class GeminiCachedStrategy(LLMCallStrategy[TOutput]):
    """
    Strategy for calling Google Gemini API with context caching.

    This strategy creates a Gemini cache for the system instruction and/or
    initial context, then uses it across all retry attempts. The cache is
    automatically refreshed if it's close to expiring, and deleted on cleanup.

    Best for: Repeated calls with large shared context (RAG, long documents).
    """

    def __init__(
        self,
        model: str,
        client: "genai.Client",
        response_parser: Callable[[Any], TOutput],
        cached_content: list["Content"],
        cache_ttl_seconds: int = 3600,
        cache_refresh_threshold: float = 0.1,  # Deprecated in favor of cache_renewal_buffer_seconds
        cache_renewal_buffer_seconds: int = 300,  # v0.2.0: Renew 5min before expiration
        auto_renew: bool = True,  # v0.2.0: Automatically renew expired caches
        config: "GenerateContentConfig | None" = None,
    ):
        """
        Initialize Gemini cached strategy with automatic cache renewal (v0.2.0).

        Args:
            model: Model name (e.g., "gemini-2.5-flash")
            client: Initialized Gemini client
            response_parser: Function to parse response into TOutput
            cached_content: Content to cache (system instructions, documents)
            cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour)
            cache_refresh_threshold: (Deprecated) Use cache_renewal_buffer_seconds instead
            cache_renewal_buffer_seconds: Renew cache this many seconds before expiration
                to avoid expiration errors (default: 300 = 5 minutes)
            auto_renew: Automatically renew expired caches in execute() (default: True)
            config: Optional generation config
        """
        if genai is None:
            raise ImportError(
                "google-genai is required for GeminiCachedStrategy. "
                "Install with: pip install 'batch-llm[gemini]'"
            )

        self.model = model
        self.client = client
        self.response_parser = response_parser
        self.cached_content = cached_content
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_refresh_threshold = cache_refresh_threshold  # Deprecated
        self.cache_renewal_buffer_seconds = cache_renewal_buffer_seconds  # v0.2.0
        self.auto_renew = auto_renew  # v0.2.0
        self.config = config

        self._cache: Any = None  # Type: CachedContent after prepare()
        self._cache_created_at: float | None = None
        self._cache_lock: Any = None  # v0.2.0: asyncio.Lock, created in prepare()

        # Detect API version (v0.2.0)
        self._api_version = self._detect_google_genai_version()
        logger.debug(f"Detected google-genai API version: {self._api_version}")

    @staticmethod
    def _detect_google_genai_version() -> str:
        """
        Detect which google-genai API version is installed.

        Returns:
            "v1.46+" if new API (with CreateCachedContentConfig)
            "v1.45" if legacy API
        """
        try:
            from google.genai.types import CreateCachedContentConfig  # noqa: F401

            return "v1.46+"
        except ImportError:
            return "v1.45"

    def _is_cache_expired(self) -> bool:
        """
        Check if cache has expired or is about to expire (v0.2.0).

        Returns:
            True if cache should be renewed, False if still valid
        """
        if self._cache is None or self._cache_created_at is None:
            return True

        cache_age = time.time() - self._cache_created_at
        expires_in = self.cache_ttl_seconds - cache_age

        return expires_in <= self.cache_renewal_buffer_seconds

    async def _find_or_create_cache(self) -> None:
        """
        Find existing cache or create new one (v0.2.0).

        Attempts to reuse existing caches with the same model to save costs.
        If no suitable cache found, creates a new one.
        """
        import asyncio

        # Initialize lock if not already done
        if self._cache_lock is None:
            self._cache_lock = asyncio.Lock()

        # Try to find existing cache with same model
        try:
            caches = await self.client.aio.caches.list()

            for cache in caches:
                # Cache model is full path: "projects/.../models/gemini-..."
                # Match by model name suffix
                if cache.model.endswith(self.model):
                    self._cache = cache

                    # CRITICAL: Use cache's actual creation time, not current time
                    # This prevents expiration detection bugs
                    if hasattr(cache, "create_time") and cache.create_time:
                        self._cache_created_at = cache.create_time.timestamp()
                    else:
                        # Fallback: assume old to trigger renewal check
                        self._cache_created_at = time.time() - self.cache_ttl_seconds

                    logger.info(
                        f"Reusing existing Gemini cache: {self._cache.name} "
                        f"(age: {time.time() - self._cache_created_at:.0f}s)"
                    )
                    return
        except Exception as e:
            logger.warning(f"Failed to list existing caches: {e}")

        # No existing cache found, create new one
        await self._create_new_cache()

    async def _create_new_cache(self) -> None:
        """Create a new Gemini cache (v0.2.0)."""
        if self._api_version == "v1.46+":
            # New API (google-genai v1.46+)
            from google.genai.types import CreateCachedContentConfig

            self._cache = await self.client.aio.caches.create(  # type: ignore[call-arg]
                model=self.model,
                config=CreateCachedContentConfig(
                    contents=self.cached_content,
                    ttl=f"{self.cache_ttl_seconds}s",
                ),
            )
        else:
            # Legacy API (google-genai v1.45 and earlier)
            self._cache = await self.client.aio.caches.create(  # type: ignore[call-arg]
                model=self.model,
                contents=self.cached_content,
                ttl=f"{self.cache_ttl_seconds}s",
            )

        self._cache_created_at = time.time()
        logger.info(
            f"Created new Gemini cache: {self._cache.name} "
            f"(TTL: {self.cache_ttl_seconds}s, API: {self._api_version})"
        )

    async def prepare(self) -> None:
        """Find or create the Gemini cache (v0.2.0)."""
        await self._find_or_create_cache()

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, TokenUsage]:
        """Execute Gemini API call with automatic cache renewal (v0.2.0).

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        # Check and renew cache if expired (proactive renewal to avoid errors)
        if self.auto_renew and self._is_cache_expired():
            logger.info(
                "Cache expired or about to expire, renewing before API call "
                f"(age: {time.time() - (self._cache_created_at or 0):.0f}s, "
                f"renewal buffer: {self.cache_renewal_buffer_seconds}s)"
            )

            # Use lock to prevent concurrent renewal
            if self._cache_lock is None:
                import asyncio

                self._cache_lock = asyncio.Lock()

            async with self._cache_lock:
                # Double-check after acquiring lock
                if self._is_cache_expired():
                    # Clear cache reference to force creation of new cache
                    self._cache = None
                    self._cache_created_at = None
                    await self._find_or_create_cache()

        if self._cache is None:
            raise RuntimeError("Cache not initialized - prepare() was not called")

        # Make the call using the cache - Google genai API, type stubs may be incomplete
        response = await self.client.aio.models.generate_content(  # type: ignore[call-arg]
            model=self.model,
            contents=prompt,
            config=self.config,
            cached_content=self._cache.name,
        )

        # Parse output
        output = self.response_parser(response)

        # Extract token usage (cached tokens counted separately)
        usage = response.usage_metadata
        tokens: TokenUsage = {
            "input_tokens": usage.prompt_token_count or 0 if usage else 0,
            "output_tokens": usage.candidates_token_count or 0 if usage else 0,
            "total_tokens": usage.total_token_count or 0 if usage else 0,
            "cached_input_tokens": (
                usage.cached_content_token_count or 0
                if usage and hasattr(usage, "cached_content_token_count")
                else 0
            ),
        }

        return output, tokens

    async def cleanup(self) -> None:
        """
        Cleanup hook - preserves cache for reuse by default (v0.2.0).

        By default, this method does NOT delete the cache. The cache remains
        active until its TTL expires, allowing reuse across multiple runs
        within the TTL window (e.g., 1 hour).

        This enables significant cost savings when running multiple batches:
        - First run: Creates cache, pays full cost
        - Subsequent runs (within TTL): Reuse cache, 70-90% cost reduction

        To delete the cache immediately (e.g., for cleanup in tests), call:
            await strategy.delete_cache()

        See docs/GEMINI_INTEGRATION.md for cache lifecycle best practices.
        """
        if self._cache:
            logger.info(
                f"Leaving cache active for reuse: {self._cache.name} "
                f"(TTL: {self.cache_ttl_seconds}s, will expire naturally)"
            )

    async def delete_cache(self) -> None:
        """
        Explicitly delete the Gemini cache (v0.2.0).

        Call this when you want to immediately delete the cache instead of
        letting it expire naturally. Useful for:
        - Test cleanup
        - One-off batch jobs where reuse isn't needed
        - Updating cached content (delete old, create new)

        Example:
            strategy = GeminiCachedStrategy(...)
            # ... use strategy ...
            await strategy.delete_cache()  # Explicit cleanup
        """
        if self._cache:
            try:
                await self.client.aio.caches.delete(name=self._cache.name)
                logger.info(f"Deleted Gemini cache: {self._cache.name}")
                self._cache = None
                self._cache_created_at = None
            except Exception as e:
                logger.warning(
                    f"Failed to delete Gemini cache '{self._cache.name}': {e}. "
                    "Cache may have already expired or been deleted."
                )


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
                "Install with: pip install 'batch-llm[pydantic-ai]'"
            )

        self.agent = agent

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, TokenUsage]:
        """Execute PydanticAI agent call.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        result = await self.agent.run(prompt)

        # Extract token usage
        usage = result.usage()
        tokens: TokenUsage = {
            "input_tokens": usage.request_tokens if usage else 0,
            "output_tokens": usage.response_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

        return result.output, tokens

    async def dry_run(self, prompt: str) -> tuple[TOutput, TokenUsage]:
        """Return mock output based on agent's result_type for dry-run mode."""
        # Try to create a mock instance of the expected output type
        try:
            from pydantic import BaseModel

            result_type = self.agent.result_type  # type: ignore[attr-defined]

            # If result_type is a Pydantic model, try to create an instance
            if isinstance(result_type, type) and issubclass(result_type, BaseModel):
                # Use model_construct to create instance without validation
                # This allows creating instances even with required fields
                mock_output: TOutput = result_type.model_construct()  # type: ignore[assignment]
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
