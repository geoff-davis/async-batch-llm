"""Concrete LLM model implementations.

Model classes wrap a provider's client and handle API calls, token extraction,
and response normalization. Strategies receive an LLMModel and call generate()
without knowing about provider-specific details.

Added in v0.6.0.
"""

import logging
import time
from typing import TYPE_CHECKING, Any

from .base import LLMResponse

# Conditional imports for optional dependencies
if TYPE_CHECKING:
    from google import genai
    from google.genai.types import Content
else:
    try:
        from google import genai
        from google.genai.types import Content
    except ImportError:
        genai = None  # type: ignore[assignment]
        Content = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


def _extract_metadata(response: Any) -> dict[str, Any] | None:
    """Extract safety ratings and finish reason from a Gemini response."""
    metadata: dict[str, Any] = {}

    try:
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Safety ratings
            if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                ratings: dict[str, str] = {}
                for rating in candidate.safety_ratings:
                    category = (
                        str(rating.category) if hasattr(rating, "category") else "UNKNOWN"
                    )
                    probability = (
                        str(rating.probability) if hasattr(rating, "probability") else "UNKNOWN"
                    )
                    ratings[category] = probability
                metadata["safety_ratings"] = ratings

            # Finish reason
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                metadata["finish_reason"] = str(candidate.finish_reason)
    except Exception as e:
        # Metadata extraction is best-effort; missing/partial attributes on the
        # provider response shouldn't break the call. Keep `except Exception` so
        # KeyboardInterrupt still propagates.
        logger.warning(f"Failed to extract response metadata: {e}", exc_info=True)

    return metadata or None


def _extract_tokens(response: Any) -> tuple[int, int, int, int]:
    """Extract token counts from a Gemini response.

    Returns:
        (input_tokens, output_tokens, total_tokens, cached_input_tokens)
    """
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        logger.debug(
            "No usage_metadata on response (%s); token counts will be zero.",
            type(response).__name__,
        )
        return 0, 0, 0, 0

    input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage_metadata, "total_token_count", 0) or 0
    cached_tokens = 0
    if hasattr(usage_metadata, "cached_content_token_count"):
        cached_tokens = getattr(usage_metadata, "cached_content_token_count", 0) or 0

    return input_tokens, output_tokens, total_tokens, cached_tokens


class GeminiModel:
    """
    LLM model backed by the Google Gemini API.

    Wraps a genai.Client and model name, handling API calls, token extraction,
    and response normalization. Implements the LLMModel protocol.

    Example:
        >>> client = genai.Client(api_key="...")
        >>> model = GeminiModel("gemini-2.5-flash", client)
        >>> response = await model.generate("Hello!")
        >>> print(response.text, response.input_tokens)

    Added in v0.6.0.
    """

    def __init__(
        self,
        model: str,
        client: "genai.Client",
        *,
        safety_settings: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
    ):
        """
        Args:
            model: Model name (e.g., "gemini-3.1-flash-lite-preview").
            client: Initialized genai.Client.
            safety_settings: Default safety settings for all calls.
            system_instruction: Default system instruction (overridable per-call).
        """
        if genai is None:
            raise ImportError(
                "google-genai is required for GeminiModel. "
                "Install with: pip install 'async-batch-llm[gemini]'"
            )

        self._model = model
        self._client = client
        self._safety_settings = safety_settings
        self._default_system_instruction = system_instruction

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float = 0.0,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response from Gemini.

        Args:
            prompt: Text prompt or list of content parts (multimodal).
            temperature: Sampling temperature.
            system_instruction: Override default system instruction.
            config: Additional provider-specific config entries.

        Returns:
            Normalized LLMResponse.
        """
        # Build config dict
        call_config: dict[str, Any] = {"temperature": temperature}

        si = system_instruction or self._default_system_instruction
        if si is not None:
            call_config["system_instruction"] = si

        if self._safety_settings:
            call_config["safety_settings"] = self._safety_settings

        if config:
            call_config.update(config)

        # Make the API call (config is built as a dict; the SDK accepts this at runtime
        # even though the type stubs say GenerateContentConfig)
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=call_config,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        )

        # Extract tokens
        input_tokens, output_tokens, total_tokens, cached_tokens = _extract_tokens(response)

        # Extract text (may be None if safety-blocked)
        text = response.text
        if text is None:
            metadata = _extract_metadata(response)
            safety_info = ""
            if metadata and "safety_ratings" in metadata:
                safety_info = f" Safety ratings: {metadata['safety_ratings']}"
            raise ValueError(
                f"Empty response from model (likely blocked by safety filter).{safety_info}"
            )

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
            metadata=_extract_metadata(response),
            raw=response,
        )


class GeminiCachedModel:
    """
    LLM model backed by Google Gemini with context caching.

    Wraps a genai.Client with cache lifecycle management. Implements the
    ManagedLLMModel protocol: call prepare() before first use, cleanup()
    when done.

    IMPORTANT — share one instance across work items.
        Create ONE GeminiCachedModel and reuse it across every LLMWorkItem that
        should share the cached context. Constructing a new instance per item
        defeats caching entirely and can cost 10× more. The framework calls
        prepare() exactly once per unique instance, so sharing is the intended
        lifecycle. See examples/example_gemini_cached.py for the pattern.

    This provides 70-90% cost savings when shared correctly.

    Example:
        >>> model = GeminiCachedModel(
        ...     "gemini-2.5-flash", client,
        ...     cached_content=[system_instruction, context_docs],
        ... )
        >>> await model.prepare()  # finds or creates cache
        >>> response = await model.generate("Process this")
        >>> await model.cleanup()  # preserves cache for reuse

    Added in v0.6.0.
    """

    def __init__(
        self,
        model: str,
        client: "genai.Client",
        cached_content: list["Content"],
        *,
        cache_ttl_seconds: int = 3600,
        cache_renewal_buffer_seconds: int = 300,
        auto_renew: bool = True,
        cache_tags: dict[str, str] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
    ):
        """
        Args:
            model: Model name (e.g., "gemini-2.5-flash").
            client: Initialized genai.Client.
            cached_content: Content to cache (system instructions, documents).
            cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour).
            cache_renewal_buffer_seconds: Renew this many seconds before expiry
                (default: 300 = 5 minutes).
            auto_renew: Auto-renew expired caches in generate() (default: True).
            cache_tags: Tags for precise cache matching.
            safety_settings: Default safety settings for all calls.
        """
        if genai is None:
            raise ImportError(
                "google-genai is required for GeminiCachedModel. "
                "Install with: pip install 'async-batch-llm[gemini]'"
            )

        if cache_renewal_buffer_seconds >= cache_ttl_seconds:
            raise ValueError(
                f"cache_renewal_buffer_seconds ({cache_renewal_buffer_seconds}) "
                f"must be less than cache_ttl_seconds ({cache_ttl_seconds})."
            )

        if 10 <= cache_ttl_seconds < 60:
            import warnings

            warnings.warn(
                f"cache_ttl_seconds ({cache_ttl_seconds}) is less than 60 seconds. "
                f"Very short TTLs defeat the purpose of caching. "
                f"Recommended minimum: 300 seconds (5 minutes).",
                UserWarning,
                stacklevel=2,
            )

        if cache_renewal_buffer_seconds < 60:
            import warnings

            warnings.warn(
                f"cache_renewal_buffer_seconds ({cache_renewal_buffer_seconds}) is less than "
                f"60 seconds. Small buffers risk renewing on every call if generation takes "
                f"longer than the buffer. Recommended minimum: 60 seconds.",
                UserWarning,
                stacklevel=2,
            )

        self._model = model
        self._client = client
        self._cached_content = cached_content
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache_renewal_buffer_seconds = cache_renewal_buffer_seconds
        self._auto_renew = auto_renew
        self._cache_tags = cache_tags or {}
        self._safety_settings = safety_settings

        self._cache: Any = None
        self._cache_created_at: float | None = None
        self._cache_lock: Any = None
        self._prepared = False


    @property
    def cache_name(self) -> str | None:
        """The name of the active cache, or None."""
        return self._cache.name if self._cache else None

    # ── Lifecycle ───────────────────────────────────────────────

    async def prepare(self) -> None:
        """Find or create the Gemini cache. Idempotent."""
        if self._prepared:
            return

        import asyncio

        if self._cache_lock is None:
            self._cache_lock = asyncio.Lock()

        async with self._cache_lock:
            if self._prepared:
                return
            await self._find_or_create_cache()
            self._prepared = True

    async def cleanup(self) -> None:
        """Preserve cache for reuse (does not delete). Idempotent."""
        if self._cache:
            logger.info(
                f"Leaving cache active for reuse: {self._cache.name} "
                f"(TTL: {self._cache_ttl_seconds}s, will expire naturally)"
            )

    async def delete_cache(self) -> None:
        """Explicitly delete the cache."""
        if self._cache:
            try:
                await self._client.aio.caches.delete(name=self._cache.name)
                logger.info(f"Deleted Gemini cache: {self._cache.name}")
                self._cache = None
                self._cache_created_at = None
                self._prepared = False
            except Exception as e:
                # Keep Exception (not BaseException) so KeyboardInterrupt still propagates;
                # cache-delete failures are best-effort — caches expire on their own.
                logger.warning(
                    f"Failed to delete Gemini cache '{self._cache.name}': {e}. "  # ty:ignore[unresolved-attribute]
                    "Cache may have already expired or been deleted.",
                    exc_info=True,
                )

    # ── Generate ────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float = 0.0,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response using the cached context.

        Args:
            prompt: Text prompt or multimodal content parts.
            temperature: Sampling temperature.
            system_instruction: Not supported with caching — raises ValueError.
            config: Additional provider-specific config entries.

        Returns:
            Normalized LLMResponse.
        """
        if system_instruction is not None:
            raise ValueError(
                "system_instruction cannot be overridden per-call with cached models. "
                "The system instruction is baked into the cache at creation time."
            )

        # Auto-renew if expired
        if self._auto_renew and self._is_cache_expired():
            logger.info(
                "Cache expired or about to expire, renewing before API call "
                f"(age: {time.time() - (self._cache_created_at or 0):.0f}s, "
                f"renewal buffer: {self._cache_renewal_buffer_seconds}s)"
            )

            import asyncio

            if self._cache_lock is None:
                self._cache_lock = asyncio.Lock()

            async with self._cache_lock:
                if self._is_cache_expired():
                    self._cache = None
                    self._cache_created_at = None
                    self._prepared = False
                    await self._find_or_create_cache()
                    self._prepared = True

        if self._cache is None:
            raise RuntimeError("Cache not initialized — call prepare() first")

        # Build config with cache reference
        call_config: dict[str, Any] = {
            "cached_content": self._cache.name,
            "temperature": temperature,
        }

        if self._safety_settings:
            call_config["safety_settings"] = self._safety_settings

        if config:
            call_config.update(config)

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=call_config,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        )

        input_tokens, output_tokens, total_tokens, cached_tokens = _extract_tokens(response)

        text = response.text
        if text is None:
            metadata = _extract_metadata(response)
            safety_info = ""
            if metadata and "safety_ratings" in metadata:
                safety_info = f" Safety ratings: {metadata['safety_ratings']}"
            raise ValueError(
                f"Empty response from model (likely blocked by safety filter).{safety_info}"
            )

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
            metadata=_extract_metadata(response),
            raw=response,
        )

    # ── Cache internals ─────────────────────────────────────────

    def _is_cache_expired(self) -> bool:
        if self._cache is None or self._cache_created_at is None:
            return True
        cache_age = time.time() - self._cache_created_at
        expires_in = self._cache_ttl_seconds - cache_age
        return expires_in <= self._cache_renewal_buffer_seconds

    async def _find_or_create_cache(self) -> None:
        try:
            caches = await self._client.aio.caches.list()

            async for cache in caches:
                if not cache.model or not cache.model.endswith(self._model):  # ty:ignore[unresolved-attribute]
                    continue

                if self._cache_tags:
                    cache_metadata = getattr(cache, "metadata", {}) or {}
                    tags_match = all(
                        cache_metadata.get(k) == v for k, v in self._cache_tags.items()
                    )
                    if not tags_match:
                        logger.debug(
                            f"Skipping cache {cache.name}: tags don't match "
                            f"(want {self._cache_tags}, has {cache_metadata})"
                        )
                        continue

                self._cache = cache
                if hasattr(cache, "create_time") and cache.create_time:
                    self._cache_created_at = cache.create_time.timestamp()
                else:
                    self._cache_created_at = time.time() - self._cache_ttl_seconds

                tag_info = f" with tags {self._cache_tags}" if self._cache_tags else ""
                age = time.time() - self._cache_created_at
                logger.info(
                    f"Reusing existing Gemini cache: {self._cache.name}{tag_info} "
                    f"(age: {age:.0f}s)"
                )
                return
        except Exception as e:
            logger.warning(f"Failed to list existing caches: {e}")

        await self._create_new_cache()

    async def _create_new_cache(self) -> None:
        from google.genai.types import CreateCachedContentConfig

        config_kwargs: dict[str, Any] = {
            "contents": self._cached_content,
            "ttl": f"{self._cache_ttl_seconds}s",
        }
        if self._cache_tags:
            config_kwargs["metadata"] = self._cache_tags

        self._cache = await self._client.aio.caches.create(
            model=self._model,
            config=CreateCachedContentConfig(**config_kwargs),
        )

        self._cache_created_at = time.time()
        tag_info = f" with tags {self._cache_tags}" if self._cache_tags else ""
        logger.info(
            f"Created new Gemini cache: {self._cache.name}{tag_info} "
            f"(TTL: {self._cache_ttl_seconds}s)"
        )
