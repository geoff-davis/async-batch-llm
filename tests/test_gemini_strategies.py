"""Comprehensive tests for Gemini strategies and models to improve coverage."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from async_batch_llm import RetryState, TokenTrackingError
from async_batch_llm.base import LLMResponse
from async_batch_llm.llm_strategies import (
    GeminiStrategy,
    LLMCallStrategy,
)
from async_batch_llm.models import (
    GeminiCachedModel,
    GeminiModel,
    _extract_metadata,
)


class AsyncIterList:
    """Wrap a list to make it async-iterable (for mocking AsyncPager)."""

    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class TestOutput(BaseModel):
    """Test output model."""

    text: str


# =============================================================================
# LLMCallStrategy base class tests
# =============================================================================


class TestLLMCallStrategyBase:
    """Tests for LLMCallStrategy abstract base class."""

    @pytest.mark.asyncio
    async def test_dry_run_default(self):
        """Test default dry_run implementation."""

        class SimpleStrategy(LLMCallStrategy[str]):
            async def execute(
                self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
            ):
                return "result", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        strategy = SimpleStrategy()
        output, tokens = await strategy.dry_run("Test prompt here")

        assert "[DRY-RUN]" in output
        assert "Test prompt here" in output
        assert tokens["input_tokens"] == 100
        assert tokens["output_tokens"] == 50
        assert tokens["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_prepare_default_noop(self):
        """Test that default prepare() is a no-op."""

        class SimpleStrategy(LLMCallStrategy[str]):
            async def execute(
                self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
            ):
                return "result", {}

        strategy = SimpleStrategy()
        await strategy.prepare()  # Should not raise

    @pytest.mark.asyncio
    async def test_cleanup_default_noop(self):
        """Test that default cleanup() is a no-op."""

        class SimpleStrategy(LLMCallStrategy[str]):
            async def execute(
                self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
            ):
                return "result", {}

        strategy = SimpleStrategy()
        await strategy.cleanup()  # Should not raise

    @pytest.mark.asyncio
    async def test_on_error_default_noop(self):
        """Test that default on_error() is a no-op."""

        class SimpleStrategy(LLMCallStrategy[str]):
            async def execute(
                self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
            ):
                return "result", {}

        strategy = SimpleStrategy()
        await strategy.on_error(ValueError("test"), 1, None)  # Should not raise


# =============================================================================
# GeminiModel tests
# =============================================================================


class TestGeminiModel:
    """Tests for GeminiModel."""

    def _create_mock_response(
        self,
        text: str = "output text",
        input_tokens: int = 10,
        output_tokens: int = 20,
        total_tokens: int = 30,
        has_safety_ratings: bool = False,
        has_finish_reason: bool = False,
    ):
        """Create a mock Gemini API response."""
        mock_response = MagicMock()
        mock_response.text = text
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = input_tokens
        mock_response.usage_metadata.candidates_token_count = output_tokens
        mock_response.usage_metadata.total_token_count = total_tokens
        mock_response.usage_metadata.cached_content_token_count = 0

        if has_safety_ratings or has_finish_reason:
            mock_candidate = MagicMock()
            if has_safety_ratings:
                mock_rating = MagicMock()
                mock_rating.category = "HARM_CATEGORY_HATE_SPEECH"
                mock_rating.probability = "LOW"
                mock_candidate.safety_ratings = [mock_rating]
            else:
                mock_candidate.safety_ratings = None

            if has_finish_reason:
                mock_candidate.finish_reason = "STOP"

            mock_response.candidates = [mock_candidate]
        else:
            mock_response.candidates = []

        return mock_response

    def _create_mock_client(self, response):
        """Create a mock Gemini client."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=response)
        return mock_client

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic generate functionality."""
        mock_response = self._create_mock_response()
        mock_client = self._create_mock_client(mock_response)

        model = GeminiModel("gemini-test", mock_client)
        llm_response = await model.generate("test prompt")

        assert llm_response.text == "output text"
        assert llm_response.input_tokens == 10
        assert llm_response.output_tokens == 20
        assert llm_response.total_tokens == 30

    @pytest.mark.asyncio
    async def test_generate_with_metadata(self):
        """Test generate extracts safety ratings and finish reason."""
        mock_response = self._create_mock_response(
            has_safety_ratings=True,
            has_finish_reason=True,
        )
        mock_client = self._create_mock_client(mock_response)

        model = GeminiModel("gemini-test", mock_client)
        llm_response = await model.generate("test prompt")

        assert llm_response.metadata is not None
        assert "safety_ratings" in llm_response.metadata
        assert "HARM_CATEGORY_HATE_SPEECH" in llm_response.metadata["safety_ratings"]
        assert llm_response.metadata["finish_reason"] == "STOP"
        assert llm_response.raw is mock_response

    @pytest.mark.asyncio
    async def test_generate_no_usage_metadata(self):
        """Test generate when response has no usage_metadata."""
        mock_response = MagicMock()
        mock_response.text = "output"
        mock_response.usage_metadata = None
        mock_response.candidates = []

        mock_client = self._create_mock_client(mock_response)

        model = GeminiModel("gemini-test", mock_client)
        llm_response = await model.generate("test prompt")

        assert llm_response.input_tokens == 0
        assert llm_response.output_tokens == 0
        assert llm_response.total_tokens == 0

    @pytest.mark.asyncio
    async def test_generate_with_system_instruction(self):
        """Test generate passes system_instruction to config."""
        mock_response = self._create_mock_response()
        mock_client = self._create_mock_client(mock_response)

        model = GeminiModel("gemini-test", mock_client, system_instruction="Be helpful")
        await model.generate("test prompt")

        call_kwargs = mock_client.aio.models.generate_content.call_args
        assert call_kwargs.kwargs["config"]["system_instruction"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_generate_none_text_raises(self):
        """Test generate raises ValueError when text is None (safety blocked)."""
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 0
        mock_response.usage_metadata.total_token_count = 10
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_response.candidates = []

        mock_client = self._create_mock_client(mock_response)

        model = GeminiModel("gemini-test", mock_client)

        with pytest.raises(ValueError, match="Empty response from model"):
            await model.generate("test prompt")

    def test_import_error_when_genai_not_available(self):
        """Test that ImportError is raised when google-genai is not available."""
        with patch("async_batch_llm.models.genai", None):
            with pytest.raises(ImportError) as exc_info:
                GeminiModel("gemini-test", MagicMock())
            assert "google-genai is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_token_usage_property(self):
        """Test that LLMResponse.token_usage returns correct dict."""
        mock_response = self._create_mock_response()
        mock_client = self._create_mock_client(mock_response)

        model = GeminiModel("gemini-test", mock_client)
        llm_response = await model.generate("test")

        tokens = llm_response.token_usage
        assert tokens["input_tokens"] == 10
        assert tokens["output_tokens"] == 20
        assert tokens["total_tokens"] == 30


# =============================================================================
# GeminiStrategy tests
# =============================================================================


class TestGeminiStrategy:
    """Tests for GeminiStrategy with mocked LLMModel."""

    def _create_mock_model(
        self,
        text: str = "parsed output",
        input_tokens: int = 10,
        output_tokens: int = 20,
        total_tokens: int = 30,
        metadata: dict | None = None,
    ):
        """Create a mock LLMModel."""
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(
            return_value=LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                metadata=metadata,
            )
        )
        return mock_model

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execute functionality."""
        mock_model = self._create_mock_model()

        strategy = GeminiStrategy(
            model=mock_model,
            response_parser=lambda r: TestOutput(text=r.text),
        )

        output, tokens = await strategy.execute("test prompt", 1, 10.0)

        assert output.text == "parsed output"
        assert tokens["input_tokens"] == 10
        assert tokens["output_tokens"] == 20
        assert tokens["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_execute_response_parser_receives_llm_response(self):
        """Test that response_parser receives LLMResponse object."""
        mock_model = self._create_mock_model(
            text="hello",
            metadata={"safety_ratings": {"HARM_CATEGORY_HATE_SPEECH": "LOW"}},
        )

        received_responses = []

        def parser(r):
            received_responses.append(r)
            return TestOutput(text=r.text)

        strategy = GeminiStrategy(model=mock_model, response_parser=parser)
        await strategy.execute("test", 1, 10.0)

        assert len(received_responses) == 1
        assert isinstance(received_responses[0], LLMResponse)
        assert received_responses[0].text == "hello"
        assert (
            received_responses[0].metadata["safety_ratings"]["HARM_CATEGORY_HATE_SPEECH"] == "LOW"
        )

    @pytest.mark.asyncio
    async def test_execute_parser_raises_with_dict(self):
        """Test that parser exceptions with __dict__ preserve token usage."""
        mock_model = self._create_mock_model()

        class CustomError(Exception):
            pass

        def failing_parser(r):
            raise CustomError("Parse failed")

        strategy = GeminiStrategy(
            model=mock_model,
            response_parser=failing_parser,
        )

        with pytest.raises(CustomError) as exc_info:
            await strategy.execute("test prompt", 1, 10.0)

        # Token usage should be attached to exception
        assert exc_info.value.__dict__["_failed_token_usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_execute_parser_raises_builtin_exception(self):
        """Test that builtin exceptions without __dict__ get wrapped."""
        mock_model = self._create_mock_model()

        def failing_parser(r):
            raise ValueError("Parse failed")

        strategy = GeminiStrategy(
            model=mock_model,
            response_parser=failing_parser,
        )

        # ValueError has __dict__, so it won't be wrapped
        with pytest.raises(ValueError) as exc_info:
            await strategy.execute("test prompt", 1, 10.0)

        assert exc_info.value.__dict__["_failed_token_usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_prepare_delegates_to_managed_model(self):
        """Test that prepare() delegates to ManagedLLMModel."""
        mock_client = MagicMock()
        mock_client.aio.caches.list = AsyncMock(return_value=AsyncIterList([]))
        mock_cache = MagicMock()
        mock_cache.name = "test-cache"
        mock_client.aio.caches.create = AsyncMock(return_value=mock_cache)

        cached_model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
            cache_ttl_seconds=3600,
        )

        strategy = GeminiStrategy(
            model=cached_model,
            response_parser=lambda r: r.text,
        )

        await strategy.prepare()

        # Should have called through to create cache
        mock_client.aio.caches.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_delegates_to_managed_model(self):
        """Test that cleanup() delegates to ManagedLLMModel."""
        mock_client = MagicMock()
        mock_client.aio.caches.list = AsyncMock(return_value=AsyncIterList([]))
        mock_cache = MagicMock()
        mock_cache.name = "test-cache"
        mock_client.aio.caches.create = AsyncMock(return_value=mock_cache)

        cached_model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
            cache_ttl_seconds=3600,
        )

        strategy = GeminiStrategy(
            model=cached_model,
            response_parser=lambda r: r.text,
        )

        await strategy.prepare()
        # cleanup should not raise
        await strategy.cleanup()

    @pytest.mark.asyncio
    async def test_prepare_noop_for_plain_model(self):
        """Test that prepare() is a no-op for non-managed models."""
        mock_model = self._create_mock_model()

        strategy = GeminiStrategy(
            model=mock_model,
            response_parser=lambda r: r.text,
        )

        # Should not raise (plain model is not ManagedLLMModel)
        await strategy.prepare()


# =============================================================================
# GeminiCachedModel tests
# =============================================================================


class TestGeminiCachedModel:
    """Tests for GeminiCachedModel."""

    def _create_mock_cache(self, name: str = "test-cache", create_time=None):
        """Create a mock cache object."""
        mock_cache = MagicMock()
        mock_cache.name = name
        mock_cache.model = "projects/test/models/gemini-test"
        if create_time:
            mock_cache.create_time = MagicMock()
            mock_cache.create_time.timestamp.return_value = create_time
        else:
            mock_cache.create_time = None
        return mock_cache

    def _create_mock_response(self, cached_tokens: int = 100):
        """Create a mock Gemini response with cached tokens."""
        mock_response = MagicMock()
        mock_response.text = "cached response text"
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30
        mock_response.usage_metadata.cached_content_token_count = cached_tokens
        mock_response.candidates = []
        return mock_response

    def _create_mock_client(self, caches=None, response=None):
        """Create a mock Gemini client."""
        mock_client = MagicMock()
        mock_client.aio.caches.list = AsyncMock(return_value=AsyncIterList(list(caches or [])))
        mock_client.aio.caches.create = AsyncMock(return_value=self._create_mock_cache())
        mock_client.aio.caches.delete = AsyncMock()
        if response:
            mock_client.aio.models.generate_content = AsyncMock(return_value=response)
        return mock_client

    def test_validation_renewal_buffer_too_large(self):
        """Test validation rejects renewal buffer >= TTL."""
        mock_client = self._create_mock_client()

        with pytest.raises(ValueError) as exc_info:
            GeminiCachedModel(
                model="gemini-test",
                client=mock_client,
                cached_content=[],
                cache_ttl_seconds=60,
                cache_renewal_buffer_seconds=60,  # Same as TTL - invalid
            )
        assert "must be less than cache_ttl_seconds" in str(exc_info.value)

    def test_validation_short_ttl_warning(self):
        """Test warning for short TTL values."""
        mock_client = self._create_mock_client()

        with pytest.warns(UserWarning) as record:
            GeminiCachedModel(
                model="gemini-test",
                client=mock_client,
                cached_content=[],
                cache_ttl_seconds=30,  # Short TTL
                cache_renewal_buffer_seconds=5,
            )

        assert len(record) == 1
        assert "less than 60 seconds" in str(record[0].message)

    def test_no_warning_for_test_ttl(self):
        """Test no warning for very short TTLs (< 10s, for testing)."""
        import warnings

        mock_client = self._create_mock_client()

        # Should not warn for TTL < 10 (testing)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GeminiCachedModel(
                model="gemini-test",
                client=mock_client,
                cached_content=[],
                cache_ttl_seconds=5,  # Very short TTL for testing
                cache_renewal_buffer_seconds=1,
            )

            # Filter for UserWarning about TTL
            ttl_warnings = [x for x in w if "less than 60 seconds" in str(x.message)]
            assert len(ttl_warnings) == 0

    @pytest.mark.asyncio
    async def test_find_existing_cache(self):
        """Test that existing cache is reused."""
        existing_cache = self._create_mock_cache(
            name="existing-cache",
            create_time=time.time() - 100,  # Created 100s ago
        )
        mock_client = self._create_mock_client(caches=[existing_cache])

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
            cache_ttl_seconds=3600,
        )

        await model.prepare()

        # Should not create new cache
        mock_client.aio.caches.create.assert_not_called()
        assert model._cache == existing_cache

    @pytest.mark.asyncio
    async def test_find_cache_with_matching_tags(self):
        """Test cache matching with tags."""
        # Cache with matching tags
        matching_cache = self._create_mock_cache(name="matching-cache", create_time=time.time())
        matching_cache.metadata = {"version": "1.0", "type": "test"}

        # Cache without matching tags
        other_cache = self._create_mock_cache(name="other-cache", create_time=time.time())
        other_cache.metadata = {"version": "2.0"}

        mock_client = self._create_mock_client(caches=[other_cache, matching_cache])

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
            cache_tags={"version": "1.0", "type": "test"},
        )

        await model.prepare()

        assert model._cache == matching_cache

    @pytest.mark.asyncio
    async def test_cache_list_failure(self):
        """Test fallback to create when cache list fails."""
        mock_client = self._create_mock_client()
        mock_client.aio.caches.list = AsyncMock(side_effect=RuntimeError("List failed"))

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )

        await model.prepare()

        # Should create new cache after list failure
        mock_client.aio.caches.create.assert_called_once()

    def test_is_cache_expired_no_cache(self):
        """Test _is_cache_expired returns True when no cache."""
        mock_client = self._create_mock_client()

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )

        assert model._is_cache_expired() is True

    def test_is_cache_expired_within_buffer(self):
        """Test _is_cache_expired returns True when within renewal buffer."""
        mock_client = self._create_mock_client()

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
            cache_ttl_seconds=100,
            cache_renewal_buffer_seconds=30,
        )

        model._cache = MagicMock()
        # Created 80s ago, so only 20s remain (< 30s buffer)
        model._cache_created_at = time.time() - 80

        assert model._is_cache_expired() is True

    @pytest.mark.asyncio
    async def test_generate_without_prepare_raises(self):
        """Test generate raises when prepare() not called."""
        mock_client = self._create_mock_client()

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
            auto_renew=False,  # Disable auto-renew to trigger error
        )

        with pytest.raises(RuntimeError) as exc_info:
            await model.generate("test")

        assert "prepare()" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_with_cached_tokens(self):
        """Test generate tracks cached tokens."""
        mock_response = self._create_mock_response(cached_tokens=500)
        mock_client = self._create_mock_client(response=mock_response)

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )

        await model.prepare()
        llm_response = await model.generate("test")

        assert llm_response.cached_input_tokens == 500
        assert llm_response.text == "cached response text"

    @pytest.mark.asyncio
    async def test_generate_auto_renew_concurrent(self):
        """Test auto-renewal uses lock for concurrent access."""
        mock_response = self._create_mock_response()
        mock_client = self._create_mock_client(response=mock_response)
        mock_client.aio.caches.create = AsyncMock(return_value=self._create_mock_cache())

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
            cache_ttl_seconds=10,
            cache_renewal_buffer_seconds=5,
            auto_renew=True,
        )

        # Manually set up expired cache
        await model.prepare()
        model._cache_created_at = time.time() - 100  # Expired

        # Generate should trigger renewal
        await model.generate("test")

        # Should have created twice (initial + renewal)
        assert mock_client.aio.caches.create.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_with_metadata(self):
        """Test generate returns metadata in LLMResponse."""
        mock_response = self._create_mock_response()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.candidates[0].finish_reason = "STOP"

        mock_client = self._create_mock_client(response=mock_response)

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )

        await model.prepare()
        llm_response = await model.generate("test")

        assert isinstance(llm_response, LLMResponse)
        assert llm_response.metadata is not None
        assert llm_response.metadata["finish_reason"] == "STOP"

    @pytest.mark.asyncio
    async def test_cleanup_preserves_cache(self):
        """Test cleanup preserves cache by default."""
        mock_client = self._create_mock_client()

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )

        await model.prepare()
        await model.cleanup()

        mock_client.aio.caches.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_cache_explicit(self):
        """Test explicit cache deletion."""
        mock_client = self._create_mock_client()

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )

        await model.prepare()
        await model.delete_cache()

        mock_client.aio.caches.delete.assert_called_once()
        assert model._cache is None

    @pytest.mark.asyncio
    async def test_delete_cache_handles_failure(self):
        """Test delete_cache handles deletion failure gracefully."""
        mock_client = self._create_mock_client()
        mock_client.aio.caches.delete = AsyncMock(side_effect=RuntimeError("Delete failed"))

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )

        await model.prepare()
        # Should not raise
        await model.delete_cache()

    def test_detect_api_version(self):
        """Test API version detection."""
        mock_client = self._create_mock_client()

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )

        # Should return a valid version string
        assert model._api_version in ["v1.45", "v1.46-v1.48", "v1.49+"]

    def test_cache_name_property(self):
        """Test cache_name property returns None when no cache."""
        mock_client = self._create_mock_client()

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )

        assert model.cache_name is None

    @pytest.mark.asyncio
    async def test_cache_name_property_after_prepare(self):
        """Test cache_name property returns name after prepare."""
        mock_client = self._create_mock_client()

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )

        await model.prepare()
        assert model.cache_name == "test-cache"

    def test_import_error_when_genai_not_available(self):
        """Test that ImportError is raised when google-genai is not available."""
        with patch("async_batch_llm.models.genai", None):
            with pytest.raises(ImportError) as exc_info:
                GeminiCachedModel(
                    model="gemini-test",
                    client=MagicMock(),
                    cached_content=[],
                )
            assert "google-genai is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_rejects_system_instruction(self):
        """Test that generate() raises when system_instruction is passed."""
        mock_response = self._create_mock_response()
        mock_client = self._create_mock_client(response=mock_response)

        model = GeminiCachedModel(
            model="gemini-test",
            client=mock_client,
            cached_content=[],
        )
        await model.prepare()

        with pytest.raises(ValueError, match="system_instruction cannot be overridden"):
            await model.generate("test", system_instruction="override")


# =============================================================================
# _extract_metadata tests
# =============================================================================


class TestExtractMetadata:
    """Tests for _extract_metadata function."""

    def test_no_candidates(self):
        """Test metadata extraction with no candidates."""
        mock_response = MagicMock()
        mock_response.candidates = []

        metadata = _extract_metadata(mock_response)
        assert metadata is None

    def test_with_safety_ratings(self):
        """Test metadata extraction with safety ratings."""
        mock_response = MagicMock()
        mock_rating = MagicMock()
        mock_rating.category = "HARM_CATEGORY_HATE_SPEECH"
        mock_rating.probability = "LOW"
        mock_candidate = MagicMock()
        mock_candidate.safety_ratings = [mock_rating]
        mock_candidate.finish_reason = None
        mock_response.candidates = [mock_candidate]

        metadata = _extract_metadata(mock_response)
        assert metadata is not None
        assert "safety_ratings" in metadata
        assert metadata["safety_ratings"]["HARM_CATEGORY_HATE_SPEECH"] == "LOW"

    def test_with_finish_reason(self):
        """Test metadata extraction with finish reason."""
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.safety_ratings = None
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]

        metadata = _extract_metadata(mock_response)
        assert metadata is not None
        assert metadata["finish_reason"] == "STOP"

    def test_exception_handling(self):
        """Test metadata extraction handles exceptions gracefully."""
        mock_response = MagicMock()
        # Make candidates access raise an exception
        type(mock_response).candidates = property(
            lambda self: (_ for _ in ()).throw(RuntimeError())
        )

        metadata = _extract_metadata(mock_response)
        assert metadata is None


# =============================================================================
# TokenTrackingError tests
# =============================================================================


class TestTokenTrackingErrorIntegration:
    """Tests for TokenTrackingError usage in strategies."""

    @pytest.mark.asyncio
    async def test_token_tracking_error_preserves_tokens(self):
        """Test that TokenTrackingError preserves token usage."""
        error = TokenTrackingError(
            "Parse failed",
            token_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

        assert str(error) == "Parse failed"
        assert error._failed_token_usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_token_tracking_error_default_empty(self):
        """Test TokenTrackingError with no token_usage."""
        error = TokenTrackingError("Error occurred")

        assert error._failed_token_usage == {}
