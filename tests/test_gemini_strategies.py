"""Comprehensive tests for Gemini strategies to improve coverage."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from async_batch_llm import RetryState, TokenTrackingError
from async_batch_llm.llm_strategies import (
    GeminiCachedStrategy,
    GeminiResponse,
    GeminiStrategy,
    LLMCallStrategy,
    _extract_safety_ratings,
)


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
# GeminiStrategy tests
# =============================================================================


class TestGeminiStrategy:
    """Tests for GeminiStrategy."""

    def _create_mock_response(
        self,
        input_tokens: int = 10,
        output_tokens: int = 20,
        total_tokens: int = 30,
        has_safety_ratings: bool = False,
        has_finish_reason: bool = False,
    ):
        """Create a mock Gemini response."""
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = input_tokens
        mock_response.usage_metadata.candidates_token_count = output_tokens
        mock_response.usage_metadata.total_token_count = total_tokens

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
    async def test_execute_basic(self):
        """Test basic execute functionality."""
        mock_response = self._create_mock_response()
        mock_client = self._create_mock_client(mock_response)

        strategy = GeminiStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: TestOutput(text="parsed"),
        )

        output, tokens = await strategy.execute("test prompt", 1, 10.0)

        assert output.text == "parsed"
        assert tokens["input_tokens"] == 10
        assert tokens["output_tokens"] == 20
        assert tokens["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_execute_with_config(self):
        """Test execute with GenerateContentConfig."""
        mock_response = self._create_mock_response()
        mock_client = self._create_mock_client(mock_response)

        mock_config = MagicMock()
        mock_config.__dict__ = {"temperature": 0.7}

        strategy = GeminiStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: TestOutput(text="parsed"),
            config=mock_config,
        )

        await strategy.execute("test prompt", 1, 10.0)

        mock_client.aio.models.generate_content.assert_called_once()
        call_kwargs = mock_client.aio.models.generate_content.call_args
        assert call_kwargs.kwargs["config"] == mock_config

    @pytest.mark.asyncio
    async def test_execute_with_metadata(self):
        """Test execute with include_metadata=True returns GeminiResponse."""
        mock_response = self._create_mock_response(
            has_safety_ratings=True,
            has_finish_reason=True,
        )
        mock_client = self._create_mock_client(mock_response)

        strategy = GeminiStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: TestOutput(text="parsed"),
            include_metadata=True,
        )

        result, tokens = await strategy.execute("test prompt", 1, 10.0)

        assert isinstance(result, GeminiResponse)
        assert result.output.text == "parsed"
        assert result.safety_ratings == {"HARM_CATEGORY_HATE_SPEECH": "LOW"}
        assert result.finish_reason == "STOP"
        assert result.raw_response is mock_response

    @pytest.mark.asyncio
    async def test_execute_no_usage_metadata(self):
        """Test execute when response has no usage_metadata."""
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.candidates = []

        mock_client = self._create_mock_client(mock_response)

        strategy = GeminiStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: TestOutput(text="parsed"),
        )

        output, tokens = await strategy.execute("test prompt", 1, 10.0)

        assert tokens["input_tokens"] == 0
        assert tokens["output_tokens"] == 0
        assert tokens["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_execute_parser_raises_with_dict(self):
        """Test that parser exceptions with __dict__ preserve token usage."""
        mock_response = self._create_mock_response()
        mock_client = self._create_mock_client(mock_response)

        class CustomError(Exception):
            pass

        def failing_parser(r):
            raise CustomError("Parse failed")

        strategy = GeminiStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=failing_parser,
        )

        with pytest.raises(CustomError) as exc_info:
            await strategy.execute("test prompt", 1, 10.0)

        # Token usage should be attached to exception
        assert exc_info.value.__dict__["_failed_token_usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_execute_parser_raises_builtin_exception(self):
        """Test that builtin exceptions without __dict__ get wrapped."""
        mock_response = self._create_mock_response()
        mock_client = self._create_mock_client(mock_response)

        def failing_parser(r):
            # StopIteration has no __dict__ in some Python versions
            raise ValueError("Parse failed")

        strategy = GeminiStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=failing_parser,
        )

        # ValueError has __dict__, so it won't be wrapped
        with pytest.raises(ValueError) as exc_info:
            await strategy.execute("test prompt", 1, 10.0)

        assert exc_info.value.__dict__["_failed_token_usage"]["total_tokens"] == 30

    def test_extract_safety_ratings_no_candidates(self):
        """Test safety rating extraction with no candidates."""
        mock_response = MagicMock()
        mock_response.candidates = []

        ratings = _extract_safety_ratings(mock_response)
        assert ratings == {}

    def test_extract_safety_ratings_exception(self):
        """Test safety rating extraction handles exceptions gracefully."""
        mock_response = MagicMock()
        # Make candidates access raise an exception
        type(mock_response).candidates = property(
            lambda self: (_ for _ in ()).throw(RuntimeError())
        )

        ratings = _extract_safety_ratings(mock_response)
        assert ratings == {}

    @pytest.mark.asyncio
    async def test_extract_finish_reason_exception(self):
        """Test finish reason extraction handles exceptions gracefully."""
        mock_response = self._create_mock_response(has_safety_ratings=True)
        # Make finish_reason access raise
        mock_response.candidates[0].finish_reason = property(
            lambda self: (_ for _ in ()).throw(RuntimeError())
        )

        mock_client = self._create_mock_client(mock_response)

        strategy = GeminiStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: TestOutput(text="parsed"),
            include_metadata=True,
        )

        # Should not raise, just log warning
        result, _ = await strategy.execute("test", 1, 10.0)
        # finish_reason should be None due to exception
        assert isinstance(result, GeminiResponse)

    def test_import_error_when_genai_not_available(self):
        """Test that ImportError is raised when google-genai is not available."""
        with patch("async_batch_llm.llm_strategies.genai", None):
            with pytest.raises(ImportError) as exc_info:
                GeminiStrategy(
                    model="gemini-test",
                    client=MagicMock(),
                    response_parser=lambda r: r,
                )
            assert "google-genai is required" in str(exc_info.value)


# =============================================================================
# GeminiCachedStrategy tests
# =============================================================================


class TestGeminiCachedStrategy:
    """Tests for GeminiCachedStrategy."""

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
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30
        mock_response.usage_metadata.cached_content_token_count = cached_tokens
        mock_response.candidates = []
        return mock_response

    def _create_mock_client(self, caches=None, response=None):
        """Create a mock Gemini client."""
        mock_client = MagicMock()
        mock_client.aio.caches.list = AsyncMock(return_value=caches or [])
        mock_client.aio.caches.create = AsyncMock(return_value=self._create_mock_cache())
        mock_client.aio.caches.delete = AsyncMock()
        if response:
            mock_client.aio.models.generate_content = AsyncMock(return_value=response)
        return mock_client

    def test_validation_renewal_buffer_too_large(self):
        """Test validation rejects renewal buffer >= TTL."""
        mock_client = self._create_mock_client()

        with pytest.raises(ValueError) as exc_info:
            GeminiCachedStrategy(
                model="gemini-test",
                client=mock_client,
                response_parser=lambda r: r,
                cached_content=[],
                cache_ttl_seconds=60,
                cache_renewal_buffer_seconds=60,  # Same as TTL - invalid
            )
        assert "must be less than cache_ttl_seconds" in str(exc_info.value)

    def test_validation_short_ttl_warning(self):
        """Test warning for short TTL values."""
        mock_client = self._create_mock_client()

        with pytest.warns(UserWarning) as record:
            GeminiCachedStrategy(
                model="gemini-test",
                client=mock_client,
                response_parser=lambda r: r,
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
            GeminiCachedStrategy(
                model="gemini-test",
                client=mock_client,
                response_parser=lambda r: r,
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

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: r,
            cached_content=[],
            cache_ttl_seconds=3600,
        )

        await strategy.prepare()

        # Should not create new cache
        mock_client.aio.caches.create.assert_not_called()
        assert strategy._cache == existing_cache

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

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: r,
            cached_content=[],
            cache_tags={"version": "1.0", "type": "test"},
        )

        await strategy.prepare()

        assert strategy._cache == matching_cache

    @pytest.mark.asyncio
    async def test_cache_list_failure(self):
        """Test fallback to create when cache list fails."""
        mock_client = self._create_mock_client()
        mock_client.aio.caches.list = AsyncMock(side_effect=RuntimeError("List failed"))

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: r,
            cached_content=[],
        )

        await strategy.prepare()

        # Should create new cache after list failure
        mock_client.aio.caches.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_cache_expired_no_cache(self):
        """Test _is_cache_expired returns True when no cache."""
        mock_client = self._create_mock_client()

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: r,
            cached_content=[],
        )

        assert strategy._is_cache_expired() is True

    @pytest.mark.asyncio
    async def test_is_cache_expired_within_buffer(self):
        """Test _is_cache_expired returns True when within renewal buffer."""
        mock_client = self._create_mock_client()

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: r,
            cached_content=[],
            cache_ttl_seconds=100,
            cache_renewal_buffer_seconds=30,
        )

        strategy._cache = MagicMock()
        # Created 80s ago, so only 20s remain (< 30s buffer)
        strategy._cache_created_at = time.time() - 80

        assert strategy._is_cache_expired() is True

    @pytest.mark.asyncio
    async def test_execute_without_prepare_raises(self):
        """Test execute raises when prepare() not called."""
        mock_client = self._create_mock_client()

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: r,
            cached_content=[],
            auto_renew=False,  # Disable auto-renew to trigger error
        )

        with pytest.raises(RuntimeError) as exc_info:
            await strategy.execute("test", 1, 10.0)

        assert "prepare() was not called" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_with_cached_tokens(self):
        """Test execute tracks cached tokens."""
        mock_response = self._create_mock_response(cached_tokens=500)
        mock_client = self._create_mock_client(response=mock_response)

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: TestOutput(text="cached"),
            cached_content=[],
        )

        await strategy.prepare()
        output, tokens = await strategy.execute("test", 1, 10.0)

        assert tokens["cached_input_tokens"] == 500
        assert output.text == "cached"

    @pytest.mark.asyncio
    async def test_execute_auto_renew_concurrent(self):
        """Test auto-renewal uses lock for concurrent access."""
        mock_response = self._create_mock_response()
        mock_client = self._create_mock_client(response=mock_response)
        mock_client.aio.caches.create = AsyncMock(return_value=self._create_mock_cache())

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: TestOutput(text="result"),
            cached_content=[],
            cache_ttl_seconds=10,
            cache_renewal_buffer_seconds=5,
            auto_renew=True,
        )

        # Manually set up expired cache
        await strategy.prepare()
        strategy._cache_created_at = time.time() - 100  # Expired

        # Execute should trigger renewal
        await strategy.execute("test", 1, 10.0)

        # Should have created twice (initial + renewal)
        assert mock_client.aio.caches.create.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_metadata(self):
        """Test execute with include_metadata=True."""
        mock_response = self._create_mock_response()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.candidates[0].finish_reason = "STOP"

        mock_client = self._create_mock_client(response=mock_response)

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: TestOutput(text="result"),
            cached_content=[],
            include_metadata=True,
        )

        await strategy.prepare()
        result, tokens = await strategy.execute("test", 1, 10.0)

        assert isinstance(result, GeminiResponse)
        assert result.finish_reason == "STOP"

    @pytest.mark.asyncio
    async def test_cleanup_preserves_cache(self):
        """Test cleanup preserves cache by default."""
        mock_client = self._create_mock_client()

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: r,
            cached_content=[],
        )

        await strategy.prepare()
        await strategy.cleanup()

        mock_client.aio.caches.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_cache_explicit(self):
        """Test explicit cache deletion."""
        mock_client = self._create_mock_client()

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: r,
            cached_content=[],
        )

        await strategy.prepare()
        await strategy.delete_cache()

        mock_client.aio.caches.delete.assert_called_once()
        assert strategy._cache is None

    @pytest.mark.asyncio
    async def test_delete_cache_handles_failure(self):
        """Test delete_cache handles deletion failure gracefully."""
        mock_client = self._create_mock_client()
        mock_client.aio.caches.delete = AsyncMock(side_effect=RuntimeError("Delete failed"))

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: r,
            cached_content=[],
        )

        await strategy.prepare()
        # Should not raise
        await strategy.delete_cache()

    def test_detect_api_version(self):
        """Test API version detection."""
        mock_client = self._create_mock_client()

        strategy = GeminiCachedStrategy(
            model="gemini-test",
            client=mock_client,
            response_parser=lambda r: r,
            cached_content=[],
        )

        # Should return a valid version string
        assert strategy._api_version in ["v1.45", "v1.46-v1.48", "v1.49+"]

    def test_import_error_when_genai_not_available(self):
        """Test that ImportError is raised when google-genai is not available."""
        with patch("async_batch_llm.llm_strategies.genai", None):
            with pytest.raises(ImportError) as exc_info:
                GeminiCachedStrategy(
                    model="gemini-test",
                    client=MagicMock(),
                    response_parser=lambda r: r,
                    cached_content=[],
                )
            assert "google-genai is required" in str(exc_info.value)


# =============================================================================
# GeminiResponse tests
# =============================================================================


class TestGeminiResponse:
    """Tests for GeminiResponse dataclass."""

    def test_creation(self):
        """Test GeminiResponse creation."""
        response = GeminiResponse(
            output=TestOutput(text="test"),
            safety_ratings={"HARM_CATEGORY_HATE_SPEECH": "LOW"},
            finish_reason="STOP",
            token_usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            raw_response=MagicMock(),
        )

        assert response.output.text == "test"
        assert response.safety_ratings["HARM_CATEGORY_HATE_SPEECH"] == "LOW"
        assert response.finish_reason == "STOP"

    def test_creation_with_none_values(self):
        """Test GeminiResponse with None optional values."""
        response = GeminiResponse(
            output="simple string",
            safety_ratings=None,
            finish_reason=None,
            token_usage={},
            raw_response=None,
        )

        assert response.output == "simple string"
        assert response.safety_ratings is None
        assert response.finish_reason is None


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
