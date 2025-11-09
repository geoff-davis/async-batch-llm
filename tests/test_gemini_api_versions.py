"""Tests for google-genai API version compatibility."""

import pytest


def test_api_version_detection():
    """Test that API version detection works correctly."""
    from batch_llm.llm_strategies import GeminiCachedStrategy

    version = GeminiCachedStrategy._detect_google_genai_version()

    # Should detect one of the two supported versions
    assert version in ["v1.45", "v1.46+"], f"Unexpected API version: {version}"


def test_api_version_matches_installed_package():
    """Test that detected version matches what's actually installed."""
    from batch_llm.llm_strategies import GeminiCachedStrategy

    version = GeminiCachedStrategy._detect_google_genai_version()

    # Try to import the new API type
    try:
        from google.genai.types import CreateCachedContentConfig  # noqa: F401

        # If import succeeds, we should detect v1.46+
        assert version == "v1.46+", (
            "CreateCachedContentConfig is importable but version detection returned v1.45"
        )
    except ImportError:
        # If import fails, we should detect v1.45
        assert version == "v1.45", (
            "CreateCachedContentConfig is not importable but version detection returned v1.46+"
        )


@pytest.mark.asyncio
async def test_gemini_cached_strategy_initialization_includes_version():
    """Test that GeminiCachedStrategy stores API version on init."""
    from unittest.mock import MagicMock

    from batch_llm.llm_strategies import GeminiCachedStrategy

    # Create a mock client (won't actually use it in this test)
    mock_client = MagicMock()

    def mock_parser(response):
        return "parsed"

    strategy = GeminiCachedStrategy(
        model="gemini-2.5-flash",
        client=mock_client,
        response_parser=mock_parser,
        cached_content=[],
        cache_ttl_seconds=3600,
    )

    # Check that _api_version was set during init
    assert hasattr(strategy, "_api_version")
    assert strategy._api_version in ["v1.45", "v1.46+"]
