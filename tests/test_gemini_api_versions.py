"""Tests for google-genai API compatibility."""


def test_google_genai_version_sufficient():
    """Test that installed google-genai meets minimum version requirement (>=1.49.0)."""
    from importlib.metadata import version

    installed = version("google-genai")
    major, minor, *_ = installed.split(".")
    assert int(major) >= 1
    assert int(major) > 1 or int(minor) >= 49, f"google-genai {installed} is below minimum 1.49.0"


def test_create_cached_content_config_available():
    """Test that CreateCachedContentConfig accepts contents parameter (v1.49+ API)."""
    import inspect

    from google.genai.types import CreateCachedContentConfig

    sig = inspect.signature(CreateCachedContentConfig.__init__)
    params = sig.parameters
    assert "contents" in params or "data" in params, (
        "CreateCachedContentConfig must accept 'contents' or 'data' parameter"
    )
