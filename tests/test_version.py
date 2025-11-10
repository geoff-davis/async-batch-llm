"""Tests for package version."""

from importlib.metadata import version


def test_version_matches_package():
    """Verify __version__ matches package metadata."""
    from batch_llm import __version__

    # Should match pyproject.toml version
    package_version = version("batch-llm")
    assert __version__ == package_version, (
        f"__version__ ({__version__}) doesn't match package metadata ({package_version})"
    )


def test_version_format():
    """Verify version follows semantic versioning."""
    from batch_llm import __version__

    # Should be in format X.Y.Z or X.Y.Z+dev
    parts = __version__.replace("+dev", "").split(".")
    assert len(parts) == 3, f"Version should have 3 parts, got: {__version__}"

    # Check each part is a number (or dev suffix)
    for part in parts:
        assert part.isdigit(), f"Version part should be numeric, got: {part}"
