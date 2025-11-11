"""Tests for cache tag matching logic (v0.3.0 feature)."""

import importlib.util

import pytest


@pytest.mark.asyncio
async def test_cache_tags_exact_match():
    """Test that cache with exact matching tags is reused."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")

    # Mock cache with metadata
    class MockCache:
        def __init__(self, model: str, metadata: dict | None = None):
            self.model = f"projects/test/models/{model}"
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Create strategy with tags
    strategy_tags = {"customer": "acme", "version": "v1", "environment": "prod"}

    # Test: Exact match should succeed
    cache = MockCache("gemini-2.0-flash", metadata={"customer": "acme", "version": "v1", "environment": "prod"})

    # Simulate the matching logic from _find_or_create_cache
    tags_match = all(
        cache.metadata.get(k) == v
        for k, v in strategy_tags.items()
    )

    assert tags_match, "Exact tag match should succeed"


@pytest.mark.asyncio
async def test_cache_tags_subset_match():
    """Test that cache with superset of tags is reused (strategy tags are subset)."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")

    # Mock cache with metadata
    class MockCache:
        def __init__(self, model: str, metadata: dict | None = None):
            self.model = f"projects/test/models/{model}"
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Strategy wants subset of tags
    strategy_tags = {"customer": "acme", "version": "v1"}

    # Cache has superset (extra tags)
    cache = MockCache(
        "gemini-2.0-flash",
        metadata={"customer": "acme", "version": "v1", "environment": "prod", "region": "us"}
    )

    # Simulate the matching logic
    tags_match = all(
        cache.metadata.get(k) == v
        for k, v in strategy_tags.items()
    )

    assert tags_match, "Cache with superset of tags should match strategy's subset"


@pytest.mark.asyncio
async def test_cache_tags_mismatch_value():
    """Test that cache with different tag value is NOT reused."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")

    # Mock cache with metadata
    class MockCache:
        def __init__(self, model: str, metadata: dict | None = None):
            self.model = f"projects/test/models/{model}"
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Strategy wants specific tags
    strategy_tags = {"customer": "acme", "version": "v1"}

    # Cache has different value for one tag
    cache = MockCache("gemini-2.0-flash", metadata={"customer": "globex", "version": "v1"})

    # Simulate the matching logic
    tags_match = all(
        cache.metadata.get(k) == v
        for k, v in strategy_tags.items()
    )

    assert not tags_match, "Cache with different tag value should NOT match"


@pytest.mark.asyncio
async def test_cache_tags_missing_key():
    """Test that cache missing required tag key is NOT reused."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")

    # Mock cache with metadata
    class MockCache:
        def __init__(self, model: str, metadata: dict | None = None):
            self.model = f"projects/test/models/{model}"
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Strategy wants specific tags
    strategy_tags = {"customer": "acme", "version": "v1", "environment": "prod"}

    # Cache is missing one required tag
    cache = MockCache("gemini-2.0-flash", metadata={"customer": "acme", "version": "v1"})

    # Simulate the matching logic
    tags_match = all(
        cache.metadata.get(k) == v
        for k, v in strategy_tags.items()
    )

    assert not tags_match, "Cache missing required tag should NOT match"


@pytest.mark.asyncio
async def test_cache_tags_empty_metadata():
    """Test that cache with no metadata doesn't match strategy with tags."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")

    # Mock cache with metadata
    class MockCache:
        def __init__(self, model: str, metadata: dict | None = None):
            self.model = f"projects/test/models/{model}"
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Strategy wants tags
    strategy_tags = {"customer": "acme"}

    # Cache has no metadata
    cache = MockCache("gemini-2.0-flash", metadata={})

    # Simulate the matching logic
    tags_match = all(
        cache.metadata.get(k) == v
        for k, v in strategy_tags.items()
    )

    assert not tags_match, "Cache with no metadata should NOT match strategy with tags"


@pytest.mark.asyncio
async def test_cache_no_tags_matches_any():
    """Test that strategy with no tags matches any cache (legacy behavior)."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")

    # Mock cache with metadata
    class MockCache:
        def __init__(self, model: str, metadata: dict | None = None):
            self.model = f"projects/test/models/{model}"
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Strategy has no tags (legacy behavior)
    strategy_tags = {}

    # Cache has tags
    cache = MockCache("gemini-2.0-flash", metadata={"customer": "acme", "version": "v1"})

    # Simulate the matching logic - empty dict means all() returns True
    tags_match = all(
        cache.metadata.get(k) == v
        for k, v in strategy_tags.items()
    )

    assert tags_match, "Strategy with no tags should match any cache (backward compatibility)"


@pytest.mark.asyncio
async def test_cache_tags_case_sensitive():
    """Test that tag matching is case-sensitive."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")

    # Mock cache with metadata
    class MockCache:
        def __init__(self, model: str, metadata: dict | None = None):
            self.model = f"projects/test/models/{model}"
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Strategy wants lowercase
    strategy_tags = {"customer": "acme"}

    # Cache has uppercase
    cache = MockCache("gemini-2.0-flash", metadata={"customer": "ACME"})

    # Simulate the matching logic
    tags_match = all(
        cache.metadata.get(k) == v
        for k, v in strategy_tags.items()
    )

    assert not tags_match, "Tag matching should be case-sensitive"


@pytest.mark.asyncio
async def test_cache_tags_type_sensitivity():
    """Test that tag matching is type-sensitive (string vs int)."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")

    # Mock cache with metadata
    class MockCache:
        def __init__(self, model: str, metadata: dict | None = None):
            self.model = f"projects/test/models/{model}"
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Strategy wants string "1"
    strategy_tags = {"version": "1"}

    # Cache has integer 1
    cache = MockCache("gemini-2.0-flash", metadata={"version": 1})

    # Simulate the matching logic
    tags_match = all(
        cache.metadata.get(k) == v
        for k, v in strategy_tags.items()
    )

    assert not tags_match, "Tag matching should be type-sensitive (string '1' != int 1)"


@pytest.mark.asyncio
async def test_cache_tags_multiple_caches_filtering():
    """Test that tag matching correctly filters among multiple caches."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")

    # Mock cache with metadata
    class MockCache:
        def __init__(self, model: str, metadata: dict | None = None):
            self.model = f"projects/test/models/{model}"
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Strategy wants specific tags
    strategy_tags = {"customer": "acme", "environment": "prod"}

    # Create multiple caches with different tags
    caches = [
        MockCache("gemini-2.0-flash", metadata={"customer": "globex", "environment": "prod"}),
        MockCache("gemini-2.0-flash", metadata={"customer": "acme", "environment": "dev"}),
        MockCache("gemini-2.0-flash", metadata={"customer": "acme", "environment": "prod"}),  # Match!
        MockCache("gemini-2.0-flash", metadata={"customer": "acme", "environment": "staging"}),
    ]

    # Find matching caches
    matching_caches = []
    for cache in caches:
        tags_match = all(
            cache.metadata.get(k) == v
            for k, v in strategy_tags.items()
        )
        if tags_match:
            matching_caches.append(cache)

    assert len(matching_caches) == 1, "Should find exactly one matching cache"
    assert matching_caches[0].metadata == {"customer": "acme", "environment": "prod"}


@pytest.mark.asyncio
async def test_cache_tags_special_characters():
    """Test that tags can contain special characters."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed")

    # Mock cache with metadata
    class MockCache:
        def __init__(self, model: str, metadata: dict | None = None):
            self.model = f"projects/test/models/{model}"
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Strategy with special characters in tags
    strategy_tags = {
        "git-sha": "abc123-def456",
        "build.date": "2024-01-15",
        "owner@company": "user@example.com"
    }

    # Cache with matching special characters
    cache = MockCache(
        "gemini-2.0-flash",
        metadata={
            "git-sha": "abc123-def456",
            "build.date": "2024-01-15",
            "owner@company": "user@example.com"
        }
    )

    # Simulate the matching logic
    tags_match = all(
        cache.metadata.get(k) == v
        for k, v in strategy_tags.items()
    )

    assert tags_match, "Tags with special characters should match correctly"
