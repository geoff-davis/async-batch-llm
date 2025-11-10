"""Tests for cache expiration with multiple workers."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import GeminiCachedStrategy


@pytest.mark.asyncio
async def test_cache_expiration_only_one_new_cache_created():
    """
    Test that when cache expires, only ONE new cache is created even with multiple workers.

    Scenario:
    1. Multiple workers process items with a cached strategy
    2. Cache expires while workers are active
    3. Only the first worker to detect expiration should create a new cache
    4. Other workers should wait and use the newly created cache
    5. Verify only ONE cache creation occurred
    """
    cache_create_count = 0
    cache_create_lock = asyncio.Lock()

    # Track which cache each worker used
    cache_usage = []

    # Mock the Gemini client
    mock_client = MagicMock()

    # Create two fake cache objects with different names
    old_cache = MagicMock()
    old_cache.name = "old-cache-123"
    old_cache.expire_time = datetime.now(timezone.utc) - timedelta(hours=1)  # Already expired
    old_cache.model = "models/gemini-2.0-flash"  # Full path format

    new_cache = MagicMock()
    new_cache.name = "new-cache-456"
    new_cache.expire_time = datetime.now(timezone.utc) + timedelta(hours=1)  # Fresh
    new_cache.model = "models/gemini-2.0-flash"  # Full path format
    new_cache.create_time = MagicMock()
    new_cache.create_time.timestamp = MagicMock(return_value=datetime.now(timezone.utc).timestamp())

    async def mock_create_cache(*args, **kwargs):
        """Mock cache creation - track how many times it's called."""
        nonlocal cache_create_count
        async with cache_create_lock:
            cache_create_count += 1
            # Simulate cache creation taking some time
            await asyncio.sleep(0.1)
        return new_cache

    mock_client.aio.caches.create = AsyncMock(side_effect=mock_create_cache)
    mock_client.aio.caches.list = AsyncMock(return_value=[])  # No existing caches

    # Mock the generate_content call
    async def mock_generate(*args, **kwargs):
        """Mock content generation - track which cache was used."""
        # In google-genai v1.46+, cached_content is passed in the config dict
        config = kwargs.get('config', {})
        cached_content = config.get('cached_content') if isinstance(config, dict) else None
        if cached_content:
            cache_usage.append(cached_content)

        # Create a mock response
        response = MagicMock()
        response.text = "Test response"
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.total_token_count = 150
        response.usage_metadata.cached_content_token_count = 80
        return response

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_generate)

    # Create the strategy with the mock client
    strategy = GeminiCachedStrategy(
        model="gemini-2.0-flash",
        client=mock_client,
        response_parser=lambda r: r.text,
        cached_content=[],
        cache_ttl_seconds=3600,
    )

    # Manually set the cache to an expired one to simulate the scenario
    strategy._cache = old_cache
    strategy._cache_lock = asyncio.Lock()
    strategy._cache_created_at = datetime.now(timezone.utc).timestamp() - 7200  # Created 2 hours ago

    config = ProcessorConfig(
        max_workers=5,  # Multiple workers
        timeout_per_item=10.0,
    )

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 10 items to process
        for i in range(10):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"item_{i}",
                    strategy=strategy,
                    prompt=f"Test prompt {i}",
                )
            )

        result = await processor.process_all()

    # Verify results
    assert result.total_items == 10
    assert result.succeeded == 10
    assert result.failed == 0

    # CRITICAL ASSERTION: Only ONE cache should have been created
    assert cache_create_count == 1, (
        f"Expected exactly 1 cache creation, but got {cache_create_count}. "
        f"Multiple workers should not create duplicate caches!"
    )

    # Verify all workers used the NEW cache (not the old expired one)
    assert len(cache_usage) == 10, "All 10 items should have used cached content"

    for cache_name in cache_usage:
        assert cache_name == new_cache.name, (
            f"Expected all workers to use new cache '{new_cache.name}', "
            f"but found usage of '{cache_name}'"
        )


@pytest.mark.asyncio
async def test_cache_expiration_during_processing():
    """
    Test cache expiration that occurs DURING processing (not before).

    Scenario:
    1. Cache is valid when processing starts
    2. Cache expires while some workers are still processing
    3. Workers that check after expiration should create/use new cache
    4. Only one new cache should be created
    """
    cache_create_count = 0
    items_processed = 0

    # Mock the Gemini client
    mock_client = MagicMock()

    # Cache starts valid but will "expire" after 3 items
    current_cache = MagicMock()
    current_cache.name = "initial-cache"
    current_cache.expire_time = datetime.now(timezone.utc) + timedelta(hours=1)
    current_cache.model = "models/gemini-2.0-flash"
    current_cache.create_time = MagicMock()
    current_cache.create_time.timestamp = MagicMock(return_value=datetime.now(timezone.utc).timestamp())

    new_cache = MagicMock()
    new_cache.name = "renewed-cache"
    new_cache.expire_time = datetime.now(timezone.utc) + timedelta(hours=1)
    new_cache.model = "models/gemini-2.0-flash"
    new_cache.create_time = MagicMock()
    new_cache.create_time.timestamp = MagicMock(return_value=datetime.now(timezone.utc).timestamp())

    async def mock_create_cache(*args, **kwargs):
        """Track cache creation."""
        nonlocal cache_create_count
        cache_create_count += 1
        await asyncio.sleep(0.05)  # Simulate creation time
        return new_cache

    mock_client.aio.caches.create = AsyncMock(side_effect=mock_create_cache)
    mock_client.aio.caches.list = AsyncMock(return_value=[])  # No existing caches

    async def mock_generate(*args, **kwargs):
        """Mock generation that simulates cache expiring mid-processing."""
        nonlocal items_processed
        items_processed += 1

        # After processing 3 items, make the cache "expire"
        if items_processed == 3:
            # Simulate expiration by backdating the expire_time
            current_cache.expire_time = datetime.now(timezone.utc) - timedelta(minutes=1)

        response = MagicMock()
        response.text = "Response"
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.total_token_count = 150
        response.usage_metadata.cached_content_token_count = 80
        return response

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_generate)

    strategy = GeminiCachedStrategy(
        model="gemini-2.0-flash",
        client=mock_client,
        response_parser=lambda r: r.text,
        cached_content=[],
        cache_ttl_seconds=3600,
    )

    # Set initial cache
    strategy._cache = current_cache
    strategy._cache_lock = asyncio.Lock()
    strategy._cache_created_at = datetime.now(timezone.utc).timestamp()

    config = ProcessorConfig(
        max_workers=3,
        timeout_per_item=10.0,
    )

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 6 items - first 3 use initial cache, last 3 should trigger renewal
        for i in range(6):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"item_{i}",
                    strategy=strategy,
                    prompt=f"Test {i}",
                )
            )

        result = await processor.process_all()

    assert result.succeeded == 6

    # Verify cache was recreated (but only once, even though multiple workers
    # may have detected expiration simultaneously)
    assert cache_create_count == 1, (
        f"Expected exactly 1 cache recreation, got {cache_create_count}"
    )


@pytest.mark.asyncio
async def test_cache_check_is_thread_safe():
    """
    Test that cache expiration checking is thread-safe.

    Verify that when multiple workers simultaneously check if cache is expired,
    they properly coordinate using locks and don't create race conditions.
    """
    check_count = 0
    create_count = 0

    mock_client = MagicMock()

    cache = MagicMock()
    cache.name = "test-cache"
    cache.expire_time = datetime.now(timezone.utc) - timedelta(hours=1)  # Expired
    cache.model = "models/gemini-2.0-flash"
    cache.create_time = MagicMock()
    cache.create_time.timestamp = MagicMock(
        return_value=(datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
    )

    new_cache = MagicMock()
    new_cache.name = "new-cache"
    new_cache.expire_time = datetime.now(timezone.utc) + timedelta(hours=1)
    new_cache.model = "models/gemini-2.0-flash"
    new_cache.create_time = MagicMock()
    new_cache.create_time.timestamp = MagicMock(return_value=datetime.now(timezone.utc).timestamp())

    async def mock_create(*args, **kwargs):
        nonlocal create_count
        create_count += 1
        # Simulate slow cache creation to increase likelihood of race conditions
        await asyncio.sleep(0.2)
        return new_cache

    mock_client.aio.caches.create = AsyncMock(side_effect=mock_create)
    mock_client.aio.caches.list = AsyncMock(return_value=[])  # No existing caches

    async def mock_generate(*args, **kwargs):
        nonlocal check_count
        check_count += 1

        response = MagicMock()
        response.text = "Response"
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.total_token_count = 150
        response.usage_metadata.cached_content_token_count = 80
        return response

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_generate)

    strategy = GeminiCachedStrategy(
        model="gemini-2.0-flash",
        client=mock_client,
        response_parser=lambda r: r.text,
        cached_content=[],
        cache_ttl_seconds=3600,
    )

    strategy._cache = cache
    strategy._cache_lock = asyncio.Lock()
    strategy._cache_created_at = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()

    config = ProcessorConfig(
        max_workers=10,  # Many workers to stress test
        timeout_per_item=10.0,
    )

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 20 items with 10 workers - high concurrency
        for i in range(20):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"item_{i}",
                    strategy=strategy,
                    prompt=f"Test {i}",
                )
            )

        result = await processor.process_all()

    assert result.succeeded == 20

    # CRITICAL: Despite 10 concurrent workers checking an expired cache,
    # only ONE cache should be created
    assert create_count == 1, (
        f"Thread safety violation! Expected 1 cache creation with {config.max_workers} workers, "
        f"but got {create_count}. The cache lock is not working correctly!"
    )
