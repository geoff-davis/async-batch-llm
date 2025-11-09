# Implementation Plan: batch-llm v0.2.0

## Executive Summary

This document outlines a comprehensive plan to address issues identified in real-world production usage of batch-llm v0.1.0, specifically around shared strategy instances for cost optimization with Gemini prompt caching.

**Target Version:** v0.2.0
**Estimated Scope:** 3-5 days of development + testing
**Breaking Changes:** Minimal (mostly additions, a few signature changes with backward compatibility)

---

## Issues Being Addressed

Based on feedback from production usage (see `BATCH_LLM_FEEDBACK.md`):

| Issue | Priority | Impact | Complexity |
|-------|----------|--------|------------|
| #1: Multiple `prepare()` calls on shared strategies | **Critical** | High | Medium |
| #2: API version incompatibility (google-genai v1.46+) | **Critical** | High | Low |
| #3: Missing safety ratings access | **High** | Medium | Medium |
| #4: No cached token tracking in `BatchResult` | **High** | Medium | Low |
| #5: Unclear cache lifecycle (cleanup() hook) | **Medium** | Medium | Low (docs) |
| #6: Cache matching precision | **Medium** | Medium | Medium |
| #7: Cache expiration error handling | **High** | High | Medium |
| #8: No per-work-item state for retry strategies | **High** | High | High |

---

## Release Strategy

### Phase 1: Core Fixes (v0.2.0)

**Issues:** #1, #2, #4, #5, #7
**Goal:** Fix critical production issues, minimal breaking changes
**Timeline:** Week 1

### Phase 2: Advanced Features (v0.3.0)

**Issues:** #3, #6, #8
**Goal:** Add per-work-item state, rich response metadata, cache management
**Timeline:** Week 2-3

This document covers **Phase 1 (v0.2.0)** only.

---

## Detailed Implementation Plan (v0.2.0)

### Issue #1: Multiple `prepare()` Calls on Shared Strategies

#### Problem

Framework calls `strategy.prepare()` once per work item. When the same strategy instance is shared across multiple work items (for caching cost optimization), `prepare()` is called multiple times concurrently, creating multiple caches.

#### Solution

Add framework-level strategy instance tracking and call `prepare()` only once per unique strategy instance.

#### Implementation

**File: `src/batch_llm/parallel.py`**

```python
class ParallelBatchProcessor:
    def __init__(self, ...):
        # ... existing code ...
        self._prepared_strategies: set[int] = set()  # Track by id()
        self._strategy_lock = asyncio.Lock()  # Protect strategy initialization

    async def _ensure_strategy_prepared(
        self, strategy: LLMCallStrategy[TOutput]
    ) -> None:
        """Ensure strategy is prepared exactly once, even with concurrent calls."""
        strategy_id = id(strategy)

        # Fast path: already prepared (no lock needed for read)
        if strategy_id in self._prepared_strategies:
            return

        # Slow path: acquire lock and prepare
        async with self._strategy_lock:
            # Double-check after acquiring lock (another worker may have prepared)
            if strategy_id in self._prepared_strategies:
                return

            await strategy.prepare()
            self._prepared_strategies.add(strategy_id)
            logger.debug(
                f"Prepared strategy {strategy.__class__.__name__} "
                f"(id={strategy_id})"
            )

    async def _process_item(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        # Ensure strategy is prepared before processing
        await self._ensure_strategy_prepared(work_item.strategy)

        # ... rest of existing code ...
```

**File: `src/batch_llm/base.py`**

```python
class BatchProcessor:
    async def cleanup(self):
        """Clean up resources including strategy cleanup."""
        # ... existing worker cleanup code ...

        # Cleanup strategies (call cleanup() once per unique strategy)
        if hasattr(self, '_prepared_strategies'):
            seen_strategies: set[int] = set()
            # Iterate through all work items to find unique strategies
            # Note: This requires tracking work items or strategies differently
            # For v0.2.0, we'll document that strategies should handle their
            # own cleanup in a reusable way (see Issue #5)
```

#### Breaking Changes

**None** - This is purely internal framework behavior.

#### Tests

**File: `tests/test_shared_strategies.py`** (new)

```python
@pytest.mark.asyncio
async def test_shared_strategy_prepare_called_once():
    """Test that prepare() is called only once for shared strategy."""
    class CountingStrategy(LLMCallStrategy[str]):
        prepare_count = 0

        async def prepare(self):
            self.prepare_count += 1
            await asyncio.sleep(0.1)  # Simulate slow preparation

        async def execute(self, prompt, attempt, timeout):
            return f"Response: {prompt}", {"input_tokens": 10}

    strategy = CountingStrategy()
    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 20 work items sharing the same strategy
        for i in range(20):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

    # All items should succeed
    assert result.succeeded == 20

    # prepare() should be called exactly once
    assert strategy.prepare_count == 1


@pytest.mark.asyncio
async def test_different_strategies_prepare_called_separately():
    """Test that different strategy instances each get prepare() called."""
    class CountingStrategy(LLMCallStrategy[str]):
        def __init__(self):
            self.prepare_count = 0

        async def prepare(self):
            self.prepare_count += 1

        async def execute(self, prompt, attempt, timeout):
            return f"Response", {"input_tokens": 10}

    strategy1 = CountingStrategy()
    strategy2 = CountingStrategy()

    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=strategy1, prompt="Test")
        )
        await processor.add_work(
            LLMWorkItem(item_id="item_2", strategy=strategy2, prompt="Test")
        )

        result = await processor.process_all()

    assert result.succeeded == 2
    assert strategy1.prepare_count == 1
    assert strategy2.prepare_count == 1
```

#### Documentation Updates

**File: `README.md`**

Add new section:

```markdown
### Sharing Strategies for Cost Optimization

For cost optimization (e.g., Gemini prompt caching), you can share a single strategy instance across all work items:

```python
# Create one cached strategy
strategy = GeminiCachedStrategy(
    model="gemini-2.5-flash",
    client=client,
    cached_content=[...],
    cache_ttl_seconds=3600,
)

# Reuse the same strategy for all items
for item in items:
    work_item = LLMWorkItem(
        item_id=item.id,
        strategy=strategy,  # Shared instance
        prompt=format_prompt(item),
    )
    await processor.add_work(work_item)
```

**Benefits:**
- Single cache created and shared across all work items
- 70-90% cost reduction with Gemini prompt caching
- Framework ensures `prepare()` is called only once

**Note:** The framework automatically handles idempotency - `prepare()` is called once per unique strategy instance, even with concurrent workers.
```

---

### Issue #2: API Version Incompatibility (google-genai v1.46+)

#### Problem

`GeminiCachedStrategy` uses old google-genai API (v1.45 and earlier):

```python
# Old API (v1.45)
cache = await client.aio.caches.create(
    model=model,
    contents=cached_content,
    ttl="3600s",
)
```

New API (v1.46+) requires:

```python
# New API (v1.46+)
from google.genai.types import CreateCachedContentConfig

cache = await client.aio.caches.create(
    model=model,
    config=CreateCachedContentConfig(
        contents=cached_content,
        ttl="3600s",
    ),
)
```

#### Solution

Auto-detect API version and use appropriate syntax.

#### Implementation

**File: `src/batch_llm/llm_strategies.py`**

```python
class GeminiCachedStrategy(LLMCallStrategy[TOutput]):
    """Strategy for calling Google Gemini API with context caching."""

    def __init__(self, ...):
        # ... existing code ...

        # Detect API version
        self._api_version = self._detect_google_genai_version()
        logger.debug(f"Detected google-genai API version: {self._api_version}")

    @staticmethod
    def _detect_google_genai_version() -> str:
        """Detect which google-genai API version is installed."""
        try:
            from google.genai.types import CreateCachedContentConfig
            return "v1.46+"
        except ImportError:
            return "v1.45"

    async def prepare(self) -> None:
        """Create the Gemini cache using appropriate API version."""
        if self._api_version == "v1.46+":
            from google.genai.types import CreateCachedContentConfig

            self._cache = await self.client.aio.caches.create(
                model=self.model,
                config=CreateCachedContentConfig(
                    contents=self.cached_content,
                    ttl=f"{self.cache_ttl_seconds}s",
                ),
            )
        else:
            # Legacy API (v1.45 and earlier)
            self._cache = await self.client.aio.caches.create(
                model=self.model,
                contents=self.cached_content,
                ttl=f"{self.cache_ttl_seconds}s",
            )

        self._cache_created_at = time.time()
        logger.info(
            f"Created Gemini cache: {self._cache.name} "
            f"(TTL: {self.cache_ttl_seconds}s)"
        )
```

#### Breaking Changes

**None** - Backward compatible with both API versions.

#### Tests

**File: `tests/test_gemini_api_versions.py`** (new)

```python
@pytest.mark.asyncio
async def test_api_version_detection():
    """Test that API version detection works correctly."""
    from batch_llm.llm_strategies import GeminiCachedStrategy

    version = GeminiCachedStrategy._detect_google_genai_version()
    assert version in ["v1.45", "v1.46+"]


# Mock-based test that doesn't require actual API calls
@pytest.mark.asyncio
async def test_gemini_cached_strategy_with_both_api_versions():
    """Test GeminiCachedStrategy works with both API versions."""
    # This would use mocking to test both code paths
    # Implementation depends on mocking strategy
    pass
```

#### Documentation Updates

**File: `docs/GEMINI_INTEGRATION.md`**

Update dependency requirements:

```markdown
## Google Gemini API Versions

batch-llm supports both old and new google-genai API versions:

- **google-genai v1.45 and earlier:** Legacy API (deprecated)
- **google-genai v1.46+:** New API with `CreateCachedContentConfig` (recommended)

The framework automatically detects which version you have installed and uses the appropriate API calls.

### Recommended Installation

```bash
# Install latest google-genai (v1.46+)
pip install 'batch-llm[gemini]'  # Installs google-genai>=1.46
```

### Legacy Version Support

If you need to use google-genai v1.45 or earlier:

```bash
pip install 'batch-llm[gemini]' 'google-genai<1.46'
```

Note: google-genai v1.45 and earlier are deprecated and may be removed in batch-llm v0.3.0.
```

---

### Issue #4: No Cached Token Tracking in BatchResult

#### Problem

`BatchResult` aggregates `total_input_tokens` and `total_output_tokens` but not `cached_input_tokens`, making it impossible to measure cache effectiveness.

#### Solution

Add `total_cached_tokens` field to `BatchResult` and aggregate from work item results.

#### Implementation

**File: `src/batch_llm/base.py`**

```python
@dataclass
class BatchResult(Generic[TOutput, TContext]):
    """Result of processing a batch of work items."""

    results: list[WorkItemResult[TOutput, TContext]]
    total_items: int = 0
    succeeded: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0  # NEW: Track cached tokens separately

    def __post_init__(self):
        """Calculate summary statistics from results."""
        self.total_items = len(self.results)
        self.succeeded = sum(1 for r in self.results if r.success)
        self.failed = sum(1 for r in self.results if not r.success)
        self.total_input_tokens = sum(
            r.token_usage.get("input_tokens", 0) for r in self.results
        )
        self.total_output_tokens = sum(
            r.token_usage.get("output_tokens", 0) for r in self.results
        )
        # NEW: Aggregate cached tokens
        self.total_cached_tokens = sum(
            r.token_usage.get("cached_input_tokens", 0) for r in self.results
        )

    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage of input tokens cached."""
        if self.total_input_tokens == 0:
            return 0.0
        return (self.total_cached_tokens / self.total_input_tokens) * 100.0

    def effective_input_tokens(self) -> int:
        """Calculate effective input tokens (actual cost after caching)."""
        # Gemini charges 10% for cached tokens
        return self.total_input_tokens - int(self.total_cached_tokens * 0.9)
```

**File: `src/batch_llm/base.py` (ProcessingStats)**

```python
@dataclass
class ProcessingStats:
    """Statistics for batch processing."""

    # ... existing fields ...
    total_cached_tokens: int = 0  # NEW: Cached tokens across all items

    def copy(self) -> dict[str, Any]:
        """Return a dictionary copy of the stats."""
        return {
            # ... existing fields ...
            "total_cached_tokens": self.total_cached_tokens,  # NEW
        }
```

#### Breaking Changes

**None** - Adding optional fields to dataclasses is backward compatible.

#### Tests

**File: `tests/test_token_tracking.py`** (new)

```python
@pytest.mark.asyncio
async def test_cached_token_aggregation():
    """Test that cached tokens are properly aggregated."""
    from batch_llm.testing import MockStrategy

    class CachedTokenStrategy(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout):
            return "Response", {
                "input_tokens": 500,
                "output_tokens": 100,
                "total_tokens": 600,
                "cached_input_tokens": 450,  # 90% cached
            }

    strategy = CachedTokenStrategy()
    config = ProcessorConfig(max_workers=2)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(10):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt="Test")
            )

        result = await processor.process_all()

    assert result.total_input_tokens == 5000
    assert result.total_output_tokens == 1000
    assert result.total_cached_tokens == 4500
    assert result.cache_hit_rate() == 90.0
    assert result.effective_input_tokens() == 950  # 500 - (450 * 0.9)


@pytest.mark.asyncio
async def test_no_cached_tokens():
    """Test that missing cached_input_tokens doesn't break aggregation."""
    class NonCachedStrategy(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout):
            return "Response", {
                "input_tokens": 100,
                "output_tokens": 50,
                # No cached_input_tokens
            }

    strategy = NonCachedStrategy()
    config = ProcessorConfig(max_workers=1)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test")
        )

        result = await processor.process_all()

    assert result.total_cached_tokens == 0
    assert result.cache_hit_rate() == 0.0
```

#### Documentation Updates

**File: `README.md`**

Add to "Working with Results" section:

```markdown
### Cache Effectiveness Metrics

When using cached strategies (e.g., `GeminiCachedStrategy`), `BatchResult` provides cache metrics:

```python
result = await processor.process_all()

print(f"Total input tokens: {result.total_input_tokens}")
print(f"Cached tokens: {result.total_cached_tokens}")
print(f"Cache hit rate: {result.cache_hit_rate():.1f}%")
print(f"Effective cost: {result.effective_input_tokens()} tokens")
```

**Example output:**
```
Total input tokens: 50000
Cached tokens: 45000
Cache hit rate: 90.0%
Effective cost: 9500 tokens  # 81% cost reduction!
```
```

---

### Issue #5: Unclear Cache Lifecycle (cleanup() Hook)

#### Problem

Documentation doesn't clarify whether `cleanup()` should delete caches or preserve them for reuse across runs.

#### Solution

Update documentation to clarify cleanup() semantics and provide best practices.

#### Implementation

**File: `src/batch_llm/llm_strategies.py`**

Update docstring:

```python
class LLMCallStrategy:
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

        **Note on Caches:**
        For reusable resources like Gemini caches with TTLs, consider letting
        them expire naturally to enable cost savings across multiple pipeline
        runs. See `GeminiCachedStrategy` for an example.

        Default: no-op
        """
        pass
```

**File: `src/batch_llm/llm_strategies.py`**

Update `GeminiCachedStrategy.cleanup()`:

```python
class GeminiCachedStrategy(LLMCallStrategy[TOutput]):
    async def cleanup(self) -> None:
        """
        Cleanup hook - preserves cache for reuse by default.

        By default, this method does NOT delete the cache. The cache remains
        active until its TTL expires, allowing reuse across multiple runs
        within the TTL window (e.g., 1 hour).

        This enables significant cost savings when running multiple batches:
        - First run: Creates cache, pays full cost
        - Subsequent runs (within TTL): Reuse cache, 70-90% cost reduction

        To delete the cache immediately (e.g., for cleanup in tests):

        ```python
        strategy = GeminiCachedStrategy(...)
        # ... use strategy ...
        await strategy.delete_cache()  # Explicit deletion
        ```
        """
        if self._cache:
            logger.info(
                f"Leaving cache active for reuse: {self._cache.name} "
                f"(expires in {self.cache_ttl_seconds}s from creation)"
            )

    async def delete_cache(self) -> None:
        """
        Explicitly delete the Gemini cache.

        Call this when you want to immediately delete the cache instead of
        letting it expire naturally. Useful for:
        - Test cleanup
        - One-off batch jobs where reuse isn't needed
        - Updating cached content (delete old, create new)
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
```

#### Breaking Changes

**Behavioral change (non-breaking):** `GeminiCachedStrategy.cleanup()` no longer deletes the cache by default. This is technically a behavioral change, but it's opt-in (cleanup is only called if you call it), and the new behavior is more sensible for production use.

Migration: If you were relying on `cleanup()` to delete caches, call `await strategy.delete_cache()` explicitly instead.

#### Tests

**File: `tests/test_cache_lifecycle.py`** (new)

```python
@pytest.mark.asyncio
async def test_cleanup_preserves_cache():
    """Test that cleanup() does not delete cache by default."""
    # Mock test - would verify cleanup() doesn't call delete
    pass


@pytest.mark.asyncio
async def test_explicit_cache_deletion():
    """Test that delete_cache() explicitly removes cache."""
    # Mock test - would verify delete_cache() calls API
    pass
```

#### Documentation Updates

**File: `docs/GEMINI_INTEGRATION.md`**

Add section:

```markdown
## Cache Lifecycle Management

### Default Behavior: Cache Reuse

By default, `GeminiCachedStrategy` preserves caches for reuse across runs:

```python
async with ParallelBatchProcessor(...) as processor:
    # ... process items ...
    result = await processor.process_all()
# Cleanup is called, but cache is preserved

# Run again within TTL window
async with ParallelBatchProcessor(...) as processor:
    # ... process more items ...
    # Reuses existing cache if found - 70-90% cost savings!
```

### Explicit Cache Deletion

To delete caches immediately:

```python
strategy = GeminiCachedStrategy(...)

async with ParallelBatchProcessor(...) as processor:
    # ... use strategy ...
    pass

# Explicitly delete cache
await strategy.delete_cache()
```

### Best Practices

**Production pipelines (multiple runs):**
- Let caches expire naturally (don't call `delete_cache()`)
- Set TTL to cover expected run frequency (e.g., 1 hour for hourly jobs)
- Monitor cache reuse via `result.cache_hit_rate()`

**Tests and one-off jobs:**
- Call `delete_cache()` for cleanup
- Or use short TTL (60s) to auto-expire quickly

**Updating prompts:**
- Delete old cache before creating new one with updated content
- Or use cache tagging/metadata to identify versions (v0.3.0 feature)
```

---

### Issue #7: Cache Expiration Error Handling

#### Problem

Long-running pipelines (>1 hour) hit cache expiration errors when Google expires the cache on their servers. No built-in detection or recovery.

#### Solution

Add proactive cache expiration detection with automatic renewal in `GeminiCachedStrategy`.

#### Implementation

**File: `src/batch_llm/llm_strategies.py`**

```python
class GeminiCachedStrategy(LLMCallStrategy[TOutput]):
    def __init__(
        self,
        model: str,
        client: "genai.Client",
        response_parser: Callable[[Any], TOutput],
        cached_content: list["Content"],
        cache_ttl_seconds: int = 3600,
        cache_renewal_buffer_seconds: int = 300,  # NEW: Renew 5min before expiration
        cache_refresh_threshold: float = 0.1,  # Deprecated in favor of buffer
        config: "GenerateContentConfig | None" = None,
        auto_renew: bool = True,  # NEW: Automatically renew expired caches
    ):
        """
        Initialize Gemini cached strategy with automatic renewal.

        Args:
            cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour)
            cache_renewal_buffer_seconds: Renew cache this many seconds before
                expiration to avoid expiration errors (default: 300 = 5 minutes)
            auto_renew: Automatically renew expired caches in execute() (default: True)
        """
        # ... existing code ...
        self.cache_renewal_buffer_seconds = cache_renewal_buffer_seconds
        self.auto_renew = auto_renew
        self._cache_lock = asyncio.Lock()  # Protect cache renewal

    def _is_cache_expired(self) -> bool:
        """Check if cache has expired or is about to expire."""
        if self._cache is None or self._cache_created_at is None:
            return True

        cache_age = time.time() - self._cache_created_at
        expires_in = self.cache_ttl_seconds - cache_age

        return expires_in <= self.cache_renewal_buffer_seconds

    async def _find_or_create_cache(self) -> None:
        """Find existing cache or create new one."""
        # Try to find existing cache with same model
        try:
            caches = await self.client.aio.caches.list()

            for cache in caches:
                # Cache model is full path: "projects/.../models/gemini-..."
                # Match by model name suffix
                if cache.model.endswith(self.model):
                    self._cache = cache

                    # CRITICAL: Use cache's actual creation time, not current time
                    if hasattr(cache, 'create_time') and cache.create_time:
                        self._cache_created_at = cache.create_time.timestamp()
                    else:
                        # Fallback: assume old to trigger renewal check
                        self._cache_created_at = (
                            time.time() - self.cache_ttl_seconds
                        )

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
        """Create a new Gemini cache."""
        api_version = self._detect_google_genai_version()

        if api_version == "v1.46+":
            from google.genai.types import CreateCachedContentConfig

            self._cache = await self.client.aio.caches.create(
                model=self.model,
                config=CreateCachedContentConfig(
                    contents=self.cached_content,
                    ttl=f"{self.cache_ttl_seconds}s",
                ),
            )
        else:
            self._cache = await self.client.aio.caches.create(
                model=self.model,
                contents=self.cached_content,
                ttl=f"{self.cache_ttl_seconds}s",
            )

        self._cache_created_at = time.time()
        logger.info(
            f"Created new Gemini cache: {self._cache.name} "
            f"(TTL: {self.cache_ttl_seconds}s)"
        )

    async def prepare(self) -> None:
        """Find or create the Gemini cache."""
        await self._find_or_create_cache()

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, TokenUsage]:
        """Execute with automatic cache renewal."""
        # Check and renew cache if expired (proactive)
        if self.auto_renew and self._is_cache_expired():
            logger.info(
                "Cache expired or about to expire, renewing before API call"
            )
            async with self._cache_lock:
                # Double-check after acquiring lock
                if self._is_cache_expired():
                    # Clear cache reference to force creation of new cache
                    self._cache = None
                    self._cache_created_at = None
                    await self._find_or_create_cache()

        # ... rest of execute() logic (make API call with cache) ...
```

**File: `src/batch_llm/classifiers/gemini.py`**

Update error classifier to detect cache expiration:

```python
class GeminiErrorClassifier(ErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        """Classify Gemini API errors."""
        error_str = str(exception)

        # Check for cache expiration FIRST (before parent class checks)
        # This is critical because ClientError may be classified differently
        if "cache" in error_str.lower() and "expired" in error_str.lower():
            return ErrorInfo(
                is_retryable=True,  # Retry - cache will be renewed
                is_rate_limit=False,
                is_timeout=False,
                error_category="cache_expired",
                message="Gemini cache expired, will renew and retry",
            )

        # ... rest of existing error classification ...
```

#### Breaking Changes

**None** - New parameters have sensible defaults, existing code continues to work.

#### Tests

**File: `tests/test_cache_expiration.py`** (new)

```python
@pytest.mark.asyncio
async def test_cache_expiration_detection():
    """Test that expired caches are detected."""
    # Mock test using strategy with short TTL
    pass


@pytest.mark.asyncio
async def test_automatic_cache_renewal():
    """Test that expired caches are automatically renewed."""
    # Mock test simulating cache expiration during execution
    pass


@pytest.mark.asyncio
async def test_cache_reuse_creation_time():
    """Test that reused caches use actual creation time, not current time."""
    # This is the critical bug mentioned in feedback
    pass


@pytest.mark.asyncio
async def test_cache_expiration_error_classification():
    """Test that cache expiration errors are classified as retryable."""
    from batch_llm.classifiers import GeminiErrorClassifier

    classifier = GeminiErrorClassifier()

    # Simulate cache expiration error
    error = Exception(
        "400 INVALID_ARGUMENT. {'error': {'code': 400, "
        "'message': 'Cache content 5025785736947826688 is expired.', "
        "'status': 'INVALID_ARGUMENT'}}"
    )

    info = classifier.classify(error)

    assert info.is_retryable is True
    assert info.error_category == "cache_expired"
```

#### Documentation Updates

**File: `docs/GEMINI_INTEGRATION.md`**

Add section:

```markdown
## Long-Running Pipelines (Cache Expiration Handling)

For pipelines running longer than your cache TTL (e.g., 3+ hours with 1-hour cache):

### Automatic Renewal (Default)

`GeminiCachedStrategy` automatically renews expired caches:

```python
strategy = GeminiCachedStrategy(
    model="gemini-2.5-flash",
    client=client,
    cached_content=[...],
    cache_ttl_seconds=3600,  # 1 hour
    cache_renewal_buffer_seconds=300,  # Renew 5min before expiration
    auto_renew=True,  # Default: automatic renewal
)
```

**How it works:**
1. Before each API call, checks if cache will expire in <5 minutes
2. If yes, creates new cache or finds existing one
3. API call uses fresh cache
4. No expiration errors!

### Manual Control

Disable automatic renewal if you want manual control:

```python
strategy = GeminiCachedStrategy(
    ...,
    auto_renew=False,  # Disable automatic renewal
)

# Manually renew cache when needed
if strategy._is_cache_expired():
    await strategy.prepare()  # Force cache renewal
```

### Cost Implications

**Automatic renewal:**
- ✅ Zero downtime (no expiration errors)
- ✅ Optimal cache reuse (always uses fresh cache)
- ⚠️  May create new cache if old one expired

**Best practice:** Set TTL to slightly longer than expected run time to maximize reuse.
```

---

## Migration Guide (v0.1.x → v0.2.0)

**File: `docs/MIGRATION_V0_2.md`** (new)

```markdown
# Migration Guide: v0.1.x → v0.2.0

## Overview

Version 0.2.0 adds critical fixes for production usage, particularly around shared strategies and Gemini caching. Most changes are backward compatible.

## Breaking Changes

### 1. GeminiCachedStrategy cleanup() Behavior

**Before (v0.1):**
```python
strategy = GeminiCachedStrategy(...)
async with ParallelBatchProcessor(...) as processor:
    # ...
    pass
# cleanup() deletes cache
```

**After (v0.2):**
```python
strategy = GeminiCachedStrategy(...)
async with ParallelBatchProcessor(...) as processor:
    # ...
    pass
# cleanup() preserves cache for reuse

# Explicitly delete if needed
await strategy.delete_cache()
```

**Migration:** If you relied on automatic cache deletion, call `await strategy.delete_cache()` explicitly.

## New Features

### Shared Strategy Optimization

You can now share strategy instances across work items without duplicate `prepare()` calls:

```python
# Create one strategy
strategy = GeminiCachedStrategy(...)

# Share across all work items
for item in items:
    await processor.add_work(
        LLMWorkItem(strategy=strategy, ...)  # Reuse same instance
    )

# prepare() is called only once!
```

### Cached Token Tracking

`BatchResult` now includes cached token metrics:

```python
result = await processor.process_all()

print(f"Cache hit rate: {result.cache_hit_rate():.1f}%")
print(f"Effective cost: {result.effective_input_tokens()} tokens")
```

### Automatic Cache Renewal

Long-running pipelines automatically renew expired caches:

```python
strategy = GeminiCachedStrategy(
    cache_ttl_seconds=3600,
    cache_renewal_buffer_seconds=300,  # Renew 5min before expiration
    auto_renew=True,  # Default
)
```

## API Compatibility

### google-genai v1.46+

Both old and new google-genai APIs are supported:

```bash
# Recommended: Install latest
pip install 'batch-llm[gemini]'  # Gets google-genai>=1.46

# Legacy support (will be removed in v0.3)
pip install 'batch-llm[gemini]' 'google-genai<1.46'
```

## Recommended Actions

1. **Update to google-genai v1.46+** for best support
2. **Review cache cleanup logic** - add explicit `delete_cache()` calls if needed
3. **Enable shared strategies** for caching cost optimization
4. **Monitor cache metrics** using `result.cache_hit_rate()`

## Support

See docs/GEMINI_INTEGRATION.md for detailed examples and best practices.
```

---

## Testing Strategy

### Test Coverage Goals

- **Unit tests:** 90%+ coverage for new code
- **Integration tests:** Key workflows (shared strategies, cache renewal)
- **Mock tests:** No API calls required for core tests
- **Manual tests:** Real API testing with Gemini (documented in examples)

### Test Organization

```text
tests/
├── test_shared_strategies.py        # Issue #1 tests
├── test_gemini_api_versions.py      # Issue #2 tests
├── test_token_tracking.py           # Issue #4 tests
├── test_cache_lifecycle.py          # Issue #5 tests
├── test_cache_expiration.py         # Issue #7 tests
└── test_backward_compatibility.py   # Ensure v0.1 code still works
```

### Running Tests

```bash
# All tests
uv run pytest

# Specific issue tests
uv run pytest tests/test_shared_strategies.py -v

# Coverage report
uv run pytest --cov=batch_llm --cov-report=html
```

---

## Documentation Updates

### Files to Update

1. **README.md**
   - Add shared strategies section
   - Add cache metrics examples
   - Update quickstart examples

2. **docs/API.md**
   - Document new BatchResult fields
   - Document GeminiCachedStrategy parameters
   - Add cache lifecycle methods

3. **docs/GEMINI_INTEGRATION.md**
   - Add API version compatibility section
   - Add cache lifecycle best practices
   - Add long-running pipeline patterns

4. **docs/MIGRATION_V0_2.md** (new)
   - Migration guide for v0.1 → v0.2

5. **CLAUDE.md**
   - Update version history
   - Add v0.2.0 features and patterns
   - Update known limitations

6. **CHANGELOG.md**
   - Document all changes, fixes, improvements

---

## Release Checklist

### Pre-Release

- [ ] All tests passing (uv run pytest)
- [ ] Linting passing (uv run ruff check src/ tests/)
- [ ] Type checking passing (uv run mypy src/batch_llm/)
- [ ] Documentation updated
- [ ] Examples updated/tested
- [ ] Migration guide written
- [ ] CHANGELOG.md updated

### Release

- [ ] Update version in pyproject.toml (0.1.0 → 0.2.0)
- [ ] Create git tag (v0.2.0)
- [ ] Build package (uv build)
- [ ] Test on TestPyPI
- [ ] Publish to PyPI
- [ ] Create GitHub release with notes

### Post-Release

- [ ] Verify installation (pip install batch-llm==0.2.0)
- [ ] Test examples with fresh install
- [ ] Update documentation site (if applicable)
- [ ] Announce release

---

## Timeline Estimate

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Week 1, Days 1-2** | 2 days | Implement Issue #1, #2, #4 with tests |
| **Week 1, Days 3-4** | 2 days | Implement Issue #5, #7 with tests |
| **Week 1, Day 5** | 1 day | Documentation updates, migration guide |
| **Week 2, Days 1-2** | 2 days | Integration testing, bug fixes |
| **Week 2, Day 3** | 1 day | Final review, release preparation |
| **Total** | ~7 days | Full implementation + testing + docs |

---

## Success Criteria

### Functional

- [ ] Shared strategies only call prepare() once
- [ ] Works with both google-genai v1.45 and v1.46+
- [ ] Cached tokens properly tracked in BatchResult
- [ ] Cache lifecycle clearly documented
- [ ] Cache expiration automatically handled
- [ ] All tests passing (90%+ coverage)

### Non-Functional

- [ ] Backward compatible with v0.1 code
- [ ] No performance regression
- [ ] Clear migration path documented
- [ ] Production-ready (tested with real Gemini API)

---

## Future Work (v0.3.0)

Issues deferred to v0.3.0:

- **Issue #3:** Rich response metadata (safety ratings, finish reason)
- **Issue #6:** Cache tagging/metadata for precise matching
- **Issue #8:** Per-work-item state for advanced retry strategies

See IMPLEMENTATION_PLAN_V0_3.md (to be created).

---

## Questions / Decisions Needed

1. **Backward compatibility:** Should we maintain v0.1 cleanup() behavior with a flag?
   - **Recommendation:** No, new behavior is better. Document migration.

2. **API version support:** How long to support google-genai <1.46?
   - **Recommendation:** Support in v0.2, deprecate with warning, remove in v0.3.

3. **Cache matching:** Should we add basic tagging in v0.2 or defer to v0.3?
   - **Recommendation:** Defer to v0.3. v0.2 is already substantial.

---

## Appendix: Code Locations

### Files to Modify

- `src/batch_llm/base.py` - BatchResult, ProcessingStats, cleanup()
- `src/batch_llm/llm_strategies.py` - GeminiCachedStrategy enhancements
- `src/batch_llm/parallel.py` - Strategy preparation tracking
- `src/batch_llm/classifiers/gemini.py` - Cache expiration detection

### Files to Create

- `tests/test_shared_strategies.py`
- `tests/test_gemini_api_versions.py`
- `tests/test_token_tracking.py`
- `tests/test_cache_lifecycle.py`
- `tests/test_cache_expiration.py`
- `tests/test_backward_compatibility.py`
- `docs/MIGRATION_V0_2.md`

### Files to Update

- `README.md`
- `docs/API.md`
- `docs/GEMINI_INTEGRATION.md`
- `CLAUDE.md`
- `CHANGELOG.md`
- `pyproject.toml` (version bump)
