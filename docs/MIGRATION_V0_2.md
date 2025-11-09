# Migration Guide: v0.1.x → v0.2.0

## Overview

Version 0.2.0 adds critical fixes for production usage, particularly around shared strategies and Gemini caching. Most changes are backward compatible with minimal breaking changes.

**Key improvements:**
- Shared strategy instances (prepare() called only once)
- Cached token tracking in BatchResult
- Automatic cache renewal for long pipelines
- google-genai v1.46+ compatibility
- Cache lifecycle clarification

---

## Breaking Changes

### 1. GeminiCachedStrategy cleanup() Behavior

**Before (v0.1):**
```python
strategy = GeminiCachedStrategy(...)
async with ParallelBatchProcessor(...) as processor:
    # ... process items ...
    pass
# cleanup() automatically deletes cache
```

**After (v0.2):**
```python
strategy = GeminiCachedStrategy(...)
async with ParallelBatchProcessor(...) as processor:
    # ... process items ...
    pass
# cleanup() preserves cache for reuse

# Explicitly delete if needed
await strategy.delete_cache()
```

**Why:** Preserving caches between runs enables 70-90% cost savings when running multiple batches within the TTL window.

**Migration:**
- If you relied on automatic cache deletion, call `await strategy.delete_cache()` explicitly
- For most production use cases, the new behavior is better (no code changes needed)

---

## New Features

### 1. Shared Strategy Optimization

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

**Benefits:**
- Single cache created and shared across all work items
- 70-90% cost reduction with Gemini prompt caching
- Framework ensures thread-safe initialization

### 2. Cached Token Tracking

`BatchResult` now includes cached token metrics:

```python
result = await processor.process_all()

print(f"Cache hit rate: {result.cache_hit_rate():.1f}%")
print(f"Effective cost: {result.effective_input_tokens()} tokens")
```

**New fields:**
- `BatchResult.total_cached_tokens` - Sum of cached input tokens
- `BatchResult.cache_hit_rate()` - Percentage of input tokens cached
- `BatchResult.effective_input_tokens()` - Actual cost after caching discount

### 3. Automatic Cache Renewal

Long-running pipelines automatically renew expired caches:

```python
strategy = GeminiCachedStrategy(
    cache_ttl_seconds=3600,  # 1 hour
    cache_renewal_buffer_seconds=300,  # Renew 5min before expiration
    auto_renew=True,  # Default: automatic renewal
)
```

**How it works:**
1. Before each API call, checks if cache will expire soon
2. If yes, creates new cache or finds existing one
3. API call uses fresh cache
4. No expiration errors!

**Benefits:**
- Pipelines > 1 hour don't fail with cache expiration errors
- Automatic renewal with configurable buffer
- Proactive renewal prevents downtime

### 4. Cache Reuse Across Runs

Caches are now preserved between runs (within TTL window):

```python
# Run 1: Creates cache at 10:00, expires at 11:00
async with ParallelBatchProcessor(...) as processor:
    # ... process 1000 items ...
    pass

# Run 2: Starts at 10:30, reuses cache (30min left)
async with ParallelBatchProcessor(...) as processor:
    # ... process 1000 more items ...
    # 70-90% cost savings!
    pass
```

**Best practices:**
- Set TTL to slightly longer than expected run frequency
- Monitor cache reuse via `result.cache_hit_rate()`
- For hourly jobs, use TTL of 3600s (1 hour)

---

## API Compatibility

### google-genai Versions

Both old and new google-genai APIs are supported:

```bash
# Recommended: Install latest
pip install 'batch-llm[gemini]'  # Gets google-genai>=1.46

# Legacy support (will be removed in v0.3)
pip install 'batch-llm[gemini]' 'google-genai<1.46'
```

The framework auto-detects which version you have and uses the appropriate API.

---

## Recommended Actions

### For All Users

1. **Update to v0.2.0:**
   ```bash
   pip install --upgrade batch-llm
   ```

2. **Review cache cleanup logic:**
   - If you explicitly delete caches, add `await strategy.delete_cache()` calls
   - For most users, no changes needed (new behavior is better)

3. **Enable shared strategies for caching:**
   ```python
   # Before: New strategy per item (creates multiple caches)
   for item in items:
       strategy = GeminiCachedStrategy(...)  # Don't do this
       work_item = LLMWorkItem(strategy=strategy, ...)

   # After: Shared strategy (creates single cache)
   strategy = GeminiCachedStrategy(...)  # Create once
   for item in items:
       work_item = LLMWorkItem(strategy=strategy, ...)  # Reuse
   ```

4. **Monitor cache metrics:**
   ```python
   result = await processor.process_all()
   print(f"Cache hit rate: {result.cache_hit_rate():.1f}%")
   ```

### For Gemini Users

1. **Update google-genai to v1.46+:**
   ```bash
   pip install --upgrade 'google-genai>=1.46'
   ```

2. **Enable auto-renewal for long pipelines:**
   ```python
   strategy = GeminiCachedStrategy(
       cache_ttl_seconds=3600,
       cache_renewal_buffer_seconds=300,  # Renew 5min before expiration
       auto_renew=True,  # Enable automatic renewal
   )
   ```

3. **Leverage cache reuse across runs:**
   - Don't call `delete_cache()` unless needed
   - Set appropriate TTL for your run frequency
   - Monitor cost savings via `effective_input_tokens()`

---

## Deprecations

### Deprecated Parameters

- `GeminiCachedStrategy.cache_refresh_threshold` - Use `cache_renewal_buffer_seconds` instead

**Before (v0.1):**
```python
strategy = GeminiCachedStrategy(
    cache_refresh_threshold=0.1,  # Deprecated: Refresh if <10% TTL remaining
)
```

**After (v0.2):**
```python
strategy = GeminiCachedStrategy(
    cache_renewal_buffer_seconds=300,  # Renew 5min before expiration
)
```

**Why:** Absolute time (seconds) is more predictable than percentage for long-running jobs.

---

## Common Migration Patterns

### Pattern 1: Test Cleanup

**Before:**
```python
async def test_something():
    strategy = GeminiCachedStrategy(...)
    async with ParallelBatchProcessor(...) as processor:
        # ... test code ...
        pass
    # Cache automatically deleted
```

**After:**
```python
async def test_something():
    strategy = GeminiCachedStrategy(...)
    async with ParallelBatchProcessor(...) as processor:
        # ... test code ...
        pass
    # Explicitly delete for tests
    await strategy.delete_cache()
```

### Pattern 2: One-Off Jobs

**Before:**
```python
# One-off job - cache deleted automatically
async with ParallelBatchProcessor(...) as processor:
    # ... process items ...
    pass
```

**After:**
```python
# One-off job - explicitly delete if cache won't be reused
strategy = GeminiCachedStrategy(...)
async with ParallelBatchProcessor(...) as processor:
    # ... process items ...
    pass

await strategy.delete_cache()  # Clean up
```

### Pattern 3: Recurring Jobs

**Before:**
```python
# Recurring job - paid full cost every time
async with ParallelBatchProcessor(...) as processor:
    # ... process items ...
    pass
# Cache deleted, next run pays full cost
```

**After:**
```python
# Recurring job - reuses cache between runs
async with ParallelBatchProcessor(...) as processor:
    # ... process items ...
    pass
# Cache preserved, next run (within TTL) gets 70-90% discount!
```

---

## Troubleshooting

### Cache Not Reused

**Symptom:** `cache_hit_rate()` is 0% on subsequent runs

**Solutions:**
1. Check that cache hasn't expired between runs
2. Verify you're using the same model name
3. Check logs for "Reusing existing Gemini cache" message
4. Ensure you're not calling `delete_cache()` between runs

### prepare() Called Multiple Times

**Symptom:** Multiple caches created when you expect one

**Solution:** Share the same strategy instance:
```python
# Wrong: Creates new strategy (and cache) per item
for item in items:
    strategy = GeminiCachedStrategy(...)  # New instance each time
    work_item = LLMWorkItem(strategy=strategy, ...)

# Right: Reuse same strategy instance
strategy = GeminiCachedStrategy(...)  # Create once
for item in items:
    work_item = LLMWorkItem(strategy=strategy, ...)  # Reuse
```

### Cache Expiration Errors

**Symptom:** "Cache content XXX is expired" errors in long pipelines

**Solution:** Enable auto-renewal:
```python
strategy = GeminiCachedStrategy(
    auto_renew=True,  # Enable automatic renewal
    cache_renewal_buffer_seconds=300,  # Renew 5min before expiration
)
```

---

## Support

### Documentation

- [README.md](../README.md) - Updated examples and features
- [docs/GEMINI_INTEGRATION.md](GEMINI_INTEGRATION.md) - Gemini-specific guide
- [docs/API.md](API.md) - Full API reference
- [IMPLEMENTATION_PLAN_V0_2.md](IMPLEMENTATION_PLAN_V0_2.md) - Technical details

### Examples

- `examples/example_gemini_direct.py` - Direct Gemini API usage
- `examples/example_gemini_smart_retry.py` - Smart retry patterns
- `examples/example_model_escalation.py` - Cost optimization

### Getting Help

- GitHub Issues: https://github.com/yourusername/batch-llm/issues
- Check logs for debug information
- Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

---

## Version History

- **v0.2.0** (Current)
  - Shared strategy optimization
  - Cached token tracking
  - Automatic cache renewal
  - google-genai v1.46+ support
  - Cache lifecycle improvements

- **v0.1.0**
  - Strategy pattern refactor
  - PydanticAI, Gemini, custom strategies
  - Framework-level timeout enforcement

- **v0.0.2.x**
  - Direct API call support
  - Race condition fixes

- **v0.0.1.x**
  - Initial release
  - PydanticAI agent support
