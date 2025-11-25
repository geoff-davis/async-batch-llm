# Migration Guide: v0.3.x → v0.4.0

This guide helps you migrate from batch-llm v0.3.x to v0.4.0, which introduces strategy lifecycle
management with context managers.

## Summary of Changes

v0.4.0 adds strategy lifecycle management using Python's context manager pattern (`async with`).
The main breaking change is that **per-item cleanup has been removed** in favor of processor-level
cleanup.

## Breaking Changes

### 1. Per-Item Cleanup Removed

**What Changed:**

- **v0.3.x**: `strategy.cleanup()` was called after each work item completed
- **v0.4.0**: `strategy.cleanup()` is only called once when exiting the context manager

**Why This Matters:**

In v0.3.x, if you had expensive resources in your strategy (like database connections or caches),
they were created in `prepare()` and destroyed in `cleanup()` for EVERY work item. This was
inefficient for shared resources.

In v0.4.0, resources are created once in `prepare()` and destroyed once in `cleanup()` when the
processor exits, which is more efficient for batch processing.

**Migration Required If:**

You rely on `cleanup()` being called after each item to release resources or save state.

**How to Migrate:**

#### Option 1: Use Context Manager (Recommended)

Wrap your processor in `async with` to enable automatic cleanup on exit:

```python
# v0.3.x - cleanup called after each item
processor = ParallelBatchProcessor(config=config)
await processor.add_work(LLMWorkItem(...))
result = await processor.process_all()
# cleanup() was called N times (once per item)

# v0.4.0 - cleanup called once on exit
async with ParallelBatchProcessor(config=config) as processor:
    await processor.add_work(LLMWorkItem(...))
    result = await processor.process_all()
    # All work completed
# cleanup() called here (once total)
```

#### Option 2: Keep Backward Compatible Behavior

If you don't use the context manager, cleanup is **never** called (backward compatible):

```python
# v0.4.0 without context manager - no cleanup
processor = ParallelBatchProcessor(config=config)
await processor.add_work(LLMWorkItem(...))
result = await processor.process_all()
# No cleanup() called - same as v0.2.0 behavior
```

This preserves backward compatibility but means resources won't be automatically cleaned up.

#### Option 3: Manual Cleanup (Not Recommended)

You can manually call cleanup if needed, but this is discouraged:

```python
processor = ParallelBatchProcessor(config=config)
try:
    await processor.add_work(LLMWorkItem(...))
    result = await processor.process_all()
finally:
    # Manually cleanup strategies
    for strategy in processor._prepared_strategies:
        await strategy.cleanup()
```

**Best Practice:** Use the context manager pattern (Option 1) for automatic resource management.

### 2. Production Caches Should Not Be Cleaned Up

**What Changed:**

If you're using strategies with long-lived resources (like production caches intended to persist
across batches), you need to make `cleanup()` a no-op.

**Example:**

```python
class ProdCachedStrategy(GeminiCachedStrategy):
    """Production strategy with persistent cache."""

    async def cleanup(self) -> None:
        """
        Don't delete cache - it should persist across batches.

        Override parent's cleanup() to prevent cache deletion.
        """
        # Do nothing - cache persists for cost optimization
        pass
```

**Why:** In v0.3.x, cleanup was called per-item so you couldn't have persistent caches. In v0.4.0,
cleanup is called once at the end, so you need to explicitly prevent cache deletion if you want it
to persist.

## New Features

### 1. RuntimeError When Adding Work After Processing Starts

**What Changed:**

Calling `add_work()` after `process_all()` has started now raises `RuntimeError`.

**Why:** This prevents race conditions and ensures all work is queued before processing begins.

**Example:**

```python
async with ParallelBatchProcessor(config=config) as processor:
    await processor.add_work(LLMWorkItem(item_id="1", ...))

    # Start processing
    result = await processor.process_all()

    # This now raises RuntimeError
    try:
        await processor.add_work(LLMWorkItem(item_id="2", ...))
    except RuntimeError as e:
        print(f"Cannot add work after processing starts: {e}")
```

**Migration:** If you need to process multiple batches, create a new processor instance for each
batch:

```python
# Process first batch
async with ParallelBatchProcessor(config=config) as processor1:
    await processor1.add_work(LLMWorkItem(item_id="1", ...))
    result1 = await processor1.process_all()

# Process second batch with new processor
async with ParallelBatchProcessor(config=config) as processor2:
    await processor2.add_work(LLMWorkItem(item_id="2", ...))
    result2 = await processor2.process_all()
```

### 2. Shared Strategy Instances

**What Changed:**

Shared strategy instances are now properly supported - they're prepared once and cleaned up once.

**Example:**

```python
# Create shared strategy for cost optimization
shared_strategy = GeminiCachedStrategy(
    model="gemini-2.0-flash",
    system_instruction="...",  # Expensive to cache
)

async with ParallelBatchProcessor(config=config) as processor:
    # Use same strategy for all items
    for i in range(100):
        await processor.add_work(
            LLMWorkItem(item_id=f"item_{i}", strategy=shared_strategy, prompt=f"...")
        )

    result = await processor.process_all()
    # shared_strategy.prepare() called once
    # shared_strategy.execute() called 100 times
# shared_strategy.cleanup() called once
```

**Benefit:** Sharing strategies saves memory and avoids duplicate cache creation costs.

## Non-Breaking Changes

### Strategy Without prepare() or cleanup()

Strategies don't need to implement `prepare()` or `cleanup()` - they're optional:

```python
class SimpleStrategy(LLMCallStrategy[str]):
    """Minimal strategy without lifecycle methods."""

    async def execute(self, prompt, attempt, timeout, state=None):
        # Just do the work
        return output, tokens

# Works fine - no prepare() or cleanup() needed
strategy = SimpleStrategy()
async with ParallelBatchProcessor(config=config) as processor:
    await processor.add_work(LLMWorkItem(strategy=strategy, ...))
    result = await processor.process_all()
```

## Migration Checklist

- [ ] Wrap all `ParallelBatchProcessor` usage in `async with` context managers
- [ ] Review custom strategies with `cleanup()` methods
  - [ ] For temporary resources: Keep cleanup implementation (will be called once on exit)
  - [ ] For persistent caches: Override `cleanup()` to be a no-op
- [ ] Update code that calls `add_work()` after `process_all()`
  - [ ] Create new processor instances for additional batches
- [ ] Run tests to verify cleanup behavior is correct
- [ ] Update documentation/examples to use context manager pattern

## Testing Your Migration

Run these tests to verify your migration:

```python
# Test 1: Verify cleanup is called with context manager
strategy = YourStrategy()
async with ParallelBatchProcessor(config=config) as processor:
    await processor.add_work(LLMWorkItem(strategy=strategy, ...))
    result = await processor.process_all()
    assert not strategy.cleanup_called, "Cleanup not called yet"
# Assert cleanup was called after exiting context
assert strategy.cleanup_called, "Cleanup should be called on exit"

# Test 2: Verify backward compatibility without context manager
strategy = YourStrategy()
processor = ParallelBatchProcessor(config=config)
await processor.add_work(LLMWorkItem(strategy=strategy, ...))
result = await processor.process_all()
assert not strategy.cleanup_called, "Cleanup not called without context manager"

# Test 3: Verify RuntimeError on late add_work()
async with ParallelBatchProcessor(config=config) as processor:
    await processor.add_work(LLMWorkItem(...))
    await processor.process_all()

    try:
        await processor.add_work(LLMWorkItem(...))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass  # Expected
```

## Need Help?

If you encounter issues during migration:

1. Check the [CHANGELOG.md](../CHANGELOG.md) for detailed changes
2. Review [test_strategy_lifecycle.py](../tests/test_strategy_lifecycle.py) for examples
3. File an issue at <https://github.com/geoff-davis/async-batch-llm/issues>

## Benefits of v0.4.0

After migration, you get:

1. **Better resource management** - Cleanup happens at the right time (once per batch)
2. **Cost optimization** - Shared strategies with persistent caches work correctly
3. **Clear lifecycle** - Prepare on first use, cleanup on exit (Pythonic context managers)
4. **Fail-fast** - Runtime errors prevent invalid usage patterns
5. **Backward compatible** - Existing code without context managers still works
