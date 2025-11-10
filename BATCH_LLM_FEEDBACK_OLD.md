# batch-llm Feedback: Shared Strategy Pattern Issues

## Context

While migrating `enrich_works.py` to use the standalone `batch-llm` package (v0.1.0), we encountered several issues related to sharing a single strategy instance across multiple work items for cost optimization with Gemini prompt caching.

## Use Case: Gemini Prompt Caching

Our enrichment pipeline processes thousands of books, each with unique data but sharing the same prompt instructions (~500 tokens). Gemini offers 90% discount on cached tokens, providing 70-75% cost reduction overall.

**Goal**: Create one Gemini cache and share it across all work items.

**Implementation**:

```python
# Create one shared strategy
strategy = create_cached_strategy(
    model_name=model_name,
    cache_ttl_seconds=3600,
)

# Use same strategy for all work items
for work in works:
    work_item = LLMWorkItem(
        item_id=work.work_key,
        strategy=strategy,  # SHARED instance
        prompt=format_work_prompt(work),
        context=work,
    )
    await processor.add_work(work_item)
```

## Issues Encountered

### Issue 1: Multiple `prepare()` Calls on Shared Strategy

**Problem**: `ParallelBatchProcessor` calls `strategy.prepare()` once per work item, even when the same strategy instance is reused. With concurrent workers, multiple workers call `prepare()` simultaneously on the shared strategy.

**Impact**:

- Created 2+ Gemini caches instead of 1
- Wasted API quota and cost
- Negated caching benefits

**Our Workaround**:

```python
class GeminiEnrichmentCachedStrategy:
    def __init__(self, ...):
        self._cache = None
        self._cache_lock = asyncio.Lock()

    async def prepare(self) -> None:
        """Create cache only once, even with concurrent calls."""
        async with self._cache_lock:
            if self._cache is not None:
                return  # Already prepared

            # Create cache...
```

**Why This Works But Isn't Ideal**:

- Requires every strategy to implement idempotency guards
- Lock overhead on every prepare() call
- Unclear from framework whether strategies should be shared

### Issue 2: API Version Incompatibility

**Problem**: `batch-llm` v0.1.0's `GeminiCachedStrategy` uses an older Gemini API:

```python
# batch-llm v0.1.0 (works with older google-genai)
self._cache = await self.client.aio.caches.create(
    model=self.model,
    contents=self.cached_content,
    ttl=f"{self.cache_ttl_seconds}s",
)
```

But `google-genai` v1.46+ changed the API:

```python
# google-genai v1.46+ (required for newer features)
self._cache = await self.client.aio.caches.create(
    model=self.model,
    config=CreateCachedContentConfig(
        contents=self.cached_content,
        ttl=f"{self.cache_ttl_seconds}s",
    ),
)
```

**Impact**: Forced to write custom strategy instead of using built-in `GeminiCachedStrategy`.

### Issue 3: Missing Safety Ratings in `GeminiCachedStrategy`

**Problem**: The built-in `GeminiCachedStrategy` uses a `response_parser` function that only receives the parsed response, not the raw Gemini response object.

**Impact**: Cannot access Gemini safety ratings (harassment, hate speech, etc.) which are critical for content filtering.

**Our Need**:

```python
# We need access to raw response
response = await client.aio.models.generate_content(...)

# Extract safety ratings from response metadata
safety_ratings = {}
for rating in response.candidates[0].safety_ratings:
    safety_ratings[rating.category] = rating.probability

# AND parse the content
metadata = parse_json_response(response.text)
```

**Current GeminiCachedStrategy Limitation**:

```python
# response_parser only gets parsed response, not raw response object
def response_parser(response) -> CanonicalBookMetadata:
    # No access to response.candidates[0].safety_ratings
    return CanonicalBookMetadata(**json.loads(response.text))
```

## Recommendations for batch-llm

### 1. Document Shared Strategy Pattern

**Add to README**:

```markdown
## Sharing Strategies Across Work Items

For cost optimization (e.g., Gemini prompt caching), you may want to share
a single strategy instance across all work items:

```python
# Create one strategy
strategy = GeminiCachedStrategy(...)

# Reuse for all items
for item in items:
    work_item = LLMWorkItem(
        strategy=strategy,  # Shared instance
        prompt=format_prompt(item),
        ...
    )
```

**Important**: When sharing strategies, ensure your `prepare()` method is
idempotent and thread-safe. The framework calls `prepare()` once per work
item, but shared strategies should only initialize once:

```python
class MySharedStrategy:
    def __init__(self):
        self._prepared = False
        self._lock = asyncio.Lock()

    async def prepare(self) -> None:
        async with self._lock:
            if self._prepared:
                return
            # ... expensive initialization ...
            self._prepared = True
```

```

### 2. Add `PrepareOnce` Mixin or Decorator

**Proposed API**:

```python
from batch_llm.mixins import PrepareOnceMixin

class MyStrategy(PrepareOnceMixin):
    async def _prepare_once(self) -> None:
        """Called exactly once, even with concurrent prepare() calls."""
        # Expensive initialization here
        pass

    # Framework calls prepare(), mixin ensures _prepare_once() runs once
```

Or as a decorator:

```python
from batch_llm.decorators import prepare_once

class MyStrategy:
    @prepare_once
    async def prepare(self) -> None:
        """Automatically guarded - only runs once."""
        # Expensive initialization
        pass
```

### 3. Update `GeminiCachedStrategy` for google-genai v1.46+

**Current Code** (batch-llm v0.1.0):

```python
self._cache = await self.client.aio.caches.create(
    model=self.model,
    contents=self.cached_content,
    ttl=f"{self.cache_ttl_seconds}s",
)
```

**Updated for v1.46+**:

```python
from google.genai.types import CreateCachedContentConfig

self._cache = await self.client.aio.caches.create(
    model=self.model,
    config=CreateCachedContentConfig(
        contents=self.cached_content,
        ttl=f"{self.cache_ttl_seconds}s",
    ),
)
```

### 4. Add Raw Response Access to `GeminiCachedStrategy`

**Current Limitation**:

```python
class GeminiCachedStrategy:
    def __init__(
        self,
        response_parser: Callable[[Any], TOutput],  # Only gets parsed response
        ...
    ):
```

**Proposed Enhancement**:

```python
from dataclasses import dataclass

@dataclass
class GeminiResponse(Generic[TOutput]):
    """Container for parsed output and raw response metadata."""
    output: TOutput
    safety_ratings: dict[str, str] | None
    finish_reason: str | None
    token_usage: dict[str, int]
    raw_response: Any  # Full response object

class GeminiCachedStrategy:
    def __init__(
        self,
        response_parser: Callable[[Any], TOutput],
        include_metadata: bool = False,  # Opt-in for backward compatibility
        ...
    ):
        self.include_metadata = include_metadata

    async def execute(self, ...) -> tuple[TOutput | GeminiResponse[TOutput], dict[str, int]]:
        response = await self.client.aio.models.generate_content(...)

        output = self.response_parser(response)

        if self.include_metadata:
            # Extract safety ratings, finish reason, etc.
            safety_ratings = self._extract_safety_ratings(response)
            return GeminiResponse(
                output=output,
                safety_ratings=safety_ratings,
                finish_reason=response.candidates[0].finish_reason,
                token_usage=token_usage,
                raw_response=response,
            ), token_usage

        return output, token_usage
```

**Usage**:

```python
strategy = GeminiCachedStrategy(
    response_parser=parse_metadata,
    include_metadata=True,  # Get full response wrapper
)

# Later, in post-processor:
result = work_item_result.output  # GeminiResponse
metadata = result.output  # Parsed CanonicalBookMetadata
safety_ratings = result.safety_ratings  # Safety ratings!
```

## Alternative: Framework-Level Strategy Lifecycle

**More Ambitious Approach**: Framework manages strategy lifecycle explicitly.

```python
# Processor manages strategy preparation
processor = ParallelBatchProcessor(...)

# Register strategies before adding work
await processor.register_strategy("enrichment", strategy)

# Work items reference strategies by name
work_item = LLMWorkItem(
    strategy_id="enrichment",  # Reference, not instance
    ...
)
```

**Benefits**:

- Framework calls `prepare()` exactly once per registered strategy
- Framework calls `cleanup()` automatically at end
- Clear ownership and lifecycle

**Drawback**: More complex API change

### Issue 4: Cached Token Tracking Not Exposed

**Problem**: The batch-llm framework's `BatchResult` doesn't track cached tokens, only `total_input_tokens` and `total_output_tokens`.

**Impact**:

- Can't measure cache effectiveness (70-75% of tokens are cached)
- Can't calculate actual cost savings
- Can't verify caching is working

**Our Workaround**:

```python
# In enrich_works.py - manually extract and sum cached tokens
total_cached_tokens = sum(
    r.token_usage.get("cached_tokens", 0)
    for r in result.results
    if r.token_usage is not None
)
token_usage.add(result.total_input_tokens, result.total_output_tokens, total_cached_tokens)
```

**Why This Is Needed**: Gemini returns `cached_content_token_count` in `usage_metadata`, but batch-llm doesn't aggregate it into `BatchResult`.

**Recommendation**: Add `total_cached_tokens` to `BatchResult`:

```python
@dataclass
class BatchResult:
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int = 0  # NEW: Track cached tokens separately
    ...
```

### Issue 5: Cache Lifecycle Unclear

**Problem**: The framework calls `cleanup()` at the end of processing, but it's unclear whether this hook should:

- Delete resources (caches, connections) for cleanup, or
- Preserve resources (caches) for reuse across runs

**Impact**:

- Initially implemented cleanup() to delete caches
- This prevented reuse and wasted the 1-hour TTL
- Cost savings disappeared because each run created a new cache

**Our Solution**: Changed cleanup() to NOT delete caches:

```python
async def cleanup(self) -> None:
    """Don't delete cache - let it expire naturally for reuse."""
    if self._cache:
        logger.info(
            f"Leaving cache active for reuse: {self._cache.name} "
            f"(expires in {self.cache_ttl_seconds}s from creation)"
        )
```

**Result**: Multiple runs within 1 hour now reuse the same cache, providing 70-75% cost savings.

**Extension - Cache Renewal for Long-Running Jobs**:

For pipelines that run longer than the cache TTL (e.g., 3+ hours), we also need to:

- Track cache creation time
- Check if cache is expired before each API call
- Automatically renew expired caches

This requires additional strategy code that batch-llm doesn't provide:

```python
def _is_cache_expired(self) -> bool:
    """Check if cache has expired."""
    if self._cache is None or self._cache_created_at is None:
        return True
    cache_age_seconds = time.time() - self._cache_created_at
    renewal_buffer = 300  # Renew 5 min before expiration
    return cache_age_seconds >= (self.cache_ttl_seconds - renewal_buffer)

async def execute(self, prompt: str, attempt: int, timeout: float):
    """Execute with automatic cache renewal."""
    # Check and renew cache if expired
    if self._is_cache_expired():
        await self.prepare()  # Creates new cache or reuses existing
    # ... continue with API call
```

**CRITICAL BUG - Cache Reuse Creation Time**:

When reusing an existing cache (found via list/search), you MUST use the cache's actual creation time, not the current time:

```python
# ❌ WRONG - Causes cache expiration to fail
for cache in caches:
    if cache.model == target_model:
        self._cache = cache
        self._cache_created_at = time.time()  # BUG: Uses current time!
        return

# ✅ CORRECT - Uses actual cache creation time
for cache in caches:
    if cache.model == target_model:
        self._cache = cache
        # Use cache's actual creation time from Google's servers
        if hasattr(cache, 'create_time') and cache.create_time:
            self._cache_created_at = cache.create_time.timestamp()
        else:
            # Fallback: assume old to trigger renewal
            self._cache_created_at = time.time() - self.cache_ttl_seconds
        return
```

**Why This Matters**:

Example with the bug:

- Cache created at 10:19, expires at 11:19 (1-hour TTL)
- New run at 11:00 finds and reuses cache
- Bug: Sets `_cache_created_at = 11:00` (should be 10:19)
- Thinks cache expires at 12:00 (11:00 + 3600s)
- Actual expiration: 11:19
- Result: API calls fail with "cache expired" between 11:19-12:00

**Impact**: Expiration detection fails, causing API errors that bypass proactive renewal.

**Recommendation**: Document the cleanup() hook's purpose more clearly:

```python
async def cleanup(self) -> None:
    """
    Cleanup hook called after all work items are processed.

    Use this for:
    - Closing connections/sessions
    - Releasing locks
    - Logging final metrics

    Do NOT use this for:
    - Deleting caches intended for reuse across runs
    - Destructive cleanup that prevents resource reuse

    Note: For reusable resources like Gemini caches with TTLs,
    consider letting them expire naturally to enable cost savings
    across multiple pipeline runs.
    """
```

### Issue 6: Cache Matching Precision

**Problem**: When checking for existing caches, we need to match:

- Model name (important)
- Cache content/prompt (ideally, but hard to check without fetching)

**Our Approach**: Simple model name matching with `.endswith()`:

```python
for cache in caches:
    # Cache model is full path: "projects/.../models/gemini-2.5-flash-lite..."
    # Our model is short name: "gemini-2.5-flash-lite-preview-09-2025"
    if cache.model.endswith(self.model_name.value):
        self._cache = cache
        logger.info(f"Reusing existing Gemini cache: {self._cache.name}")
        return
```

**Trade-off**:

- ✅ Simple and fast (no need to fetch cache content)
- ⚠️ Will reuse cache even if prompt changed
- ⚠️ If prompt changes, LLM will get inconsistent instructions

**Better Solution**: Framework could support cache tagging:

```python
# When creating cache, add metadata
self._cache = await self.client.aio.caches.create(
    model=self.model_name.value,
    config=CreateCachedContentConfig(...),
    metadata={"prompt_version": "v2", "purpose": "enrichment"},  # NEW
)

# When looking for cache, match on metadata
for cache in caches:
    if (cache.model.endswith(self.model_name.value) and
        cache.metadata.get("prompt_version") == "v2" and
        cache.metadata.get("purpose") == "enrichment"):
        return cache
```

### Issue 7: Cache Expiration Error Handling

**Problem**: When caches expire on Google's servers (after TTL), subsequent API calls fail with:

```
400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Cache content 5025785736947826688 is expired.', 'status': 'INVALID_ARGUMENT'}}
```

This happens in multi-hour pipelines where:

- Local expiration detection doesn't catch all edge cases
- Google expires cache before local check detects it
- Workers continue using stale cache reference

**Impact**:

- Pipeline failures in long-running jobs
- No automatic recovery from expired caches
- Requires manual intervention to restart

**Our Workaround**: Two-part solution in custom error classifier + strategy:

**Part 1: Error Classifier** (enrich_works.py)

```python
class EnrichmentErrorClassifier(GeminiErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        # CRITICAL: Check cache expiration FIRST, before parent class
        # Parent class may classify ClientError differently, preventing our check
        error_str = str(exception)
        if "Cache content" in error_str and "expired" in error_str.lower():
            return ErrorInfo(
                is_retryable=True,  # Retry - cache will be renewed
                is_rate_limit=False,
                is_timeout=False,
                error_category="cache_expired",
            )

        # ... other custom error checks (JSON, validation, etc.) ...

        # Delegate to parent class LAST
        return super().classify(exception)
```

**Part 2: Proactive Renewal** (enrich.py)

```python
async def execute(self, prompt: str, attempt: int, timeout: float):
    """Execute with proactive cache renewal."""
    # Check if cache needs renewal before processing
    if self._is_cache_expired():
        logger.info("Cache expired or about to expire, renewing before next API call")
        # Force cache renewal by clearing current cache reference
        async with self._cache_lock:
            self._cache = None
            self._cache_created_at = None
        await self.prepare()  # Creates new cache or reuses existing

    # ... continue with API call
```

**Why This Works**:

- **Ordering is critical**: Cache check must come BEFORE parent class delegation
- Error classifier ensures expired cache errors are retried (not permanent failures)
- Proactive renewal (5-min buffer) prevents most expiration errors
- Forced cache reset ensures prepare() creates/finds new cache instead of reusing stale ref
- Combined approach handles both proactive and reactive scenarios

**Gotcha**: If cache expiration check happens after `super().classify()`, the parent class may classify the `ClientError` as non-retryable before your custom check runs. Always check cache expiration FIRST.

**Recommendation for batch-llm**:

**Option 1: Built-in Cache Expiration Handling** (preferred)

Add cache expiration detection to `GeminiCachedStrategy`:

```python
class GeminiCachedStrategy:
    def __init__(self, cache_ttl_seconds: int = 3600, ...):
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_renewal_buffer = 300  # Renew 5min before expiration
        self._cache_created_at: float | None = None

    def _is_cache_expired(self) -> bool:
        """Check if cache has expired or is about to expire."""
        if self._cache is None or self._cache_created_at is None:
            return True
        import time
        cache_age = time.time() - self._cache_created_at
        return cache_age >= (self.cache_ttl_seconds - self.cache_renewal_buffer)

    async def execute(self, prompt: str, attempt: int, timeout: float):
        """Execute with automatic cache renewal."""
        # Renew cache if expired or about to expire
        if self._is_cache_expired():
            async with self._cache_lock:
                self._cache = None
                self._cache_created_at = None
            await self.prepare()

        # ... continue with API call
```

### Option 2: Document Error Classification Pattern

Add to README/documentation:

```markdown
## Handling Cache Expiration in Long Pipelines

For pipelines running longer than your cache TTL (e.g., 3+ hours with 1-hour cache):

1. **Add custom error classifier** to detect and retry cache expiration errors:

   ```python
   class MyCachedErrorClassifier(GeminiErrorClassifier):
       def classify(self, exception: Exception) -> ErrorInfo:
           # IMPORTANT: Check BEFORE calling super().classify()
           # Parent may classify ClientError as non-retryable
           error_str = str(exception)
           if "Cache content" in error_str and "expired" in error_str.lower():
               return ErrorInfo(is_retryable=True, error_category="cache_expired")
           return super().classify(exception)

   processor = ParallelBatchProcessor(
       error_classifier=MyCachedErrorClassifier()
   )
   ```

   **Critical**: Check cache expiration BEFORE delegating to parent class.

1. **Implement proactive renewal** in your strategy's execute() method with a 5-minute buffer.

```

**Benefit of Option 1**: Users get cache expiration handling out-of-the-box, no custom code needed.

**Benefit of Option 2**: Simpler framework, users can customize renewal behavior.

### Issue 8: Per-Work-Item State for Advanced Retry Strategies

**Problem**: Cannot implement sophisticated retry strategies that need per-work-item state across retry attempts.

**Use Case**: Multi-stage validation recovery with network error resilience:

1. **Stage 1**: Full prompt at temperature 0.0
2. **Stage 2**: Partial recovery at temperature 0.0 (fix only failed fields - 81% cheaper)
3. **Stage 3**: Full prompt at temperature 0.25
4. **Stage 4**: Partial recovery at temperature 0.25

**Desired Retry Logic**:

- ValidationError → advance to next stage, increment total_prompts
- Network/timeout error → retry same stage, increment total_prompts
- Fail when total_prompts > threshold (e.g., 5)

**Why This Is Better**:

- ✅ Partial recovery attempts (stages 2, 4) are 81% cheaper than full retries
- ✅ Network errors at stage 4 only retry stage 4 (not stages 1-3)
- ✅ Total prompt limit prevents runaway costs from repeated network errors
- ✅ Progressive temperature only when partial recovery fails

**Current Framework Limitation**: Cannot persist state between retry attempts:

```python
class MyStrategy:
    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Problem: Where to store?
        # - Which stage are we on? (1-4)
        # - How many total prompts have we made?
        # - What was the ValidationError from previous stage?
        # - What was the partial_data for partial recovery?

        # Strategy instance is SHARED across all work items
        # → Can't use instance variables (state collision)

        # execute() only receives (prompt, attempt, timeout)
        # → No work item context or per-item state
```

**Root Causes**:

1. **Strategy shared across work items** - Instance variables cause state collision
2. **execute() signature** - No access to work item context or per-item state
3. **on_error() signature** - No access to work item context to store error details
4. **No state parameter** - Framework doesn't provide state storage/retrieval mechanism

**Recommendation**: Add per-work-item state management to framework.

### Proposed API: State Parameter

Add optional `state` parameter that strategies can use to persist data across retry attempts:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class RetryState:
    """Mutable state that persists across retry attempts for a work item."""
    data: dict[str, Any]  # User-defined state

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

class Strategy(Protocol):
    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,  # NEW: Per-work-item state
    ) -> tuple[TOutput, dict[str, int]]:
        ...

    async def on_error(
        self,
        error: Exception,
        attempt: int,
        state: RetryState | None = None,  # NEW: Access to state
    ) -> None:
        ...
```

**Usage Example**: Multi-stage validation recovery

```python
class MultiStageStrategy:
    async def execute(self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None):
        # Initialize state on first attempt
        if state is None or attempt == 1:
            state = RetryState(data={"stage": 1, "total_prompts": 0})

        stage = state.get("stage", 1)
        total_prompts = state.get("total_prompts", 0)

        # Enforce total prompt limit
        if total_prompts >= 5:
            raise ValueError("Exceeded maximum prompts (5)")

        try:
            if stage in [1, 3]:
                # Full prompt stages
                temp = 0.0 if stage == 1 else 0.25
                result = await self._call_gemini(prompt, temp)
                state.set("total_prompts", total_prompts + 1)
                return result

            elif stage in [2, 4]:
                # Partial recovery stages
                temp = 0.0 if stage == 2 else 0.25
                last_error = state.get("last_validation_error")
                partial_data = state.get("partial_data")
                result = await self._partial_recovery(last_error, partial_data, temp)
                state.set("total_prompts", total_prompts + 1)
                return result

        except ValidationError as e:
            # Validation error: advance to next stage
            state.set("stage", stage + 1)
            state.set("last_validation_error", e)
            state.set("partial_data", e.partial_data)
            state.set("total_prompts", total_prompts + 1)
            raise  # Re-raise for framework to retry

        except (NetworkError, TimeoutError):
            # Network error: retry same stage
            state.set("total_prompts", total_prompts + 1)
            raise  # Re-raise for framework to retry

    async def on_error(self, error: Exception, attempt: int, state: RetryState | None = None):
        # Can inspect/modify state if needed
        if state:
            logger.info(f"Stage {state.get('stage')}, Total prompts: {state.get('total_prompts')}")
```

**Framework Implementation Notes**:

1. **State Storage**: Framework maintains `Dict[work_item_id, RetryState]`
2. **State Lifecycle**:
   - Created on first attempt
   - Passed to `execute()` and `on_error()` on every attempt
   - Cleared when work item succeeds or exhausts retries
3. **Thread Safety**: State access is serialized per work item (no concurrent retries for same item)
4. **Backward Compatibility**: `state` parameter is optional, existing strategies work unchanged

**Benefits**:

- ✅ Enables sophisticated multi-stage retry strategies
- ✅ Strategies can track custom metrics (stage, total_prompts, error history)
- ✅ No state collision (each work item has isolated state)
- ✅ Clean API (no global state or threading issues)
- ✅ Backward compatible (optional parameter)

**Alternative Simpler Approach**: Pass work item context to execute()

```python
async def execute(
    self,
    prompt: str,
    attempt: int,
    timeout: float,
    context: Any = None,  # NEW: Work item context (user-defined)
) -> tuple[TOutput, dict[str, int]]:
    ...
```

This allows strategies to use work item ID as key for instance-level state dict, but requires manual state management and cleanup.

## Summary

The shared strategy pattern is valuable for cost optimization but currently requires:

1. Manual idempotency guards in every strategy
2. Custom strategies for newer API versions
3. Custom strategies to access response metadata
4. Manual tracking and aggregation of cached tokens
5. Careful lifecycle management (cleanup vs. reuse)
6. Manual cache matching logic
7. Custom error classifiers and renewal logic for cache expiration
8. Cannot implement advanced multi-stage retry strategies without per-work-item state

**Quick Wins**:

- Document shared strategy pattern and best practices
- Add `PrepareOnceMixin` for idempotency
- Update `GeminiCachedStrategy` for google-genai v1.46+
- Add `include_metadata` option to expose safety ratings
- Add `total_cached_tokens` to `BatchResult`
- Clarify cleanup() hook purpose in documentation
- Add built-in cache expiration handling or document the pattern

**Longer Term**:

- Consider explicit strategy lifecycle management in framework
- Support cache metadata/tagging for better matching
- Built-in cache renewal with configurable TTL and renewal buffer
- Add per-work-item state management for advanced retry strategies (see Issue #8)

## Our Implementation

For reference, our working implementation is in:

- `backend/ingest/openlibrary/enrichment/enrich.py`: `GeminiEnrichmentCachedStrategy`
- `backend/ingest/openlibrary/enrichment/enrich_works.py`: `EnrichmentErrorClassifier` + usage pattern

Key features:

- Async lock for idempotency
- google-genai v1.46 compatibility
- Safety rating extraction
- Cached token tracking and reporting
- Automatic cache renewal for long-running jobs (5-minute buffer)
- Cache expiration error detection and retry (in error classifier)
- Proactive cache renewal before execute() calls
- Intelligent cache reuse (checks existing caches before creating new ones)
- Shared across all work items (tested with 2-10 workers)
