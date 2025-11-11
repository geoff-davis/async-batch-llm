# Migration Guide: v0.2.x → v0.3.0

## Overview

Version 0.3.0 adds advanced retry strategies, safety ratings access, and cache tagging based on production
feedback. **All changes are backward compatible** - existing code will continue to work without modification.

**Key improvements:**

- Per-work-item retry state for sophisticated multi-stage strategies
- Access to Gemini safety ratings and response metadata
- Cache tagging for precise cache matching
- Enhanced error classifier extensibility

---

## Breaking Changes

**None.** Version 0.3.0 is fully backward compatible with v0.2.x.

All new features are opt-in:

- Retry state is optional (strategies without `state` parameter work unchanged)
- Metadata access requires `include_metadata=True`
- Cache tags are optional
- Existing code requires no changes

---

## New Features

### 1. Per-Work-Item Retry State

Enable sophisticated multi-stage retry strategies that maintain state across retry attempts.

**Basic Usage:**

```python
from batch_llm import LLMCallStrategy, RetryState

class MultiStageStrategy(LLMCallStrategy[Output]):
    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,  # NEW: Optional state parameter
    ):
        # Access state across retries
        if state:
            stage = state.get("stage", 1)
            total_attempts = state.get("total_attempts", 0)

            # Store state for next retry
            state.set("total_attempts", total_attempts + 1)

        # ... rest of execute logic ...
```

#### Advanced Example: Multi-Stage Validation Recovery

Implement cost-optimized retry strategy with different stages:

```python
from batch_llm import LLMCallStrategy, RetryState
from pydantic import ValidationError

class SmartRetryStrategy(LLMCallStrategy[BookMetadata]):
    """Multi-stage retry with partial recovery for 81% cost savings."""

    STAGES = {
        1: {"type": "full", "temperature": 0.0},
        2: {"type": "partial", "temperature": 0.0},  # 81% cheaper
        3: {"type": "full", "temperature": 0.25},
        4: {"type": "partial", "temperature": 0.25},  # 81% cheaper
    }

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ):
        # Initialize state on first attempt
        if state is None:
            state = RetryState()

        if attempt == 1:
            state.set("stage", 1)
            state.set("total_prompts", 0)

        stage = state.get("stage", 1)
        total_prompts = state.get("total_prompts", 0)

        # Enforce total prompt limit (prevent runaway costs)
        if total_prompts >= 5:
            raise ValueError(f"Exceeded max prompts: {total_prompts}/5")

        stage_config = self.STAGES.get(stage, self.STAGES[4])

        try:
            if stage_config["type"] == "full":
                # Full prompt with all instructions
                result = await self._call_llm(prompt, stage_config["temperature"])
            else:
                # Partial recovery - only fix failed fields (81% cheaper)
                last_error = state.get("last_validation_error")
                partial_data = state.get("partial_data")
                result = await self._partial_recovery(
                    last_error, partial_data, stage_config["temperature"]
                )

            state.set("total_prompts", total_prompts + 1)
            return result, tokens

        except ValidationError as e:
            # Validation failed - advance to next stage
            state.set("stage", min(stage + 1, 4))
            state.set("last_validation_error", str(e))
            state.set("partial_data", e.partial_data if hasattr(e, 'partial_data') else None)
            state.set("total_prompts", total_prompts + 1)
            raise  # Framework will retry with new stage

        except (ConnectionError, TimeoutError) as e:
            # Network error - retry same stage (don't advance)
            state.set("total_prompts", total_prompts + 1)
            raise  # Framework will retry same stage

    async def on_error(
        self,
        exception: Exception,
        attempt: int,
        state: RetryState | None = None,
    ):
        """Log state for debugging."""
        if state:
            logger.info(
                f"Retry {attempt}: Stage {state.get('stage')}, "
                f"Total prompts: {state.get('total_prompts')}"
            )
```

**Benefits:**

- ✅ Partial recovery stages are **81% cheaper** than full retries
- ✅ Network errors don't waste progress (retry same stage)
- ✅ Total prompt limit prevents runaway costs
- ✅ Progressive temperature escalation only when needed
- ✅ State is isolated per work item (no collisions)

**RetryState API:**

```python
from batch_llm import RetryState

state = RetryState()

# Store values
state.set("key", value)

# Retrieve values with optional default
value = state.get("key", default=None)

# Clear all state
state.clear()

# Serialize for debugging
state_dict = state.to_dict()
state = RetryState.from_dict(state_dict)
```

**See Also:**

- `examples/example_multi_stage_retry.py` - Complete multi-stage example
- `examples/example_partial_recovery.py` - Partial recovery pattern

---

### 2. Gemini Safety Ratings and Metadata

Access safety ratings, finish reasons, and raw response metadata from Gemini.

**Before (v0.2.x):**

```python
strategy = GeminiCachedStrategy(
    response_parser=lambda r: parse_json(r.text),
)

# No access to safety ratings or finish_reason
```

**After (v0.3.0):**

```python
from batch_llm import GeminiCachedStrategy, GeminiResponse

strategy = GeminiCachedStrategy(
    response_parser=lambda r: parse_json(r.text),
    include_metadata=True,  # NEW: Enable metadata access
)

# In post-processor or after processing:
result = work_item_result.output  # GeminiResponse object

# Access parsed output
metadata = result.output  # Your parsed BookMetadata

# Access safety ratings
if result.safety_ratings:
    hate_speech = result.safety_ratings.get("HARM_CATEGORY_HATE_SPEECH")
    harassment = result.safety_ratings.get("HARM_CATEGORY_HARASSMENT")

    if hate_speech == "HIGH" or harassment == "HIGH":
        logger.warning(f"High-risk content detected: {work_item.item_id}")
        # Filter or flag content

# Access finish reason
if result.finish_reason == "SAFETY":
    logger.warning("Response blocked by safety filters")

# Access full response object
raw = result.raw_response
```

**GeminiResponse API:**

```python
@dataclass
class GeminiResponse(Generic[TOutput]):
    output: TOutput                          # Your parsed output
    safety_ratings: dict[str, str] | None    # Safety ratings by category
    finish_reason: str | None                # Why generation stopped
    token_usage: dict[str, int]              # Token counts
    raw_response: Any                        # Full Gemini response object
```

**Safety Rating Categories:**

Common Gemini safety categories:

- `HARM_CATEGORY_HATE_SPEECH`
- `HARM_CATEGORY_HARASSMENT`
- `HARM_CATEGORY_SEXUALLY_EXPLICIT`
- `HARM_CATEGORY_DANGEROUS_CONTENT`

Probability levels: `"NEGLIGIBLE"`, `"LOW"`, `"MEDIUM"`, `"HIGH"`

**Content Filtering Example:**

```python
async def filter_unsafe_content(result: WorkItemResult):
    """Filter content based on safety ratings."""
    if not isinstance(result.output, GeminiResponse):
        return

    safety = result.output.safety_ratings or {}

    # Check for high-risk content
    high_risk_categories = [
        cat for cat, prob in safety.items()
        if prob in ["HIGH", "MEDIUM"]
    ]

    if high_risk_categories:
        logger.warning(
            f"Flagged {result.item_id}: {high_risk_categories}"
        )
        # Mark for manual review
        await db.flag_for_review(result.item_id, high_risk_categories)

processor = ParallelBatchProcessor(
    config=config,
    post_processor=filter_unsafe_content,
)
```

**Backward Compatibility:**

```python
# include_metadata=False (default) - returns parsed output directly
strategy = GeminiCachedStrategy(response_parser=parser)
result.output  # Your parsed type (e.g., BookMetadata)

# include_metadata=True - returns GeminiResponse wrapper
strategy = GeminiCachedStrategy(response_parser=parser, include_metadata=True)
result.output  # GeminiResponse[BookMetadata]
result.output.output  # Your parsed type
```

**See Also:**

- `examples/example_gemini_safety_ratings.py` - Complete safety ratings example
- `docs/GEMINI_INTEGRATION.md` - Gemini-specific features

---

### 3. Cache Tagging for Precise Matching

Tag caches with metadata to prevent accidental reuse when prompts change.

**Problem (v0.2.x):**

```python
# Day 1: Create cache with prompt v1
strategy = GeminiCachedStrategy(
    cached_content=[{"role": "user", "parts": [{"text": "Prompt version 1"}]}],
)

# Day 2: Change prompt but reuse old cache (BUG!)
strategy = GeminiCachedStrategy(
    cached_content=[{"role": "user", "parts": [{"text": "Prompt version 2"}]}],
)
# May reuse v1 cache → LLM gets inconsistent instructions
```

**Solution (v0.3.0):**

```python
# Day 1: Create cache with tags
strategy = GeminiCachedStrategy(
    cached_content=[{"role": "user", "parts": [{"text": "Prompt version 1"}]}],
    cache_tags={
        "prompt_version": "v1",
        "purpose": "enrichment",
        "dataset": "openlibrary",
    },
)

# Day 2: Different tags → creates new cache
strategy = GeminiCachedStrategy(
    cached_content=[{"role": "user", "parts": [{"text": "Prompt version 2"}]}],
    cache_tags={
        "prompt_version": "v2",  # Different tag
        "purpose": "enrichment",
        "dataset": "openlibrary",
    },
)
# Won't reuse v1 cache - tags don't match
```

**Tag Matching Logic:**

```python
# Cache is reused only if:
# 1. Model name matches AND
# 2. All tags match (if tags provided)

# Example: This will NOT reuse cache from above
strategy = GeminiCachedStrategy(
    cache_tags={
        "prompt_version": "v1",  # Matches
        "purpose": "enrichment",  # Matches
        "dataset": "books",       # Different! Won't match
    },
)
```

**Best Practices:**

```python
# Version your prompts
cache_tags = {
    "prompt_version": "v3",
    "schema_version": "2024-01",
}

# Separate by purpose/dataset
cache_tags = {
    "purpose": "enrichment",  # vs "classification", "summarization"
    "dataset": "openlibrary",  # vs "gutenberg", "archive"
}

# Include model parameters that affect output
cache_tags = {
    "temperature": "0.0",
    "output_format": "json",
}
```

**Graceful Degradation:**

If google-genai doesn't support cache metadata:

- Framework logs warning: `"Cache tags requested but not supported by API"`
- Falls back to model-name-only matching
- No errors, continues working

**See Also:**

- `examples/example_gemini_cache_tags.py` - Complete cache tagging example
- `docs/GEMINI_INTEGRATION.md` - Cache management strategies

---

## Optional Enhancements

### PrepareOnceMixin (New)

Helper for custom strategies that need idempotency (though v0.2.0's framework tracking makes this less critical).

**Usage:**

```python
from batch_llm.mixins import PrepareOnceMixin
from batch_llm import LLMCallStrategy

class MyCustomStrategy(PrepareOnceMixin, LLMCallStrategy[Output]):
    async def _prepare_once(self) -> None:
        """Expensive initialization - called exactly once."""
        self.connection = await create_expensive_connection()
        self.cache = await initialize_cache()
        logger.info("Strategy prepared once")

    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Use self.connection and self.cache
        response = await self.connection.generate(prompt)
        return response, tokens
```

**When to Use:**

- ✅ Extra safety for shared strategies
- ✅ Clear intent documentation
- ✅ Use outside ParallelBatchProcessor
- ❌ Not needed with v0.2.0+ framework tracking (prepare() called once by framework)

---

## Migration Checklist

### From v0.2.x to v0.3.0

**No code changes required!** But you may want to adopt new features:

#### Consider Retry State If

- [ ] You have multi-stage retry logic
- [ ] You want partial recovery to reduce costs
- [ ] You need different handling for ValidationError vs NetworkError
- [ ] You want to enforce total prompt limits

**Action:** Review `examples/example_multi_stage_retry.py` and implement if beneficial.

#### Consider Safety Ratings If

- [ ] You process user-generated content
- [ ] You need content moderation
- [ ] You want to filter harmful outputs
- [ ] You need audit trails for safety

**Action:** Set `include_metadata=True` and add post-processor for filtering.

#### Consider Cache Tagging If

- [ ] You frequently change prompts
- [ ] You run multiple different pipelines
- [ ] You've had cache reuse bugs
- [ ] You want explicit cache versioning

**Action:** Add `cache_tags` to your `GeminiCachedStrategy` initialization.

---

## Examples

All new features have complete working examples:

### Retry State Examples

- `examples/example_multi_stage_retry.py` - Multi-stage validation recovery
- `examples/example_partial_recovery.py` - Partial recovery pattern (81% cost savings)
- `examples/example_retry_state_basics.py` - Basic retry state usage

### Safety Ratings Examples

- `examples/example_gemini_safety_ratings.py` - Content filtering with safety ratings
- `examples/example_safety_audit.py` - Audit logging for compliance

### Cache Tagging Examples

- `examples/example_gemini_cache_tags.py` - Cache versioning and tagging
- `examples/example_multi_pipeline_caching.py` - Multiple pipelines with separate caches

---

## Configuration Reference

### New Parameters

**GeminiCachedStrategy:**

```python
strategy = GeminiCachedStrategy(
    # ... existing params ...

    # NEW in v0.3.0:
    include_metadata: bool = False,         # Enable safety ratings access
    cache_tags: dict[str, str] | None = None,  # Tags for cache matching
)
```

**LLMCallStrategy.execute():**

```python
async def execute(
    prompt: str,
    attempt: int,
    timeout: float,
    state: RetryState | None = None,  # NEW: Optional retry state
) -> tuple[TOutput, dict[str, int]]:
    ...
```

**LLMCallStrategy.on_error():**

```python
async def on_error(
    exception: Exception,
    attempt: int,
    state: RetryState | None = None,  # NEW: Optional retry state
) -> None:
    ...
```

---

## Troubleshooting

### Retry State Not Persisting

**Symptom:** State resets between retries

**Solution:** Ensure you're using the `state` parameter correctly:

```python
# ❌ WRONG - Creating new state each time
async def execute(self, prompt, attempt, timeout, state=None):
    state = RetryState()  # Don't create new state!
    state.set("count", state.get("count", 0) + 1)

# ✅ CORRECT - Use provided state
async def execute(self, prompt, attempt, timeout, state=None):
    if state is None:
        state = RetryState()  # Only if framework didn't provide one
    state.set("count", state.get("count", 0) + 1)
```

### Safety Ratings Always None

**Symptom:** `result.output.safety_ratings` is always `None`

**Solutions:**

1. Ensure `include_metadata=True`:

   ```python
   strategy = GeminiCachedStrategy(
       response_parser=parser,
       include_metadata=True,  # Required!
   )
   ```

2. Check that response is actually `GeminiResponse`:

   ```python
   if isinstance(result.output, GeminiResponse):
       ratings = result.output.safety_ratings
   ```

3. Some Gemini models/prompts may not include safety ratings

### Cache Tags Not Matching

**Symptom:** New cache created instead of reusing tagged cache

**Solutions:**

1. **Verify tags are identical:**

   ```python
   # All tags must match exactly
   cache_tags = {"version": "v1", "dataset": "books"}
   # Won't match: {"version": "v1"}  (missing dataset)
   # Won't match: {"version": "v1", "dataset": "Books"}  (case-sensitive)
   ```

2. **Check for API support:**

   ```python
   # Look for warning in logs:
   # "Cache tags requested but not supported by API"
   ```

3. **Use model name matching as fallback:**

   ```python
   # If tags not supported, ensure model names match exactly
   model = "gemini-2.5-flash"  # Use consistent model names
   ```

---

## Best Practices

### Retry State

**DO:**

- ✅ Use for multi-stage strategies
- ✅ Enforce total prompt limits to prevent runaway costs
- ✅ Log state before clearing for debugging
- ✅ Store minimal state (only what's needed)

**DON'T:**

- ❌ Store large objects in state (keep it lightweight)
- ❌ Assume state is always provided (check for None)
- ❌ Modify state in `on_error()` unless needed

### Safety Ratings

**DO:**

- ✅ Use for user-generated content
- ✅ Log safety events for audit trails
- ✅ Have fallback behavior if ratings unavailable
- ✅ Consider cultural/regional differences in thresholds

**DON'T:**

- ❌ Rely solely on safety ratings (defense in depth)
- ❌ Block all content with MEDIUM ratings (may be overly strict)
- ❌ Assume ratings are always provided

### Cache Tagging

**DO:**

- ✅ Version your prompts explicitly
- ✅ Use semantic tags (purpose, dataset, version)
- ✅ Handle graceful degradation (API may not support tags)
- ✅ Document your tagging schema

**DON'T:**

- ❌ Use too many tags (harder to match)
- ❌ Include dynamic values (timestamps, UUIDs)
- ❌ Rely on tags for security (they're for matching only)

---

## Performance Impact

### Retry State

- **Memory:** ~1KB per work item (negligible)
- **CPU:** Minimal overhead for dict operations
- **Concurrency:** No impact (state is per work item)

### Safety Ratings

- **API:** No additional API calls (included in response)
- **Memory:** ~1KB per response (negligible)
- **CPU:** Minimal overhead for extraction

### Cache Tagging

- **API:** No additional API calls
- **Memory:** Negligible (tags stored in cache metadata)
- **CPU:** Minimal overhead for tag comparison

**Overall:** v0.3.0 features have **<1% performance impact**.

---

## Support

### Documentation

- [README.md](../README.md) - Updated with v0.3.0 features
- [docs/API.md](API.md) - Full API reference
- [docs/GEMINI_INTEGRATION.md](GEMINI_INTEGRATION.md) - Gemini-specific guide
- [docs/IMPLEMENTATION_PLAN_V0_3.md](IMPLEMENTATION_PLAN_V0_3.md) - Technical details

### Examples

All features have complete working examples in `examples/` directory.

### Getting Help

- GitHub Issues: <https://github.com/geoff-davis/batch-llm/issues>
- Check logs for debug information
- Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

---

## Version History

- **v0.3.0** (Current)
  - Per-work-item retry state for advanced strategies
  - Gemini safety ratings and metadata access
  - Cache tagging for precise matching
  - PrepareOnceMixin helper

- **v0.2.0**
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
