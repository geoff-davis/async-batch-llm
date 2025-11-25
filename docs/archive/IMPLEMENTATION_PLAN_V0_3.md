# Implementation Plan: v0.3.0

## Overview

Version 0.3.0 focuses on remaining production feedback items from BATCH_LLM_FEEDBACK.md that were deferred
from v0.2.0, plus additional enhancements based on user needs.

**Target Date:** TBD
**Status:** Planning

---

## Items from BATCH_LLM_FEEDBACK.md

### ✅ Already Implemented in v0.2.0

1. **Issue #1: Multiple prepare() calls** - ✅ Fixed with framework-level strategy tracking
2. **Issue #2: API version incompatibility** - ✅ Fixed with auto-detection for google-genai v1.46+
3. **Issue #4: Cached token tracking** - ✅ Added `total_cached_tokens`, `cache_hit_rate()`, `effective_input_tokens()`
4. **Issue #5: Cache lifecycle unclear** - ✅ Clarified with cleanup() preserving caches, added delete_cache()
5. **Issue #7: Cache expiration handling** - ✅ Added auto-renewal with configurable buffer

### 🔜 Deferred to v0.3.0

#### Issue #3: Missing Safety Ratings in GeminiCachedStrategy

**Priority:** Medium
**Effort:** Medium
**Breaking:** No (opt-in feature)

**Problem:**

- Users cannot access Gemini safety ratings (harassment, hate speech, etc.)
- Current `response_parser` only gets parsed response, not raw response object
- Critical for content filtering use cases

**Proposed Solution:**

Add optional `include_metadata` parameter and `GeminiResponse` wrapper:

```python
from dataclasses import dataclass
from typing import Generic, TypeVar, Any

TOutput = TypeVar('TOutput')

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
        # ... existing params ...
        include_metadata: bool = False,  # NEW: Opt-in for metadata
    ):
        self.include_metadata = include_metadata

    async def execute(self, ...) -> tuple[TOutput | GeminiResponse[TOutput], dict[str, int]]:
        response = await self.client.aio.models.generate_content(...)

        output = self.response_parser(response)

        if self.include_metadata:
            safety_ratings = self._extract_safety_ratings(response)
            return GeminiResponse(
                output=output,
                safety_ratings=safety_ratings,
                finish_reason=response.candidates[0].finish_reason,
                token_usage=token_usage,
                raw_response=response,
            ), token_usage

        return output, token_usage

    def _extract_safety_ratings(self, response) -> dict[str, str]:
        """Extract safety ratings from Gemini response."""
        ratings = {}
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'safety_ratings'):
                for rating in candidate.safety_ratings:
                    ratings[rating.category] = rating.probability
        return ratings
```

**Usage:**

```python
strategy = GeminiCachedStrategy(
    response_parser=parse_metadata,
    include_metadata=True,  # Enable metadata access
)

# In post-processor:
if isinstance(result.output, GeminiResponse):
    parsed_data = result.output.output
    safety = result.output.safety_ratings
    if safety.get("HARM_CATEGORY_HATE_SPEECH") == "HIGH":
        # Filter content
        pass
```

**Implementation Steps:**

1. Add `GeminiResponse` dataclass to `llm_strategies.py`
2. Add `include_metadata` parameter to `GeminiCachedStrategy.__init__`
3. Add `_extract_safety_ratings()` helper method
4. Update `execute()` to return `GeminiResponse` when enabled
5. Add tests for metadata extraction
6. Update documentation with safety ratings example

**Test Coverage:**

- Test metadata extraction with mocked Gemini response
- Test safety ratings parsing
- Test finish_reason extraction
- Test backward compatibility (include_metadata=False)
- Test with real Gemini API (integration test)

---

#### Issue #6: Cache Matching Precision

**Priority:** Low
**Effort:** Medium
**Breaking:** No (enhancement)

**Problem:**

- Current cache matching only checks model name (`.endswith()`)
- Will reuse cache even if prompt/content changed
- Can lead to inconsistent LLM instructions

**Current Behavior:**

```python
# Finds first cache for model, ignores prompt content
for cache in caches:
    if cache.model.endswith(self.model_name):
        return cache  # May have different prompt!
```

**Proposed Solution:**

Add cache tagging/metadata support:

```python
class GeminiCachedStrategy:
    def __init__(
        self,
        # ... existing params ...
        cache_tags: dict[str, str] | None = None,  # NEW: Cache tags for matching
    ):
        self.cache_tags = cache_tags or {}

    async def _create_cache(self):
        """Create new cache with tags."""
        # Check if API supports metadata
        try:
            config = CreateCachedContentConfig(
                contents=self.cached_content,
                ttl=f"{self.cache_ttl_seconds}s",
                metadata=self.cache_tags,  # Add tags
            )
        except TypeError:
            # Fallback for older API without metadata support
            config = CreateCachedContentConfig(
                contents=self.cached_content,
                ttl=f"{self.cache_ttl_seconds}s",
            )

        return await self.client.aio.caches.create(
            model=self.model,
            config=config,
        )

    async def _find_existing_cache(self):
        """Find cache matching model AND tags."""
        caches = await self.client.aio.caches.list()

        for cache in caches:
            # Match model name
            if not cache.model.endswith(self.model):
                continue

            # Match tags (if provided)
            if self.cache_tags:
                cache_metadata = getattr(cache, 'metadata', {})
                if not all(
                    cache_metadata.get(k) == v
                    for k, v in self.cache_tags.items()
                ):
                    continue  # Tags don't match

            return cache

        return None
```

**Usage:**

```python
strategy = GeminiCachedStrategy(
    model="gemini-2.5-flash",
    cache_tags={
        "prompt_version": "v2",
        "purpose": "enrichment",
        "dataset": "openlibrary",
    },
)
```

**Implementation Steps:**

1. Add `cache_tags` parameter to `GeminiCachedStrategy.__init__`
2. Update `_create_cache()` to include metadata (with fallback)
3. Update `_find_existing_cache()` to match on tags
4. Add tests for tag matching
5. Document cache tagging pattern
6. Add migration note about tag matching

**Considerations:**

- Check if google-genai API supports cache metadata (may be newer feature)
- Provide graceful fallback if metadata not supported
- Document that cache matching is best-effort

---

#### Issue #8: Per-Work-Item State for Advanced Retry Strategies

**Priority:** High
**Effort:** High
**Breaking:** No (additive API change)

**Problem:**

- Cannot implement sophisticated multi-stage retry strategies
- No way to persist state between retry attempts for same work item
- Shared strategies mean instance variables cause state collision

**Use Case:**

Multi-stage validation recovery:

1. Stage 1: Full prompt at temp 0.0
2. Stage 2: Partial recovery at temp 0.0 (fix only failed fields - 81% cheaper)
3. Stage 3: Full prompt at temp 0.25
4. Stage 4: Partial recovery at temp 0.25

Different errors need different handling:

- ValidationError → advance to next stage
- Network error → retry same stage
- Limit total prompts to prevent runaway costs

**Proposed Solution:**

Add `RetryState` parameter to strategy methods:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RetryState:
    """Mutable state that persists across retry attempts for a work item."""
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self.data[key] = value

    def clear(self) -> None:
        """Clear all state."""
        self.data.clear()

# Update LLMCallStrategy protocol
class LLMCallStrategy(Protocol[TOutput]):
    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,  # NEW: Optional state
    ) -> tuple[TOutput, dict[str, int]]:
        ...

    async def on_error(
        self,
        exception: Exception,
        attempt: int,
        state: RetryState | None = None,  # NEW: Optional state
    ) -> None:
        ...
```

**Framework Implementation:**

```python
class ParallelBatchProcessor:
    def __init__(self, ...):
        # ... existing init ...
        self._retry_states: dict[str, RetryState] = {}  # item_id -> state
        self._retry_states_lock = asyncio.Lock()

    async def _get_retry_state(self, item_id: str) -> RetryState:
        """Get or create retry state for work item."""
        async with self._retry_states_lock:
            if item_id not in self._retry_states:
                self._retry_states[item_id] = RetryState()
            return self._retry_states[item_id]

    async def _clear_retry_state(self, item_id: str) -> None:
        """Clear retry state after completion."""
        async with self._retry_states_lock:
            self._retry_states.pop(item_id, None)

    async def _process_item_with_retries(self, work_item):
        # Get state for this work item
        state = await self._get_retry_state(work_item.item_id)

        try:
            # ... retry loop ...
            for attempt in range(1, max_attempts + 1):
                try:
                    # Pass state to execute()
                    output, token_usage = await strategy.execute(
                        prompt=work_item.prompt,
                        attempt=attempt,
                        timeout=timeout,
                        state=state,  # NEW
                    )
                    # Success - clear state
                    await self._clear_retry_state(work_item.item_id)
                    return success_result

                except Exception as e:
                    # Pass state to on_error()
                    await strategy.on_error(e, attempt, state=state)  # NEW
                    # ... retry logic ...

        finally:
            # Always clear state when done (success or failure)
            await self._clear_retry_state(work_item.item_id)
```

**Usage Example:**

```python
class MultiStageStrategy(LLMCallStrategy[Output]):
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

        # Enforce total prompt limit
        if total_prompts >= 5:
            raise ValueError(f"Exceeded max prompts ({total_prompts}/5)")

        try:
            if stage in [1, 3]:
                # Full prompt stages
                temp = 0.0 if stage == 1 else 0.25
                result = await self._call_llm(prompt, temp)
                state.set("total_prompts", total_prompts + 1)
                return result, tokens

            elif stage in [2, 4]:
                # Partial recovery stages (81% cheaper)
                temp = 0.0 if stage == 2 else 0.25
                last_error = state.get("last_validation_error")
                partial_data = state.get("partial_data")
                result = await self._partial_recovery(last_error, partial_data, temp)
                state.set("total_prompts", total_prompts + 1)
                return result, tokens

        except ValidationError as e:
            # Validation error: advance to next stage
            state.set("stage", stage + 1)
            state.set("last_validation_error", str(e))
            state.set("partial_data", getattr(e, 'partial_data', None))
            state.set("total_prompts", total_prompts + 1)
            raise  # Framework will retry with new stage

        except (ConnectionError, TimeoutError):
            # Network error: retry same stage
            state.set("total_prompts", total_prompts + 1)
            raise  # Framework will retry same stage

    async def on_error(
        self,
        exception: Exception,
        attempt: int,
        state: RetryState | None = None,
    ):
        if state:
            logger.info(
                f"Attempt {attempt}, Stage {state.get('stage')}, "
                f"Total prompts: {state.get('total_prompts')}"
            )
```

**Implementation Steps:**

1. Create `RetryState` class in `batch_llm/core/retry_state.py`
2. Update `LLMCallStrategy` protocol with optional `state` parameter
3. Add `_retry_states` dict to `ParallelBatchProcessor`
4. Add `_get_retry_state()` and `_clear_retry_state()` methods
5. Update `_process_item_with_retries()` to pass state
6. Update all built-in strategies to accept (but ignore) state parameter
7. Add comprehensive tests for state management
8. Add example multi-stage strategy to examples/
9. Document retry state pattern in README
10. Add migration guide section

**Test Coverage:**

- Test state creation and retrieval
- Test state persistence across retries
- Test state clearing on success
- Test state clearing on final failure
- Test concurrent work items have isolated state
- Test multi-stage strategy example
- Test backward compatibility (strategies without state parameter)

**Backward Compatibility:**

- `state` parameter is optional with default `None`
- Existing strategies work unchanged (they don't use state)
- No breaking changes to existing APIs

---

## Additional Enhancements

### Enhancement #1: PrepareOnceMixin for User Strategies

**Priority:** Low
**Effort:** Low
**Breaking:** No

**Rationale:**
While v0.2.0 fixed framework-level prepare() tracking, users writing custom strategies still need to implement
idempotency manually. A mixin makes this easier.

**Proposed Solution:**

```python
# batch_llm/mixins/prepare_once.py
import asyncio
from typing import Any

class PrepareOnceMixin:
    """Mixin to ensure prepare() is called only once, even with concurrent calls."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._prepare_once_lock = asyncio.Lock()
        self._prepare_once_done = False

    async def prepare(self) -> None:
        """Call _prepare_once() exactly once, with thread-safe initialization."""
        async with self._prepare_once_lock:
            if self._prepare_once_done:
                return
            await self._prepare_once()
            self._prepare_once_done = True

    async def _prepare_once(self) -> None:
        """Override this method with your expensive initialization.

        This will be called exactly once, even with concurrent prepare() calls.
        """
        raise NotImplementedError("Subclasses must implement _prepare_once()")
```

**Usage:**

```python
from async_batch_llm.mixins import PrepareOnceMixin
from async_batch_llm.llm_strategies import LLMCallStrategy

class MyCustomStrategy(PrepareOnceMixin, LLMCallStrategy[Output]):
    async def _prepare_once(self) -> None:
        """Expensive initialization - called exactly once."""
        self.connection = await create_connection()
        self.cache = await create_cache()

    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Use self.connection and self.cache
        pass
```

**Note:** With v0.2.0's framework-level tracking, this mixin is less critical but still useful for:

- Documentation/clarity of intent
- Extra safety layer
- Use outside of ParallelBatchProcessor

---

### Enhancement #2: Improved Error Classifier Extensibility

**Priority:** Low
**Effort:** Low
**Breaking:** No

**Problem:**
Current pattern for extending error classifiers requires careful ordering:

```python
class MyClassifier(GeminiErrorClassifier):
    def classify(self, exception):
        # MUST check custom cases BEFORE super().classify()
        if "cache expired" in str(exception):
            return ErrorInfo(...)
        return super().classify(exception)  # AFTER custom checks
```

**Proposed Solution:**

Add hooks for pre/post classification:

```python
class ExtensibleErrorClassifier:
    """Base classifier with extension hooks."""

    def classify(self, exception: Exception) -> ErrorInfo:
        """Final classification with extension hooks."""
        # Pre-hook: Check custom patterns first
        pre_result = self.classify_custom(exception)
        if pre_result is not None:
            return pre_result

        # Default classification
        result = self.classify_default(exception)

        # Post-hook: Modify default result if needed
        return self.post_classify(exception, result)

    def classify_custom(self, exception: Exception) -> ErrorInfo | None:
        """Override to add custom classification logic.

        Return ErrorInfo to override default classification.
        Return None to fall through to default classification.
        """
        return None

    def classify_default(self, exception: Exception) -> ErrorInfo:
        """Default classification logic. Override in subclasses."""
        raise NotImplementedError()

    def post_classify(self, exception: Exception, info: ErrorInfo) -> ErrorInfo:
        """Override to modify classification after default logic.

        Useful for adjusting retry behavior based on context.
        """
        return info
```

**Usage:**

```python
class MyCachedClassifier(GeminiErrorClassifier):
    def classify_custom(self, exception):
        """Custom checks run FIRST."""
        if "cache expired" in str(exception):
            return ErrorInfo(is_retryable=True, error_category="cache_expired")
        return None  # Fall through to default Gemini classification

    def post_classify(self, exception, info):
        """Modify default results if needed."""
        # Example: Make all validation errors use higher temp on retry
        if info.error_category == "validation_error":
            info.suggested_temperature = 0.5
        return info
```

**Implementation:**

- Add to `strategies/errors.py`
- Update `GeminiErrorClassifier` and `DefaultErrorClassifier` to use pattern
- Add documentation example

---

## Release Checklist

### Phase 1: Issue #8 - Retry State (High Priority)

- [ ] Implement `RetryState` class
- [ ] Update `LLMCallStrategy` protocol
- [ ] Update `ParallelBatchProcessor` with state management
- [ ] Update built-in strategies for compatibility
- [ ] Add comprehensive tests (15+ tests)
- [ ] Add multi-stage example to `examples/`
- [ ] Update README with retry state documentation
- [ ] Update API.md with RetryState API
- [ ] Add to MIGRATION_V0_3.md

### Phase 2: Issue #3 - Safety Ratings (Medium Priority)

- [ ] Implement `GeminiResponse` dataclass
- [ ] Add `include_metadata` parameter
- [ ] Add `_extract_safety_ratings()` method
- [ ] Update `execute()` to return wrapped response
- [ ] Add tests for metadata extraction (8+ tests)
- [ ] Add safety ratings example to examples/
- [ ] Update README with safety ratings section
- [ ] Update GEMINI_INTEGRATION.md with metadata docs
- [ ] Add to MIGRATION_V0_3.md

### Phase 3: Issue #6 - Cache Tagging (Low Priority)

- [ ] Add `cache_tags` parameter
- [ ] Update `_create_cache()` with metadata
- [ ] Update `_find_existing_cache()` with tag matching
- [ ] Add graceful fallback for older API
- [ ] Add tests for tag matching (6+ tests)
- [ ] Document cache tagging pattern
- [ ] Add example to GEMINI_INTEGRATION.md
- [ ] Add to MIGRATION_V0_3.md

### Phase 4: Enhancements (Optional)

- [ ] Implement `PrepareOnceMixin`
- [ ] Add tests for PrepareOnceMixin
- [ ] Document mixin usage
- [ ] Consider ExtensibleErrorClassifier pattern

### Phase 5: Documentation & Release

- [ ] Update CHANGELOG.md with v0.3.0 changes
- [ ] Create MIGRATION_V0_3.md
- [ ] Update README.md with new features
- [ ] Update API.md with new APIs
- [ ] Run full test suite (target: 117+ tests, 80%+ coverage)
- [ ] Run linters and type checkers
- [ ] Update version in pyproject.toml to 0.3.0
- [ ] Create git tag v0.3.0
- [ ] Build and publish to PyPI

---

## Breaking Changes

**None planned.** All changes are additive and backward compatible:

- `state` parameter is optional (defaults to `None`)
- `include_metadata` is opt-in (defaults to `False`)
- `cache_tags` is optional (defaults to empty dict)
- Existing strategies and code work unchanged

---

## Success Criteria

1. **Retry State:**
   - Users can implement multi-stage retry strategies
   - State is isolated per work item (no collisions)
   - Example demonstrates 81% cost savings with partial recovery

2. **Safety Ratings:**
   - Users can access Gemini safety ratings
   - Opt-in feature doesn't affect existing code
   - Clear example for content filtering

3. **Cache Tagging:**
   - Users can tag caches for precise matching
   - Graceful fallback if API doesn't support metadata
   - Prevents accidental cache reuse with different prompts

4. **Quality:**
   - All tests passing (117+ tests)
   - Coverage maintained or improved (80%+)
   - No regressions in existing functionality
   - Clear migration guide for upgrading

---

## Timeline Estimate

- **Issue #8 (Retry State):** 2-3 days
- **Issue #3 (Safety Ratings):** 1-2 days
- **Issue #6 (Cache Tagging):** 1 day
- **Documentation & Testing:** 1-2 days
- **Total:** 5-8 days

---

## Open Questions

1. **Retry State Cleanup:** Should state be cleared on final failure, or preserved for post-mortem analysis?
   - **Recommendation:** Clear on both success and failure, but log state to logger.debug() before clearing

2. **Safety Ratings API:** Does google-genai consistently provide safety_ratings, or only sometimes?
   - **Action:** Test with various Gemini models and prompts

3. **Cache Metadata Support:** What version of google-genai added metadata support?
   - **Action:** Check google-genai changelog and add version check if needed

4. **State Serialization:** Should RetryState support JSON serialization for debugging?
   - **Recommendation:** Yes, add `to_dict()` and `from_dict()` methods

---

## Post-v0.3.0 Backlog

Items to consider for future versions:

- **Strategy Lifecycle Management:** Explicit `register_strategy()` pattern (Issue from feedback)
- **Prometheus Metrics:** Built-in Prometheus endpoint (mentioned in features)
- **Dynamic Worker Scaling:** Adjust workers based on load (mentioned in future enhancements)
- **Persistent Queue:** Redis/DB-backed queue (mentioned in limitations)
- **Batch API Support:** True batch API for 50% cost savings (mentioned in limitations)
- **Multi-Process Support:** Distributed locks for multi-process scenarios (mentioned in limitations)
- **More Error Classifiers:** OpenAI, Anthropic, etc. (mentioned in contributing)
