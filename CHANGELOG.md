# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2025-11-25

Release focused on developer experience improvements and code quality.

### Added

- **`TokenTrackingError` export** - The `TokenTrackingError` class is now exported from the main
  `batch_llm` package for users who want to catch it explicitly when handling failed LLM calls
  with token usage tracking.
- **Type aliases** - Added convenience type aliases for the most common use case (string input,
  typed output, no context):
  - `SimpleBatchProcessor[T]` → `ParallelBatchProcessor[str, T, None]`
  - `SimpleWorkItem[T]` → `LLMWorkItem[str, T, None]`
  - `SimpleResult[T]` → `WorkItemResult[T, None]`

### Changed

- **Deprecation warnings** - Added `DeprecationWarning` for legacy constructor parameters on
  `ParallelBatchProcessor`:
  - `max_workers` → Use `ProcessorConfig(max_workers=...)`
  - `timeout_per_item` → Use `ProcessorConfig(timeout_per_item=...)`
  - `rate_limit_cooldown` → Use `ProcessorConfig(rate_limit=RateLimitConfig(cooldown_seconds=...))`
- **Module docstring** - Updated to accurately describe the v0.1+ strategy pattern API instead
  of the removed v0.0.x integration modes.
- **Python classifiers** - Added Python 3.13 and 3.14 to pyproject.toml classifiers (tests already
  run on these versions in CI).
- **Config auto-validation** - `ProcessorConfig`, `RetryConfig`, and `RateLimitConfig` now
  automatically validate on construction via `__post_init__`. Invalid values raise `ValueError`
  immediately instead of waiting for explicit `validate()` call. This is a **behavior change**
  for code that creates configs with invalid values and validates later.
- **Logging format** - Replaced emoji characters in log messages with ASCII text alternatives
  for better terminal compatibility:
  - `✓` → `[OK]`
  - `✗` → `[FAIL]`
  - `⚠️` → `[WARN]`
  - `ℹ️` → `[INFO]`

### Fixed

- **Code duplication** - Deduplicated the `TokenTrackingError` class which was defined inline
  3 times in `llm_strategies.py`. Now defined once in `strategies/errors.py`.
- **Code duplication** - Deduplicated `_extract_safety_ratings` method from `GeminiStrategy`
  and `GeminiCachedStrategy` into a module-level function.

## [0.4.0] - 2025-01-14

Major release adding strategy lifecycle management with context managers.

### Added

- **Strategy lifecycle management** - Hybrid approach using context managers for prepare/cleanup
  - Strategies are prepared once when first used (via existing `_ensure_strategy_prepared()`)
  - Strategies are cleaned up once on exit when using `async with` context manager
  - Backward compatible: without context manager, no cleanup is called
  - Prevents adding work after process_all() starts (raises `RuntimeError`)
  - Supports shared strategy instances (prepared once, cleaned up once)
  - Comprehensive test coverage in `test_strategy_lifecycle.py`

### Changed

- **BREAKING: Per-item cleanup removed** - Strategies are no longer cleaned up after each item
  - Previously: `strategy.cleanup()` called after each work item completed
  - Now: `strategy.cleanup()` only called in `__aexit__` when using context manager
  - Migration: Wrap processor in `async with` to enable automatic cleanup
  - For production caches that should persist, make `cleanup()` a no-op
- **Base class tracking** - `BatchProcessor` now tracks unique strategy instances and processing
  state

### Fixed

- **Cancellation propagation** - Added regression test and explicit `asyncio.CancelledError`
  handling so `ParallelBatchProcessor` no longer swallows cancellations inside retries,
  middlewares, or rate-limit coordination. Ensures shutdown behaves correctly on Python 3.10+.

## [0.3.6] - 2025-01-13

Minor release fixing mypy compatibility and improving code quality standards.

### Fixed

- **Mypy compatibility** - Changed `TokenTrackingError` from dynamic base class `type(e)` to static `Exception`
  base (mypy doesn't support dynamic base classes)
- **Linter warnings** - Replaced try/import/except pattern with `importlib.util.find_spec()` for optional
  dependency checks in test_cache_tag_matching.py
- **Documentation linting** - Fixed all 173 markdown linting issues across README.md and docs/
  - Line length violations (wrapped at 120 chars)
  - Missing code block language specifiers
  - Ordered list numbering consistency
  - Bold text used as headings
  - Broken internal links

### Changed

- **Pre-commit checklist** - Added to CLAUDE.md emphasizing quality checks (tests, linter, mypy, markdown-lint)
  before ANY commit
- **Development workflow** - Updated with common mypy issues and solutions

## [0.3.5] - 2025-01-13

Critical bug fix for token tracking when items fail validation.

### Fixed

- **Token tracking for failed items** - Fixed critical bug where failed items showed 0 tokens consumed
  - Previously caused 20-30% cost underestimation when failure rates were high
  - Now correctly tracks tokens even when validation fails
  - Applies to all three built-in strategies: GeminiStrategy, GeminiCachedStrategy, PydanticAIStrategy
  - Exceptions now carry `_failed_token_usage` in `__dict__` for framework to aggregate
  - Critical for accurate production cost tracking and budget planning

### Changed

- **Token extraction timing** - All strategies now extract token usage BEFORE parsing/validation
- **Exception handling** - Enhanced to preserve token usage through exception chain

### Added

- **Test coverage** - Added `test_token_tracking_on_failure.py` to verify fix works correctly
- **MockAgent enhancement** - Added `tokens_per_call` parameter for token tracking tests

## [0.3.4] - 2025-01-12

Compatibility release for google-genai v1.49.0+.

### Fixed

- **google-genai v1.49.0+ compatibility** - Updated GeminiCachedStrategy to handle API changes
  - Detection of three API versions: v1.45, v1.46-v1.48, v1.49+
  - Proper handling of `CreateCachedContentConfig` vs `CachedContent` parameter types
  - Automatic version detection during strategy initialization

### Added

- **API version tests** - Added `test_gemini_api_versions.py` to verify version detection works correctly

## [0.3.3] - 2025-01-11

Bug fix release for proactive rate limiting.

### Fixed

- **Proactive rate limiting** - Fixed bug where `check_rate_limit_proactively()` wasn't being called
  - Now properly prevents hitting API limits by checking before each batch
  - Reduces wasted API calls and improves throughput

### Added

- **Default error classifier** - Added rate limit detection to default classifier (not just Gemini-specific)

## [0.3.2] - 2025-01-11

Bug fix release for error classification.

### Fixed

- **Error classifiers** - Fixed classifiers to prevent wasting retries on logic bugs
  - ValidationError, TypeError, AttributeError, KeyError now marked as non-retryable
  - Saves API costs by not retrying programmer errors

## [0.3.1] - 2025-01-11

Bug fix release for process_all() state contamination.

### Fixed

- **State contamination** - Fixed bug where reusing processor with process_all() contaminated state
  - Each process_all() call now gets fresh state
  - Safe to call process_all() multiple times on same processor instance

## [0.3.0] - 2025-01-10

This release adds advanced retry patterns for multi-stage LLM strategies,
safety ratings access for content moderation, and precise cache tagging for production deployments.

### ⚠️ Breaking Changes

**None** - This release is 100% backward compatible. All new features are opt-in with default values that
preserve existing behavior.

---

### Added

#### Core Features

- **RetryState for multi-stage strategies** (#8, HIGH priority)
  - New `RetryState` class for per-work-item mutable state that persists across all retry attempts
  - Enables advanced retry patterns like partial recovery and progressive prompting
  - Dictionary-style API: `get()`, `set()`, `delete()`, `clear()`, `__contains__()`
  - Automatically created by framework and passed to `execute()` and `on_error()`
  - Each work item gets its own isolated `RetryState` instance
  - Use cases:
    - **Partial recovery**: Parse what succeeded, retry only failed parts (81% cost savings)
    - **Multi-stage strategies**: Track state across validation/formatting/final output stages
    - **Progressive prompting**: Build increasingly detailed prompts based on previous failures
    - **Error tracking**: Count different error types per work item
  - Example:

    ```python
    async def execute(self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None):
        if state and attempt > 1:
            # Use state from previous attempts
            partial_results = state.get("partial_results", {})
            # Retry with context from previous failures
    ```

- **GeminiResponse for safety ratings access** (#3, MEDIUM priority)
  - New `GeminiResponse[TOutput]` generic wrapper class for Gemini API responses
  - Access safety ratings, finish reasons, and raw response metadata
  - Opt-in via `include_metadata=True` parameter on `GeminiStrategy` and `GeminiCachedStrategy`
  - Fields:
    - `output: TOutput` - The parsed output
    - `safety_ratings: dict[str, str] | None` - Content safety ratings (HARM_CATEGORY_HATE_SPEECH, etc.)
    - `finish_reason: str | None` - Why generation stopped (STOP, MAX_TOKENS, SAFETY, etc.)
    - `token_usage: dict[str, int]` - Token counts
    - `raw_response: Any` - Full Google API response object
  - Use cases:
    - Content moderation and filtering
    - Safety rating logging for compliance
    - Debugging generation issues (finish_reason)
    - Accessing provider-specific metadata
  - Example:

    ```python
    strategy = GeminiCachedStrategy(
        model="gemini-2.0-flash",
        client=client,
        response_parser=parse_output,
        cached_content=content,
        include_metadata=True,  # Opt-in for safety ratings
    )
    result = await processor.process_all()
    if isinstance(result.results[0].output, GeminiResponse):
        ratings = result.results[0].output.safety_ratings
        if ratings.get("HARM_CATEGORY_HATE_SPEECH") == "HIGH":
            # Handle unsafe content
    ```

- **Cache tagging for precise cache matching** (#6, LOW priority)
  - New `cache_tags` parameter for `GeminiCachedStrategy`
  - Attach custom metadata tags to caches for precise identification
  - Prevents accidental cache reuse across experiments/versions
  - Tags are checked during cache matching in `_find_or_create_cache()`
  - Version-aware: Falls back gracefully on older google-genai versions
  - Use cases:
    - Multi-tenant applications (tag by customer_id)
    - A/B testing (tag by experiment variant)
    - Version control (tag by schema version or prompt version)
    - Environment isolation (tag by "production" vs "staging")
  - Example:

    ```python
    strategy = GeminiCachedStrategy(
        model="gemini-2.0-flash",
        client=client,
        response_parser=parse_output,
        cached_content=content,
        cache_tags={
            "customer_id": "acme-corp",
            "schema_version": "v2",
            "experiment": "new-prompt-A"
        }
    )
    ```

#### API Changes (Backward Compatible)

- **Updated `LLMCallStrategy` protocol**:
  - `execute()` now accepts optional `state: RetryState | None = None` parameter
  - `on_error()` now accepts optional `state: RetryState | None = None` parameter
  - Default `None` ensures backward compatibility with existing strategies

- **Updated `GeminiStrategy`**:
  - New parameter: `include_metadata: bool = False`
  - Return type: `TOutput | GeminiResponse[TOutput]`
  - Helper method: `_extract_safety_ratings()` for parsing response metadata

- **Updated `GeminiCachedStrategy`**:
  - New parameter: `include_metadata: bool = False`
  - New parameter: `cache_tags: dict[str, str] | None = None`
  - Return type: `TOutput | GeminiResponse[TOutput]`
  - Enhanced `_find_or_create_cache()` to check tags when matching caches
  - Enhanced `_create_new_cache()` to include metadata/tags when creating caches

#### Documentation

- **New implementation plan**: `docs/IMPLEMENTATION_PLAN_V0_3.md`
  - Detailed technical specifications for all three features
  - Priority ranking and use case analysis
  - Test requirements and success criteria
  - Timeline estimates

- **New migration guide**: `docs/MIGRATION_V0_3.md`
  - Complete v0.2 → v0.3 upgrade instructions
  - Emphasizes 100% backward compatibility
  - Code examples for all new features
  - Use case guides and best practices

#### Testing

- **12 new comprehensive tests** (141 tests total, all passing)
  - `tests/test_v0_3_features.py` - Comprehensive test coverage for all v0.3.0 features
  - **RetryState tests** (5 tests):
    - `test_retry_state_persistence` - State persists across retry attempts
    - `test_retry_state_isolation` - Each work item gets its own state
    - `test_retry_state_operations` - Dictionary operations (get/set/delete/clear)
    - `test_retry_state_none_backward_compatibility` - Works when state is None
    - `test_on_error_receives_state` - on_error receives the same state as execute
  - **GeminiResponse tests** (3 tests):
    - `test_gemini_response_with_metadata` - Safety ratings extraction
    - `test_gemini_response_without_metadata` - Backward compatibility (include_metadata=False)
    - `test_gemini_response_generic_type` - Works with different output types
  - **Cache tagging tests** (2 tests):
    - `test_cache_tags_isolation` - Different tags stored correctly
    - `test_cache_tags_none_default` - Tags default to empty dict
  - **Integration tests** (2 tests):
    - `test_retry_state_with_gemini_response` - Features work together
    - `test_shared_strategy_with_retry_state` - Shared strategies with per-item state

### Changed

- **ParallelBatchProcessor** now creates `RetryState` instance per work item
- **ParallelBatchProcessor** passes `retry_state` to `strategy.execute()` and `strategy.on_error()`
- All built-in strategies updated to accept optional `state` parameter

### Performance

- **No performance regression** - All additions are opt-in
- **RetryState is lightweight** - Simple dict wrapper, minimal memory overhead
- **GeminiResponse is zero-cost when disabled** - Only overhead when `include_metadata=True`
- **Cache tags are checked efficiently** - Simple dict comparison during cache matching

### Test Coverage

- **76% overall coverage** (meets requirement)
- **141 tests total** (129 existing + 12 new)
- **All tests passing** including edge cases and integration scenarios

---

## [0.2.0] - 2025-01-09

This release addresses critical production issues identified from real-world usage,
particularly around shared strategy instances for cost optimization with Gemini prompt caching.

### ⚠️ Breaking Changes

#### GeminiCachedStrategy cleanup() Behavior

`cleanup()` now **preserves** caches for reuse by default (previously deleted them).
This enables 70-90% cost savings when running multiple batches within the TTL window.

**Before (v0.1):**

```python
await strategy.cleanup()  # Deleted cache
```

**After (v0.2):**

```python
await strategy.cleanup()  # Preserves cache for reuse

# Explicitly delete if needed
await strategy.delete_cache()
```

**Migration:** If you relied on automatic cache deletion, call `await strategy.delete_cache()` explicitly.
For most production use cases, the new behavior is better (no code changes needed).

See **[Migration Guide](docs/MIGRATION_V0_2.md)** for complete upgrade instructions.

---

### Added

#### Core Features

- **Shared strategy optimization** (#1)
  - Framework now tracks unique strategy instances by `id()`
  - `prepare()` called only once per unique strategy instance, even with concurrent workers
  - Thread-safe via double-checked locking pattern
  - Enables cost optimization: single cache shared across all work items
  - New internal method: `_ensure_strategy_prepared()`

- **Cached token tracking** (#4)
  - `BatchResult.total_cached_tokens` - Sum of cached input tokens across all work items
  - `BatchResult.cache_hit_rate()` - Calculate cache hit rate as percentage
  - `BatchResult.effective_input_tokens()` - Calculate actual cost after caching discount (90% savings)
  - `ProcessingStats.total_cached_tokens` - Track cached tokens in stats

- **Automatic cache renewal** (#7)
  - New parameter: `GeminiCachedStrategy.cache_renewal_buffer_seconds` (default: 300s)
  - New parameter: `GeminiCachedStrategy.auto_renew` (default: True)
  - Proactive cache expiration detection via `_is_cache_expired()`
  - Automatic renewal before API calls to prevent expiration errors
  - Critical bug fix: Reused caches now use actual creation time (not current time)

- **Cache reuse across runs** (#5, #6)
  - `GeminiCachedStrategy._find_or_create_cache()` - Find existing caches or create new ones
  - Caches matched by model name suffix
  - Preserves caches between pipeline runs (within TTL window)
  - Enables 70-90% cost savings for recurring jobs

- **`GeminiCachedStrategy.delete_cache()` method** (#5)
  - Explicit cache deletion for tests and one-off jobs
  - Separated from `cleanup()` hook for better lifecycle control

#### API Compatibility

- **google-genai v1.46+ support** (#2)
  - Auto-detects API version via `_detect_google_genai_version()`
  - Uses `CreateCachedContentConfig` for v1.46+
  - Falls back to legacy API for v1.45 and earlier
  - Both versions fully supported

#### Documentation

- **Enhanced `LLMCallStrategy.cleanup()` docstring**
  - Clarified use cases (connections, locks, metrics)
  - Documented what NOT to use it for (cache deletion)
  - Added note on cache lifecycle best practices

- **Updated README.md**
  - New section: "Shared Strategies for Cost Optimization"
  - Enhanced "Token Tracking" section with cache metrics examples
  - Added cache hit rate and effective cost calculation examples

- **New migration guide**: `docs/MIGRATION_V0_2.md`
  - Complete v0.1 → v0.2 upgrade instructions
  - Breaking changes with code examples
  - New features with usage patterns
  - Troubleshooting guide

- **Implementation plan**: `docs/IMPLEMENTATION_PLAN_V0_2.md`
  - Detailed technical specifications
  - Issue analysis and solutions
  - Test strategy and coverage goals

### Changed

- `GeminiCachedStrategy.prepare()` now calls `_find_or_create_cache()` (reuses existing caches)
- `GeminiCachedStrategy.execute()` now includes automatic cache renewal logic
- `GeminiCachedStrategy.cleanup()` preserves caches by default (breaking change)

### Deprecated

- `GeminiCachedStrategy.cache_refresh_threshold` - Use `cache_renewal_buffer_seconds` instead
  - Percentage-based threshold less predictable than absolute time for long jobs
  - Will be removed in v0.3.0

### Fixed

- **Multiple `prepare()` calls on shared strategies** (#1)
  - Framework called `prepare()` once per work item, not once per unique strategy
  - With concurrent workers, created multiple caches instead of one
  - Fixed via `_ensure_strategy_prepared()` with double-checked locking

- **Cache expiration errors in long pipelines** (#7)
  - Pipelines longer than cache TTL failed with "cache expired" errors
  - Fixed via proactive renewal with configurable buffer

- **Cache reuse creation time bug** (#7)
  - Reused caches incorrectly used current time instead of actual creation time
  - Caused expiration detection to fail (thought cache was newer than it was)
  - Fixed to use `cache.create_time.timestamp()` from Google's servers

### Tests

- **17 new tests added** (95 tests total, all passing)
  - `test_shared_strategies.py` - 6 tests for shared strategy optimization
  - `test_gemini_api_versions.py` - 3 tests for API version detection
  - `test_token_tracking.py` - 8 tests for cached token tracking

- **Updated existing tests**
  - `test_gemini_cached_strategy_lifecycle` - Updated for new cleanup() behavior
  - `test_gemini_cached_strategy_auto_renewal` - Renamed and updated for auto-renewal

### Performance

- **No performance regression** - All optimizations are additions
- **Shared strategies reduce API calls** - One cache creation instead of multiple
- **Token tracking is O(n)** - Simple sum over results

---

## [0.1.0] - TBD

### ⚠️ Breaking Changes

This major release introduces the **LLM Call Strategy Pattern**,
providing a flexible, provider-agnostic architecture for batch LLM processing.

#### Removed Parameters

- **`agent=` parameter removed** from `LLMWorkItem` - Use `strategy=` instead
- **`client=` parameter removed** from `LLMWorkItem` - Use `strategy=` instead

#### Migration Required

All code using `LLMWorkItem` must be updated:

```python
# ❌ Old (v0.0.x) - No longer works
work_item = LLMWorkItem(
    item_id="item_1",
    agent=agent,  # or client=client
    prompt="Test prompt",
)

# ✅ New (v0.1) - Use strategy
from batch_llm import PydanticAIStrategy

strategy = PydanticAIStrategy(agent=agent)
work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="Test prompt",
)
```

See **[Migration Guide](docs/archive/MIGRATION_V0_1.md)** for complete upgrade instructions.

---

### Added

#### Core Features

- **`LLMCallStrategy` abstract base class** - Universal interface for any LLM provider
  - `prepare()` - Initialize resources before processing
  - `execute()` - Execute LLM call with retry support
  - `on_error()` - Handle errors and adjust retry behavior (new in this release)
  - `cleanup()` - Clean up resources after processing
- **`on_error()` callback for intelligent retry strategies**
  - Called automatically when `execute()` raises an exception
  - Enables error-type-aware retry logic (validation vs. network vs. rate limit errors)
  - Allows state tracking across retry attempts
  - Use cases:
    - **Smart model escalation**: Only escalate to expensive models on validation errors, not network errors
    - **Smart retry prompts**: Build better retry prompts based on which fields failed validation
    - **Error tracking**: Distinguish and count different error types
  - Non-breaking: Default no-op implementation
  - Framework catches and logs exceptions in `on_error()` to prevent crashes
- **Proactive rate limiting** to prevent hitting API rate limits
  - Configure via `ProcessorConfig.max_requests_per_minute`
  - Throttles requests before they hit the API (prevents 429 errors)
  - Uses `aiolimiter` for token bucket rate limiting
  - Coordinates across all workers (shared limiter instance)
  - Complements reactive rate limit handling (cooldown after 429s)
  - Optional: Set to `None` to disable (default behavior)
  - Validation warns if rate limit is lower than worker count

#### Built-in Strategies

- **`PydanticAIStrategy`** - Wraps PydanticAI agents for batch processing
- **`GeminiStrategy`** - Direct Google Gemini API calls without caching
- **`GeminiCachedStrategy`** - Gemini API calls with automatic context caching
  - Automatic cache creation on `prepare()`
  - Automatic cache TTL refresh during processing
  - Automatic cache deletion on `cleanup()`
  - Configurable cache TTL and refresh threshold

#### Documentation

- **`docs/API.md`** - Complete API reference documentation
  - Added `TokenUsage` TypedDict documentation
  - Added `FrameworkTimeoutError` exception documentation
  - Documented `LLMCallStrategy.dry_run()` method
  - **Documented `LLMCallStrategy.on_error()` callback** with 3 complete use case examples
    - Smart model escalation (validation errors only)
    - Smart retry with partial parsing
    - Error type tracking
  - Updated strategy lifecycle description to include `on_error` call sequence
  - Updated `ErrorInfo` field documentation (error_category, is_timeout)
  - Added missing `ProcessorConfig.progress_callback_timeout` field
  - Updated all code examples to use `TokenUsage`
- **`docs/archive/MIGRATION_V0_1.md`** - Comprehensive v0.0.x → v0.1 migration guide
- **`README.md`** - Comprehensive improvements
  - Added complete table of contents with 40+ section links
  - Added **Configuration Reference** section (200+ lines)
  - Added **Best Practices** section (120+ lines)
  - Added **Troubleshooting** section (180+ lines)
  - Added **FAQ** section (180+ lines) with 15+ Q&A
  - **Updated Strategy Pattern section** to include `on_error` method
  - **Updated Smart Retry section** to demonstrate `on_error` callback usage
  - **Updated Model Escalation section** to show smart escalation with `on_error`
  - **Updated FAQ** with `on_error` callback approach for adaptive prompts
  - Enhanced Middleware & Observers documentation
  - Improved Testing section with 3 approaches
  - Updated all code examples to use `TokenUsage` TypedDict
  - Fixed mutable default argument in progressive temperature example
- **`examples/example_openai.py`** - OpenAI integration examples
- **`examples/example_anthropic.py`** - Anthropic Claude integration examples
- **`examples/example_langchain.py`** - LangChain integration examples (including RAG)
- **`examples/example_llm_strategies.py`** - All built-in strategies with examples
- **`examples/example_smart_model_escalation.py`** - Smart model escalation using `on_error` callback
  - Only escalates to expensive models on validation errors
  - Retries with same cheap model on network/rate limit errors
  - Demonstrates 60-80% cost savings vs. always using best model
  - Includes comparison with blind escalation strategy
- **`examples/example_gemini_smart_retry.py`** - Enhanced with `on_error` callback documentation
  - Shows how to use `on_error` to track validation errors cleanly
  - Demonstrates building targeted retry prompts based on which fields failed
- **All example files** - Updated to use `TokenUsage` TypedDict consistently

#### Testing

- **`batch_llm.testing.MockAgent`** - Mock agent for testing without API calls
- Comprehensive test coverage for all strategies
- Strategy lifecycle tests (prepare/execute/cleanup)
- **New `on_error` callback tests** (4 comprehensive tests):
  - `test_on_error_callback_called` - Verifies callback is invoked with correct parameters
  - `test_on_error_callback_with_state` - Tests state tracking across retries (validation vs. network errors)
  - `test_on_error_callback_exception_handling` - Ensures buggy callbacks don't crash processor
  - `test_on_error_not_called_on_success` - Confirms callback only runs on errors

---

### Changed

#### Architecture

- **Strategy pattern** replaces direct agent/client parameters
  - Cleaner separation of concerns
  - Framework handles timeout enforcement at top level
  - Strategies no longer need `asyncio.wait_for()` wrappers
- **Improved timeout enforcement** - Framework-level with `asyncio.wait_for()`
  - Consistent behavior across all strategies
  - Timeout parameter still passed to `execute()` for informational purposes
  - Removed redundant timeout wrappers from built-in strategies
- **Enhanced strategy execution lifecycle** to include error callback
  - Framework now calls `strategy.on_error(exception, attempt)` when `execute()` raises
  - Error callback invoked before retry logic, allowing strategies to adjust behavior
  - Exceptions in `on_error()` are caught and logged (won't crash processing)
  - Type guard added for mypy compliance

#### Type System

- **`LLMWorkItem` now accepts `strategy=`** instead of `agent=` or `client=`
- Generic type parameters preserved: `LLMWorkItem[TInput, TOutput, TContext]`
- Better type safety with strategy pattern

#### Internal

- Refactored `_process_work_item_direct()` to use strategy lifecycle
- Improved error handling in strategy execution
- Better resource cleanup with context managers

---

### Fixed

- **Timeout enforcement bug** - Custom strategies now respect timeouts consistently
  - Framework wraps all `strategy.execute()` calls in `asyncio.wait_for()`
  - Previously, custom strategies could ignore timeout parameter
  - All 61 tests now pass (was 60 passing, 1 skipped)
- **Test coverage** - Fixed `test_custom_strategy_timeout_handling` (previously skipped)
- **Error classification logic bug handling** - Error classifiers now properly distinguish logic bugs from
  transient failures
  - `DefaultErrorClassifier` and `GeminiErrorClassifier` now explicitly check for logic bug exceptions
    (`ValueError`, `TypeError`, `AttributeError`, etc.)
  - Logic bugs are marked as non-retryable to avoid wasting retry attempts and tokens on deterministic failures
  - Pydantic `ValidationError` is explicitly marked as retryable (LLM might generate valid output on retry)
  - Generic `Exception` instances remain retryable (allows custom transient errors and test mocks)
  - Prevents wasting `max_attempts` retries on programming errors that won't be fixed by retrying
  - Added regression test `test_logic_bugs_fail_fast` to ensure logic bugs fail after 1 attempt
  - Fixed `test_token_usage_tracked_across_retries` exception chain construction

---

### Migration Path

**Upgrading from v0.0.x?** Follow these steps:

1. **Read the Migration Guide**: `docs/archive/MIGRATION_V0_1.md`
2. **Update imports**: Add `PydanticAIStrategy`, `GeminiStrategy`, or `GeminiCachedStrategy`
3. **Wrap your agents/clients**: Create strategy instances
4. **Update LLMWorkItem**: Replace `agent=` or `client=` with `strategy=`
5. **Test thoroughly**: Verify timeout and retry behavior

**Estimated migration time**: 15-60 minutes for most codebases

**Benefits**:

- ✅ Support for any LLM provider (OpenAI, Anthropic, LangChain, etc.)
- ✅ Better caching with automatic lifecycle management
- ✅ More reliable timeout enforcement
- ✅ Cleaner, more maintainable code
- ✅ Easy to create custom strategies

---

## [0.0.2] - 2025-10-20

### Fixed

- Fixed crash when middleware's `before_process()` returns `None` - now properly preserves original `item_id`
- Fixed stats race condition where re-queued rate-limited items inflated the total count
- Improved token usage extraction robustness with multiple fallback strategies for different LLM providers
- Fixed all linting issues (20 total: 2 in source code, 18 in tests)

### Added

- `max_queue_size` configuration option to prevent memory issues with large batches (default: 0 = unlimited)
- 3 new tests for edge cases and bug fixes
- `docs/internal/` directory for development documentation (gitignored)

### Changed

- Token extraction now uses a robust helper method with 3 fallback strategies
- Moved 16 internal documentation files to `docs/internal/` for cleaner repository
- Updated `CLAUDE.md` with ruff workflow reminder
- **BREAKING**: Removed unused `batch_size` parameter from `BatchProcessor` and `ParallelBatchProcessor`

### Removed

- `batch_size` parameter (was unused and ignored)

### Documentation

- Created comprehensive bug fix documentation in `docs/internal/BUG_FIXES_V2.0.2.md`
- Updated code review with completion status
- Cleaned up repository root (70% reduction in visible markdown files)

## [0.0.1] - 2025-10-19

### Added

- Optional dependencies: `pydantic-ai` and `google-genai` are now optional extras
- Comprehensive Gemini integration guide (`docs/GEMINI_INTEGRATION.md`)
- Working Gemini direct API example (`examples/example_gemini_direct.py`)
- Installation options: `[pydantic-ai]`, `[gemini]`, `[all]`, `[dev]`

### Fixed

- Direct call timeout enforcement - now properly wraps calls in `asyncio.wait_for()`
- Middleware `on_error` now called after retry exhaustion (not just for non-retryable errors)
- Middleware execution order - `after_process` now runs in reverse order (onion pattern)

### Changed

- Core dependency now only `pydantic>=2.0.0` (was also `pydantic-ai` and `google-genai`)
- String annotations for `Agent` type to work without `pydantic-ai` installed

### Documentation

- `OPTIONAL_DEPENDENCIES.md` - Complete installation guide
- Migration guide for v0.0.0 → v0.0.1
- Updated README with installation options

## [0.0.0] - 2025-10-19

### Added

- Initial PyPI package release
- Provider-agnostic error classification system
- Pluggable rate limit strategies (ExponentialBackoff, FixedDelay)
- Middleware pipeline for extensible processing
- Observer pattern for monitoring and metrics
- Configuration-based setup with `ProcessorConfig`
- `GeminiErrorClassifier` for Google Gemini API error handling
- `MetricsObserver` for tracking processing statistics
- `MockAgent` for testing without API calls
- Comprehensive test suite with pytest
- Support for Python 3.10+

### Changed

- Refactored to src-layout for better packaging
- Improved error handling with retryable error detection
- Enhanced documentation with installation instructions
- Updated examples to use new configuration system

### Features

- `ParallelBatchProcessor` - Async parallel LLM request processing
- Work queue management with context passing
- Post-processing hooks for custom logic
- Partial failure handling
- Token usage tracking
- Timeout support per item
- Type-safe with generics support

## [0.0.0-alpha] - Internal

### Added

- Initial implementation for internal use
- Basic parallel processing
- PydanticAI integration
- Work item and result models
