# batch-llm Improvement Plan

**Created:** 2025-01-10
**Based on:** Package Review (PACKAGE_REVIEW_2025_01_10.md)
**Current Version:** 0.3.0
**Target Version:** 0.3.1 (patches), 0.4.0 (features)

---

## Overview

This document outlines a prioritized plan for improving the batch-llm package based on a comprehensive review
that identified 27 opportunities for improvement across code quality, testing, documentation, performance, and
user experience.

**Current Status:** Production-ready (8/10 score)
**Goal:** Increase to 9/10 with focused improvements

---

## Phase 1: Critical Fixes (Week 1)

**Target:** v0.3.1 patch release
**Goal:** Fix critical issues that affect correctness and user experience

### 1.1 Fix Version Mismatch ⚠️ CRITICAL

**Priority:** P0 - BLOCKER
**Effort:** 15 minutes
**Owner:** TBD

**Problem:**

- `pyproject.toml` declares `version = "0.2.0"`
- `src/batch_llm/__init__.py` declares `__version__ = "0.1.0"`
- Users checking `batch_llm.__version__` get wrong information

**Implementation:**

1. Update `src/batch_llm/__init__.py`:

   ```python
   # Remove hardcoded version
   # OLD: __version__ = "0.1.0"

   # NEW: Use importlib.metadata
   from importlib.metadata import version, PackageNotFoundError

   try:
       __version__ = version("batch-llm")
   except PackageNotFoundError:
       # Package not installed (e.g., running from source)
       __version__ = "0.0.0+dev"
   ```

1. Update `pyproject.toml` to correct version:

   ```toml
   version = "0.3.0"  # Match actual release
   ```

1. Add test:

   ```python
   def test_version_matches_package():
       """Verify __version__ matches package metadata."""
       from batch_llm import __version__
       from importlib.metadata import version
       assert __version__ == version("batch-llm")
   ```

**Validation:**

- Run `python -c "import batch_llm; print(batch_llm.__version__)"`
- Should print "0.3.0"
- Test passes in CI

**Related Issues:** None
**Documentation:** Update CHANGELOG.md with fix

---

### 1.2 Add Shared Strategy Lifecycle Tests

**Priority:** P0 - CRITICAL
**Effort:** 2-3 hours
**Owner:** TBD

**Problem:**

- v0.2.0 shared strategy optimization (70-90% cost savings) has no tests
- Critical feature could break silently
- `_ensure_strategy_prepared()` logic at `parallel.py:162-194` untested

**Implementation:**

Create `tests/test_shared_strategy_lifecycle.py`:

```python
"""Tests for shared strategy lifecycle (v0.2.0 feature)."""

import asyncio
import pytest
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import LLMCallStrategy


@pytest.mark.asyncio
async def test_shared_strategy_prepare_called_once():
    """Verify prepare() called exactly once for shared strategy."""
    prepare_count = 0
    cleanup_count = 0
    execute_count = 0

    class CountingStrategy(LLMCallStrategy[str]):
        async def prepare(self):
            nonlocal prepare_count
            prepare_count += 1
            await asyncio.sleep(0.01)  # Simulate slow prepare

        async def cleanup(self):
            nonlocal cleanup_count
            cleanup_count += 1

        async def execute(self, prompt, attempt, timeout, state):
            nonlocal execute_count
            execute_count += 1
            return f"Result: {prompt}", {"input_tokens": 10, "output_tokens": 5}

    strategy = CountingStrategy()  # One instance
    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add 100 items all sharing the same strategy
        for i in range(100):
            await processor.add_work(LLMWorkItem(
                item_id=f"item_{i}",
                strategy=strategy,  # SHARED
                prompt=f"Test {i}"
            ))

        result = await processor.process_all()

    # Critical assertions
    assert result.total_items == 100
    assert result.succeeded == 100
    assert prepare_count == 1, "prepare() should be called exactly once"
    assert execute_count == 100, "execute() should be called per item"
    assert cleanup_count == 100, "cleanup() should be called per item"


@pytest.mark.asyncio
async def test_shared_strategy_concurrent_prepare():
    """Verify no race condition with concurrent workers."""
    prepare_count = 0
    prepare_lock = asyncio.Lock()

    class ConcurrentStrategy(LLMCallStrategy[str]):
        async def prepare(self):
            nonlocal prepare_count
            async with prepare_lock:
                prepare_count += 1
            await asyncio.sleep(0.1)  # Long prepare to stress test

        async def execute(self, prompt, attempt, timeout, state):
            return "Result", {"input_tokens": 10, "output_tokens": 5}

    strategy = ConcurrentStrategy()
    config = ProcessorConfig(max_workers=20, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i in range(100):
            await processor.add_work(LLMWorkItem(
                item_id=f"item_{i}",
                strategy=strategy,
                prompt=f"Test {i}"
            ))

        result = await processor.process_all()

    assert result.succeeded == 100
    assert prepare_count == 1, "Race condition: prepare() called multiple times"


@pytest.mark.asyncio
async def test_different_strategies_prepare_separately():
    """Verify different strategy instances each get prepare() called."""
    prepare_counts = {}

    class TrackedStrategy(LLMCallStrategy[str]):
        def __init__(self, name: str):
            self.name = name

        async def prepare(self):
            prepare_counts[self.name] = prepare_counts.get(self.name, 0) + 1

        async def execute(self, prompt, attempt, timeout, state):
            return f"{self.name}: {prompt}", {"input_tokens": 10}

    strategy_a = TrackedStrategy("A")
    strategy_b = TrackedStrategy("B")
    strategy_c = TrackedStrategy("C")

    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add items with different strategies
        for i in range(10):
            await processor.add_work(LLMWorkItem(
                item_id=f"a_{i}", strategy=strategy_a, prompt="Test"
            ))
        for i in range(10):
            await processor.add_work(LLMWorkItem(
                item_id=f"b_{i}", strategy=strategy_b, prompt="Test"
            ))
        for i in range(10):
            await processor.add_work(LLMWorkItem(
                item_id=f"c_{i}", strategy=strategy_c, prompt="Test"
            ))

        result = await processor.process_all()

    assert result.total_items == 30
    assert result.succeeded == 30
    assert prepare_counts["A"] == 1
    assert prepare_counts["B"] == 1
    assert prepare_counts["C"] == 1
```

**Validation:**

- All tests pass with 100% success rate
- Run with `pytest tests/test_shared_strategy_lifecycle.py -v`
- Add to CI pipeline

**Related Issues:** #1 (from v0.2.0)
**Documentation:** Update README with test results

---

### 1.3 Document Shared Strategy Cost Optimization

**Priority:** P0 - CRITICAL
**Effort:** 1-2 hours
**Owner:** TBD

**Problem:**

- Users might create new strategy instances per item
- Miss 70-90% cost savings with Gemini caching
- Not prominently documented in README

**Implementation:**

#### 1.3.1 Update README.md

Add prominent section in "Quick Start" or "Features":

```markdown
## 💰 Critical: Shared Strategies for Cost Optimization

**IMPORTANT:** When using `GeminiCachedStrategy`, reuse the SAME strategy instance across all work items to share the cache. This provides **70-90% cost savings**.

### ❌ Wrong: New Strategy Per Item (Expensive!)

```python
# This creates a NEW cache for each item - wastes money!
for document in documents:
    strategy = GeminiCachedStrategy(...)  # NEW INSTANCE per loop
    work_item = LLMWorkItem(
        item_id=document.id,
        strategy=strategy,  # Different strategy each time
        prompt=format_prompt(document)
    )
    await processor.add_work(work_item)
```

**Cost:** 100 items × $0.10 per item = **$10.00**

### ✅ Right: Reuse Strategy (70-90% Savings!)

```python
# Create ONE strategy, reuse for all items
strategy = GeminiCachedStrategy(...)  # ONE INSTANCE

for document in documents:
    work_item = LLMWorkItem(
        item_id=document.id,
        strategy=strategy,  # REUSE same strategy
        prompt=format_prompt(document)
    )
    await processor.add_work(work_item)
```

**Cost:** 100 items × $0.03 per item (with caching) = **$3.00** ✅

**Savings: $7.00 (70%) on this batch alone!**

### How It Works

When you reuse a strategy:

1. Framework calls `prepare()` **once** (creates one cache)
2. All work items share that cache
3. Cached tokens get 90% discount from Gemini
4. Overall cost reduction: 70-90%

See the Shared Strategies documentation for details.

```text
(end of markdown code block)
```

#### 1.3.2 Update GeminiCachedStrategy Docstring

Add to `src/batch_llm/llm_strategies.py`:

```python
class GeminiCachedStrategy(LLMCallStrategy[TOutput]):
    """
    Gemini strategy with automatic context caching (v0.2.0).

    **CRITICAL FOR COST OPTIMIZATION:**
    Create ONE instance and reuse it across all work items to share the cache.
    This provides 70-90% cost savings compared to creating new instances per item.

    Example (CORRECT usage):
        >>> # Create one strategy
        >>> strategy = GeminiCachedStrategy(
        ...     model="gemini-2.0-flash",
        ...     client=client,
        ...     response_parser=lambda r: str(r.text),
        ...     cached_content=[system_instruction, context_docs],
        ... )
        >>>
        >>> # Reuse for all items
        >>> for doc in documents:
        ...     work_item = LLMWorkItem(strategy=strategy, ...)  # REUSE
        ...     await processor.add_work(work_item)

    Wrong usage (creates new cache per item):
        >>> for doc in documents:
        ...     strategy = GeminiCachedStrategy(...)  # NEW instance - expensive!
        ...     work_item = LLMWorkItem(strategy=strategy, ...)

    Cost comparison (100 items with 500 cached tokens):
    - Wrong: $10.00 (no caching benefit)
    - Right: $3.00 (70% savings)
    """
```

#### 1.3.3 Add FAQ Entry

Add to README FAQ:

```markdown
### Why are my costs so high with GeminiCachedStrategy?

You're probably creating a new strategy instance for each work item:

```python
# ❌ WRONG - creates new cache per item
for item in items:
    strategy = GeminiCachedStrategy(...)  # NEW instance
    await processor.add_work(LLMWorkItem(strategy=strategy, ...))
```

Instead, create ONE strategy and reuse it:

```python
# ✅ RIGHT - shares cache across all items (70-90% savings)
strategy = GeminiCachedStrategy(...)  # ONE instance
for item in items:
    await processor.add_work(LLMWorkItem(strategy=strategy, ...))  # REUSE
```

See the Cost Optimization section for details.

```text
(end of markdown code block)
```

**Validation:**

- README section is visible and prominent
- Docstring shows up in IDE hover
- FAQ addresses common mistake

**Related Issues:** User confusion from production usage
**Documentation:** Update CHANGELOG.md noting documentation improvements

---

## Phase 2: High-Impact Improvements (Sprint 1)

**Target:** v0.3.1 or v0.3.2
**Goal:** Improve code quality, type safety, and test coverage

### 2.1 Reduce Type Ignores (21 → <10)

**Priority:** P1 - HIGH
**Effort:** 4-6 hours
**Owner:** TBD

**Problem:** 21 `# type: ignore` comments mask potential type safety issues

**Implementation:**

#### Step 1: Fix TokenUsage Default Factory

**File:** `src/batch_llm/base.py`

```python
# Before
token_usage: TokenUsage = field(default_factory=dict)  # type: ignore[assignment]

# After
def _empty_token_usage() -> TokenUsage:
    """Create an empty token usage dict with proper types."""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
    }

@dataclass
class WorkItemResult(Generic[TOutput, TContext]):
    token_usage: TokenUsage = field(default_factory=_empty_token_usage)
```

#### Step 2: Fix Optional Import Type Ignores

**File:** `src/batch_llm/llm_strategies.py`

```python
# Before
try:
    from pydantic_ai import Agent
except ImportError:
    Agent = Any  # type: ignore[misc,assignment]

# After
try:
    from pydantic_ai import Agent as PydanticAgent
    Agent = PydanticAgent
except ImportError:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from pydantic_ai import Agent
    else:
        Agent = Any  # Only ignore when not type checking
```

#### Step 3: Fix Google API Type Stubs

**File:** `src/batch_llm/llm_strategies.py`

For incomplete third-party type stubs, use `cast()`:

```python
# Before
self._cache = await self.client.aio.caches.create(...)  # type: ignore[call-arg]

# After
from typing import cast
self._cache = cast(
    "CachedContent",
    await self.client.aio.caches.create(...)
)
```

**Target:** Reduce from 21 to <10 type ignores
**Validation:** Run `mypy src/batch_llm/ --ignore-missing-imports`

---

### 2.2 Fix Line Length Violations

**Priority:** P1 - HIGH
**Effort:** 1 hour
**Owner:** TBD

**Files:** `base.py`, `parallel.py`, others

**Implementation:**

Run ruff and fix all E501 violations:

```bash
uv run ruff check src/ tests/ --select E501 --fix
```

Manual fixes for complex cases:

```python
# Before (113 chars)
f"item_id must be a non-empty string (got {type(self.item_id).__name__}: {repr(self.item_id)}). "

# After
raise ValueError(
    f"item_id must be a non-empty string "
    f"(got {type(self.item_id).__name__}: {repr(self.item_id)}). "
    f"Provide a unique string identifier for this work item."
)
```

**Validation:** `ruff check src/ tests/` shows 0 E501 violations

---

### 2.3 Add Resource Cleanup for Failed prepare()

**Priority:** P1 - HIGH
**Effort:** 1 hour
**Owner:** TBD

**File:** `src/batch_llm/parallel.py:625-730`

**Implementation:**

```python
async def _process_item_with_retries(
    self, work_item: LLMWorkItem[TInput, TOutput, TContext], worker_id: int
) -> WorkItemResult[TOutput, TContext]:
    """Wrapper that applies retry logic and strategy lifecycle."""
    strategy = self._get_strategy(work_item)
    retry_state = RetryState()

    # Track if prepare succeeded for cleanup decision
    prepare_succeeded = False

    try:
        # Ensure strategy is prepared
        await self._ensure_strategy_prepared(strategy)
        prepare_succeeded = True

        # Process with retries
        for attempt in range(1, self.config.retry.max_attempts + 1):
            try:
                return await self._process_item(
                    work_item, worker_id, attempt_number=attempt,
                    strategy=strategy, retry_state=retry_state
                )
            except Exception as e:
                # ... existing retry logic ...

    except Exception as e:
        # ... existing error handling ...
        raise

    finally:
        # Always cleanup, even if prepare failed
        if strategy is not None:
            try:
                await strategy.cleanup()
            except Exception as cleanup_error:
                logger.warning(
                    f"Cleanup failed for {work_item.item_id}: {cleanup_error}"
                )
```

**Validation:** Test that cleanup() called even when prepare() fails

---

### 2.4 Document Worker Count Selection

**Priority:** P1 - HIGH
**Effort:** 1 hour
**Owner:** TBD

**Implementation:**

Add section to README:

```markdown
## Configuration Guide

### Choosing max_workers

The optimal worker count depends on your use case:

#### 1. Rate-Limited APIs (OpenAI, Anthropic, Gemini)

**Recommended:** 5-10 workers

```python
config = ProcessorConfig(max_workers=5)
```

**Why:** Too many workers hit rate limits immediately. Start with 5, increase gradually while monitoring
`rate_limit_count` in metrics.

**With proactive rate limiting:**

```python
config = ProcessorConfig(
    max_workers=4,  # Conservative: 300 req/min ÷ 60 × 0.8 buffer
    max_requests_per_minute=300,  # Set based on your API tier
)
```

#### 2. Unlimited APIs (Local Models, Self-Hosted)

**Recommended:** `min(cpu_count() * 2, 20)`

```python
import os
config = ProcessorConfig(max_workers=min(os.cpu_count() * 2, 20))
```

**Why:** CPU-bound operations benefit from more workers up to 2× CPU count. Cap at 20 to avoid diminishing returns.

#### 3. Testing and Debugging

**Recommended:** 2 workers

```python
config = ProcessorConfig(max_workers=2)
```

**Why:** Easier to debug with less concurrency. Logs are easier to follow.

#### 4. Monitoring and Tuning

Monitor these metrics to optimize:

```python
result = await processor.process_all()
stats = await processor.get_stats()

print(f"Rate limits hit: {stats['rate_limit_count']}")
print(f"Items/sec: {result.total_items / stats['duration']:.2f}")
```

- **If rate_limit_count > 0:** Reduce workers or enable proactive rate limiting
- **If items/sec is low:** Check if workers are idle (increase count)
- **If errors spike:** Check logs for timeout or validation issues

```text
(end of markdown code block)
```

**Validation:** User feedback on clarity

---

### 2.5 Add Tests for Cache Tag Matching (v0.3.0)

**Priority:** P1 - HIGH
**Effort:** 2 hours
**Owner:** TBD

**Implementation:**

Add to `tests/test_v0_3_features.py` or create `tests/test_cache_tags.py`:

```python
@pytest.mark.asyncio
async def test_cache_tags_matching_real_caches():
    """Test that cache tags correctly filter cache matches."""
    try:
        from batch_llm.llm_strategies import GeminiCachedStrategy
        import google.genai as genai
        from google.genai.types import Content
    except ImportError:
        pytest.skip("google-genai not installed")

    # This test uses actual cache creation logic (no API calls)
    # Tests the matching algorithm in _find_or_create_cache()

    # Mock cache objects
    class MockCache:
        def __init__(self, model: str, metadata: dict):
            self.model = model
            self.metadata = metadata or {}
            self.name = f"cache_{model}_{id(self)}"

    # Test: Matching tags should reuse cache
    strategy_a = GeminiCachedStrategy(
        model="gemini-2.0-flash",
        client=None,  # Won't actually call API
        response_parser=lambda x: str(x),
        cached_content=[],
        cache_tags={"customer": "acme", "version": "v1"}
    )

    # Simulate existing cache with matching tags
    existing_cache = MockCache(
        model="gemini-2.0-flash",
        metadata={"customer": "acme", "version": "v1"}
    )

    # Test matching logic
    tags_match = all(
        existing_cache.metadata.get(k) == v
        for k, v in strategy_a.cache_tags.items()
    )
    assert tags_match, "Tags should match"

    # Test: Different tags should NOT reuse cache
    existing_cache_b = MockCache(
        model="gemini-2.0-flash",
        metadata={"customer": "globex", "version": "v1"}
    )

    tags_match_b = all(
        existing_cache_b.metadata.get(k) == v
        for k, v in strategy_a.cache_tags.items()
    )
    assert not tags_match_b, "Tags should NOT match"
```

**Validation:** Tests pass, cover all tag matching scenarios

---

## Phase 3: Performance & Polish (Sprint 2)

**Target:** v0.4.0
**Goal:** Optimize performance and improve reliability

### 3.1 Optimize Slow-Start Calculations

**Priority:** P2 - MEDIUM
**Effort:** 2-3 hours
**Impact:** 5-10% performance improvement

**File:** `src/batch_llm/parallel.py:300-314`

**Implementation:**

```python
class ParallelBatchProcessor:
    def __init__(self, config: ProcessorConfig, ...):
        # ... existing init ...

        # Pre-calculate slow-start delays (v0.4.0 optimization)
        self._slow_start_delays: list[float] = []
        if self.rate_limit_strategy.slow_start_items > 0:
            for i in range(self.rate_limit_strategy.slow_start_items):
                _, delay = self.rate_limit_strategy.should_apply_slow_start(i)
                self._slow_start_delays.append(delay)

    async def _worker(self, worker_id: int) -> None:
        """Worker that processes items from queue."""
        while True:
            # ... get work_item ...

            # Optimized slow-start (v0.4.0)
            should_delay = False
            delay = 0.0

            async with self._rate_limit_lock:
                if self._slow_start_active:
                    idx = self._items_since_resume
                    if idx < len(self._slow_start_delays):
                        delay = self._slow_start_delays[idx]
                        should_delay = True
                        self._items_since_resume += 1
                    else:
                        self._slow_start_active = False

            if should_delay:
                await asyncio.sleep(delay)

            # ... process item ...
```

**Validation:**

- Benchmark with 20 workers, 1000 items
- Measure lock contention reduction

---

### 3.2 Add Debug Logging to Token Extraction

**Priority:** P2 - MEDIUM
**Effort:** 30 minutes

**File:** `src/batch_llm/parallel.py:550-608`

**Implementation:**

```python
def _extract_token_usage(self, exception: Exception) -> TokenUsage:
    """Extract token usage from exception chain."""
    try:
        # ... existing extraction logic ...
    except Exception as e:
        logger.debug(
            f"Failed to extract token usage from {type(exception).__name__}: {e}. "
            "Returning 0 tokens. This is normal for non-LLM exceptions."
        )

    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
```

---

### 3.3 Add Validation for Cache Parameters

**Priority:** P2 - MEDIUM
**Effort:** 30 minutes

**File:** `src/batch_llm/llm_strategies.py:356-406`

**Implementation:**

```python
def __init__(
    self,
    model: str,
    # ... other params ...
    cache_ttl_seconds: int = 3600,
    cache_renewal_buffer_seconds: int = 300,
    # ... other params ...
):
    # Validate buffer < ttl
    if cache_renewal_buffer_seconds >= cache_ttl_seconds:
        raise ValueError(
            f"cache_renewal_buffer_seconds ({cache_renewal_buffer_seconds}) "
            f"must be less than cache_ttl_seconds ({cache_ttl_seconds}). "
            f"Typical value: 5-10 minutes (300-600 seconds)."
        )

    # ... rest of init ...
```

---

### 3.4 Improve Error Context

**Priority:** P2 - MEDIUM
**Effort:** 1 hour

**File:** `src/batch_llm/strategies/errors.py`

**Implementation:**

```python
class FrameworkTimeoutError(TimeoutError):
    """Raised when framework-level timeout is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        item_id: str | None = None,
        elapsed: float | None = None,
        timeout_limit: float | None = None,
    ):
        super().__init__(message)
        self.item_id = item_id
        self.elapsed = elapsed
        self.timeout_limit = timeout_limit

# Usage in parallel.py:
raise FrameworkTimeoutError(
    f"Framework timeout after {elapsed:.1f}s (limit: {timeout:.1f}s)",
    item_id=work_item.item_id,
    elapsed=elapsed,
    timeout_limit=timeout,
) from timeout_exc
```

---

### 3.5 Improve RetryState API Consistency

**Priority:** P2 - MEDIUM
**Effort:** 30 minutes

**File:** `src/batch_llm/base.py:24-100`

**Implementation:**

```python
def delete(self, key: str, raise_if_missing: bool = False) -> None:
    """
    Delete a value from the state.

    Args:
        key: State key to delete
        raise_if_missing: If True, raise KeyError if key doesn't exist.
                         If False, silently ignore missing keys.

    Raises:
        KeyError: If key doesn't exist and raise_if_missing=True
    """
    if raise_if_missing:
        del self.data[key]
    else:
        self.data.pop(key, None)
```

---

## Phase 4: Documentation & Testing ✅ COMPLETED

**Target:** v0.4.x (achieved in v0.3.x)
**Goal:** Comprehensive documentation and test coverage

**Status:** All tasks completed (2025-01-10)

### 4.1 Add Missing Docstrings ✅

**Status:** COMPLETED
**Priority:** P3 - LOW
**Effort:** 2-3 hours

**Files:** Multiple

**Completion Notes:**

- Added docstrings to all missing methods identified by `pydocstyle`
- Methods documented: `MetricsObserver.reset()`, `ProcessorConfig.validate()`, `RetryState.__contains__()`

---

### 4.2 Create Migration Guides ✅

**Status:** COMPLETED (Commit: 791a1e0)
**Priority:** P2 - MEDIUM
**Effort:** 2-3 hours

**Completion Notes:**

- Created `docs/MIGRATION.md` as comprehensive migration guide index
- Links all existing migration guides: v0.0→v0.1, v0.1→v0.2, v0.2→v0.3
- Added "Migration Path Finder" for users upgrading from any version
- Updated README.md to link to new migration index
- Includes version history table and deprecation policy

---

### 4.3 Add Error Handling to Examples ✅

**Status:** COMPLETED
**Priority:** P3 - LOW
**Effort:** 1 hour

**Files:** `examples/*.py`

**Completion Notes:**

- Added API key checks to all example files
- Examples now provide helpful error messages when API keys are missing
- Includes links to where users can get API keys

---

### 4.4 Add Integration Tests ✅

**Status:** COMPLETED (Commit: 94d65f6)
**Priority:** P2 - MEDIUM
**Effort:** 4-6 hours

**File:** `tests/test_integration.py`

**Completion Notes:**

- Created 7 integration tests with real API calls
- Tests for: Gemini basic generation, cached strategy, safety ratings, validation retries
- Tests automatically skip when API keys are not present
- Placeholder tests for future OpenAI/Anthropic implementations
- Added `integration` marker to pyproject.toml
- Tests deselected by default, run with: `pytest -m integration -v`

---

### 4.5 Add Performance Benchmarks ✅

**Status:** COMPLETED (Commit: 94d65f6)
**Priority:** P3 - LOW
**Effort:** 4-6 hours

**File:** `tests/test_performance.py`

**Completion Notes:**

- Created 8 performance benchmark tests
- Benchmarks: throughput (single/scaling), memory usage, framework overhead, stats performance
- Tests compare shared vs unique strategy performance
- All benchmarks print detailed results and verify performance expectations
- Added `benchmark` marker to pyproject.toml
- Tests deselected by default, run with: `pytest -m benchmark -v -s`

---

### 4.6 Add Worst-Case Rate Limit Test ✅

**Status:** COMPLETED
**Priority:** P2 - MEDIUM
**Effort:** 2 hours

**File:** `tests/test_worst_case_rate_limit.py`

**Completion Notes:**

- Created comprehensive worst-case rate limit tests
- Tests all workers hitting rate limit simultaneously using barriers
- Verifies coordination and no deadlocks occur
- Includes tests for cache expiration thread safety with multiple workers
- Tests run with standard pytest, no special marker needed

---

## Phase 5: User Experience (Future)

**Target:** v0.5.0
**Goal:** Convenience features and polish

### 5.1 Add Built-in Progress Bar

**Priority:** P3 - MEDIUM
**Effort:** 4-6 hours

**Implementation:** Optional integration with `rich` or `tqdm`

```python
from batch_llm.progress import RichProgressBar

processor = ParallelBatchProcessor(
    config=config,
    progress_callback=RichProgressBar(),  # Optional convenience
)
```

---

### 5.2 Improve Dry-Run Documentation

**Priority:** P3 - LOW
**Effort:** 1 hour

Add example to README showing dry-run testing

---

### 5.3 Add Input Validation

**Priority:** P2 - MEDIUM
**Effort:** 2 hours

**File:** `src/batch_llm/base.py:145-157`

```python
def __post_init__(self):
    """Validate work item fields."""
    if not self.item_id or not isinstance(self.item_id, str):
        raise ValueError(...)

    if self.strategy is None:
        raise ValueError(
            f"strategy must not be None for {self.item_id}. "
            f"Provide an LLMCallStrategy instance."
        )

    if len(self.prompt) > 1_000_000:  # 1MB limit
        logger.warning(
            f"Very large prompt for {self.item_id}: {len(self.prompt)} chars. "
            f"May exceed model context limits."
        )
```

---

## Code Quality Improvements

### 6.1 Consistent Error Message Truncation

**Priority:** P3 - LOW
**Effort:** 30 minutes

**Implementation:**

```python
# Add constant
ERROR_MESSAGE_MAX_LENGTH = 200

# Use throughout:
error = f"{type(e).__name__}: {str(e)[:ERROR_MESSAGE_MAX_LENGTH]}"
```

---

### 6.2 Document Thread Safety

**Priority:** P3 - LOW
**Effort:** 1 hour

Add comments documenting thread-safety assumptions in:

- `_is_cache_expired()`
- `_progress_tasks` cleanup

---

## Tracking and Metrics

### Success Criteria

**Phase 1 (Critical):**

- ✅ Version mismatch fixed
- ✅ Shared strategy tests added (3+ tests)
- ✅ Cost optimization documented prominently

**Phase 2 (High Impact):**

- ✅ Type ignores reduced to <10
- ✅ Line length violations: 0
- ✅ Resource cleanup added
- ✅ Worker count guide added
- ✅ Cache tag tests added

**Phase 3 (Performance):**

- ✅ Slow-start optimized (5-10% improvement)
- ✅ Debug logging added
- ✅ Parameter validation added

### Progress Tracking

Track progress in GitHub Issues:

- Tag issues with phase labels: `phase-1-critical`, `phase-2-high-impact`, etc.
- Tag issues with type labels: `bug`, `enhancement`, `documentation`, `testing`
- Use milestones: `v0.3.1`, `v0.4.0`, etc.

### Review Cadence

- **Weekly:** Review Phase 1 progress
- **Bi-weekly:** Review Phases 2-3 progress
- **Monthly:** Review backlog prioritization

---

## Appendix: Full Issue List

### By Priority

**P0 - Critical (3 issues):**

1. Fix version mismatch
2. Add shared strategy tests
3. Document cost optimization

**P1 - High (5 issues):**
4. Reduce type ignores
5. Fix line length violations
6. Add resource cleanup
7. Document worker count
8. Add cache tag tests

**P2 - Medium (11 issues):**
9. Optimize slow-start
10. Add debug logging
11. Add parameter validation
12. Improve error context
13. Improve RetryState API
14. Add migration guides
15. Add integration tests
16. Add worst-case rate limit test
17. Add input validation
18. Others...

**P3 - Low (8 issues):**
19. Add docstrings
20. Add example error handling
21. Performance benchmarks
22. Progress bar integration
23. Consistent error truncation
24. Others...

---

## Change Log

**2025-01-10:** Initial plan created based on package review

---

## References

- Package Review: `PACKAGE_REVIEW_2025_01_10.md`
- v0.2.0 Features: `CHANGELOG.md`
- v0.3.0 Features: `docs/IMPLEMENTATION_PLAN_V0_3.md`
- Architecture Notes: `CLAUDE.md`
