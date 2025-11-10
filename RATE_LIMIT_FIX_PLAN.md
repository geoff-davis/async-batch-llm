# Rate Limit Coordination Fix - Detailed Implementation Plan

## Problem Summary

When multiple workers hit rate limits simultaneously, the current implementation deadlocks. The core issue is complex state management across concurrent workers during rate limit cooldown cycles.

## Root Causes Identified

1. **Item Re-queueing Issue**: When a worker hits a rate limit, it re-queues the item BEFORE waiting for cooldown. This means:
   - Multiple workers can re-queue the SAME item multiple times
   - Queue never empties even after cooldown completes
   - Workers keep picking up duplicate work

2. **Event State Race Conditions**:
   - Worker A completes cooldown, sets event
   - Worker A immediately picks up new work and starts processing
   - Worker B arrives late, clears event for NEW cooldown
   - Worker A is now processing with event cleared → deadlock when it needs to wait

3. **Generation Counter Limitations**:
   - Current implementation tracks generations but doesn't prevent item duplication
   - Workers still re-queue items causing queue to never drain
   - The synchronization problem is correct but the work distribution is broken

## Proposed Solution: Stateful Item Tracking

Instead of re-queueing items, track rate-limited items separately and redistribute them after cooldown.

### Architecture Overview

```
Current (broken):
Worker hits rate limit → Re-queue item → Call _handle_rate_limit() → Raise exception

Proposed (fixed):
Worker hits rate limit → Store item in _rate_limited_items → Call _handle_rate_limit() → Return from worker loop
After cooldown: Coordinator redistributes all _rate_limited_items back to queue
```

### Key Changes

1. **Add Rate-Limited Item Storage**
   ```python
   self._rate_limited_items: list[LLMWorkItem] = []  # Protected by _rate_limit_lock
   ```

2. **Modify Rate Limit Detection** (in `_handle_exception`):
   ```python
   if error_info.is_rate_limit:
       # DON'T re-queue here - store for later redistribution
       async with self._rate_limit_lock:
           self._rate_limited_items.append(work_item)

       # Coordinate cooldown (one worker becomes coordinator)
       await self._handle_rate_limit(worker_id)

       # Return None to signal "item handled, continue loop"
       return None  # Instead of raising exception
   ```

3. **Coordinator Redistributes After Cooldown** (in `_finalize_cooldown`):
   ```python
   async def _finalize_cooldown(self, start_time: float, error: Exception | None) -> None:
       actual_duration = max(0.0, time.time() - start_time)

       async with self._rate_limit_lock:
           self._items_since_resume = 0
           self._in_cooldown = False

           # Redistribute rate-limited items back to queue
           items_to_redistribute = self._rate_limited_items.copy()
           self._rate_limited_items.clear()

           self._rate_limit_event.set()  # Resume all workers

       # Re-queue outside lock to avoid holding it during queue operations
       for item in items_to_redistribute:
           await self._queue.put(item)
   ```

4. **Update Worker Loop** (in `_worker`):
   ```python
   try:
       result = await self._process_item_with_retries(work_item, worker_id)
   except RateLimitHandled:
       # Item was stored in _rate_limited_items, will be redistributed after cooldown
       self._queue.task_done()
       continue
   except Exception as e:
       # ... existing error handling
   ```

5. **Simplify `_handle_rate_limit()`**:
   ```python
   async def _handle_rate_limit(self, worker_id: int):
       """Handle rate limit by pausing all workers and coordinating cooldown."""

       # Check if we're already in cooldown atomically
       async with self._rate_limit_lock:
           if self._in_cooldown:
               # Another worker is coordinating - just record we're waiting
               current_generation = self._cooldown_generation
               should_wait = True
           else:
               # We're the coordinator
               self._in_cooldown = True
               self._cooldown_generation += 1
               self._slow_start_active = True
               self._consecutive_rate_limits += 1
               self._rate_limit_event.clear()
               consecutive = self._consecutive_rate_limits
               generation = self._cooldown_generation
               should_wait = False

       # Wait for cooldown if we're not the coordinator
       if should_wait:
           await self._rate_limit_event.wait()
           return

       # We're the coordinator - perform cooldown
       # ... existing cooldown logic ...

       await self._finalize_cooldown(pause_started_at, cooldown_error)
   ```

## Implementation Steps

### Phase 1: Add Infrastructure (Low Risk)
1. Add `_rate_limited_items` list to `__init__`
2. Create `RateLimitHandled` exception class
3. Add tests to verify list operations are thread-safe

### Phase 2: Modify Rate Limit Detection (Medium Risk)
1. Update `_handle_exception` to store items instead of re-queueing
2. Change exception handling to return None instead of raising
3. Update worker loop to handle None return value
4. Run existing rate limit tests to verify no regression

### Phase 3: Implement Redistribution (Medium Risk)
1. Add redistribution logic to `_finalize_cooldown`
2. Ensure items are re-queued outside lock
3. Add logging to track redistribution
4. Test with simple rate limit scenarios

### Phase 4: Cleanup & Validation (Low Risk)
1. Remove old re-queue code from `_handle_exception`
2. Simplify `_handle_rate_limit` (no more RateLimitException)
3. Update comments and docstrings
4. Run full test suite

### Phase 5: Enable Worst-Case Tests (Validation)
1. Remove `@pytest.mark.skip` from worst-case tests
2. Run tests with increased logging
3. Verify all three scenarios pass
4. Add metrics to track redistribution counts

## Alternative Approaches Considered

### Alternative 1: Semaphore-Based Rate Limiting
**Idea**: Use `asyncio.Semaphore` to limit concurrent requests instead of Event-based pausing.

**Pros**:
- Simpler state management
- No need for cooldown coordination

**Cons**:
- Doesn't handle "pause all workers" requirement
- Can't implement slow-start mechanism
- Doesn't prevent hitting rate limits (reactive vs proactive)

**Decision**: Not suitable - we need to pause ALL workers, not just limit concurrency.

### Alternative 2: Single Coordinator Worker
**Idea**: Designate one worker as "coordinator" that handles all rate-limited items.

**Pros**:
- No coordination needed between workers
- Clear ownership of rate-limited items

**Cons**:
- Coordinator becomes bottleneck
- Doesn't scale well
- Complex handoff protocol needed

**Decision**: Too complex, doesn't solve core issue.

### Alternative 3: Queue Deduplication
**Idea**: Deduplicate items in queue using item_id before processing.

**Pros**:
- Prevents duplicate processing
- Minimal changes to existing code

**Cons**:
- Queue modifications are expensive (O(n) search)
- Doesn't solve event state race conditions
- Band-aid solution, not addressing root cause

**Decision**: Not addressing the real problem.

## Why Stateful Tracking is Best

1. **Eliminates Duplication**: Items are stored once, redistributed once
2. **Clear Ownership**: Coordinator owns redistribution
3. **Atomic Operations**: All item storage/retrieval happens under lock
4. **Event Clarity**: Event is only cleared/set by coordinator
5. **Testable**: Can verify no items are lost or duplicated
6. **Backward Compatible**: Doesn't change public API

## Testing Strategy

### Unit Tests
1. Test rate-limited item storage under concurrent access
2. Test redistribution logic in isolation
3. Test coordinator selection (first worker to acquire lock)

### Integration Tests
1. Single worker rate limit (baseline)
2. Two workers hitting rate limit simultaneously
3. Ten workers hitting rate limit simultaneously ✅ (current worst-case test)
4. Cascading rate limits (multiple cooldown cycles)
5. Mixed success and rate limit scenarios

### Stress Tests
1. 100 items, 20 workers, 50% rate limit
2. Verify no items lost
3. Verify no items duplicated
4. Verify all workers resume after cooldown

## Metrics to Track

Add counters to monitor:
- `rate_limited_items_stored`: Items stored during rate limit
- `rate_limited_items_redistributed`: Items redistributed after cooldown
- `rate_limit_generations`: Total number of cooldown cycles
- `rate_limit_participants`: Workers that detected each rate limit

## Rollback Plan

If implementation fails:
1. Revert to previous commit (current generation counter approach)
2. Keep worst-case tests skipped
3. Document why approach didn't work in BATCH_LLM_FEEDBACK.md

## Success Criteria

1. ✅ All 151 existing tests pass
2. ✅ All 3 worst-case rate limit tests pass (currently skipped)
3. ✅ No items lost or duplicated (verified by test assertions)
4. ✅ Performance regression < 5% (measure with existing benchmarks)
5. ✅ Clean separation of concerns (coordinator vs waiters)

## Timeline Estimate

- Phase 1: 30 minutes (add infrastructure)
- Phase 2: 1 hour (modify rate limit detection)
- Phase 3: 1 hour (implement redistribution)
- Phase 4: 30 minutes (cleanup)
- Phase 5: 1 hour (enable and debug tests)

**Total: ~4 hours** for complete implementation and validation

## Open Questions

1. **Q**: Should we limit the size of `_rate_limited_items`?
   **A**: No - if we hit that many rate limits, we have bigger problems. Document as limitation.

2. **Q**: What if cooldown fails (exception in `_finalize_cooldown`)?
   **A**: Items would remain in `_rate_limited_items` forever. Need try/catch to always redistribute.

3. **Q**: Can multiple cooldowns overlap (cascading)?
   **A**: No - `_in_cooldown` flag prevents this. Second wave waits for first to complete.

4. **Q**: What about proactive rate limiting?
   **A**: This fix is orthogonal - proactive limits reduce rate limit frequency, but when they DO hit, this fix ensures proper handling.

## References

- Current implementation: `src/batch_llm/parallel.py` lines 458-548 (`_handle_rate_limit`)
- Rate limit detection: `src/batch_llm/parallel.py` lines 951-984 (`_handle_exception`)
- Worker loop: `src/batch_llm/parallel.py` lines 288-365 (`_worker`)
- Worst-case tests: `tests/test_worst_case_rate_limit.py`
- Investigation: `BATCH_LLM_FEEDBACK.md`
