# Observers API Reference

## ProcessorObserver

::: async_batch_llm.observers.ProcessorObserver

## BaseObserver

::: async_batch_llm.observers.BaseObserver

## MetricsObserver

::: async_batch_llm.observers.MetricsObserver

## Event contracts

Interface for observers that monitor processing events.

```python
class ProcessorObserver(ABC):
    @abstractmethod
    async def on_event(
        self, event: ProcessingEvent, data: dict[str, Any]
    ) -> None: ...
```

**Events:**

- `BATCH_STARTED`: `{total, max_workers, start_time}`
- `BATCH_COMPLETED`: `{processed, succeeded, failed, total, total_tokens,
  cached_input_tokens, total_admission_wait_seconds,
  max_admission_wait_seconds, total_quota_wait_seconds,
  max_quota_wait_seconds, quota_wait_p50/p95/p99_seconds,
  estimated_input_tokens, estimated_output_tokens, reserved_tokens,
  reported_reconciliation_tokens, refunded_tokens, underestimated_tokens,
  unknown_usage_attempts, known_zero_usage_attempts,
  token_estimation_failures, quota_scope_count,
  admission_wait_p50/p95/p99_seconds,
  execution_p50/p95/p99_seconds, structured_output_recoveries,
  structured_output_retries_avoided, structured_output_recovery_reasons,
  duration}`
- `WORKER_STARTED` / `WORKER_STOPPED`: `{worker_id}`
- `ITEM_STARTED`: `{item_id, worker_id}`
- `ITEM_ADMITTED`: `{item_id, worker_id, attempt, wait_seconds, capacity,
  startup_ramp_wait_seconds}`
- `QUOTA_ADMITTED`: `{item_id, worker_id, attempt, try_number,
  quota_scope_id, wait_seconds, request_reserved, estimated_input_tokens,
  estimated_output_tokens, estimated_total_tokens, reserved_tokens,
  rpm_configured, tpm_configured, limited_by}`
- `QUOTA_RECONCILED`: `{item_id, worker_id, attempt, try_number,
  quota_scope_id, reserved_tokens, reported_tokens, known_usage,
  delta_tokens, disposition}`
- `ITEM_COMPLETED`: `{item_id, duration, tokens, admission_wait_seconds,
  structured_output_recovered, structured_output_recovery_reason,
  structured_output_retries_avoided}`
- `ITEM_FAILED`: `{item_id, submission_index, error_type, error_category}`
- `ITEM_REPLAYED`: `{item_id, submission_index, success, error_type, error_category}`
- `RATE_LIMIT_HIT`: `{item_id, worker_id}`
- `COOLDOWN_STARTED`: `{worker_id, duration, consecutive}`
- `COOLDOWN_ENDED`: `{duration, error?}`

`add_work()` raises public `BatchAdmissionClosedError` (a `RuntimeError`) when
finish, shutdown, or batch abort has stopped admission. Rejected items receive
no submission index. Use a new processor for additional work.

Observers may be duck-typed objects with a callable `on_event(event, data)`;
subclassing `ProcessorObserver` is optional. The callback must return an awaitable.
Invalid callbacks are rejected at processor construction. Observer exceptions and
callback timeouts are logged without failing the item.

Each finalized item emits exactly one of `ITEM_COMPLETED`, `ITEM_FAILED`, or
`ITEM_REPLAYED`, after its checkpoint and before any fail-fast `BATCH_ABORTED` event.
A budget stop is different: the provider attempt that reaches the budget emits
`BATCH_ABORTED` (`kind="budget_exceeded"`) before that item's own terminal event.
Middleware-filtered items count as failed; middleware recovery and `after_process`
changes determine the final event. Fatal checkpoint failures and cancellation before
finalization produce no terminal item event or processed-stat increment. Early
shutdown may abandon accepted items, so processed counts can be lower than total.
Replay does not emit an additional success/failure event or consume live tokens.
For live provider successes, `duration` and `tokens` retain final-attempt values;
new recovery-only success events use the final result's timing and token totals.

**Cleanup note:**

- Preferred: wrap `ParallelBatchProcessor` in `async with` so the ordered close runs automatically.
- If you do not use a context manager, call `await processor.shutdown()` after `process_all()`. It
  stops workers and callbacks, releases admission state, runs strategy cleanups, and closes the
  artifact store. Repeated calls are safe and retry only what did not complete.

---
