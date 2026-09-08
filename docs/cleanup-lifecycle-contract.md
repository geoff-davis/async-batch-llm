# Cleanup and stream-terminal contract

This is the behavioral specification for resource cleanup and stream
finalization in `ParallelBatchProcessor`, `LLMGateway`, `ExecutorHost`, and
the high-level `call_result()` / `process_prompts()` / `process_stream()`
surfaces. `tests/test_cleanup_contract.py` holds one executable test per
clause; a change to this document without a matching test change is a bug.

The implementation lives in `src/async_batch_llm/_internal/cleanup.py`.

## 1. Timing and ordering

- There is no global cleanup deadline, no per-step time budget, and no
  cleanup timeout exception. Cleanup proceeds sequentially through dependency
  phases: runtime tasks and callbacks, then admission resources, then
  prepared strategies, then the artifact store.
- A completed ordinary failure in one step does not prevent sibling steps
  from being attempted.
- A still-running resource is an ordering barrier. A dependent resource is
  never closed while the resource it depends on is still running.
- Resource phases are discovered after preceding barriers finish, including
  strategies whose preparation started or completed during the gateway drain.
  If a private barrier wait is interrupted, dependent steps wait for the next
  explicit close; an ordinary failure reported by an already-settled owned
  task does not itself leave a live barrier.
- The gateway in-flight drain and the artifact-store close run to completion.
- Batch `process_all()` leaves resource teardown, including artifact-store
  close, to context exit or an explicit close. Streaming finalization runs
  the ordered close before deciding its terminal, including waiting for
  admission callbacks and closing strategies before the store.
- `WORKER_CANCELLATION_TIMEOUT` and `PROGRESS_TASK_CANCELLATION_TIMEOUT`
  (two seconds each) are diagnostic thresholds only: one warning is logged
  when they elapse, the wait continues, and no strategy or store cleanup and
  no clean end-of-stream happens while a worker, progress callback, or
  callback thread is still alive.
- Any cleanup step still running after `CLEANUP_SLOW_WARNING_SECONDS` logs
  one warning and continues unchanged.

## 2. Cancellation deliveries

Cancellation of the task that is closing a resource is counted explicitly.

- A cancellation that is already propagating when cleanup is entered (a
  `finally` or `__aexit__` reached by cancellation) counts as the first
  delivery.
- The first delivery is deferred: teardown finishes in order, the
  cancellation stays primary over ordinary cleanup errors (which are logged),
  and the original cancellation is re-raised after teardown completes.
- The second delivery is a forced abort: the current private cleanup
  operation is cancelled, no further steps start, dependent resources are not
  closed, the abandoned step and the skipped steps are logged, and the second
  cancellation propagates immediately. The forced abort deliberately
  sacrifices durability. It is the operator's escape from a cleanup callback
  that never returns.

## 3. Portable cancellation classification

- Each cleanup step runs in a private task and is awaited through a detached
  completion future, so the awaiting task's own cancellation is never linked
  to the step task. A `CancelledError` that reaches the awaiting task is
  therefore always a caller cancellation.
- A `CancelledError` raised inside a step, or a step task cancelled by a
  third party before it ran, is a cleanup-step interruption. It is reported
  as `CleanupInterruptedError` (an ordinary `Exception`), the step is not
  checkpointed, and a later explicit close retries it.
- `Task.cancelling()`, `Task.uncancel()`, and exception attributes are not
  used. Behavior is identical on every supported Python version.
- Owned quota wakes remain tracked across rescheduling and unsolicited
  failures until shutdown observes their outcomes. A concurrently set stop
  event never relabels an already-cancelled private child as owner-cancelled.
  A lost quota wake schedules a replacement so reservations continue. An
  externally cancelled cooldown releases waiting workers if finalization has
  not yet started, including cancellation before the owned task starts;
  shutdown still reports the interruption. An interrupted finalization is
  retried only by a later explicit close.

## 4. Retry and idempotency

- A cleanup step that completed successfully is checkpointed and never
  repeated by a later close.
- A failed or interrupted step is retried only by a later explicit close
  call, never repeatedly within one call.
- Automatic context or convenience-API exit does not retry a close already
  attempted by stream finalization. It applies that attempt's report even if
  the consumer never read the terminal, then releases the pending report.
  The reported stream terminal remains unchanged if a later explicit close
  successfully retries failed steps.
- Concurrent close calls on one owner share the same in-flight attempt.
- User-defined `cleanup()` methods must be idempotent and safe to call after
  a partially completed earlier attempt, because the framework cannot
  checkpoint operations inside a user callback.
- Failed outcomes and their tracebacks are not retained after the close call
  that reported them returns.
- Lifecycle state is one-way: open, then closing, then closed. A failed or
  aborted close stays closing; a later close resumes the unfinished steps.
  Strategy closing starts after admitted runtime work drains; preparation
  needed by that work is allowed during draining. Once the strategy phase is
  closing, new strategy preparation is rejected.

## 5. Stream finalization

- The streaming end-of-stream is published only after successful
  finalization, never from an unconditional `finally`.
- The stream has exactly one terminal decision, made once. A worker crash,
  a finalization failure, a cancelled finalizer, or a shutdown before
  `finish()` all decide a failure terminal; a completed finalization decides
  success. The first decision wins.
- The terminal is durable: it survives `cleanup()`, later results that were
  published after the decision, repeated `results()` calls, and concurrent
  consumers. A consumer may receive already-queued results first but always
  ends by raising the original failure and never observes clean completion
  after a failure.
- An ordinary failure is re-raised as the original exception instance. A
  cancelled finalization is raised as `StreamFinalizationError` whose
  `__cause__` is the cancellation, because the consumer itself was not
  cancelled. `KeyboardInterrupt` and `SystemExit` are never converted; the
  terminal records the failure for any consumer and the finalizer re-raises
  them so they retain their type.
- A live worker, post-processor, progress callback, or callback thread
  prevents successful finalization. Synchronous progress and post-processing
  use separate owned pools so slow progress callbacks cannot starve writes.

## 6. Precedence and changelog

- When a close call has no pre-existing exception, the first ordinary cleanup
  failure is raised after every step has been attempted; later failures are
  logged with tracebacks.
- A body exception or a caller cancellation is never replaced by an ordinary
  cleanup error. `KeyboardInterrupt` and `SystemExit` raised by a cleanup
  step take precedence over a deferred cancellation and retain their type.
- High-level convenience APIs preserve a completed result when the only
  cleanup failures came from user strategy `cleanup()`; runtime, admission,
  and artifact-store failures propagate.
- Breaking behavior changes are recorded in `CHANGELOG.md` with the
  repository's inline `**BREAKING**` marker.
