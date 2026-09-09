# Review context: settled decisions and scope

Paste the relevant parts of this file into a review agent's prompt. It has
no memory across rounds and will otherwise re-report accepted trade-offs as
defects. See "Review protocol" in `CLAUDE.md` for how the loop runs.

Prune this file at each release: it describes decisions in flight, not
permanent architecture. Anything that outlives the release belongs in
`CLAUDE.md`, `CHANGELOG.md`, or `docs/`.

## Settled for v0.23.1 — do not re-litigate

Each of these was raised, examined, and deliberately kept. Report a *new*
consequence if you find one; don't re-report the decision itself.

- **Batch `process_all()` leaves the artifact store open** until context
  exit or an explicit `shutdown()` / `cleanup()`. Declared BREAKING, so
  strategy teardown always precedes store close. Every documented usage
  goes through `process_prompts` / `process_stream`, which close in a
  `finally`.
- **There is no cleanup deadline.** `WORKER_CANCELLATION_TIMEOUT` and
  `PROGRESS_TASK_CANCELLATION_TIMEOUT` are diagnostic: one warning, then
  the wait continues. A worker that swallows cancellation blocks teardown.
  The escape hatch is a second cancellation of the closing caller, which
  force-aborts; a single `wait_for` or one Ctrl-C deliberately does not.
- **Synchronous callbacks are an ordering barrier.** `post_processor_timeout`
  and `progress_callback_timeout` stop the *waiter*, not the thread; close
  joins the callback pools.
- **Owned tasks are never cancelled by their owner.** The rate-limit
  cooldown and quota wake are stopped with an event and joined through a
  detached future. A cancelled owned task is therefore a third party's by
  construction, and is reported as `CleanupInterruptedError` at close —
  including after a fully successful batch. This is what makes the
  classification portable without `Task.cancelling()`.
- **`Task.cancelling()`, `uncancel()`, and exception attributes are banned.**
  Behavior must be identical on Python 3.10 and 3.13.
- **`before_process` runs once per logical item, including items that then
  replay**; `after_process` runs only on a newly executed success. Filtering
  therefore always wins over a historical success.
- **Artifact identity and input-serialization errors are per-item failures**
  (`artifact_preparation_error`), not batch-fatal. Fail-fast is opt-in via
  `GuardrailConfig.abort_on_error_categories`.
- **Abort and deadline terminals are checkpointed for audit but are never
  replay eligible**, including under `REUSE_ALL`, and legacy records in
  those categories are excluded on read.
- **Only batch-abort and batch-deadline audit writes are best-effort.**
  Their artifact preparation/append errors are logged. Per-item deadline
  checkpoint errors still raise, as do ordinary execution checkpoint errors.
  Audit preparation and append have no guardrail timeout; draining an abort
  may write one row per queued item and can block on a stalled store.
- **Capacity warnings are attributed to the submitting caller.** A filter
  targeting `module="async_batch_llm.*"` no longer matches them; match the
  caller's module or the message instead. Streaming captures the consumer's
  location before spawning the producer; module registries are created only
  when issuing a warning, preserving standard filtering and deduplication.

## Current review handoff

See `reports/v0.23.1_review_handoff.md` for the frozen review target, base
revision, finding-to-test mapping, policy matrix, and validation evidence.
The later fixes were layered on uncommitted work without saved pre-fix
snapshots; the handoff explicitly distinguishes passing regressions from
unavailable fail-first evidence. Do not stash the live workspace to review it.

## Out of scope

- The exception side channels `_abl_admission_wait_seconds`,
  `_abl_work_item_timing`, and `_abl_error_info`. Deferred by decision.
- `async_batch_llm_v0.23.1_v0.24_codex_plan.md`, which is intentionally
  untracked and not part of the branch.

## Weak evidence

Findings supported only by these have not held up:

- A passing suite. It says nothing about an uncovered path.
- Reading a diff without running it. Half of the "regressions" raised this
  way were pre-existing; run the same probe against the parent commit.
- One artifact backend. `JsonlArtifactStore` and `SqliteArtifactStore` have
  diverged more than once; check both whenever replay, record eligibility,
  or lookup changes.
