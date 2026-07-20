# Deadlines and Fail-Fast Guardrails

Guardrails are opt-in. Their defaults preserve normal completion-order,
retry, streaming, token-accounting, and cancellation behavior.

```python
from async_batch_llm import AbortMode, GuardrailConfig, ProcessorConfig

config = ProcessorConfig(
    max_workers=20,
    attempt_timeout=30,
    guardrails=GuardrailConfig(
        total_timeout_per_item=180,
        batch_timeout=3600,
        abort_on_error_categories=frozenset({
            "authentication",
            "insufficient_balance",
        }),
        abort_mode=AbortMode.DRAIN_ACTIVE,
    ),
)
```

Timeouts must be finite and greater than zero. All deadline calculations use a
monotonic clock.

## Per-attempt timeout versus total item deadline

`ProcessorConfig.attempt_timeout` retains its existing meaning: it limits one
`strategy.execute()` attempt after provider-capacity admission. Retries can each
receive another attempt timeout.

`GuardrailConfig.total_timeout_per_item` is one end-to-end logical-item budget.
It starts when execution of an accepted item begins and includes:

- coordinated cooldown and startup-ramp waits;
- pre-execution middleware and strategy error callbacks;
- proactive rate-limiter waits;
- provider-capacity admission;
- every provider attempt;
- retry cooldowns and backoff.

Every wait is bounded by the remaining budget. A provider call receives the
lower of `attempt_timeout` and the remaining total budget, and no retry or new
provider attempt starts after expiry. Completed-attempt timing and token usage
remain on the terminal result.

Postprocessing and artifact persistence occur after the logical provider
execution pipeline and are not included in the total item deadline. They have
their own timeout/durability semantics. The queue wait before a worker picks up
an item is also outside the item deadline; use `batch_timeout` to bound the
whole run. In `LLMCallPool`, the pool semaphore wait is outside the executor's
item deadline; use `submit_timeout` when the caller needs one budget that also
includes pool admission.

Total expiry is terminal and non-retryable:

- exception: `ItemDeadlineExceeded`;
- `error_category`: `framework_total_item_timeout`.

The existing per-attempt framework timeout remains distinct as
`framework_execution_timeout`.

## Batch deadlines

`batch_timeout` starts when `process_all()` or streaming execution starts, not
when a processor object is constructed. At expiry, source consumption stops,
no new provider attempt starts, and every already accepted item receives one
terminal result. Work not yet pulled from an async source was never accepted
and is not materialized.

Queued or interrupted collateral items use `BatchDeadlineExceeded` and
`error_category="batch_deadline_exceeded"`. `process_all()` and
`process_prompts()` return completed and collateral results with
`result.termination.kind == "batch_timeout"`. `process_stream()` yields all
terminal results for accepted work and then ends normally. Low-level streaming
callers can inspect `processor.termination` after completion.

The abort mode controls provider calls already in flight:

- `AbortMode.DRAIN_ACTIVE`: let a provider call already registered as active
  finish, but do not start another retry or provider call afterward.
- `AbortMode.CANCEL_ACTIVE`: cancel active provider calls and convert unfinished
  accepted work into batch-deadline results.

External caller cancellation is different: it still propagates after worker,
producer, queue, and strategy cleanup and is not mislabeled as a deadline.

## Configurable fail-fast

`abort_on_error_categories` is empty by default. A configured category trips
the shared abort controller only after an item reaches terminal failure; an
intermediate retryable attempt does not abort a batch that could still recover.
The first concurrent trigger wins and records its category and item ID.

Already completed results are preserved. Queued accepted items do not call the
provider and receive `BatchAbortedError` with
`error_category="batch_aborted"`. The active-call behavior follows the same
`abort_mode` described above. The returned batch reports
`termination.kind == "fail_fast"`.

Choose categories that indicate a batch-wide condition. Good candidates when
your provider classifier supplies reliable status information include:

- `authentication` (HTTP 401);
- `permission_denied` (HTTP 403), when permission is account/model-wide; and
- `insufficient_balance`.

Do not use `client_error` as a blanket default: malformed input or validation
can be item-specific. The provider-neutral classifier does not invent auth or
permission categories when reliable status data is unavailable.

## End-to-end checkpointed run

```python
from pathlib import Path
from async_batch_llm import (
    AbortMode,
    ArtifactIdentity,
    GuardrailConfig,
    JsonlArtifactStore,
    ProcessorConfig,
    ResumePolicy,
    process_prompts,
)

store = JsonlArtifactStore(
    "run.jsonl",
    identity=ArtifactIdentity(
        provider="openai",
        model="example-model",
        prompt_version="invoice-v4",
        parser_version="invoice-schema-v2",
        application_version="billing-pipeline-v7",
    ),
)

config = ProcessorConfig(
    max_workers=20,
    attempt_timeout=30,
    guardrails=GuardrailConfig(
        total_timeout_per_item=180,
        batch_timeout=3600,
        abort_on_error_categories=frozenset({
            "authentication",
            "insufficient_balance",
        }),
        abort_mode=AbortMode.DRAIN_ACTIVE,
    ),
)

result = await process_prompts(
    strategy,
    prompts,
    config=config,
    artifact_store=store,
    resume=ResumePolicy.REUSE_SUCCESSES,
    preserve_order=True,
)

Path("summary.json").write_text(result.to_json(), encoding="utf-8")
```

Serialization, artifact I/O, and programming failures remain exceptions. They
are not disguised as controlled guardrail termination.
