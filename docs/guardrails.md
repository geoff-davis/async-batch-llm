# Deadlines, Budgets and Fail-Fast Guardrails

Guardrails are opt-in. Their defaults preserve normal completion-order,
retry, streaming, token-accounting, and cancellation behavior.

```python
from async_batch_llm import AbortMode, ErrorCategory, GuardrailConfig, ProcessorConfig

config = ProcessorConfig(
    max_workers=20,
    attempt_timeout=30,
    guardrails=GuardrailConfig(
        total_timeout_per_item=180,
        batch_timeout=3600,
        abort_on_error_categories=frozenset({
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.INSUFFICIENT_BALANCE,
        }),
        abort_mode=AbortMode.DRAIN_ACTIVE,
    ),
)
```

Timeouts must be finite numbers (not booleans) greater than zero, or `None`.
All deadline calculations use a
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

A per-attempt `attempt_timeout` expiry is distinct: it is retryable, with
`error_category="framework_timeout"` and
`timing.timeout_category="framework_execution_timeout"`. Put `framework_timeout`,
not `framework_execution_timeout`, in `abort_on_error_categories`.

`timing.timeout_category` records where the time ran out (see
[`TimeoutCategory`](api/core.md#timeoutcategory) for the values). A deadline reached
while the item waited for provider capacity is `admission_timeout`, for both item
and batch deadlines. The elapsed capacity wait still counts toward
`admission_wait_seconds`, and it's kept when an abort ends the wait too.

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

`abort_on_error_categories` is empty by default. Use `ErrorCategory` members for the
built-in categories (see [the full list](api/core.md#errorcategory)); plain strings,
including categories returned by a custom classifier, work too. A configured category trips
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

- `ErrorCategory.AUTHENTICATION` (HTTP 401);
- `ErrorCategory.PERMISSION_DENIED` (HTTP 403), when permission is
  account/model-wide; and
- `ErrorCategory.INSUFFICIENT_BALANCE`.

Depending on your setup, these are also batch-wide:

- `ErrorCategory.QUOTA_EXHAUSTED` (Gemini daily quota);
- `ErrorCategory.TOKEN_ESTIMATOR_REQUIRED` and `ErrorCategory.QUOTA_SCOPE_ERROR`
  (configuration failures that affect every item).

Do not use `client_error` as a blanket default: malformed input or validation
can be item-specific. The provider-neutral classifier does not invent auth or
permission categories when reliable status data is unavailable.

## Token and cost budgets

`max_total_tokens` and `max_total_cost` cap what one processor run spends. Both
are opt-in soft caps.

Provisional: this API is new in v0.27 and may change in a minor release; see
[API stability](stability.md#provisional-through-10).

```python
from async_batch_llm import AttemptUsage, GuardrailConfig, ProcessorConfig

def price(usage: AttemptUsage) -> float:
    # Your prices, in your currency. ABL ships no price table.
    return (
        usage.usage.get("input_tokens", 0) * 0.15
        + usage.usage.get("output_tokens", 0) * 0.60
    ) / 1_000_000

config = ProcessorConfig(
    max_workers=20,
    guardrails=GuardrailConfig(
        max_total_tokens=5_000_000,
        max_total_cost=25.0,
        cost_function=price,
    ),
)
```

**What counts.** Every physical provider attempt whose usage the provider
reported, success or failure, including retries and rate-limited tries. Usage is
counted as each attempt finishes, from the same observation that tokens-per-minute
admission reconciles. These are not counted:

- replayed results;
- dry-run calls;
- attempts that never started a provider call;
- attempts whose usage the provider didn't report (they appear in
  `budget_unknown_usage_attempts`);
- usage a strategy adds only later, from `on_error`.

So the budget can be lower than a result's `token_usage`.

**Reaching a cap.** When the total reaches a cap (equality counts), the run
stops like a batch deadline, with `termination.kind == "budget_exceeded"`:

- no new provider call or retry starts;
- `abort_mode` decides whether calls already running drain or are cancelled;
- accepted items that didn't finish receive `BatchBudgetExceeded` with
  `error_category="batch_budget_exceeded"`.

The attempt that reached the cap keeps its outcome if it succeeded. If it failed,
its result becomes `batch_budget_exceeded`, with the provider error kept as the
exception's `__context__`.

**Overshoot.** Final usage can exceed the cap by the usage of the attempt that
reached it plus other attempts that had already started, which is at most
`max_workers` attempts in total. There is no numeric bound unless each attempt's
usage is bounded, for example with `max_tokens`. With `max_workers=1` and a cap of
10 tokens, a single call that reports 100 overshoots by 90. Providers may still
bill calls that `AbortMode.CANCEL_ACTIVE` cancels.

**Cost function.** `max_total_cost` requires `cost_function`, which is called once
per attempt with known usage. It receives an `AttemptUsage`: the item ID, attempt
and try numbers, the strategy, a read-only copy of the token usage, and whether the
call succeeded. Pass the `strategy` to price model escalation per attempt.

- It must be synchronous and fast: it runs on the event loop, and a blocking call
  can't be interrupted.
- It must return a finite, non-negative number.
- If it raises or returns anything else (including an awaitable or a bool), the
  run stops with `budget_exceeded`, tokens are still counted, and
  `budget_cost_complete` becomes `False`. Its exception message is not logged.
- A `cost_function` without `max_total_cost` only tracks cost.

**Reporting.** `await processor.get_stats()` adds `budget_tokens_used`,
`budget_cost_used`, `budget_cost_complete`, and `budget_unknown_usage_attempts`
when a budget or cost function is configured. The termination reason names the
cap and the amount used.

**Scope.** Budgets apply to processor runs (`process_all()`, streaming,
`process_prompts()`, `process_stream()`). `call()`, `call_result()`, and
`LLMCallPool` reject them with `ValueError` rather than ignoring a safety cap.
Every run starts with a fresh budget, including a resumed run.

**Artifacts.** `batch_budget_exceeded` results are audit records like other
batch-abort results: written best-effort and never replayed. A `BatchResult`
serialized with `termination.kind == "budget_exceeded"` can't be read by v0.26
or earlier.

## Cache hit-rate floor

A prompt cache that stops hitting can multiply a run's input cost without any call
failing. Every processor run counts cache reads and writes per *cache group*, and
`min_cache_hit_rate` stops the run when a group's recent hit rate falls below a floor.

Provisional: new in 1.1; it may change in a minor release.

```python
from async_batch_llm import GuardrailConfig, ProcessorConfig

config = ProcessorConfig(
    max_workers=20,
    guardrails=GuardrailConfig(
        min_cache_hit_rate=0.8,  # None (the default) turns the floor off
        cache_hit_window=50,     # measured calls per group in the sliding window
    ),
)
```

**Groups.** An attempt's group is the `cache_group` key in its metadata:

- OpenAI-compatible models (OpenAI, OpenRouter, DeepSeek) use the request's
  `prompt_cache_key` if one is set, else `si:` plus a 12-character hash of the
  system instruction.
- `GeminiModel` uses the system-instruction hash. `GeminiCachedModel` uses
  `gemini-cache:` plus a hash of the cached content, which stays the same when the
  cache is renewed.
- `PydanticAIStrategy` sets no group. `CallableStrategy` and custom strategies can
  set `metadata["cache_group"]` themselves.
- Attempts without a group are counted together as the `None` group.

A group longer than 128 characters is replaced by its SHA-256. A run tracks at most
64 groups; attempts for further groups are counted together in
`cache_groups_overflow`, with one warning.

**What counts.** Provider attempts in this run that returned a result. Failed
attempts, replayed results and dry-run calls aren't counted. An attempt is
*measured* when its usage is known and reports positive input tokens; others are
counted as `unmeasured_calls` and never move the rate.

**The floor.** Each group keeps a window of its last `cache_hit_window` measured
attempts. Once a group's window is full, a window hit rate (cached input tokens
over input tokens) below `min_cache_hit_rate` stops the run like
[configurable fail-fast](#configurable-fail-fast): `termination.kind ==
"fail_fast"`, `termination.error_category == "cache_hit_rate_below_floor"`,
`abort_mode` applies, and items that hadn't started end `batch_aborted`. The window
is also the warm-up, and each group has its own, so a group's first cache writes
aren't held against it. With a floor configured, a 65th group also stops the run,
since the floor can't be enforced for groups it doesn't track. If the same attempt
reaches a token or cost budget, the run ends `budget_exceeded` instead.

**Miss reasons.** A strategy can put a provider's explanation of a miss in
`metadata["cache_miss_reason"]` (a string, kept to 64 characters). Each group
reports its latest one as `last_miss_reason`, and the floor's stop reason ends with
it. No built-in model sets it yet.

**Reporting.** `await processor.get_stats()` and the `BATCH_COMPLETED` event carry
`cache_groups`, a list sorted by input tokens, and `cache_groups_overflow` (`None`
unless there was overflow). Each entry has `group`, `measured_calls`,
`unmeasured_calls`, `input_tokens`, `cached_input_tokens`,
`cache_write_input_tokens`, `hit_rate` (cumulative) and `last_miss_reason`, plus
`window_hit_rate` when a floor is configured (`None` until the window is full).
`BatchResult` has no per-group view: results include replays and add up retries,
which isn't what the floor measures.

**Scope.** Processor runs only. `call()`, `call_result()` and `LLMCallPool` reject
`min_cache_hit_rate` with `ValueError`. A resumed run starts with fresh counts.

## End-to-end checkpointed run

```python
from pathlib import Path
from async_batch_llm import (
    AbortMode,
    ArtifactIdentity,
    ErrorCategory,
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
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.INSUFFICIENT_BALANCE,
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

## Retry delay cannot fit

If the actual jittered retry delay is at least the remaining item deadline,
ABL fails immediately with `ItemDeadlineExceeded` instead of spending that time
waiting. Its message and cause retain the last provider error; the attempt timing
keeps that error's category and reports zero retry-backoff wait. The final result
uses `framework_total_item_timeout`, so `abort_on_error_categories` matches that
final category rather than the provider category. The existing guardrail audit
and non-replayable-record behavior remains in effect.
