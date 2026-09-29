# Core API Reference

## ParallelBatchProcessor

::: async_batch_llm.ParallelBatchProcessor

## LLMWorkItem

::: async_batch_llm.LLMWorkItem

## WorkItemResult

::: async_batch_llm.WorkItemResult

## AttemptTiming

::: async_batch_llm.AttemptTiming

## WorkItemTiming

::: async_batch_llm.WorkItemTiming

## ProcessorConfig

::: async_batch_llm.ProcessorConfig

## GuardrailConfig

::: async_batch_llm.GuardrailConfig

## AbortMode

::: async_batch_llm.AbortMode

## AttemptUsage

::: async_batch_llm.AttemptUsage

## BatchBudgetExceeded

::: async_batch_llm.BatchBudgetExceeded

## StartupRampConfig

::: async_batch_llm.StartupRampConfig

## BatchResult

::: async_batch_llm.BatchResult

## BatchTermination

::: async_batch_llm.BatchTermination

## Grounding

::: async_batch_llm.Grounding

## GroundingSource

::: async_batch_llm.GroundingSource

## ToolCall

::: async_batch_llm.ToolCall

## Lifecycle exceptions

::: async_batch_llm.StreamFinalizationError

::: async_batch_llm.CleanupInterruptedError

## Admission closure

::: async_batch_llm.BatchAdmissionClosedError

`add_work()` raises `BatchAdmissionClosedError` after finish, shutdown, or abort.
Rejected submissions receive no submission index. Use a new processor for further work.

### `async def cleanup() -> None` / `async def shutdown() -> None`

Release every owned resource in dependency order. `cleanup()`, `shutdown()`,
and `async with` exit all run the same ordered close:

1. Runtime tasks and callbacks: the stream finalizer, workers, queued work,
   progress callbacks, background post-processors, and callback threads.
2. Admission resources (quota scopes and the rate-limit coordinator).
3. Prepared strategies (`strategy.cleanup()`).
4. The artifact store, including after an early stream exit or a failed run.

```python
await processor.shutdown()
```

Behavior (see `docs/cleanup-lifecycle-contract.md` for the full contract):

- **Every step is attempted.** A failing strategy `cleanup()` does not stop
  sibling strategies or the artifact store from closing. Every failure is
  logged with a traceback; with no body exception, the first failure is
  raised after all steps have run. Inside `async with`, a body exception
  stays primary and cleanup failures are only logged.
- **Repeat calls are safe.** A step that succeeded is never repeated. A step
  that failed or was interrupted is retried by the next explicit close.
  Concurrent calls share one in-flight attempt. Because a retry re-invokes a
  strategy's whole `cleanup()`, user `cleanup()` implementations must be
  idempotent and safe after a partially completed earlier attempt.
- **No deadline.** Cleanup waits for the artifact store and the gateway drain
  to finish. The two-second worker and progress thresholds only log a warning
  and keep waiting; any step still running after 30 seconds logs one warning.
  A dependent resource is never closed while the resource it depends on is
  still running, including synchronous callback threads.
- **Cancellation.** Cancelling the task that is closing once defers that
  cancellation until cleanup finishes, then re-raises it (cleanup errors are
  logged, not raised). Cancelling it a second time force-aborts: the running
  step is abandoned, later steps are skipped and logged, and the cancellation
  propagates immediately. This is the operator's escape from a cleanup that
  never returns and deliberately sacrifices durability.
- **Interruptions.** A strategy `cleanup()` that raises `CancelledError`
  itself, or whose private task is cancelled by something else, surfaces as
  `CleanupInterruptedError` (an ordinary `Exception`); the caller's task is
  not cancelled and the next close retries the step.
- `KeyboardInterrupt` and `SystemExit` raised by a cleanup step keep their
  type and take precedence over a deferred cancellation.
- Once a close has started, preparing a new strategy raises `RuntimeError`.

### Response metadata (`WorkItemResult.metadata`)

Provider metadata (Gemini safety ratings and finish reason, OpenRouter
provider/routed model, etc.) flows into `WorkItemResult.metadata` — a plain
`dict[str, Any] | None`. The parsed output stays in `WorkItemResult.output`;
you no longer wrap it in a separate response object.

Conservative structured-output recovery uses three reserved metadata keys:
`structured_output_recovered`, `structured_output_recovery_reason`, and
`structured_output_retries_avoided`. Read them through the corresponding typed
properties on `LLMResponse` or `WorkItemResult`.

> **Removed:** the old `GeminiResponse` wrapper and the `include_metadata`
> opt-in were removed in v0.6.0. Read metadata off `result.metadata` instead.
> For Gemini safety ratings specifically, `result.metadata["safety_ratings"]`
> carries them (the deprecated `result.gemini_safety_ratings` field is still
> backfilled for compatibility).

**Usage:**

```python
result = await processor.process_all()
first = result.results[0]
ratings = (first.metadata or {}).get("safety_ratings")
if ratings and ratings.get("HARM_CATEGORY_HATE_SPEECH") == "HIGH":
    log_flagged_content(first.output)
```

---

### Typed auxiliary output (grounding, reasoning, tool calls, logprobs)

> **Experimental.** This surface is new (v0.16.0) and hasn't seen much
> real-world use yet — the reserved-key dict shapes and the typed views may
> change in a future minor release while they stabilize. The `metadata`
> dict channel itself is stable; if you persist these shapes, read them
> back defensively.

Provider-specific structured output travels through `metadata` under four
**reserved keys** with documented plain-dict shapes (JSON-serializable, so
persisting `metadata` as-is works):

| Key | Shape | Emitted by |
| --- | ----- | ---------- |
| `grounding` | `{"sources": [{"uri", "title"}], "queries": [str], "supports": [dict]}` | Gemini models, when the response carries `google_search` grounding |
| `reasoning` | `str` — the model's reasoning/thinking trace | OpenAI-compatible models (`reasoning_content`, e.g. DeepSeek, falling back to `reasoning`, e.g. OpenRouter) |
| `tool_calls` | `[{"id": str\|None, "name": str, "arguments": str}]` — `arguments` is the raw JSON string | OpenAI-compatible models |
| `logprobs` | provider-shaped `dict`/`list` (via `model_dump()`) | OpenAI-compatible models, when requested |

Both `LLMResponse` and `WorkItemResult` expose **lazy read-only typed
views** over these keys — parsed from `metadata` on each access, never
cached, never stored twice:

```python
result = await processor.process_all()
item = result.results[0]

if item.grounding:                       # Grounding | None
    for source in item.grounding.sources:  # list[GroundingSource]
        print(source.uri, source.title)
    print(item.grounding.queries)          # list[str]

print(item.reasoning)                    # str | None
for call in item.tool_calls or []:       # list[ToolCall] | None
    print(call.name, call.arguments)     # arguments: raw JSON string
print(item.logprobs)                     # Any | None (provider-shaped)
```

The view classes (`Grounding`, `GroundingSource`, `ToolCall`) are exported
at the top level. Parsing is lenient: malformed metadata yields `None` (or
drops the bad entry) rather than raising — which also means a future shape
change degrades to `None` views rather than errors.

**Boundaries:**

- `tool_calls` is **visibility only** — the framework never executes tools.
  Feed them to your own dispatch loop (or use an agent framework via
  `PydanticAIStrategy`). Covered for OpenAI-compatible providers only this
  phase (Gemini function-call parts are not extracted yet).
- A response whose `content` is `null` (e.g. a pure tool-call turn) still
  raises `EmptyResponseError` before any result exists, so `tool_calls`
  surfaces only when the model returned text alongside the calls.
- Auxiliary output does not survive empty/safety-blocked responses — the
  call fails first.

---

## TokenUsage

::: async_batch_llm.TokenUsage
