# Single Call & Gateway

Convenience surfaces for running individual calls through the full resilience
pipeline — error-type-aware retries, the coordinated rate-limit cooldown, and
token accounting — without constructing a
[`ParallelBatchProcessor`](core.md#parallelbatchprocessor).

- **`call` / `call_result`** — one prompt, one result. No queue, workers, or result
  stream are created.
- **`LLMGateway`** — a long-lived, shared entry point for the request path: many
  concurrent callers, one coordinated cooldown, concurrency bounded by a
  semaphore.

For large bulk jobs, keep using `ParallelBatchProcessor` /
[`process_prompts`](core.md); these surfaces are for single calls, not batches.

## Example

```python
from async_batch_llm import OpenAIModel, OpenAIStrategy, call, call_result, LLMGateway, ProcessorConfig

strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-4o-mini"))

# One prompt through the full pipeline — no queue, workers, or result stream.
summary = await call(strategy, "Summarize: ...")        # output, or raises

# A long-lived, shared entry point for a web service's request path.
async with LLMGateway(
    strategy,
    config=ProcessorConfig(max_workers=5),
    max_pending=100,     # admission cap: reject instantly when saturated
    submit_timeout=30,   # per-caller latency budget (seconds)
) as gw:
    reply = await gw.submit("Answer this one request")
```

## Failure semantics

- **`call()` / `LLMGateway.submit()`** re-raise the **provider's own exception**
  on failure, preserving its type. [`LLMCallError`](#llmcallerror) is raised only
  when no provider exception was preserved — a middleware filter-skip, or the
  gateway's `max_pending` / `submit_timeout` rejections.
- **`call_result()` / `LLMGateway.submit_result()`** never raise for a request
  failure; they return the full
  [`WorkItemResult`](core.md#workitemresult) — inspect `success`, `error`,
  `token_usage`, `metadata`, and the originating `exception`.

```python
result = await call_result(strategy, "Summarize: ...")
if not result.success:
    print(result.error, result.exception)   # exception preserves the provider type
```

The gateway drains already-admitted requests on `aclose()` (the `async with`
exit) before cleaning up the shared strategy, so in-flight calls aren't cut off
by shutdown; `submit_timeout` bounds how long that drain can take.

## call

::: async_batch_llm.call

## call_result

::: async_batch_llm.call_result

## LLMGateway

::: async_batch_llm.LLMGateway

## LLMCallError

::: async_batch_llm.LLMCallError
