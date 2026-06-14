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

## call

::: async_batch_llm.call

## call_result

::: async_batch_llm.call_result

## LLMGateway

::: async_batch_llm.LLMGateway

## LLMCallError

::: async_batch_llm.LLMCallError
