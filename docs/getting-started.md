# Getting Started

This path starts with the smallest public API and adds operational controls one
at a time. Built-in provider wrappers are optional; the final section shows how
to bring an existing async client.

## 1. Installation

Install the extra for your provider. Add `progress` for a tqdm terminal bar:

```bash
pip install 'async-batch-llm[openai,progress]'
export OPENAI_API_KEY='...'
```

Available provider extras are `openai`, `gemini`, `openrouter`, `deepseek`, and
`pydantic-ai`. The credential-free
[embedded application example](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_callable_application.py)
and [Colab notebook](https://colab.research.google.com/github/geoff-davis/async-batch-llm/blob/main/notebooks/async_batch_llm_quickstart.ipynb)
need only the core package plus the optional progress extra.

## 2. First small batch

`llm()` constructs a tested built-in strategy. `concurrency=` aligns workers,
provider admission, and the connection pool when the model supports resizing.

```python
import asyncio
from async_batch_llm import llm, process_prompts

async def main():
    batch = await process_prompts(
        llm("openai:gpt-4o-mini"),
        ["Summarize document A", "Summarize document B"],
        concurrency=10,
    )
    print(batch.summary())

asyncio.run(main())
```

The factory also supports `gemini:`, `openrouter:`, and `deepseek:` model
specifications. It reads each provider's normal environment variable.

## 3. Read successes and failures

Every accepted item receives one terminal `WorkItemResult`; failures are data,
not exception objects mixed into an output list.

```python
for result in batch.results:
    if result.success:
        print(result.item_id, result.output)
    else:
        print(result.item_id, result.error_category, result.exception)

for item_id, output in batch.outputs(with_ids=True):
    print("successful output:", item_id, output)
```

Results are collected in completion order. Pass `preserve_order=True` to
`process_prompts()`, or call `batch.in_input_order()`, when submission order is
more useful.

## 4. Print a summary

`BatchResult.summary()` reports terminal counts, retries, current-run token
usage, wall time, timing percentiles, and failure categories:

```python
print(batch.summary())
```

Replayed token totals remain attached to replayed items for auditing but are
separated from current-run consumption in the summary.

## 5. Add progress

```python
batch = await process_prompts(
    strategy,
    prompts,
    concurrency=10,
    progress=True,
)
```

The bundled reporter sees every exact count but coalesces terminal rendering to
`ProcessorConfig.progress_refresh_interval_seconds` (0.1 seconds by default).
It renders the first observation and one final exact state. Lazy async sources
may increase the displayed total while running. Without tqdm, a slower
coalesced logging fallback is used.

A custom callback still receives every completed item and retains the existing
timeout/thread behavior:

```python
def on_progress(completed: int, total: int, item_id: str) -> None:
    metrics.gauge("batch.completed", completed)

batch = await process_prompts(strategy, prompts, progress=on_progress)
```

`progress_interval` is different: it controls processor log lines by item
count, not the bundled terminal refresh cadence.

## 6. Add checkpoint and resume

```python
from async_batch_llm import JsonlArtifactStore, ResumePolicy

store = JsonlArtifactStore("runs/summaries.jsonl")
batch = await process_prompts(
    strategy,
    prompts,
    artifact_store=store,
    resume=ResumePolicy.REUSE_SUCCESSES,
)
```

Built-in strategies provide a stable zero-configuration identity. Checkpoint
compatibility still includes item ID, prompt, participating context, provider,
model, and identity versions. A compatible success is returned with
`replayed_from_artifact=True` without another provider call.

For an arbitrary callable, provide an explicit `ArtifactIdentity`; ABL cannot
safely infer the provider, route, parser, or application version from a Python
function.

## 7. Stream from an async source

Use `process_stream()` when the source is lazy or the result set is large:

```python
from collections.abc import AsyncIterator
from async_batch_llm import ProcessorConfig, process_stream

async def source() -> AsyncIterator[tuple[str, str]]:
    async for row in repository.iter_documents(page_size=500):
        yield row.id, f"Summarize:\n{row.text}"

config = ProcessorConfig(
    concurrency=32,
    max_queue_size=128,
    max_result_queue_size=64,
    attempt_timeout=30,
)

async for result in process_stream(strategy, source(), config=config):
    await repository.save_result(result)
```

`max_queue_size` bounds accepted input waiting for workers.
`max_result_queue_size` bounds completed results waiting for the consumer. An
active worker can temporarily hold one additional completed result, and the
consumer owns its current item. `process_prompts()` still retains the complete
final `BatchResult`.

## 8. Bring an existing async client

`CallableStrategy` adapts an async operation to the same `ItemExecutor` used by
built-in strategies:

```python
from async_batch_llm import ArtifactIdentity, CallOutcome, CallableStrategy

async def invoke(prompt, *, attempt, timeout, state):
    response = await existing_client.generate(prompt, timeout=timeout)
    return CallOutcome(
        output=response.text,
        token_usage=response.usage,
        metadata={"route": response.route},
    )

strategy = CallableStrategy(
    invoke,
    identity=ArtifactIdentity(
        provider="application-gateway",
        model="summary-route",
        parser_version="summary-v2",
    ),
)
```

Put item-specific validation feedback or escalation state in the supplied
`RetryState`, never on a strategy shared by concurrent items. Avoid letting
both ABL and an upstream gateway run the same transport retry policy at full
strength.

See [Use Your Existing Async Client](callable-integration.md) for lifecycle,
token accounting, cancellation, classification, and shared-call composition.

## 9. Production guides

- [Choosing Your Limits](choosing-your-limits.md)
- [Production Checklist](production-checklist.md)
- [Deadlines and Fail-Fast Guardrails](guardrails.md)
- [Bounded Work and Backpressure](bounded-work.md)
- [Results, Artifacts, and Resume](results-and-artifacts.md)
- [Troubleshooting and FAQ](troubleshooting.md)
- [Compare alternatives](comparison.md)
