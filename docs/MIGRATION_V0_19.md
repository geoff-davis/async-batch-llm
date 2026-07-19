# Migration Guide: v0.18.x → v0.19.0

**Nothing breaks in v0.19.** Every change is additive or a deprecation with
the old spelling still accepted. Upgrade first, migrate at your own pace.

## Summary of changes

| Change | Breaking? | Required action |
| --- | --- | --- |
| `llm("provider:model")` strategy factory | No | Optional one-liner; the two-object form remains fully supported |
| `timeout_per_item` → `attempt_timeout` | No | Rename when convenient; old name warns, both-at-once raises |
| `concurrency=` unified knob | No | Optional; replaces hand-aligning workers/admission/pool |
| `BatchResult.summary()` / `.outputs()` | No | Optional; replaces hand-rolled reporting loops |
| `BatchResult.wall_time_seconds` field | No | New optional field; old serialized records restore as `None` |
| Zero-config `JsonlArtifactStore("run.jsonl")` | No | Optional; explicit `ArtifactIdentity` unchanged and recommended for versioned pipelines |
| `progress=True` on `process_prompts`/`process_stream` | No | Optional; install the `[progress]` extra for a tqdm bar |
| `progress_callback` fires per item | Unlikely | Only if you relied on the old every-`progress_interval` cadence |

## Two-object construction → `llm()` (optional)

Before:

```python
from async_batch_llm import OpenAIModel, OpenAIStrategy

strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-4o-mini"))
```

After:

```python
from async_batch_llm import llm

strategy = llm("openai:gpt-4o-mini")  # reads OPENAI_API_KEY
```

The factory covers `gemini:`, `openai:`, `openrouter:`, and `deepseek:`
prefixes and returns exactly the strategy objects the explicit form builds,
so error-classifier auto-selection and client lifecycle are unchanged.
Keyword arguments forward to the model constructor
(`llm("deepseek:deepseek-v4-flash", thinking=False)`), and
`response_parser=`/`temperature=`/`generation_config=` forward to the
strategy. Keep the explicit two-object form for custom clients, Gemini
cached models, and custom strategies — it is not deprecated.

## `timeout_per_item` → `attempt_timeout`

The per-attempt execution timeout is now named `attempt_timeout`, ending the
confusion with `GuardrailConfig.total_timeout_per_item` (which bounds the
whole logical item — admission waits, cooldowns, calls, and retries):

```python
config = ProcessorConfig(attempt_timeout=30.0)   # was: timeout_per_item=30.0
```

- `timeout_per_item` still works — as a keyword, positionally, and as a
  config attribute — and emits a `DeprecationWarning`.
- Passing **both** names raises `ValueError`.
- Removal is targeted for the next major release.

## Per-knob concurrency → `concurrency=`

Before, aligning throughput required three knobs (and the httpx pool was a
documented footgun):

```python
model = DeepSeekModel.from_api_key("deepseek-v4-flash", max_connections=64)
config = ProcessorConfig(max_workers=64, max_provider_concurrency=64)
```

After, one knob sizes whatever you did not set explicitly:

```python
batch = await process_prompts(strategy, prompts, concurrency=64)
# or: ProcessorConfig(concurrency=64)
```

With `concurrency=N`, built-in models created via `llm()` or
`from_api_key(...)` *without* an explicit `max_connections` get their
connection pool resized to `N` before the first request. Explicit values for
any individual knob always win; the capacity warning now fires only on a
real contradiction (an explicit client capacity smaller than the requested
concurrency), never on an override.

## Progress reporting

`progress=True` gives a tqdm bar (`pip install 'async-batch-llm[progress]'`)
or interval logging without the extra. If you already use
`progress_callback=`, note it now fires for **every** completed item rather
than every `progress_interval` items — `progress_interval` still controls
the built-in log-line cadence. Callbacks that assumed the sparser cadence
should rate-limit themselves.

## New result ergonomics (nothing to migrate)

```python
batch = await process_prompts(strategy, prompts)
print(batch.summary())                       # complete post-run report
for item_id, output in batch.outputs(with_ids=True):
    ...
```

`batch.wall_time_seconds` is stamped by `process_all()`/`process_prompts()`
and survives dict/JSON/JSONL round-trips; records serialized by earlier
versions restore it as `None` and `summary()` shows `n/a`.
