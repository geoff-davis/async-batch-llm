# Migrating from v0.18.x to v0.20.0

v0.19.0 was not published. Its planned onboarding and ergonomics improvements
are included in v0.20.0. Upgrade directly from v0.18.x; no v0.19 package, tag,
or intermediate checkpoint conversion exists.

Most v0.20 changes are additive. The two renamed access patterns retain
warning-emitting compatibility shims.

## 1. Changes requiring user action

No v0.18 call site must change immediately. Migrate these deprecated spellings
before the next major release:

```python
# Before (still works with DeprecationWarning)
config = ProcessorConfig(timeout_per_item=30)
rate = batch.cache_hit_rate()

# After
config = ProcessorConfig(attempt_timeout=30)
rate = batch.cache_hit_rate
```

`attempt_timeout` still limits one call to `strategy.execute()`, not the whole
logical item. Use `GuardrailConfig.total_timeout_per_item` for admission,
cooldowns, calls, and retries together. Passing both timeout names raises
`ValueError` rather than choosing one silently.

Smart-retry strategies must keep validation counts, partial output, escalation
stage, and retry-prompt feedback in the work item's `RetryState`. A strategy or
client shared by concurrent items should contain only shared, concurrency-safe
resources and configuration.

## 2. Additive features

### High-level construction and execution

`llm("provider:model")` creates the same built-in strategy objects as explicit
model/strategy construction. `process_prompts()` collects a `BatchResult`;
`process_stream()` yields terminal results in completion order.

```python
batch = await process_prompts(
    llm("openai:gpt-4o-mini"),
    prompts,
    concurrency=16,
    progress=True,
)
print(batch.summary())
for item_id, output in batch.outputs(with_ids=True):
    save(item_id, output)
```

The unified `concurrency=N` knob sizes unset worker, provider-admission, and
resizable built-in connection-pool limits together. Explicit individual limits
still take precedence.

The bundled `progress=True` reporter coalesces terminal rendering by
`ProcessorConfig.progress_refresh_interval_seconds`. User callbacks retain
per-item delivery. `progress_interval` still controls processor log lines by
item count.

### Checkpoints and bounded streaming

Built-in strategies can use `JsonlArtifactStore("run.jsonl")` without an
explicit identity. Provider/model identity is inferred; prompt and context
remain part of each compatibility fingerprint. Explicit `ArtifactIdentity`
remains preferable for production prompt/parser/application versioning.

Set `max_result_queue_size` to bound completed results waiting for a
`process_stream()` consumer independently of `max_queue_size`, which bounds
accepted input waiting for workers:

```python
config = ProcessorConfig(
    concurrency=32,
    max_queue_size=128,
    max_result_queue_size=64,
)
```

Both queue bounds default to zero (unbounded), preserving v0.18 behavior.
`process_prompts()` still retains its final collected batch.

### Existing async clients

`CallableStrategy` and `CallOutcome` wrap an existing async SDK, gateway, or
application service in the existing executor:

```python
async def invoke(prompt, *, attempt, timeout, state):
    response = await client.generate(prompt, timeout=timeout)
    return CallOutcome(response.text, token_usage=response.usage)

strategy = CallableStrategy(
    invoke,
    identity=ArtifactIdentity(provider="internal", model="summary-route"),
)
```

Arbitrary callables require a stable explicit identity when artifacts are
enabled. ABL never fingerprints a lambda, closure, object ID, or `repr()`.
Use `TokenTrackingError` when a billed response fails parsing so failed-attempt
usage reaches the terminal result.

### Shared calls

`LLMCallPool` is the preferred name for the in-process shared executor.
`LLMGateway` remains an exact, warning-free alias:

```python
from async_batch_llm import LLMCallPool, LLMGateway

assert LLMCallPool is LLMGateway
```

It is not an HTTP gateway, router, credential store, or background queue.

## 3. Compatibility shims

| v0.18 form | v0.20 preferred form | v0.20 behavior |
| --- | --- | --- |
| `timeout_per_item=30` | `attempt_timeout=30` | Old form works and warns |
| `batch.cache_hit_rate()` | `batch.cache_hit_rate` | Old call works and warns |
| `LLMGateway` | `LLMCallPool` | Exact alias; no warning |
| Explicit model plus strategy | `llm()` when convenient | Explicit form remains supported |
| Separate concurrency knobs | `concurrency=` when aligned | Explicit overrides remain supported |

Existing tuple-returning custom `LLMCallStrategy` subclasses continue to work.
Work-item prompts remain strings.

## 4. Suggested migration sequence

1. Upgrade in a branch and run the v0.18 test suite unchanged.
2. Rename `timeout_per_item` and property-style `cache_hit_rate` call sites.
3. Audit shared strategies for item-specific mutable recovery fields; move
   them to `RetryState`.
4. Adopt `llm()`, `concurrency=`, progress, summaries, and `outputs()` where
   they simplify code.
5. Keep explicit provider construction where a custom client or parser needs
   it.
6. Add `max_queue_size` and `max_result_queue_size` before scaling lazy runs.
7. Version artifact identity deliberately and test one compatible resume plus
   one intentional invalidation.
8. Prefer `LLMCallPool` in new code; rename old `LLMGateway` references only
   when convenient.

## 5. Before and after

Before:

```python
model = OpenAIModel.from_api_key("gpt-4o-mini", max_connections=10)
strategy = OpenAIStrategy(model)
config = ProcessorConfig(max_workers=10, timeout_per_item=30)
batch = await process_prompts(strategy, prompts, config=config)
print(batch.succeeded, batch.total_items)
```

After:

```python
strategy = llm("openai:gpt-4o-mini")
config = ProcessorConfig(concurrency=10, attempt_timeout=30)
batch = await process_prompts(strategy, prompts, config=config, progress=True)
print(batch.summary())
```

The after form is optional convenience, not a replacement for explicit models
or low-level processors.

## 6. Removed or deferred functionality

v0.20 removes no published v0.18 public execution surface. It does not add
non-string work items, a generic job runner, a network gateway, provider-native
batch submission, distributed workers, scheduling, DAGs, adaptive concurrency,
token-per-minute admission, or SQLite artifacts.

## 7. Checkpoint compatibility

The v1 result/artifact schema from v0.18 remains. Older records without the
additive `wall_time_seconds` field restore it as `None`. No synthetic v0.19
format exists.

Replay remains deliberately strict: item ID, prompt, participating context,
artifact schema, and complete identity must match. New `CallableStrategy`
artifacts need explicit identity because callable semantics cannot be inferred.
Changing prompt/model/parser/application identity correctly causes live
execution. `REUSE_SUCCESSES` reruns prior failures; `REUSE_ALL` may replay them.

## 8. Gateway and callable composition

ABL may wrap a LiteLLM client or another real gateway. The gateway can own
provider normalization, credentials, routing, and centralized policy while ABL
owns one application's bounded run, validation recovery, deadlines, terminal
outcomes, and checkpoints.

Avoid running the same transport retry policy at full strength in both layers.
ABL cannot account for provider attempts hidden inside the gateway. Put
application-specific recovery in each item's `RetryState`, not on the shared
`CallableStrategy`.

Continue with [Troubleshooting and FAQ](troubleshooting.md) and the
[v0.20 release notes](release-notes-v0.20.md).
