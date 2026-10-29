# Prompt caching: getting hits and keeping them

A prompt cache that stops hitting doesn't make any call fail. It makes every call
cost more, sometimes several times more, and nothing in the results says so unless
you look. This guide covers how to get cache hits, how to see them per job, and how
to stop a run, or alert, when they drop.

The statistics, the floor and `CacheGroupTracker` are provisional: new in 1.1, and
they may change in a minor release ([API stability](stability.md#provisional-through-10)).
For the exact rules, see [Cache hit-rate floor](guardrails.md#cache-hit-rate-floor).

A runnable, credential-free walkthrough of everything below is in
`examples/example_prompt_caching.py`.

## 1. Get cache hits

Every supported provider caches a prompt *prefix*. Put the static part first and the
per-item part last:

- Put long instructions, examples and reference text in `system_instruction` (or a
  system message), and only the item in the prompt.
- Keep the static part byte-for-byte identical across calls. A timestamp, a
  shuffled example list or an item id near the start makes every prefix unique.
- Don't join static instructions and per-item data in one user message. That
  writes the whole prompt to the cache on every call and never reads it back.
  When a provider caches by prefix only (DeepSeek, for example), a static prefix at
  the start of the user message works too, as long as nothing per-item comes first.

Providers also need a minimum prefix length (about 1,024 tokens for OpenAI) before
they cache anything. Provider specifics:
[OpenAI](OPENAI_INTEGRATION.md#getting-cache-hits),
[OpenRouter](OPENROUTER_INTEGRATION.md#prompt-caching) (including Anthropic's opt-in
`cache_control` markers), and [Gemini](GEMINI_INTEGRATION.md) (`GeminiCachedModel`
for explicit context caching).

## 2. What each provider supports

The statistics and the floor work with any strategy. Each call needs cached-token
counts in its usage (`cached_input_tokens`, and `cache_write_input_tokens` where the
provider reports writes) and a group label in its metadata (`cache_group`).

| Built-in model | Cached tokens counted | Group set automatically | `cache_group=` label |
| --- | --- | --- | --- |
| `OpenAIModel` | yes | `prompt_cache_key`, else a system-instruction hash | yes |
| `OpenRouterModel` | yes, when the upstream reports them | `prompt_cache_key`, else a system-instruction hash | yes |
| `DeepSeekModel` | yes, from DeepSeek's cache-hit field | system-instruction hash (DeepSeek takes no key) | yes |
| `GeminiModel` | yes | system-instruction hash | yes |
| `GeminiCachedModel` | yes | a stable hash of the cached content | yes |
| `PydanticAIStrategy` | yes | none (calls count as ungrouped) | no |
| Custom strategy or `CallableStrategy` | if your usage reports them | your `metadata["cache_group"]` | n/a |

Two features are OpenAI-specific:

- **`prompt_cache_key`** on `OpenAIModel` and `OpenRouterModel` routes calls that
  share a key to the same cache. DeepSeek's and Gemini's caches take no key.
- **Miss reasons.** OpenAI's Responses API can explain a miss
  (`prompt_cache_diagnostics`); the reason becomes the call's `cache_miss_reason`.
  Other providers don't explain misses, so `last_miss_reason` stays `None` unless your
  strategy sets it.

## 3. Name your cache groups

Statistics and the floor are kept per *cache group*: the calls that should share a
cache. A built-in model picks a group for you (table above), but a label you choose
reads better and keeps jobs apart:

```python
from async_batch_llm import DeepSeekModel, OpenAIModel

# A label for every call this model makes. It is never sent to the provider.
model = DeepSeekModel.from_api_key("deepseek-chat", cache_group="tier1-dedupe")

# OpenAI: send a cache key on every call; it also names the group.
model = OpenAIModel.from_api_key(
    "gpt-6-luna",
    system_instruction=LONG_STATIC_INSTRUCTIONS,
    prompt_cache_key="classify-v3",  # or "auto": a hash of the system instruction
)
```

A strategy can override the model's label with
`OpenAIStrategy(model, generation_config={"cache_group": "..."})` (any built-in model
strategy takes `generation_config`), and a custom `execute()` can pass
`config={"cache_group": "..."}` to `model.generate()` for one call. The order is: a metadata extractor's
`cache_group`, then the per-call label, then the model's label, then the cache key,
then the system-instruction hash.

Calls with none of these share one ungrouped bucket (`group` is `None`), which the
floor also checks. Two unrelated keyless jobs in one run land in that bucket together,
so label them.

For a custom strategy or `CallableStrategy`, put the group in the metadata you
return:

```python
from async_batch_llm import CallOutcome

async def invoke(prompt, *, attempt, timeout, state):
    response = await client.generate(prompt)
    return CallOutcome(
        output=response.text,
        token_usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cached_input_tokens": response.usage.cache_read_tokens,
        },
        metadata={"cache_group": "extract-v2"},
    )
```

**One hot key.** OpenAI's guide says traffic above roughly 15 requests per minute for
one prefix and key spills to other machines and caches less well. With many workers on
one key, split the key (for example `classify-v3-0` through `classify-v3-3`, chosen per
item) and give the models one `cache_group` label so they report together, or leave the
labels apart to compare the shards.

## 4. Read the statistics

Every processor run keeps per-group statistics, with or without a floor:

```python
async with ParallelBatchProcessor[None, str, None](config=config) as processor:
    ...
    result = await processor.process_all()
    stats = await processor.get_stats()

for group in stats["cache_groups"]:  # most input tokens first
    print(group["group"], f"{group['hit_rate']:.0%}", group["last_miss_reason"])
```

Each entry has `measured_calls`, `unmeasured_calls`, `input_tokens`,
`cached_input_tokens`, `cache_write_input_tokens`, the cumulative `hit_rate` (cached
over input tokens), `window_hit_rate` over the group's last `cache_hit_window`
measured calls, and `last_miss_reason`. The `BATCH_COMPLETED` observer event carries
the same lists. Calls without usage (or with no input tokens) count as
`unmeasured_calls` and never move a rate.

Run-wide totals are on the result: `result.total_cached_tokens`,
`result.total_cache_write_tokens` and `result.cache_hit_rate` (a percentage, 0-100,
unlike the per-group rates, which are fractions).

**The ceiling isn't 100%.** The hit rate counts every input token, and the per-item
part is never cached. A 2,000-token static prefix with 100-token items tops out at
about 95%, and a group's first call, which writes the cache, reads 0%.

## 5. Stop a run when hits drop

```python
from async_batch_llm import GuardrailConfig, ProcessorConfig

config = ProcessorConfig(
    max_workers=20,
    guardrails=GuardrailConfig(
        min_cache_hit_rate=0.7,
        cache_hit_window=50,
    ),
)
```

Once a group has `cache_hit_window` measured calls, a window hit rate below
`min_cache_hit_rate` stops the run like [configurable
fail-fast](guardrails.md#configurable-fail-fast). Check for it on the result:

```python
from async_batch_llm import ErrorCategory

if result.termination.error_category == ErrorCategory.CACHE_HIT_RATE_BELOW_FLOOR:
    print(result.termination.reason)
    # "Cache hit rate for group 'classify' fell to 57.1% over the last 50
    #  measured calls (floor 70.0%); last miss reason: input_changed"
```

Calls that hadn't started end `batch_aborted`, and `abort_mode` decides what happens
to calls already running. With an [artifact store](results-and-artifacts.md) and
`ResumePolicy`, fix the prompt and rerun: completed items replay, and only the
rest are called. The resumed run starts with fresh statistics.

**Choosing the values.**

- **Floor:** set it below your ceiling (section 4) with room for noise, and above
  the rate a broken cache gives. If a working job reads 90% and a broken one 0%, 0.7
  catches the break without false alarms. Measure a healthy run first, using the
  statistics without a floor.
- **Window:** it is also the warm-up, because no group is checked until its window
  is full. Make it comfortably larger than `max_workers`: calls that start together
  before the cache is written all miss. Larger windows react later but trip less on
  noise. The default is 50.
- **Many groups:** each group has its own window, so a small group may never be
  checked. A run tracks at most 64 groups; with a floor set, a 65th group stops the
  run, because the floor can't be enforced for groups it doesn't track.

The floor applies to processor runs only. `call()`, `call_result()` and
`LLMCallPool` reject it; check a pool yourself (next section).

## 6. Long-lived pools and resumed jobs

`LLMCallPool` keeps the same per-group statistics, without a floor:

```python
stats = pool.cache_stats()  # same shape as get_stats(); always includes window_hit_rate
pool.reset_cache_stats()  # start counting again, e.g. after deploying a new prompt
```

The cumulative `hit_rate` of a long-lived pool moves slowly, so watch
`window_hit_rate`, or check a floor yourself with the pool's tracker:

```python
for breach in pool.cache_tracker.below_floor(0.7):
    logger.warning(breach.message)  # the same message a processor run would stop with
```

To keep the statistics across a restart, save the tracker and pass the restored one
to the next pool:

```python
import json

from async_batch_llm import CacheGroupTracker, LLMCallPool

saved = json.dumps(pool.cache_tracker.to_dict())  # at a checkpoint or on shutdown

# Next process:
tracker = CacheGroupTracker.from_dict(json.loads(saved))
pool = LLMCallPool(strategy, config=config, cache_tracker=tracker)
```

The saved state includes every group's window, so the restored tracker continues
exactly, warm-up included. A restored tracker keeps its own `window`, whatever
`cache_hit_window` the new config sets.

**Without saved state**, rebuild a tracker from saved results:

```python
tracker = CacheGroupTracker(window=50)
for row in saved_results:  # WorkItemResult objects
    tracker.record_result(row)
```

This reads lower than the live statistics did when retries were billed. A result's
`token_usage` adds up every attempt of its item, failed ones included, while live
tracking counts only the attempt that returned. Failed results are skipped.

`CacheGroupTracker` also works on its own, outside the framework:
`tracker.record(group, usage, miss_reason=...)` counts one call, with the same rules.
A tracker isn't thread-safe; use it from one event loop or thread.

## See also

- [Cache hit-rate floor](guardrails.md#cache-hit-rate-floor): the exact counting and
  floor rules.
- [OpenAI prompt caching](OPENAI_INTEGRATION.md#getting-cache-hits) and
  [OpenRouter prompt caching](OPENROUTER_INTEGRATION.md#prompt-caching).
- `CachedTokenRates` and `BatchResult.estimated_cost()` for pricing cached and written
  tokens ([API reference](api/core.md)).
