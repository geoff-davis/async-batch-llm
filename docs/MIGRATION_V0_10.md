# Migration Guide: v0.8.x → v0.10.0

This guide covers the changes shipped across v0.9.0 (first-class OpenAI
and OpenRouter providers) and v0.10.0 (response metadata reaching
`WorkItemResult`). Both releases were designed to be **mostly additive** —
existing code keeps working unchanged. The one real breaking surface is
narrow and called out below.

## Summary of changes

| Change                                                     | Breaking? | Required action                                                    |
|------------------------------------------------------------|-----------|--------------------------------------------------------------------|
| `LLMCallStrategy.execute()` return shape (3-tuple)         | Sometimes | Only if you call `.execute()` directly on a built-in strategy      |
| `WorkItemResult.metadata` field added                      | No        | None                                                               |
| `WorkItemResult.gemini_safety_ratings` deprecated          | No        | Read `metadata['safety_ratings']` instead (eventually)             |
| `CachedTokenRates` constants                               | No        | Pass to `effective_input_tokens()` for accurate non-Gemini billing |
| `OpenAIModel`, `OpenRouterModel` and friends               | No        | Optional new providers; install `[openai]` / `[openrouter]` extras |
| `OpenAIErrorClassifier`, `OpenRouterErrorClassifier`       | No        | Optional; pass to `ParallelBatchProcessor` if you want them        |
| `DeepSeekModel` / `DeepSeekStrategy`                       | No        | Optional new provider; install the `[deepseek]` extra              |
| `ModelStrategy` base class                                 | No        | Internal dedupe; built-in strategies behave identically            |
| `temperature=None` to omit the parameter                   | No        | Use it for OpenAI reasoning models (o1/o3) that reject temperature |
| `effective_input_tokens()` warns on implicit default       | Sometimes | Pass an explicit `CachedTokenRates` rate to silence the warning    |
| `ErrorInfo.suggested_wait` now honored as a cooldown floor | No        | None; only a parsed `Retry-After` sets it now                      |

## Breaking change: 3-tuple `execute()` return shape

### What changed

`LLMCallStrategy.execute()` now returns a 3-tuple:

```python
(output, token_usage, metadata)
```

where `metadata: dict[str, Any] | None` is forwarded into
`WorkItemResult.metadata`. All four built-in strategies — `GeminiStrategy`,
`OpenAIStrategy`, `OpenRouterStrategy`, `PydanticAIStrategy` — return the
3-tuple shape.

The framework still **accepts** the legacy 2-tuple `(output, token_usage)`
from custom strategies via a compat shim. Returning 2-tuple from your
custom strategy keeps working; you just don't get metadata in
`WorkItemResult`.

### Why this matters

Most users never call `strategy.execute()` directly — the framework drives
it. The only place this breaks is **direct callers**, typically:

- Unit tests that exercise a strategy in isolation
- Custom processor implementations that drive strategies themselves

In those cases the 2-tuple unpack now fails with `ValueError: too many
values to unpack`.

### Migration required if

You have code like:

```python
output, tokens = await my_strategy.execute(prompt, attempt, timeout)
```

…where `my_strategy` is one of the built-in strategies.

### How to migrate

Unpack three values:

```python
# Before
output, tokens = await strategy.execute(prompt, 1, 30.0)

# After
output, tokens, metadata = await strategy.execute(prompt, 1, 30.0)

# If you don't care about metadata in the test:
output, tokens, _ = await strategy.execute(prompt, 1, 30.0)
```

If you have a **custom** `LLMCallStrategy` and want to opt into the
metadata path so users see provider info in `WorkItemResult.metadata`,
update your `execute()` to return a 3-tuple:

```python
class MyStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        response = await self.client.generate(prompt)
        tokens = {...}
        metadata = {"model": self.model, "finish_reason": response.stop_reason}
        return response.text, tokens, metadata  # 3-tuple
```

Returning 2-tuple still works — your strategy just yields
`metadata=None` on the resulting `WorkItemResult`.

## Deprecation: `WorkItemResult.gemini_safety_ratings`

### What changed

`WorkItemResult` gained a generic `metadata: dict[str, Any] | None` field.
Gemini safety ratings now live there under the `safety_ratings` key, same
shape as before. The legacy `gemini_safety_ratings` field is still
populated automatically (backfilled from `metadata['safety_ratings']` via
`__post_init__`) for backward compatibility, but it's slated for removal
alongside the 2-tuple compat shim.

### How to migrate

Switch reads from the named field to the metadata key when you next touch
the relevant code:

```python
# Before
ratings = result.gemini_safety_ratings

# After
ratings = (result.metadata or {}).get("safety_ratings")
```

No urgency — both paths work today and will continue to work for at
least one more minor release. We'll add a `DeprecationWarning` when the
removal date is set.

## New: provider-aware billing with `CachedTokenRates`

### What changed

`BatchResult.effective_input_tokens()` previously hardcoded Gemini's 90%
cached-token discount, producing wrong numbers for OpenAI (50% discount),
Anthropic (90% on cache reads, 25% premium on writes), and DeepSeek
(90%). It now accepts a `cached_token_rate` parameter, with named
constants on `CachedTokenRates`:

```python
from async_batch_llm import CachedTokenRates

billable = result.effective_input_tokens(CachedTokenRates.OPENAI)
billable = result.effective_input_tokens(CachedTokenRates.ANTHROPIC_READ)
billable = result.effective_input_tokens(CachedTokenRates.DEEPSEEK)
```

The default rate is `CachedTokenRates.GEMINI` for backward compat — if
you don't pass anything, you get exactly the same number as before.

> **v0.10.0 behavior change:** calling `effective_input_tokens()` *without*
> an explicit rate now emits a `UserWarning` **when cached tokens are
> present** (the implicit Gemini rate is wrong for other providers). The
> returned number is unchanged; pass an explicit `CachedTokenRates` constant
> to silence the warning. The no-cache case stays silent.

### Migration required if

You're using `effective_input_tokens()` with a non-Gemini provider and
care about the accuracy of the result. Pass the matching constant (which
also silences the new warning).

### Mixed-provider batches

For OpenRouter batches that mix providers per item, do per-item
arithmetic using `WorkItemResult.metadata['provider']`:

```python
PROVIDER_RATES = {
    "OpenAI": CachedTokenRates.OPENAI,
    "Anthropic": CachedTokenRates.ANTHROPIC_READ,
    "DeepSeek": CachedTokenRates.DEEPSEEK,
}

total_billable = 0
for r in result.results:
    if not r.success:
        continue
    provider = (r.metadata or {}).get("provider")
    rate = PROVIDER_RATES.get(provider, CachedTokenRates.OPENAI)
    cached = r.token_usage.get("cached_input_tokens", 0)
    inp = r.token_usage.get("input_tokens", 0)
    total_billable += inp - int(cached * (1.0 - rate))
```

See [`OPENROUTER_INTEGRATION.md`](OPENROUTER_INTEGRATION.md) for more.

## New: built-in OpenAI / OpenRouter providers

### What changed

You no longer have to write a custom `LLMCallStrategy` for OpenAI or
OpenRouter. New built-ins:

```python
# OpenAI
from async_batch_llm import OpenAIModel, OpenAIStrategy
model = OpenAIModel.from_api_key("gpt-4o-mini")  # reads OPENAI_API_KEY
strategy = OpenAIStrategy(model)

# OpenRouter
from async_batch_llm import OpenRouterModel, OpenRouterStrategy
model = OpenRouterModel.from_api_key("anthropic/claude-haiku-4-5")  # OPENROUTER_API_KEY
strategy = OpenRouterStrategy(model)
```

Install the matching extras:

```bash
pip install 'async-batch-llm[openai]'
pip install 'async-batch-llm[openrouter]'
```

The previous custom-strategy patterns in `examples/example_openai.py`
(pre-v0.9) still work; the example file has been rewritten to use the
new built-ins. If you're maintaining a custom strategy and want to
switch, deep dives are at [`OPENAI_INTEGRATION.md`](OPENAI_INTEGRATION.md)
and [`OPENROUTER_INTEGRATION.md`](OPENROUTER_INTEGRATION.md).

### Migration required if

Never required. The new providers are additive. Switch when convenient
to drop the custom-strategy code.

## New: built-in DeepSeek provider (v0.10.0)

`DeepSeekModel` / `DeepSeekStrategy` call DeepSeek's API directly. Unlike a
generic `OpenAICompatibleModel` pointed at DeepSeek, `DeepSeekModel` reads
DeepSeek's native `prompt_cache_hit_tokens` field into `cached_input_tokens`
(DeepSeek doesn't use OpenAI's nested `prompt_tokens_details.cached_tokens`),
so cache telemetry is accurate.

```python
from async_batch_llm import CachedTokenRates, DeepSeekModel, DeepSeekStrategy

model = DeepSeekModel.from_api_key("deepseek-chat")  # reads DEEPSEEK_API_KEY
strategy = DeepSeekStrategy(model)
# DeepSeek cache reads cost 10% of normal:
billable = result.effective_input_tokens(CachedTokenRates.DEEPSEEK)
```

```bash
pip install 'async-batch-llm[deepseek]'
```

DeepSeek is OpenAI-compatible, so reuse `OpenAIErrorClassifier` for its
error handling.

## New: `temperature=None` to omit the parameter (v0.10.0)

The `LLMModel` protocol, all built-in models, and `ModelStrategy` now accept
`temperature=None`, which **omits** the parameter from the API call so the
provider uses its own default. This is required for OpenAI reasoning models
(o1/o3/etc.) that reject an explicit `temperature`:

```python
model = OpenAIModel.from_api_key("o1-mini")
strategy = OpenAIStrategy(model, temperature=None)  # don't send temperature
```

The default is still `0.0`, so existing code is unaffected.

## Behavior change: `ErrorInfo.suggested_wait` is now honored (v0.10.0)

`suggested_wait` was previously populated but unused. It is now applied by the
`RateLimitCoordinator` as a **floor** on the cooldown (the `RateLimitStrategy`
may wait longer, never shorter). To avoid a hardcoded constant silently
overriding your configured cooldown, the field now carries **only genuine
server signals**: `OpenAIErrorClassifier` parses the `Retry-After` header, and
the old `DEFAULT_RATE_LIMIT_WAIT` fallbacks were removed (the `RateLimitStrategy`
owns the default cooldown). No action required unless you built a custom
classifier that relied on the old constant being surfaced.

## Compatibility cheatsheet

| You wrote                                                   | After upgrade                                                     |
|-------------------------------------------------------------|-------------------------------------------------------------------|
| `out, toks = await s.execute(...)` (built-in)               | **Update to 3-tuple unpack**                                      |
| `out, toks = await s.execute(...)` (custom)                 | Still works (custom returns 2-tuple)                              |
| Custom `LLMCallStrategy.execute()` returning 2-tuple        | Still works; opt into 3-tuple for metadata                        |
| `result.output`, `result.token_usage`, `result.error`, etc. | Unchanged                                                         |
| `result.gemini_safety_ratings`                              | Still populated (deprecated)                                      |
| `result.effective_input_tokens()`                           | Still defaults to Gemini rate; **warns** if cached tokens present |
| `temperature=0.0` (default)                                 | Unchanged; pass `None` to omit for reasoning models               |

## See also

- [`CHANGELOG.md`](https://github.com/geoff-davis/async-batch-llm/blob/main/CHANGELOG.md)
  for the full release-by-release detail.
- Issue [#8](https://github.com/geoff-davis/async-batch-llm/issues/8) for
  the design discussion behind the metadata change, including why we
  chose the 3-tuple-with-shim approach over a `RetryState`-based handoff.
