# OpenRouter Integration

OpenRouter ([openrouter.ai](https://openrouter.ai)) exposes a unified
OpenAI-compatible API that fronts many upstream providers — Anthropic,
OpenAI, Google, DeepSeek, Mistral, Together, Fireworks, and more. First-class
support arrived in v0.9.0 via `OpenRouterModel`, `OpenRouterStrategy`, and
`OpenRouterErrorClassifier`.

## Installation

```bash
pip install 'async-batch-llm[openrouter]'
```

(Internally this installs the `openai` SDK; OpenRouter speaks the OpenAI
chat-completions wire format with a different `base_url`.)

## Authentication

Set `OPENROUTER_API_KEY`, or pass `api_key=` to
`OpenRouterModel.from_api_key()`. Keys live at
[openrouter.ai/keys](https://openrouter.ai/keys).

## Quick start

```python
import asyncio
from async_batch_llm import (
    LLMWorkItem,
    OpenRouterErrorClassifier,
    OpenRouterModel,
    OpenRouterStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
)


async def main() -> None:
    model = OpenRouterModel.from_api_key(
        "anthropic/claude-haiku-4-5",
        api_key="sk-or-...",
        # Optional — used by OpenRouter for app attribution / leaderboard.
        referer="https://my-app.example.com",
        title="My App",
    )
    strategy = OpenRouterStrategy(model)
    config = ProcessorConfig(max_workers=5, timeout_per_item=60.0)

    async with ParallelBatchProcessor[None, str, None](
        config=config,
        error_classifier=OpenRouterErrorClassifier(),
    ) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="hi", strategy=strategy, prompt="Hello!")
        )
        result = await processor.process_all()

    print(result.results[0].output)


asyncio.run(main())
```

## Model selection

OpenRouter model ids are prefixed with the upstream provider:

- `anthropic/claude-sonnet-4-5`, `anthropic/claude-haiku-4-5`
- `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/o1`
- `google/gemini-2.5-flash`, `google/gemini-2.5-pro`
- `deepseek/deepseek-chat-v3.1`, `deepseek/deepseek-reasoner`
- `meta-llama/llama-3.3-70b-instruct`
- `mistralai/mistral-large-latest`

See [openrouter.ai/models](https://openrouter.ai/models) for the full
catalog.

## Provider routing

OpenRouter lets you constrain which upstream host serves a request via the
`extra_body["provider"]` configuration. Forward it through `OpenRouterModel`:

```python
model = OpenRouterModel.from_api_key(
    "deepseek/deepseek-chat-v3.1",
    api_key="sk-or-...",
    extra_body={
        "provider": {
            # Try Fireworks first, then Together; fall back to others.
            "order": ["Fireworks", "Together"],
            # Or whitelist:
            # "allow_fallbacks": False,
            # "data_collection": "deny",
        }
    },
)
```

Full reference:
[openrouter.ai/docs/features/provider-routing](https://openrouter.ai/docs/features/provider-routing).

## Prompt caching

Caching behavior depends on which upstream provider serves your request.
**This is the most common gotcha when migrating between providers — read
this section.**

| Upstream provider               | Caching behavior                                         | Action required         |
|---------------------------------|----------------------------------------------------------|-------------------------|
| OpenAI                          | Automatic for prompts > ~1024 tokens                     | None                    |
| Gemini (implicit caching)       | Automatic for repeated long prefixes                     | None                    |
| DeepSeek                        | Automatic, on-disk context cache                         | None                    |
| Anthropic                       | **Opt-in** via `cache_control` markers on message blocks | Build structured prompt |
| Gemini explicit `CachedContent` | Not exposed via OpenRouter                               | Use `GeminiCachedModel` |

`cached_input_tokens` is populated when the upstream cache hits;
`BatchResult.total_cached_tokens` aggregates across the batch. To estimate
billable tokens, pass the matching rate from `CachedTokenRates`:

```python
from async_batch_llm import CachedTokenRates

# Pick the rate for whichever upstream actually served your request
# (visible in LLMResponse.metadata['provider']).
billable = result.effective_input_tokens(CachedTokenRates.OPENAI)
billable = result.effective_input_tokens(CachedTokenRates.ANTHROPIC_READ)
billable = result.effective_input_tokens(CachedTokenRates.DEEPSEEK)
```

Anthropic also charges a 25% premium on cache *writes*, which this helper
does not model.

### Per-item provider metadata for mixed-provider batches

As of v0.10.0, `OpenRouterStrategy` (and the other built-in strategies)
forwards `LLMResponse.metadata` straight through to
`WorkItemResult.metadata`. For OpenRouter that includes the upstream
provider name (`"Anthropic"`, `"OpenAI"`, `"DeepSeek"`, etc.), the
actually-routed model, and `finish_reason`. No custom parser needed.

```python
from async_batch_llm import (
    CachedTokenRates,
    LLMWorkItem,
    OpenRouterModel,
    OpenRouterStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
)

# Per-provider rate lookup.
PROVIDER_RATES = {
    "OpenAI": CachedTokenRates.OPENAI,
    "Anthropic": CachedTokenRates.ANTHROPIC_READ,
    "DeepSeek": CachedTokenRates.DEEPSEEK,
}

model = OpenRouterModel.from_api_key("openrouter/auto")
strategy = OpenRouterStrategy(model)

# After process_all():
total_billable = 0
for r in result.results:
    if not r.success:
        continue
    provider = (r.metadata or {}).get("provider")
    rate = PROVIDER_RATES.get(provider, CachedTokenRates.OPENAI)
    cached = r.token_usage.get("cached_input_tokens", 0)
    inp = r.token_usage.get("input_tokens", 0)
    discount = int(cached * (1.0 - rate))
    total_billable += inp - discount
```

`BatchResult.effective_input_tokens()` takes a single rate, so it's
appropriate when every item in the batch uses the same upstream. For
mixed batches, use the per-item arithmetic above.

If you need the provider info inside your output type (e.g. to feed into a
strict Pydantic model rather than read from `WorkItemResult.metadata`), a
custom `response_parser` still works:

```python
from dataclasses import dataclass


@dataclass
class TaggedOutput:
    text: str
    provider: str | None


strategy = OpenRouterStrategy(
    model,
    response_parser=lambda r: TaggedOutput(
        text=r.text,
        provider=(r.metadata or {}).get("provider"),
    ),
)
```

Both paths (reading `WorkItemResult.metadata` or capturing into a typed
output) are fully supported.

### Reasoning traces, tool calls, and logprobs

`OpenRouterModel` shares the OpenAI-compatible extractor, so reasoning
traces (`message.reasoning` on OpenRouter, `message.reasoning_content` on
DeepSeek-style upstreams), tool calls, and logprobs land under reserved
`metadata` keys with typed views on each per-item `WorkItemResult` (not the
batch-level `BatchResult`): `item_result.reasoning`,
`item_result.tool_calls`, `item_result.logprobs`. See
[Typed auxiliary output](API.md#typed-auxiliary-output-grounding-reasoning-tool-calls-logprobs)
for shapes and boundaries (**experimental** — shapes may change while they
stabilize).

### Anthropic prompt caching via OpenRouter

Anthropic requires you to mark the message blocks you want cached with
`cache_control: {"type": "ephemeral"}`. OpenRouter passes the markers
through. The cleanest way to do this through the framework is a small custom
strategy that builds the message list inside `execute()`:

```python
from async_batch_llm.base import LLMResponse, RetryState, TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy


class CachedAnthropicStrategy(LLMCallStrategy[str]):
    def __init__(self, model: OpenRouterModel, system_text: str) -> None:
        self.model = model
        self.system_text = system_text

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[str, TokenUsage]:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": prompt},
        ]
        response: LLMResponse = await self.model.generate(messages)
        return response.text, response.token_usage
```

The first call writes the cache (paid at 1.25× normal input rate). Calls
within ~5 minutes (default ephemeral TTL) hit the cache (paid at 0.1× normal
rate). Use a 1-hour TTL with `{"type": "ephemeral", "ttl": "1h"}` for
longer-lived caches if your account supports it.

### DeepSeek caching note

DeepSeek's automatic cache uses native field names
(`prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`) rather than OpenAI's
`prompt_tokens_details.cached_tokens`. Whether OpenRouter normalizes those
into `cached_tokens` depends on the upstream host and OpenRouter's current
mapping. If you specifically want reliable DeepSeek cache telemetry, call
DeepSeek's API directly with the built-in `DeepSeekModel` (v0.10.0), which
reads those native fields into `cached_input_tokens`:

```python
from async_batch_llm import DeepSeekModel, DeepSeekStrategy

model = DeepSeekModel.from_api_key("deepseek-chat")  # reads DEEPSEEK_API_KEY
strategy = DeepSeekStrategy(model)
```

## Cross-provider batches

Because every model is just a different `OpenRouterModel(model="...")`,
mixing providers in one batch is trivial:

```python
anthropic = OpenRouterModel.from_api_key(
    "anthropic/claude-haiku-4-5", api_key=KEY,
)
openai = OpenRouterModel.from_api_key(
    "openai/gpt-4o-mini", api_key=KEY,
)

async with ParallelBatchProcessor(...) as processor:
    await processor.add_work(LLMWorkItem(
        item_id="anthropic_q",
        strategy=OpenRouterStrategy(anthropic),
        prompt="...",
    ))
    await processor.add_work(LLMWorkItem(
        item_id="openai_q",
        strategy=OpenRouterStrategy(openai),
        prompt="...",
    ))
    result = await processor.process_all()
```

## Error handling

`OpenRouterErrorClassifier` extends `OpenAIErrorClassifier` and adds:

- 502 with body containing `no_provider_available` (no upstream host could
  serve the request) → retryable, `network_error`. Without the override
  these would otherwise look like generic server errors.

Everything else (rate limits, timeouts, 4xx vs 5xx) inherits from the OpenAI
classifier.

## See also

- [`docs/OPENAI_INTEGRATION.md`](OPENAI_INTEGRATION.md) — the OpenAI sibling.
- [`examples/example_openrouter.py`](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_openrouter.py)
  — runnable example with the cross-provider and cached-Anthropic patterns.
