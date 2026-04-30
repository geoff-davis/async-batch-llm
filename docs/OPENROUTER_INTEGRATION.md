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
`BatchResult.total_cached_tokens` aggregates across the batch.

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
mapping. If you specifically want reliable DeepSeek cache telemetry, calling
DeepSeek's API directly (an upcoming `DeepSeekModel`) is the better path.

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
