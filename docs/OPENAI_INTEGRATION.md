# OpenAI Integration

First-class OpenAI support arrived in v0.9.0 via `OpenAIModel`,
`OpenAIStrategy`, and `OpenAIErrorClassifier`.

## Installation

```bash
pip install 'async-batch-llm[openai]'
```

## Authentication

Set the `OPENAI_API_KEY` environment variable, or pass `api_key=` directly to
`OpenAIModel.from_api_key()`.

## Quick start

```python
import asyncio
from async_batch_llm import (
    LLMWorkItem,
    OpenAIErrorClassifier,
    OpenAIModel,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
)


async def main() -> None:
    model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-...")
    strategy = OpenAIStrategy(model)
    config = ProcessorConfig(max_workers=5, timeout_per_item=30.0)

    async with ParallelBatchProcessor[None, str, None](
        config=config,
        error_classifier=OpenAIErrorClassifier(),
    ) as processor:
        await processor.add_work(
            LLMWorkItem(item_id="hello", strategy=strategy, prompt="Hi!")
        )
        result = await processor.process_all()

    print(result.results[0].output)


asyncio.run(main())
```

## Choosing a model

`OpenAIModel` accepts any model id the OpenAI chat completions endpoint
serves: `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini`, etc. Reasoning models
(`o1`, `o3`) work, but if you need reasoning summaries or server-side tools,
the [Responses API](https://platform.openai.com/docs/api-reference/responses)
is a better fit; that's a future addition (`OpenAIResponsesModel`).

## Structured output

The simplest path is to ask for JSON in the prompt and parse it via
`response_parser`:

```python
from pydantic import BaseModel


class Sentiment(BaseModel):
    sentiment: str
    confidence: float


model = OpenAIModel.from_api_key(
    "gpt-4o-mini",
    api_key="sk-...",
    extra_body={"response_format": {"type": "json_object"}},
    system_instruction='Respond with JSON: {"sentiment": ..., "confidence": ...}',
)
strategy = OpenAIStrategy(
    model,
    response_parser=lambda r: Sentiment.model_validate_json(r.text),
)
```

For OpenAI specifically, `client.chat.completions.parse(response_format=...)`
also works — wrap it in a custom strategy that calls `parse()` directly. Kept
out of `OpenAIModel.generate()` so the same class can serve every
OpenAI-compatible provider.

## Prompt caching

OpenAI automatically caches prompt prefixes longer than ~1024 tokens. No
client action is required — `cached_input_tokens` is populated on hits, and
`BatchResult.total_cached_tokens` aggregates across the batch.

```python
from async_batch_llm import CachedTokenRates

result = await processor.process_all()
print(f"input={result.total_input_tokens} cached={result.total_cached_tokens}")
print(f"cache hit rate: {result.cache_hit_rate():.1f}%")
# OpenAI charges 50% of normal for cached tokens — pass the matching rate.
print(f"billable tokens: {result.effective_input_tokens(CachedTokenRates.OPENAI)}")
```

`effective_input_tokens()` defaults to `CachedTokenRates.GEMINI` (10% rate)
for backward compatibility with pre-v0.9.0 versions — **always pass an
explicit rate when working with OpenAI** to get accurate numbers. Note that
Anthropic charges a 25% premium on cache *writes* over the normal input
price; that write premium is not modeled by this helper.

## Error handling

`OpenAIErrorClassifier` understands the openai SDK's exception hierarchy:

- `RateLimitError` → retryable, rate-limit category, 60s default cooldown.
- `APITimeoutError` → retryable, timeout.
- `APIConnectionError` → retryable, network.
- `APIStatusError` → branches on `status_code`:
  - 429 → rate limit.
  - 408/425/500/502/503/504 → retryable server error.
  - 400/401/403/404/422 → not retryable (client error / auth / config).
- Pydantic `ValidationError` → retryable (LLM may produce valid output on
  retry).
- `ValueError`/`TypeError`/etc. → not retryable (logic bug).

Pass it to the processor:

```python
processor = ParallelBatchProcessor(
    config=config,
    error_classifier=OpenAIErrorClassifier(),
)
```

## Convenience constructor

```python
OpenAIModel.from_api_key(
    model="gpt-4o-mini",
    api_key="sk-...",
    base_url=None,                # override SDK default if needed
    system_instruction="...",     # default system message
    extra_headers={...},          # forwarded on every request
    extra_body={"response_format": {...}},  # default per-request kwargs
    timeout=30.0,                 # forwarded to AsyncOpenAI
)
```

## Subclassing for other OpenAI-compatible providers

`OpenAICompatibleModel` is exported so you can target Together, Fireworks,
local vLLM, etc. with a few lines:

```python
from async_batch_llm import OpenAICompatibleModel


class TogetherModel(OpenAICompatibleModel):
    _default_base_url = "https://api.together.xyz/v1"
    _install_extras = "openai"
```

## See also

- [`docs/OPENROUTER_INTEGRATION.md`](OPENROUTER_INTEGRATION.md) — the
  multi-provider sibling.
- [`examples/example_openai.py`](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_openai.py)
  — runnable example.
