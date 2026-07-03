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

> **Reasoning models reject an explicit `temperature`.** Pass
> `temperature=None` (on the strategy or per `generate()` call) to omit the
> parameter entirely so the model uses its default — otherwise the call fails:
>
> ```python
> model = OpenAIModel.from_api_key("o1-mini")
> strategy = OpenAIStrategy(model, temperature=None)
> ```

## Structured output

Use the `json_mode=True` convenience to request JSON, and the built-in
`pydantic_json_parser` helper to parse it. The parser strips markdown code
fences before validating, so providers that wrap JSON in ```` ```json ... ``` ````
(DeepSeek does this even in JSON mode) validate cleanly instead of burning
retries on the fence characters:

```python
from pydantic import BaseModel

from async_batch_llm import OpenAIModel, OpenAIStrategy, pydantic_json_parser


class Sentiment(BaseModel):
    sentiment: str
    confidence: float


model = OpenAIModel.from_api_key(
    "gpt-4o-mini",
    api_key="sk-...",
    json_mode=True,  # adds response_format={"type": "json_object"}
    system_instruction='Respond with JSON: {"sentiment": ..., "confidence": ...}',
)
strategy = OpenAIStrategy(model, pydantic_json_parser(Sentiment))
```

`json_mode=True` is shorthand for
`extra_body={"response_format": {"type": "json_object"}}`; an explicit
`response_format` you pass in `extra_body` takes precedence. Most providers
still require the word "JSON" somewhere in the prompt for JSON mode to engage.

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
explicit rate when working with OpenAI** to get accurate numbers. As of
v0.10.0, calling it without an explicit rate while cached tokens are present
emits a `UserWarning` for exactly this reason; passing
`CachedTokenRates.OPENAI` silences it. Note that Anthropic charges a 25%
premium on cache *writes* over the normal input price; that write premium is
not modeled by this helper.

## Reasoning traces, tool calls, and logprobs

The OpenAI-compatible models (`OpenAIModel`, `OpenRouterModel`,
`DeepSeekModel`) surface additional structured output under reserved
`metadata` keys, readable through typed views on the result — see
[Typed auxiliary output](API.md#typed-auxiliary-output-grounding-reasoning-tool-calls-logprobs)
for the full contract:

- **`reasoning`** — the model's reasoning/thinking trace, read from
  `message.reasoning_content` (DeepSeek reasoner models) with a fallback to
  `message.reasoning` (OpenRouter). Access via `result.reasoning`.
- **`tool_calls`** — tool/function calls the model requested, as
  `[{"id", "name", "arguments"}]` with `arguments` kept as the raw JSON
  string. Access via `result.tool_calls` (a `list[ToolCall] | None`).
  Visibility only — the framework never executes tools; note that a pure
  tool-call turn (`content=null`) raises `EmptyResponseError`, so calls
  surface only alongside returned text.
- **`logprobs`** — the provider logprobs object (as a plain dict via
  `model_dump()`), when you requested it, e.g.
  `OpenAIStrategy(model, generation_config={"logprobs": True})`. Access via
  `result.logprobs`.

Each key is emitted only when present on the response, so default payloads
are unchanged unless you asked the model for these features.

## Error handling

`OpenAIErrorClassifier` understands the openai SDK's exception hierarchy:

- `RateLimitError` → retryable, rate-limit category. If the response carries a
  `Retry-After` header, it's parsed into `ErrorInfo.suggested_wait`, which the
  `RateLimitCoordinator` honors as a *floor* on the cooldown (the
  `RateLimitStrategy` still owns the default duration when there's no header).
- `APITimeoutError` → retryable, timeout.
- `APIConnectionError` → retryable, network.
- `APIStatusError` → branches on `status_code`:
  - 429 → rate limit.
  - 402 → not retryable, `insufficient_balance` category, with a remediation
    hint ("top up your prepaid DeepSeek balance"). Auth has passed, so this
    otherwise looks like a generic bug; the hint is logged at WARNING when the
    item gives up. Stops a dead balance from silently burning every retry.
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
    max_connections=50,           # size the httpx pool to match max_workers
    timeout=30.0,                 # forwarded to AsyncOpenAI
)
```

## Connection pool sizing (`max_connections`)

The openai SDK uses httpx's default connection pool (~100 connections). If you
raise `ProcessorConfig(max_workers=...)` above that, the extra workers just
block waiting for a connection — **throughput plateaus with no warning**. This
bites high-concurrency providers like DeepSeek hardest (it allows thousands of
concurrent connections, so the ~100 default — not the API — is your ceiling).

Pass `max_connections` to size the pool to your worker count:

```python
# Match the pool to max_workers (a little headroom doesn't hurt).
model = OpenAIModel.from_api_key("gpt-4o-mini", max_connections=150)
config = ProcessorConfig(max_workers=150, timeout_per_item=60.0)
```

`max_connections` sets both `max_connections` and `max_keepalive_connections`
on the underlying `httpx.AsyncClient`. It's a convenience for the common case;
if you need finer control, build your own `http_client=httpx.AsyncClient(...)`
and pass that instead (the two are mutually exclusive).

> **Slow-start, too.** Even with the pool raised, the default
> `RateLimitConfig` slow-start ramp bounds *time-to-full-throughput* on the
> first ~50 items. If you're chasing peak throughput, tune that as well.

## Subclassing for other OpenAI-compatible providers

`OpenAICompatibleModel` is exported so you can target Together, Fireworks,
local vLLM, etc. with a few lines:

```python
from async_batch_llm import OpenAICompatibleModel


class TogetherModel(OpenAICompatibleModel):
    _default_base_url = "https://api.together.xyz/v1"
    _install_extras = "openai"
```

The built-in `DeepSeekModel` is exactly this pattern — it additionally
overrides `_extract_tokens` to read DeepSeek's native cache-hit field. Read
its source for a worked example of customizing token extraction.

## See also

- [`docs/OPENROUTER_INTEGRATION.md`](OPENROUTER_INTEGRATION.md) — the
  multi-provider sibling.
- `DeepSeekModel` / `DeepSeekStrategy` — direct DeepSeek access with native
  cache-hit tracking (install `[deepseek]`); see
  [`examples/example_deepseek.py`](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_deepseek.py).
- [`examples/example_openai.py`](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_openai.py)
  — runnable example.
