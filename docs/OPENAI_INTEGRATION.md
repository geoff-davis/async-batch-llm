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
    model = OpenAIModel.from_api_key("gpt-6-luna", api_key="sk-...")
    strategy = OpenAIStrategy(model)
    config = ProcessorConfig(max_workers=5, attempt_timeout=30.0)

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

OpenAI credential resolution remains with the installed SDK, including supported
admin-key and workload-identity authentication. Missing credentials raise
`ValueError`; other SDK configuration errors retain their original type.

## Retry ownership

`from_api_key()` defaults the OpenAI SDK's `max_retries` to zero, including
OpenRouter and DeepSeek models. The framework owns retries so its attempt counts,
timeouts, and quota accounting reflect provider requests. An explicit
`max_retries=N` overrides this default and is preserved when a client reopens.

If you supply an SDK client yourself, construct it with `max_retries=0` to
avoid retries inside a framework attempt. See the
[OpenAI SDK retry documentation](https://github.com/openai/openai-python#retries).

## Choosing a model

`OpenAIModel` accepts any OpenAI model id: `gpt-6-luna`, `gpt-6.1-sol`,
`gpt-6-astra`, and so on.

## Responses API (the default since v0.27)

`OpenAIModel` calls OpenAI's
[Responses API](https://platform.openai.com/docs/api-reference/responses) by
default and sends `store=False`, so requests aren't kept for later retrieval.
Set `store` in `generation_config` or `extra_body` to change that. For the
previous Chat Completions behavior, pass `api_surface="chat_completions"`:

```python
model = OpenAIModel.from_api_key("gpt-6-luna", api_surface="chat_completions")
```

Switching surfaces changes automatic artifact identity: Responses results aren't
replayed as Chat Completions results, and the reverse. v0.26 checkpoints replay only
with `api_surface="chat_completions"`. See
[the v0.27 migration guide](migration/v0.27.md#openaimodel-uses-the-responses-api).

Chat-style request fields are translated, so most existing configurations
keep working:

| You pass | Sent to the Responses API |
| --- | --- |
| `max_tokens` / `max_completion_tokens` | `max_output_tokens` |
| `response_format` (JSON object or JSON schema), `json_mode=True` | `text.format` |
| `reasoning_effort` | `reasoning.effort` |
| `logprobs=True`, `top_logprobs` | `include=["message.output_text.logprobs"]`, `top_logprobs` |
| Chat `tools` / `tool_choice` | Responses function tools. An omitted `strict` becomes `false`, matching Chat. |
| Chat message lists, including image parts, assistant `tool_calls` and `tool` replies | Responses input items |

Fields with no Responses equivalent (`n`, `stop`, `seed`,
`presence_penalty`, `frequency_penalty`, `logit_bias`, legacy
`functions`/`function_call`), unsupported content parts (audio, files in Chat
form), and `stream`/`background` raise `ValueError` before any request is
sent. Use the Chat Completions opt-out for those.

Results keep the Chat Completions vocabulary for `metadata["finish_reason"]`:

- `"stop"` for a completed response;
- `"tool_calls"` when the model called functions;
- `"length"` when it stopped at the token limit;
- `"content_filter"` when a filter cut it off.

The raw Responses status is in `metadata["response_status"]`.
`metadata["reasoning_tokens"]` reports reasoning tokens. They're already
included in `output_tokens`, so cost math is unchanged.

Other metadata keys, each emitted only when the response has a value:

- Responses surface: `api_surface` (always `"responses"`), `provider_request_id`
  (the response id), `model`, `refusal` (refusal text returned alongside output),
  plus the reserved `reasoning`, `tool_calls`, and `logprobs` keys described below.
- Chat Completions surface: the shared OpenAI-compatible keys (`model`,
  `finish_reason`, and so on) plus `reasoning_tokens` from
  `completion_tokens_details`.

Some outcomes differ from Chat Completions:

- A completed response that contains **only function calls** succeeds with
  `text=""`, `finish_reason="tool_calls"`, and the calls in
  `result.tool_calls`. The Chat Completions surface still raises
  `EmptyResponseError` for a textless reply.
- A response cut off at the token limit succeeds with the partial text and
  `finish_reason="length"`, as on Chat Completions. Without any text, it raises
  `EmptyResponseError`.
- A refusal raises `EmptyResponseError`.
- A failed response raises `ProviderResponseError` with the usage attached.
  Billing (`insufficient_quota`) and rate-limit failures classify as usual.

**Other endpoints.** The Responses default applies only to `OpenAIModel`. For
vLLM, Together, proxies, or any other OpenAI-compatible server, use
`OpenAICompatibleModel`, which stays on Chat Completions (see
[Other OpenAI-compatible providers](#other-openai-compatible-providers)).
`OpenRouterModel` is unaffected. There's no automatic fallback: pointing
`OpenAIModel` at a server without a `/responses` endpoint fails at request
time. The same goes for `OPENAI_BASE_URL` and a client that exposes
`responses.create` but talks to a server without the endpoint.

> **Temperature is omitted by default.** Built-in models send no `temperature`
> unless you pass one, so each model uses its provider default. Some models or
> reasoning modes reject an explicit value, so leave it unset for them. GPT-6 models
> reason by default and accept a temperature only with reasoning effort `none`. For
> more repeatable output, turn reasoning off and pass one explicitly:
>
> ```python
> model = OpenAIModel.from_api_key("gpt-6-luna")
> strategy = OpenAIStrategy(
>     model, temperature=0.0, generation_config={"reasoning_effort": "none"}
> )
> ```

## Structured output

Trailing-markdown recovery uses the same Pydantic JSON validation as ordinary
responses. Strict models therefore accept JSON representations of dates and tuples
consistently on both paths.

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
    "gpt-6-luna",
    api_key="sk-...",
    json_mode=True,  # adds response_format={"type": "json_object"}
    system_instruction='Respond with JSON: {"sentiment": ..., "confidence": ...}',
)
strategy = OpenAIStrategy(
    model,
    pydantic_json_parser(Sentiment, recover_trailing_markdown=True),
)
```

`json_mode=True` is shorthand for
`extra_body={"response_format": {"type": "json_object"}}`; an explicit
`response_format` you pass in `extra_body` takes precedence. Most providers
still require the word "JSON" somewhere in the prompt for JSON mode to engage.

Parsing remains strict first. When `recover_trailing_markdown=True`, a failed
strict parse gets one conservative fallback: decode exactly one complete
top-level JSON object or array with Python's JSON decoder, accept the remainder
only when it is the recognized closing fence (three backticks, with or without
the observed trailing underscore), then run normal Pydantic schema validation.
Malformed JSON, scalar values, multiple JSON values, arbitrary prose, and
schema-invalid data are not accepted and proceed through the configured
validation retry policy.

Successful recovery preserves provider metadata and token/cost accounting. It
sets `result.structured_output_recovered`,
`result.structured_output_recovery_reason`, and
`result.structured_output_retries_avoided`; processor stats and
`MetricsObserver` aggregate the same signal. Leave the option off when trailing
content should always be treated as a hard validation failure.

For strict schema output, pass a `json_schema` `response_format`. No custom
strategy is needed; on the Responses surface it's sent as `text.format` with the
same `name`, `schema`, and `strict` fields:

```python
from pydantic import BaseModel

from async_batch_llm import OpenAIModel, OpenAIStrategy, pydantic_json_parser

class Sentiment(BaseModel):
    model_config = {"extra": "forbid"}  # strict mode needs additionalProperties: false

    sentiment: str
    confidence: float

model = OpenAIModel.from_api_key("gpt-6-luna", api_key="sk-...")
strategy = OpenAIStrategy(
    model,
    pydantic_json_parser(Sentiment),
    generation_config={
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment",
                "schema": Sentiment.model_json_schema(),
                "strict": True,
            },
        }
    },
)
```

The SDK's own parse helpers (`client.responses.parse(...)`, or
`client.chat.completions.parse(...)` with `api_surface="chat_completions"`) also
work from a custom strategy if you want the SDK to build the schema.

## Prompt caching

OpenAI automatically caches prompt prefixes longer than ~1024 tokens. No
client action is required — `cached_input_tokens` is populated on hits, and
`BatchResult.total_cached_tokens` aggregates across the batch.

```python
from async_batch_llm import CachedTokenRates

result = await processor.process_all()
print(f"input={result.total_input_tokens} cached={result.total_cached_tokens}")
print(f"cache hit rate: {result.cache_hit_rate:.1f}%")
# CachedTokenRates.OPENAI is 50% (gpt-4o-era pricing); newer models often
# charge less for cached input. Check your model's pricing and pass that rate.
print(f"billable tokens: {result.effective_input_tokens(CachedTokenRates.OPENAI)}")
```

Always pass an explicit rate. Omitting it is deprecated since v0.27 and the
argument becomes required in 1.0. Until then an omitted rate falls back to
`CachedTokenRates.GEMINI` (10%) and warns: a `UserWarning` when cached tokens
are present, since the Gemini rate is wrong for OpenAI, and a
`DeprecationWarning` otherwise. Note that Anthropic charges a 25%
premium on cache *writes* over the normal input price; that write premium is
not modeled by this helper.

## Reasoning traces, tool calls, and logprobs

The OpenAI-compatible models (`OpenAIModel`, `OpenRouterModel`,
`DeepSeekModel`) surface additional structured output under reserved
`metadata` keys, readable through typed views on each per-item
`WorkItemResult` (or `LLMResponse`) — not on the batch-level `BatchResult` — see
[Typed auxiliary output](api/core.md#typed-auxiliary-output-grounding-reasoning-tool-calls-logprobs)
for the shapes and boundaries (**experimental** — shapes may change while
they stabilize):

- **`reasoning`** — the model's reasoning/thinking trace. On Chat
  Completions it comes from `message.reasoning_content` (DeepSeek reasoner
  models), falling back to `message.reasoning` (OpenRouter). On the Responses
  API it comes from reasoning-item text when the provider returns it; `OpenAIModel`
  falls back to the joined reasoning summaries (`DeepSeekModel` doesn't). OpenAI returns summaries only when you ask for
  them, e.g. `generation_config={"reasoning": {"summary": "auto"}}`.
  Encrypted reasoning is never read. Access via `item_result.reasoning`.
- **`tool_calls`** — function calls the model requested, as
  `[{"id", "name", "arguments"}]` with `arguments` kept as the raw JSON
  string. Access via `item_result.tool_calls` (a `list[ToolCall] | None`).
  Visibility only: the framework never executes tools. On the Responses API a
  function-call-only turn succeeds with empty text. On Chat Completions a pure
  tool-call turn (`content=null`) raises `EmptyResponseError`, so calls surface
  only alongside returned text. Built-in Responses tools such as web search are
  not function calls and aren't included.
- **`logprobs`** — the provider's logprobs as plain JSON, when you requested
  them, e.g. `OpenAIStrategy(model, generation_config={"logprobs": True})`. On
  Chat Completions this is the logprobs object; on the Responses API it's a list
  of per-token entries. Access via `item_result.logprobs`.

Each key is emitted only when present on the response, so default payloads
are unchanged unless you asked the model for these features.

## Error handling

`OpenAIErrorClassifier` understands the openai SDK's exception hierarchy:

- `RateLimitError` → retryable, rate-limit category. If the response carries a
  `retry-after-ms` (preferred) or `Retry-After` header, it is parsed into
  `ErrorInfo.suggested_wait`, which the
  `RateLimitCoordinator` honors as a *floor* capped by `rate_limit.max_cooldown_seconds` (the
  `RateLimitStrategy` still owns the default duration when there's no header).
- `APITimeoutError` → retryable, timeout.
- `APIConnectionError` → retryable, network.
- `APIStatusError` → branches on `status_code`:
  - 429 → rate limit.
  - 402 → not retryable, `insufficient_balance` category, with a remediation
    hint. The hint is shared across providers and currently names DeepSeek as
    its example ("top up your prepaid DeepSeek balance"). Auth has passed, so this
    otherwise looks like a generic bug; the hint is logged at WARNING when the
    item gives up. Stops a dead balance from silently burning every retry.
  - 408/425/500/502/503/504 → retryable server error.
  - 400/401/403/404/405/409/410/422 → not retryable (client error / auth / config).
- An explicit `insufficient_quota` code (for example on a 429, or on a failed
  Responses `ProviderResponseError`) → `insufficient_balance`, not retryable, and
  no coordinated cooldown.
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
    model="gpt-6-luna",
    api_key="sk-...",
    base_url=None,                # override SDK default if needed
    system_instruction="...",     # default system message
    extra_headers={...},          # forwarded on every request
    extra_body={"response_format": {...}},  # default per-request kwargs
    json_mode=False,              # True adds response_format={"type": "json_object"}
    max_connections=50,           # size the httpx pool to match max_workers
    metadata_extractors=None,     # extra metadata keys (see api/core.md)
    api_surface="responses",      # or "chat_completions"
    timeout=30.0,                 # other kwargs are forwarded to AsyncOpenAI
)
```

`OpenAICompatibleModel`, `OpenRouterModel`, and `DeepSeekModel` take the same
arguments except `api_surface`; `DeepSeekModel` adds its own `api_surface` and
`thinking` options.

## Connection pool sizing (`max_connections`)

Without `max_connections`, the openai SDK builds its own httpx client with
`max_connections=1000` and `max_keepalive_connections=100` (the SDK's
`DEFAULT_CONNECTION_LIMITS`; the same values in openai 1.66.2, the minimum ABL
supports, and in current 3.x releases). That has two effects at high concurrency:

- Above 100 concurrent requests, connections beyond the keep-alive limit are
  closed after each response and reopened for the next one, so you pay extra
  TCP/TLS handshakes.
- Above 1000 concurrent requests, extra workers block waiting for a connection,
  and that wait counts against `attempt_timeout`.

ABL can't see the SDK's default pool, so it doesn't warn or gate on it. The
exception is `ProcessorConfig(concurrency=N)`: before the first request it
rebuilds an owned default-pool client with a pool of `N`.

Pass `max_connections` to size the pool to your worker count:

```python
# Match the pool to max_workers (a little headroom doesn't hurt).
model = OpenAIModel.from_api_key("gpt-6-luna", max_connections=150)
config = ProcessorConfig(max_workers=150, attempt_timeout=60.0)
```

`max_connections` sets both `max_connections` and `max_keepalive_connections`
on the underlying `httpx.AsyncClient`. It's a convenience for the common case;
if you need finer control, build your own `http_client=httpx.AsyncClient(...)`
and pass that instead (the two are mutually exclusive).

ABL records `max_connections` as `model.max_concurrency`; `ModelStrategy`
forwards it, and processors/gateways warn when `max_workers` exceeds the known
capacity. The shared executor automatically gates attempts at that capacity
before `attempt_timeout` starts. For a user-supplied client, declare the limit
on `ProcessorConfig` because ABL cannot inspect the transport reliably:

```python
import httpx
from openai import AsyncOpenAI

http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
    timeout=httpx.Timeout(60),
)
client = AsyncOpenAI(api_key="sk-...", http_client=http_client)
model = OpenAIModel("gpt-6-luna", client)  # caller owns and closes client
config = ProcessorConfig(
    max_workers=100,
    max_provider_concurrency=64,
    attempt_timeout=60,
)
```

The explicit limit keeps attempts from waiting for httpx connections inside
`strategy.execute()`, where such a wait would count against
`attempt_timeout`. See the
[timeout and concurrency semantics](production-checklist.md#4-timeout-and-concurrency-semantics)
for the full boundary.

> **Post-rate-limit slow-start:** `RateLimitConfig.slow_start_*` applies only
> after a rate-limit cooldown. It does not ramp the initial batch startup.

## Other OpenAI-compatible providers

`OpenAICompatibleModel` targets any Chat Completions server (vLLM, Together,
Fireworks, proxies) without subclassing:

```python
from async_batch_llm import OpenAICompatibleModel, OpenAIStrategy, llm

model = OpenAICompatibleModel.from_api_key(
    "meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="token",  # falls back to OPENAI_API_KEY
)
strategy = OpenAIStrategy(model)

# Or through the factory; base_url is required for this prefix.
strategy = llm(
    "openai-compatible:meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
)
```

For a provider you use often, a small subclass keeps the URL in one place:

```python
from async_batch_llm import OpenAICompatibleModel

class TogetherModel(OpenAICompatibleModel):
    _default_base_url = "https://api.together.xyz/v1"
    _install_extras = "openai"
```

The built-in `DeepSeekModel` is exactly this pattern. It also overrides
`_extract_tokens` to read DeepSeek's native cache-hit field; read its source for
a worked example of customizing token extraction.

## See also

- [`docs/OPENROUTER_INTEGRATION.md`](OPENROUTER_INTEGRATION.md) — the
  multi-provider sibling.
- `DeepSeekModel` / `DeepSeekStrategy` — direct DeepSeek access with native
  cache-hit tracking (install `[deepseek]`); see
  [`examples/example_deepseek.py`](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_deepseek.py).
- [`examples/example_openai.py`](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_openai.py)
  — runnable example.
