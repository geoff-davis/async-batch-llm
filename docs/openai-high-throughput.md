# OpenAI-Compatible High-Throughput Guide

Use this configuration pattern for OpenAI, DeepSeek, OpenRouter, and other
OpenAI-compatible chat-completions endpoints when throughput is high enough that
connection pools, startup bursts, and timeout tails matter.

## Model-Owned Client

`from_api_key(max_connections=N)` lets ABL size and own the httpx client. The
model advertises `N`, so execution attempts are admitted before
`timeout_per_item` starts and never queue invisibly inside the transport.

```python
from async_batch_llm import (
    DeepSeekModel,
    DeepSeekStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    RateLimitConfig,
    RetryConfig,
    StartupRampConfig,
)

model = DeepSeekModel.from_api_key(
    "deepseek-v4-flash",
    thinking=False,
    max_connections=64,
    timeout=60.0,
    max_retries=0,  # let ABL own retry visibility/accounting
)
strategy = DeepSeekStrategy(model)
config = ProcessorConfig(
    max_workers=64,
    timeout_per_item=60.0,
    retry=RetryConfig(max_attempts=3, max_rate_limit_retries=20),
    rate_limit=RateLimitConfig(cooldown_seconds=30.0),
    startup_ramp=StartupRampConfig(
        initial_concurrency=4,
        concurrency_step=4,
        ramp_interval_seconds=1.0,
        max_concurrency=64,
        jitter_seconds=0.05,
    ),
)

async with ParallelBatchProcessor(config=config) as processor:
    ...
```

Set `max_workers` to the amount of useful application concurrency and
`max_connections` to the provider-call capacity. They are often equal. A larger
worker count can still help when middleware or post-processing occupies workers;
ABL will admit only the model's advertised capacity into `strategy.execute()`.

## User-Supplied Client

ABL cannot resize or reliably inspect a caller-supplied `AsyncOpenAI` client's
httpx pool. Configure the transport explicitly and declare the same capacity on
`ProcessorConfig`:

```python
import httpx
from openai import AsyncOpenAI
from async_batch_llm import OpenAIModel, OpenAIStrategy, ProcessorConfig

http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=32, max_keepalive_connections=32),
    timeout=httpx.Timeout(connect=10, read=60, write=30, pool=10),
)
client = AsyncOpenAI(
    api_key="sk-...",
    http_client=http_client,
    max_retries=0,
)
model = OpenAIModel("gpt-4o-mini", client)
strategy = OpenAIStrategy(model)
config = ProcessorConfig(
    max_workers=100,
    max_provider_concurrency=32,
    timeout_per_item=60,
)

# The caller owns a directly supplied client.
await client.close()
```

## Timeout Layers

| Layer | Scope |
| --- | --- |
| httpx connect/read/write/pool timeout | One transport operation |
| OpenAI SDK timeout | One SDK request |
| `timeout_per_item` | One `strategy.execute()` attempt, after ABL admission |
| Retry backoff and coordinated cooldown | Outside `timeout_per_item` |
| Gateway `submit_timeout` | Full caller wall time, including all waits/retries |

Avoid multiplying hidden SDK retries by ABL retries. Setting SDK
`max_retries=0` gives ABL complete attempt timing, classification, and failed-token
accounting. If SDK retries remain enabled, treat one `strategy.execute()` as the
outer attempt and size `timeout_per_item` for all SDK work inside it.

## Startup Ramp vs Cooldown

- Use `StartupRampConfig` when a cold endpoint or new connection pool is
  sensitive to an immediate burst. It starts at `initial_concurrency` and adds
  `concurrency_step` every `ramp_interval_seconds` until the lowest configured,
  advertised, or ramp maximum is reached.
- Lower `max_workers` when the provider has a stable hard concurrency limit and
  extra application workers add no value.
- Use `RateLimitConfig` for reactive behavior after a 429/quota response: one
  coordinated cooldown, followed by the existing post-cooldown slow-start.

Startup-ramp and provider-capacity wait occur before `timeout_per_item`.
`WorkItemResult.timing` separates them from execution, cooldown, and backoff.

## Gateway Services

For request-serving applications, construct one long-lived `LLMGateway` so all
callers share capacity and cooldown state. Set `max_pending` for load shedding
and `submit_timeout` for the end-to-end request budget:

```python
async with LLMGateway(
    strategy,
    config=config,
    max_pending=100,
    submit_timeout=30,
) as gateway:
    response = await gateway.submit("Summarize this request")
```

## Troubleshooting

| Symptom | Inspect | Corrective action |
| --- | --- | --- |
| Throughput plateaus below provider capacity | `max_connections`, `max_provider_concurrency`, execution p95 | Align pool/admission limits; raise only after measuring |
| High admission p95, normal execution p95 | `admission_wait_*` and startup-ramp wait | Raise capacity if the provider allows it, or lower worker/pending load |
| Framework execution timeouts | Attempt `timeout_category`, execution/provider duration | Raise per-attempt timeout or fix provider/client latency |
| httpx pool timeouts with a custom client | `httpx.Limits`, pool timeout, explicit ABL capacity | Set `max_provider_concurrency` no higher than the httpx pool |
| Initial 429/5xx burst | First-attempt errors and startup timing | Enable `StartupRampConfig`; do not confuse it with post-429 slow-start |
| Repeated 429s after recovery | Cooldown duration and rate-limit retry count | Lower steady capacity/RPM or increase coordinated cooldown |
| Long request tail in a service | Gateway admission and `submit_timeout` | Bound `max_pending`; set an end-to-end submit budget |

For the exact timing boundary, see the
[production checklist](production-checklist.md#4-timeout-and-concurrency-semantics).
