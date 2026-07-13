# async-batch-llm

**async-batch-llm runs independent LLM calls concurrently with production-grade
retries, coordinated rate-limit cooldowns, bounded input buffering, resumable
checkpoints, deadlines, and token accounting—including failed attempts.**

Use it for results you need during the current workflow. For latency-tolerant
jobs, provider batch APIs may offer lower prices in exchange for a longer
turnaround.

Provider-agnostic (OpenAI, Anthropic, Google, DeepSeek, OpenRouter, PydanticAI, or your own)
through a simple strategy pattern; built on asyncio for I/O-bound throughput.

[![PyPI version](https://badge.fury.io/py/async-batch-llm.svg)](https://badge.fury.io/py/async-batch-llm)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/geoff-davis/async-batch-llm/workflows/Tests/badge.svg)](https://github.com/geoff-davis/async-batch-llm/actions)
[![Coverage](https://raw.githubusercontent.com/geoff-davis/async-batch-llm/python-coverage-comment-action-data/badge.svg)](https://github.com/geoff-davis/async-batch-llm/tree/python-coverage-comment-action-data)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://geoff-davis.github.io/async-batch-llm/)

**📚 [Read the Documentation](https://geoff-davis.github.io/async-batch-llm/)**

**Start here:** [Quick start](#quick-start) ·
[Choose an execution surface](#choose-an-execution-surface) ·
[Production-safe resumable run](#production-safe-resumable-run) ·
[Why not just use `gather`?](#why-not-just-use-gather) ·
[Documentation](#documentation)

---

## Quick Start

### Installation

Install the provider extra you need. For OpenAI:

```bash
pip install 'async-batch-llm[openai]'
```

Other extras are `pydantic-ai`, `gemini`, `openrouter`, `deepseek`, and `all`.
The core package alone is `pip install async-batch-llm`. The `uv sync` and
`make` workflows in [Development Setup](#development-setup) are for contributors
working from a clone, not for applications installing from PyPI.

### Run a batch

```python
import asyncio

from async_batch_llm import OpenAIModel, OpenAIStrategy, ProcessorConfig, process_prompts


async def main() -> None:
    strategy = OpenAIStrategy(
        OpenAIModel.from_api_key("gpt-4o-mini")  # reads OPENAI_API_KEY
    )
    result = await process_prompts(
        strategy,
        ["Summarize document A", "Summarize document B"],
        # Ten worker tasks; provider-capacity admission may allow fewer calls.
        config=ProcessorConfig(max_workers=10),
    )

    print(f"{result.succeeded}/{result.total_items} succeeded")
    for item in result.results:  # completion order by default
        print(item.item_id, item.success, item.output or item.error)


asyncio.run(main())
```

Pass `(item_id, prompt)` pairs to control IDs, or `(item_id, prompt, context)`
triples to carry application data into each result. Collected results default
to completion order. Pass `preserve_order=True`, or call
`result.in_input_order()`, when stable submission order is required.

For incremental handling, use `process_stream` with a bounded work queue:

```python
from async_batch_llm import ProcessorConfig, process_stream

config = ProcessorConfig(max_workers=50, max_queue_size=200)

async for item in process_stream(strategy, huge_prompt_source, config=config):
    await save(item)  # results arrive in completion order
```

`max_queue_size` bounds pending input and applies producer backpressure. The
result handoff queue is not bounded, so consumers should process results
promptly rather than launching an unbounded number of background save tasks.
`process_prompts` always retains every result by design.

### Choose an execution surface

| Need | API |
| --- | --- |
| Collect a finite run | `process_prompts()` |
| Handle results incrementally | `process_stream()` |
| Execute one resilient request | `call()` / `call_result()` |
| Share limits across service requests | `LLMGateway` |
| Customize queueing and lifecycle | `ParallelBatchProcessor` |

The batch, stream, single-call, and gateway surfaces share the same execution,
retry, timing, and token-accounting pipeline.

### Production-safe resumable run

With an application strategy and prompt source already defined:

```python
from pathlib import Path

from async_batch_llm import (
    AbortMode,
    ArtifactIdentity,
    GuardrailConfig,
    JsonlArtifactStore,
    ProcessorConfig,
    ResumePolicy,
    process_prompts,
)

store = JsonlArtifactStore(
    "runs/invoice-extraction.jsonl",
    identity=ArtifactIdentity(
        provider="openai",
        model="gpt-4o-mini",
        prompt_version="invoice-v4",
        parser_version="invoice-schema-v2",
        application_version="billing-pipeline-v7",
    ),
)
config = ProcessorConfig(
    max_workers=20,
    timeout_per_item=30,  # one provider attempt
    guardrails=GuardrailConfig(
        total_timeout_per_item=180,  # all waits and retries for one item
        batch_timeout=3600,
        abort_on_error_categories=frozenset(
            {"authentication", "insufficient_balance"}
        ),
        abort_mode=AbortMode.DRAIN_ACTIVE,
    ),
)

result = await process_prompts(
    strategy,
    prompts,
    config=config,
    artifact_store=store,
    resume=ResumePolicy.REUSE_SUCCESSES,
    preserve_order=True,
)

print(result.termination.kind, result.succeeded, result.failed)
Path("summary.json").write_text(result.to_json(), encoding="utf-8")
```

Each newly executed terminal result is flushed before it is returned or
streamed. Replay compatibility includes item ID, prompt, participating context,
and the complete artifact identity—not merely `item_id`.
`REUSE_SUCCESSES` reruns prior failures; `REUSE_ALL` also replays compatible
terminal failures.

Raw prompts and contexts are excluded from artifacts by default. Outputs and
metadata are included by default because successful replay needs the output;
they may contain sensitive application data. Replayed token usage remains on
the result for audit, while live processor statistics exclude those historical
tokens from spending in the current run.

See [Results, Artifacts, and Resume](docs/results-and-artifacts.md) and
[Deadlines and Fail-Fast Guardrails](docs/guardrails.md) for schema,
privacy, durability, and timeout details. A complete version is available in
[`examples/example_production_resume.py`](examples/example_production_resume.py).

## A sense of scale

One concrete [GSM8K test-split run](docs/benchmarks.md) is more useful than an
abstract "fast". Treat these as order-of-magnitude examples, not promises:
model latency, account limits, pricing, and network all move the numbers.

- **Thirty independent calls: seconds instead of a minute.** The serial race took
  39–65 s. With a worker pool it finished in 2.1–4.2 s on the uncapped providers
  (15.6–19.1× faster). The throttle-capped Gemini 2.5 run used 5 workers and
  landed at 8.1 s (5.0×).
- **A thousand-call pool actually fills.** At equal concurrency on 1,000 prompts,
  async-batch-llm processed 72 items/s on DeepSeek and 108 items/s on Gemini 3.1,
  versus 58 and 55 items/s for a fair `Semaphore` pool. The exact multiple is
  run-specific; the durable point is bounded workers and backpressure without
  scheduling every coroutine up front.
- **The full 1,319-item split made provider tradeoffs visible.** DeepSeek Flash
  completed for **$0.054** at 97.0% accuracy; Gemini 3.1 cost **$0.433** at
  96.6%; Gemini 2.5 cost **$0.261** at 95.4% but took ~21.5 minutes because it
  was capped
  at 5 workers. The package does not make provider calls cheaper; it makes the
  cost/latency/accuracy tradeoffs visible and keeps the provider swap small.
- **Retries are part of the run, not an afterthought.** DeepSeek recovered 9
  parse failures; the rough Gemini 2.5 run recorded 120 retries, 41 model
  escalations, and transient 503s/timeouts, finishing at 95.4% accuracy with 2
  permanent errors. A bare `gather` loop would make that error handling and cost
  accounting your problem.

See the [benchmarks](docs/benchmarks.md) for methodology and the full tables.

## Why not just use `gather`?

The 90% version is a semaphore and a `gather`. Here's what those few lines *don't* handle:

```python
import asyncio

sem = asyncio.Semaphore(20)  # cap concurrency

async def call_one(prompt: str) -> str:
    async with sem:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

results = await asyncio.gather(*(call_one(p) for p in prompts))
```

- **429 / rate limits** — no coordinated cooldown; every task keeps hammering a throttled endpoint.
- **Validation failures** — a malformed/unparseable response is just *returned*; no retry, no
  "escalate to a smarter or thinking model when the output is bad".
- **Transient errors** — one raised exception loses the whole batch; `return_exceptions=True` only
  trades that for `Exception` objects salted through your results to hand-filter.
- **Cost** — no idea what you spent, and *zero* accounting for tokens burned on failed attempts.
- **Input memory** — `gather()` materializes every coroutine up front; it does
  not provide lazy input consumption or producer backpressure.

async-batch-llm *is* that loop with the operational layer filled in — coordinated cooldowns,
error-type-aware retries, token/cost accounting (including failures), bounded input streaming, and a
one-line provider swap.

## When NOT to use this

- **You can wait hours.** If the job is latency-tolerant, provider batch APIs
  may offer a discount in exchange for delayed results. Check the current
  pricing and turnaround guarantees for the provider and model you use.
- **The operational layer is unnecessary.** For a small script where retries,
  validation, token accounting, coordinated cooldowns, and bounded input do not
  matter, `asyncio.gather` with a semaphore may be enough. If you want those
  guarantees for one request or a service path without batch queueing, use
  [`call()` / `LLMGateway`](#single-calls-and-a-shared-gateway).

---

## Advanced processor control

### Direct processor API

For custom queueing, per-item context, or fine-grained lifecycle control, drive
the `ParallelBatchProcessor` directly — `process_prompts` is a thin wrapper over it:

```python
import asyncio
from async_batch_llm import (
    LLMWorkItem,
    OpenAIModel,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
)

documents = ["Document 1 text...", "Document 2 text..."]

async def main():
    model = OpenAIModel.from_api_key("gpt-4o-mini")   # reads OPENAI_API_KEY
    strategy = OpenAIStrategy(model)
    config = ProcessorConfig(max_workers=10)  # provider admission may impose a lower limit

    async with ParallelBatchProcessor(config=config) as processor:
        for i, doc in enumerate(documents):
            await processor.add_work(
                LLMWorkItem(item_id=f"doc_{i}", strategy=strategy, prompt=f"Summarize: {doc}")
            )
        result = await processor.process_all()

    print(f"Succeeded: {result.succeeded}/{result.total_items}")
    print(f"Tokens used: {result.total_input_tokens + result.total_output_tokens}")

asyncio.run(main())
```

Switching providers is a one-line change (`DeepSeekModel` / `GeminiModel` /
`OpenRouterModel`, or a custom strategy). For **structured output** pass a
`response_parser` (or use `PydanticAIStrategy`); for **smart retries**,
**caching**, and **observability**, see [Core Features](#core-features) below and
the [`examples/`](examples/) directory.

`pydantic_json_parser(Model)` is strict by default. For providers that
occasionally append a stray closing Markdown fence after one otherwise valid
JSON object or array, opt into conservative recovery with
`pydantic_json_parser(Model, recover_trailing_markdown=True)`. Recovery never
repairs malformed JSON or discards prose/multiple values, and is visible through
`WorkItemResult.structured_output_recovered` and aggregate metrics.

---

## Core Features

### Any LLM Provider

Built-in strategies for common providers:

- **`PydanticAIStrategy`** - PydanticAI agents with structured output
- **`GeminiStrategy`** - Google Gemini; returns response text by default or accepts a
  `response_parser` for structured output. Wrap a `GeminiCachedModel` for context caching.
- **`OpenAIStrategy`** - OpenAI (`OpenAIModel.from_api_key(...)`)
- **`OpenRouterStrategy`** - OpenRouter, a single OpenAI-compatible API in front of Anthropic,
  Google, DeepSeek, etc. (`OpenRouterModel.from_api_key(...)`)
- **`DeepSeekStrategy`** - DeepSeek direct, with native cache-hit token tracking
  (`DeepSeekModel.from_api_key(...)`)

`OpenAIStrategy`/`OpenRouterStrategy`/`DeepSeekStrategy` are thin subclasses of `ModelStrategy`;
their models all extend `OpenAICompatibleModel`, so any OpenAI-compatible endpoint (Together,
Fireworks, vLLM, …) works by subclassing it. Built-in usage is a two-liner:

```python
from async_batch_llm import OpenAIModel, OpenAIStrategy

model = OpenAIModel.from_api_key("gpt-4o-mini")  # reads OPENAI_API_KEY
strategy = OpenAIStrategy(model)
```

For a provider without a built-in, write a custom strategy:

```python
from async_batch_llm import LLMCallStrategy, TokenUsage

class MyProviderStrategy(LLMCallStrategy[str]):
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
        response = await self.client.generate(prompt, model=self.model)
        tokens: TokenUsage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        # 3-tuple (output, tokens, metadata); metadata may be None.
        return response.text, tokens, None
```

See the [`examples/`](examples/) directory for OpenAI, OpenRouter, DeepSeek, Anthropic,
LangChain, and more.

#### DeepSeek quickstart

DeepSeek can support high concurrency, subject to your account and the current
service limits. The footgun: the OpenAI SDK's httpx connection pool may be
smaller than `max_workers`, so raising the worker count past the pool size gives
no extra throughput (workers just block on the pool). Size `max_connections` to
match `max_workers`:

```python
from async_batch_llm import DeepSeekModel, DeepSeekStrategy

model = DeepSeekModel.from_api_key(
    "deepseek-v4-flash",   # reads DEEPSEEK_API_KEY; V4 defaults thinking ON (pricey for batch)
    thinking=False,        # turn it off for classification/extraction
    max_connections=150,   # size the httpx pool to your max_workers
)
strategy = DeepSeekStrategy(model)
```

Models built this way advertise `max_connections` to the strategy. The batch
processor and gateway automatically admit only that many execute attempts before
`timeout_per_item` starts. For a caller-supplied OpenAI/httpx client, set
`ProcessorConfig(max_provider_concurrency=N)` explicitly. Admission timing is
available on each `WorkItemResult` and in processor/observer metrics. See the
[production timeout and concurrency semantics](docs/production-checklist.md#4-timeout-and-concurrency-semantics).

For burst-sensitive endpoints, `StartupRampConfig` can begin at a small
concurrency and step up to the steady limit without charging ramp wait to the
execution timeout. `WorkItemResult.timing` records per-try admission, ramp,
execution, provider, cooldown, backoff, classification, and timeout details.
See the [high-throughput guide](docs/openai-high-throughput.md).

[`examples/example_deepseek.py`](examples/example_deepseek.py) has the full
version: JSON mode with markdown-fence-tolerant parsing (`pydantic_json_parser`),
`402 Insufficient Balance` handling, and cache-hit token accounting
(`CachedTokenRates.DEEPSEEK`).

### Automatic Retries

Configure retry behavior with exponential backoff and jitter:

```python
from async_batch_llm import RetryConfig

config = ProcessorConfig(
    max_workers=5,
    timeout_per_item=30.0,
    retry=RetryConfig(
        max_attempts=3,             # budget for content/transport failures
        initial_wait=1.0,
        exponential_base=2.0,
        jitter=True,                # Prevents thundering herd
        max_rate_limit_retries=20,  # rate-limit retries are budgeted separately
    ),
)
```

The framework automatically retries on validation errors, network errors, and other transient failures.

**Rate limits don't consume your retry budget.** `max_attempts` bounds *content
and transport* failures (bad/invalid output, timeouts, connection errors, 5xx).
A `429` / quota / coordinated-cooldown signal is a "wait and try again", not a
failed attempt — so it's retried at the **same logical attempt number** after the
cooldown and is bounded separately by `max_rate_limit_retries` (default 20; set
to `0` to make rate limits fail immediately). When that separate budget is
exceeded the item fails with a `RateLimitRetriesExceeded` error (token usage
included, like any other exhausted failure).

This keeps the `attempt` number that `execute()` sees meaningful: a
model-escalation strategy (escalate to a smarter/thinking model on attempt ≥ 2)
escalates because the *output* was bad over `max_attempts` tries — never just
because the endpoint was busy.

For **error-*type*-aware** retries — retry the cheap model on transient/rate-limit errors, but
escalate to a smarter or thinking model only when the *output* is bad — see
[`examples/example_smart_model_escalation.py`](examples/example_smart_model_escalation.py).

### Rate Limiting

Coordinated rate limit handling across all workers:

```python
from async_batch_llm import RateLimitConfig

config = ProcessorConfig(
    rate_limit=RateLimitConfig(
        cooldown_seconds=60.0,        # Pause after rate limit
        backoff_multiplier=2.0,       # Increase cooldown on repeated limits
        slow_start_items=50,          # Gradual ramp-up over 50 items
        slow_start_initial_delay=2.0, # Start slow
        slow_start_final_delay=0.1,   # Ramp to full speed
    ),
)
```

When any worker hits a rate limit (429 error), **all workers pause** during cooldown, then gradually
resume to prevent immediate re-limiting.

### Single calls and a shared gateway

The resilience layer above — error-aware retries, the coordinated cooldown, token accounting — is
also reachable *without* the batch processor, for the cases where parallelism isn't the point:

```python
from async_batch_llm import OpenAIModel, OpenAIStrategy, call, LLMGateway

strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-4o-mini"))

# One prompt through the full pipeline — no queue, workers, or result stream.
summary = await call(strategy, "Summarize: ...")          # returns output, or raises

# A long-lived, shared entry point for a web service's request path. Many
# concurrent callers share one cooldown (one caller's 429 throttles everyone,
# then slow-starts); a semaphore bounds global concurrency. Opt into
# load-shedding with max_pending (admission cap) and submit_timeout (latency budget).
async with LLMGateway(
    strategy, config=ProcessorConfig(max_workers=5), max_pending=100, submit_timeout=30
) as gw:
    reply = await gw.submit("Answer this one request")
```

On failure, `call()` / `gw.submit()` re-raise the provider's own exception (preserving its type;
`LLMCallError` only when none was preserved). `call_result()` / `gw.submit_result()` instead return
the full `WorkItemResult` — `success`, `error`, `token_usage`, `metadata`, and the originating
`exception` — without raising. See [`examples/example_single_call.py`](examples/example_single_call.py)
and [`examples/example_gateway.py`](examples/example_gateway.py).
For a large input, do not wrap every gateway call in one `asyncio.gather()`;
that still materializes one task per item. Use `process_stream` or a bounded
outer task window as shown in the
[bounded-work guide](docs/bounded-work.md#gateway-task-counts).

### Cost Optimization with Caching

Share a single cached strategy across all work items to avoid paying for the same context repeatedly:

```python
from async_batch_llm import GeminiCachedModel, GeminiStrategy
from google import genai

client = genai.Client(api_key="your-api-key")

# Wrap a cached model in GeminiStrategy. Create ONE cached model and
# share it across all work items; a model per item defeats cache reuse.
cached_model = GeminiCachedModel(
    model="gemini-2.0-flash",
    client=client,
    cached_content=[
        genai.types.Content(
            role="user",
            parts=[genai.types.Part(text="Large document context...")],
        )
    ],
    cache_ttl_seconds=3600,  # 1 hour
    auto_renew=True,         # Automatic renewal for long pipelines
)
strategy = GeminiStrategy(model=cached_model)

async with ParallelBatchProcessor(config=config) as processor:
    for item in items:
        await processor.add_work(
            LLMWorkItem(
                item_id=item.id,
                strategy=strategy,  # Shared instance
                prompt=format_prompt(item),
            )
        )

    result = await processor.process_all()

# Framework calls prepare() once per shared strategy (creates cache).
# All items share the cache. Cache pricing varies by provider and model.
# Cleanup runs once when the processor context exits; the Gemini cache stays
# alive until TTL expiry unless you call cached_model.delete_cache().
```

Caching can reduce repeated-input charges, but rates and minimum cache sizes
vary by provider and model. Use the provider's current pricing with the token
counts reported by this package; do not treat the examples here as a price
table.

### Token & cost accounting

Every `BatchResult` aggregates input / cached / output tokens — across retries,
and recovered from failed attempts. Turn them into an estimate with
`estimated_cost` by supplying the rates you intend to use:

```python
from async_batch_llm import CachedTokenRates

print(f"Cache hit rate: {result.cache_hit_rate():.1f}%")
cost = result.estimated_cost(
    input_per_mtok=0.15, output_per_mtok=0.60,   # $ per 1M tokens
    cached_token_rate=CachedTokenRates.OPENAI,   # per-provider cache rate
)
print(f"Estimated cost: ${cost:.4f}")
```

`BatchResult.to_json()` and `to_jsonl()` provide strict, versioned,
provider-neutral exports. Unsupported application values raise instead of
falling back to `repr()`; use an encoder/decoder pair when typed reconstruction
is required.

### Observability

Metrics observers, lifecycle events (`ITEM_*`, `RATE_LIMIT_HIT`, `COOLDOWN_*`, …)
with JSON / Prometheus export, plus middleware and progress callbacks. See
[Advanced Patterns → Custom Observers / Middleware](docs/examples/advanced.md).

---

## Common Use Cases

Worked, runnable versions of the usual jobs — structured extraction with
validation retry, document summarization, RAG with shared context caching, and
saving results to a DB via a `post_processor` — live in
[`examples/`](examples/) and the [Basic Usage](docs/examples/basic.md) guide.

## Advanced Patterns

`RetryState` persists across an item's attempts, which unlocks **error-type-aware**
strategies — progressive temperature, **smart model escalation** (cheap model first, escalate to a
smarter/thinking model only on bad *output*), and partial-field recovery. Savings depend on the
workload, validation rate, and current model prices.
Because rate limits don't advance the attempt number, escalation tracks genuine quality failures, not
throttling.

→ Full walkthroughs in **[Advanced Patterns](docs/examples/advanced.md)**, runnable in
[`examples/example_smart_model_escalation.py`](examples/example_smart_model_escalation.py) and
[`examples/example_gemini_smart_retry.py`](examples/example_gemini_smart_retry.py).

---

## Configuration & tuning

`ProcessorConfig` (with nested `RetryConfig`, `RateLimitConfig`, and
`GuardrailConfig`) controls workers, the per-attempt `timeout_per_item`, the
end-to-end `total_timeout_per_item`, the batch deadline, fail-fast categories,
retry budgets (`max_attempts`, plus `max_rate_limit_retries` — rate limits don't
burn your retry budget), rate-limit cooldown + slow-start, proactive limiting,
progress reporting, and queueing. Full field reference:
**[API reference → ProcessorConfig](docs/API.md)**.

For the operational decisions — worker count per provider, sizing `max_connections` to `max_workers`,
the `RLIMIT_NOFILE` footgun, timeout-vs-retry-budget interaction, rate-limit tuning, and the
bounded-input streaming pattern — see the **[Production Checklist](docs/production-checklist.md)**.

## Testing

Test without spending on API calls — dry-run mode, `MockAgent` (simulates latency, rate limits, and
errors), and small-batch integration tests. See the **[Testing guide](docs/testing.md)**.

---

## Examples

Check out the [`examples/`](examples/) directory for complete working examples:

- [`example_production_resume.py`](examples/example_production_resume.py) - Checkpoints, replay,
  deadlines, fail-fast, and ordered collection
- [`example_llm_strategies.py`](examples/example_llm_strategies.py) - All built-in strategies
- [`example_single_call.py`](examples/example_single_call.py) - One resilient call, no batch machinery
- [`example_gateway.py`](examples/example_gateway.py) - Shared rate-limited gateway for a request path
- [`example_openai.py`](examples/example_openai.py) - OpenAI integration
- [`example_openrouter.py`](examples/example_openrouter.py) - OpenRouter (multi-provider)
- [`example_deepseek.py`](examples/example_deepseek.py) - DeepSeek with native cache-hit tracking
- [`example_anthropic.py`](examples/example_anthropic.py) - Anthropic Claude
- [`example_langchain.py`](examples/example_langchain.py) - LangChain & RAG
- [`example_gemini_direct.py`](examples/example_gemini_direct.py) - Direct Gemini API
- [`example_gemini_smart_retry.py`](examples/example_gemini_smart_retry.py) - Smart retry patterns
- [`example_smart_model_escalation.py`](examples/example_smart_model_escalation.py) - Cost optimization
- [`example_embeddings.py`](examples/example_embeddings.py) - Batch embedding generation (OpenAI + Gemini)
- [`example_context_manager.py`](examples/example_context_manager.py) - Resource management
- [`example_batch_benchmark.py`](examples/example_batch_benchmark.py) - Flagship bulk-benchmark demo

---

## Documentation

- **[Getting Started](docs/getting-started.md)** - Installation, strategies, and first batch
- **[Production Checklist](docs/production-checklist.md)** - Concurrency, limits, timeouts, and operations
- **[Results, Artifacts, and Resume](docs/results-and-artifacts.md)** - Serialization, privacy, checkpoints, and replay
- **[Deadlines and Fail-Fast Guardrails](docs/guardrails.md)** - Item and batch deadlines, abort modes, and categories
- **[Bounded Work and Backpressure](docs/bounded-work.md)** - Lazy sources, queue sizing, and streaming lifecycle
- **[Testing](docs/testing.md)** - Deterministic tests without provider calls
- **[Core API Reference](docs/api/core.md)** - Results, configuration, and processor APIs
- **[Artifact API Reference](docs/api/artifacts.md)** - Stores, identity, resume policies, and errors

Browse the rendered **[full documentation](https://geoff-davis.github.io/async-batch-llm/)**
for search, provider integration guides, worked examples, and migration guides.

---

## Contributing

Contributions welcome! Areas of interest:

- Additional provider strategies (AWS Bedrock, Azure OpenAI, etc.)
- Improved error classification
- Performance optimizations
- Documentation improvements

### Development Setup

```bash
# Clone and install
git clone https://github.com/geoff-davis/async-batch-llm.git
cd async-batch-llm
uv sync --all-extras

# Run tests
uv run pytest

# Run all checks (lint + typecheck + test + markdown-lint)
make ci
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Questions?** Open an issue on [GitHub](https://github.com/geoff-davis/async-batch-llm/issues) or check the [documentation](https://geoff-davis.github.io/async-batch-llm/).
