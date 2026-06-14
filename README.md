# async-batch-llm

**Run thousands of individual LLM calls in parallel — with coordinated rate-limit handling,
error-type-aware retries, and per-call cost accounting — when you need the results *now*, not from
a 24-hour batch API.**

Provider-agnostic (OpenAI, Anthropic, Google, DeepSeek, OpenRouter, PydanticAI, or your own)
through a simple strategy pattern; built on asyncio for I/O-bound throughput.

[![PyPI version](https://badge.fury.io/py/async-batch-llm.svg)](https://badge.fury.io/py/async-batch-llm)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/geoff-davis/async-batch-llm/workflows/Tests/badge.svg)](https://github.com/geoff-davis/async-batch-llm/actions)
[![Coverage](https://raw.githubusercontent.com/geoff-davis/async-batch-llm/python-coverage-comment-action-data/badge.svg)](https://github.com/geoff-davis/async-batch-llm/tree/python-coverage-comment-action-data)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://geoff-davis.github.io/async-batch-llm/)

**📚 [Read the Documentation](https://geoff-davis.github.io/async-batch-llm/)**

---

## A sense of scale

From a sample [GSM8K test-split run](docs/benchmarks.md) — illustrative, not a spec
(numbers shift with provider, account limits, and network):

- **~16–19× faster than serial** — 30 problems took ~40–65 s one-at-a-time vs ~2–4 s through the
  pool (even a provider throttle-capped to 5 workers ran 5×). Concurrency collapses wall time.
- **The full 1,319-problem test split for ~$0.05** on DeepSeek Flash — vs ~$0.43 on a Gemini run at
  the *same* 95–97% accuracy (~8× cheaper), with the per-provider cost breakdown printed for free.
- **At least as fast as a hand-rolled `Semaphore` + `gather` pool** — it edged ahead in this run (a
  bounded worker pool runs a fixed N tasks instead of scheduling every coroutine up front) — and,
  unlike a bare pool, *survives* the 429s/503s it would otherwise drop: retrying validation errors,
  escalating the model on bad output, riding out throttling.

See the [benchmarks](docs/benchmarks.md) for methodology and the full tables.

## vs. rolling your own

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
- **Memory** — `gather()` materializes every coroutine up front; you can't stream a million prompts
  through constant memory.

async-batch-llm *is* that loop with the operational layer filled in — coordinated cooldowns,
error-type-aware retries, token/cost accounting (including failures), bounded-memory streaming, and
a one-line provider swap.

## When NOT to use this

- **You can wait hours.** If the job is latency-tolerant, the providers' own **batch APIs**
  (OpenAI / Anthropic / Gemini Batch) run ~50% cheaper with results in up to 24 h. This library is
  for *real-time* bulk — results now, at full price.
- **It's a handful of calls.** For a one-off script over a few dozen prompts, a bare
  `asyncio.gather` (optionally with a semaphore) is fine — don't take the dependency.
  That said, if you want this library's *resilience* (error-aware retries, a coordinated
  rate-limit cooldown, token accounting) for a single call or a web service's request
  path *without* the batch machinery, reach for
  [`call()` / `LLMGateway`](#single-calls-and-a-shared-gateway) instead.

---

## Quick Start

### Installation

```bash
# Basic installation
pip install async-batch-llm

# With PydanticAI support (recommended for structured output)
pip install 'async-batch-llm[pydantic-ai]'

# With Google Gemini support
pip install 'async-batch-llm[gemini]'

# With OpenAI support
pip install 'async-batch-llm[openai]'

# With OpenRouter support (multi-provider via one OpenAI-compatible API)
pip install 'async-batch-llm[openrouter]'

# With DeepSeek support (direct DeepSeek API, native cache-hit tracking)
pip install 'async-batch-llm[deepseek]'

# With everything
pip install 'async-batch-llm[all]'

# Alternatively, using the uv workflow from this repo's Makefile:
uv venv && uv sync
```

Once dependencies are installed, run the pinned tooling via `make check-all` so your local Ruff/mypy
versions match CI (all Makefile targets call `uv run` to use the synced environment).

### Basic Example

The fastest way in is `process_prompts` — hand it a strategy and an iterable of
prompts, get a `BatchResult` back. Item ids are auto-generated for bare strings:

```python
import asyncio
from async_batch_llm import OpenAIModel, OpenAIStrategy, ProcessorConfig, process_prompts

documents = ["Document 1 text...", "Document 2 text..."]

async def main():
    strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-4o-mini"))  # reads OPENAI_API_KEY

    result = await process_prompts(
        strategy,
        [f"Summarize: {doc}" for doc in documents],
        config=ProcessorConfig(max_workers=10),  # up to 10 calls in flight
    )

    print(f"Succeeded: {result.succeeded}/{result.total_items}")
    for r in result.successes:
        print(r.item_id, "->", r.output)

asyncio.run(main())
```

Want results as they finish (e.g. to write each to disk)? Stream them. With a
bounded `max_queue_size`, the producer applies **backpressure** — so you can
stream a **million prompts (or an unbounded source) through constant memory**,
since work isn't all buffered up front:

```python
from async_batch_llm import process_stream, ProcessorConfig

config = ProcessorConfig(max_workers=50, max_queue_size=200)  # ~constant memory

async for result in process_stream(strategy, huge_prompt_source, config=config):
    if result.success:
        await save(result.item_id, result.output)   # results arrive in completion order
```

`prompts` can be any sync **or** async iterable. Pass `(item_id, prompt)` pairs
instead of bare strings to control ids — or `(item_id, prompt, context)` triples
to carry per-item data through to the result — and forward any processor option
(`post_processor`, `observers`, `error_classifier`, …) as a keyword argument.
The error classifier is auto-selected from the strategy when you don't pass one.

Need the low-level controls? `processor.start()` / `add_work()` / `finish()` /
`results()` is the streaming mode `process_stream` is built on (workers run
while you add work — a bounded queue is backpressure, not a deadlock).

#### Full control (advanced)

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
    config = ProcessorConfig(max_workers=10)          # up to 10 calls in flight at once

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

DeepSeek allows **thousands of concurrent connections** — far more than most
providers — so one asyncio batch can drive very high throughput. The footgun:
the openai SDK defaults to httpx's ~100-connection pool, so raising `max_workers`
past that gives no extra throughput (workers just block on the pool). Size
`max_connections` to match `max_workers`:

```python
from async_batch_llm import DeepSeekModel, DeepSeekStrategy

model = DeepSeekModel.from_api_key(
    "deepseek-v4-flash",   # reads DEEPSEEK_API_KEY; V4 defaults thinking ON (pricey for batch)
    thinking=False,        # turn it off for classification/extraction
    max_connections=150,   # size the httpx pool to your max_workers
)
strategy = DeepSeekStrategy(model)
```

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
# then slow-starts); a semaphore bounds global concurrency.
async with LLMGateway(strategy, config=ProcessorConfig(max_workers=5)) as gw:
    reply = await gw.submit("Answer this one request")
```

`call_result()` / `gw.submit_result()` return the full `WorkItemResult` (token usage, metadata, error)
instead of raising. See [`examples/example_single_call.py`](examples/example_single_call.py) and
[`examples/example_gateway.py`](examples/example_gateway.py).

### Cost Optimization with Caching

Share a single cached strategy across all work items to avoid paying for the same context repeatedly:

```python
from async_batch_llm import GeminiCachedModel, GeminiStrategy
from google import genai

client = genai.Client(api_key="your-api-key")

# v0.6.0+: wrap a cached model in GeminiStrategy. Create ONE cached
# model and share it across all work items — constructing a new model
# per item would defeat caching and can cost 10x more.
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
# All items share the cache (cached tokens are billed at 10% of the normal rate).
# Cleanup runs once when the processor context exits; the Gemini cache stays
# alive until TTL expiry unless you call cached_model.delete_cache().
```

**Cost Example:**

- Without caching: 100 items × $0.10 = **$10.00**
- With shared caching: 100 items × $0.03 = **$3.00** (assuming cached tokens are billed at 10% of the original rate)

### Token & cost accounting

Every `BatchResult` aggregates input / cached / output tokens — across retries,
and recovered from failed attempts. Turn them into money with `estimated_cost`,
which applies the per-provider cache discount:

```python
from async_batch_llm import CachedTokenRates

print(f"Cache hit rate: {result.cache_hit_rate():.1f}%")
cost = result.estimated_cost(
    input_per_mtok=0.15, output_per_mtok=0.60,   # $ per 1M tokens
    cached_token_rate=CachedTokenRates.OPENAI,   # per-provider cache rate
)
print(f"Estimated cost: ${cost:.4f}")
```

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
smarter/thinking model only on bad *output* — typically 60–80% cheaper), and partial-field recovery.
Because rate limits don't advance the attempt number, escalation tracks genuine quality failures, not
throttling.

→ Full walkthroughs in **[Advanced Patterns](docs/examples/advanced.md)**, runnable in
[`examples/example_smart_model_escalation.py`](examples/example_smart_model_escalation.py) and
[`examples/example_gemini_smart_retry.py`](examples/example_gemini_smart_retry.py).

---

## Configuration & tuning

`ProcessorConfig` (with nested `RetryConfig` / `RateLimitConfig`) controls workers, per-attempt
timeout, retry budgets (`max_attempts`, plus `max_rate_limit_retries` — rate limits don't burn your
retry budget), rate-limit cooldown + slow-start, proactive limiting, progress reporting, and
queueing. Full field reference: **[API reference → ProcessorConfig](docs/API.md)**.

For the operational decisions — worker count per provider, sizing `max_connections` to `max_workers`,
the `RLIMIT_NOFILE` footgun, timeout-vs-retry-budget interaction, rate-limit tuning, and the
constant-memory streaming pattern — see the **[Production Checklist](docs/production-checklist.md)**.

## Testing

Test without spending on API calls — dry-run mode, `MockAgent` (simulates latency, rate limits, and
errors), and small-batch integration tests. See the **[Testing guide](docs/testing.md)**.

---

## Examples

Check out the [`examples/`](examples/) directory for complete working examples:

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
- [`example_context_manager.py`](examples/example_context_manager.py) - Resource management
- [`example_batch_benchmark.py`](examples/example_batch_benchmark.py) - Flagship bulk-benchmark demo

---

## Documentation

- **[Full Documentation](https://geoff-davis.github.io/async-batch-llm/)** - Getting started, examples, and API reference
- **[API Reference](https://geoff-davis.github.io/async-batch-llm/api/core/)** - Complete API documentation
- **[Migration Guides](https://geoff-davis.github.io/async-batch-llm/migration/v0.4/)** - Version upgrade guides

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
