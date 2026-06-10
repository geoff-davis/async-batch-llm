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

From a sample [GSM8K test-split run](docs/examples/bulk-benchmark.md) — illustrative, not a spec
(numbers shift with provider, account limits, and network):

- **~17× faster than serial** — 30 problems took ~57 s one-at-a-time vs ~3.4 s through the pool.
  Concurrency is what collapses wall time.
- **The full 1,319-problem test split for ~$0.05** on DeepSeek Flash, with the per-provider
  token/cost breakdown printed for free.
- **Throughput on par with a hand-tuned `asyncio.gather` pool** — the framework matches a good
  semaphore-bounded pool on raw speed. Its edge is *surviving the failures a bare pool drops*:
  retrying validation errors, escalating the model on bad output, and riding out 429 cooldowns
  instead of shedding results.

See the [bulk benchmark](docs/examples/bulk-benchmark.md) for methodology and the full tables.

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
instead of bare strings to control ids, and forward any processor option
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

### Token Tracking

Track token usage across all requests, including cached tokens:

```python
result = await processor.process_all()

# Basic token counts
print(f"Input tokens: {result.total_input_tokens}")
print(f"Output tokens: {result.total_output_tokens}")

# Cache metrics
print(f"Cached tokens: {result.total_cached_tokens}")
print(f"Cache hit rate: {result.cache_hit_rate():.1f}%")
# Pass a per-provider rate from CachedTokenRates (GEMINI / OPENAI /
# ANTHROPIC_READ / DEEPSEEK) for an accurate billable-token estimate.
# Calling it without an explicit rate defaults to the Gemini rate AND warns
# when cached tokens are present (the rate is wrong for other providers).
from async_batch_llm import CachedTokenRates

# Billable *input tokens* after the cache discount (a token count, not a price):
print(f"Billable input tokens: {result.effective_input_tokens(CachedTokenRates.OPENAI):,}")

# Or estimate spend directly from per-million-token prices:
cost = result.estimated_cost(
    input_per_mtok=0.15,   # $ per 1M input tokens
    output_per_mtok=0.60,  # $ per 1M output tokens
    cached_token_rate=CachedTokenRates.OPENAI,
)
print(f"Estimated cost: ${cost:.4f}")
```

### Observability

Monitor processing with metrics, middleware, and event observers:

```python
from async_batch_llm import MetricsObserver

# Collect metrics
metrics = MetricsObserver()

# Observers receive lifecycle events:
# - BATCH_STARTED / BATCH_COMPLETED
# - WORKER_STARTED / WORKER_STOPPED
# - ITEM_STARTED / ITEM_COMPLETED / ITEM_FAILED
# - RATE_LIMIT_HIT / COOLDOWN_STARTED / COOLDOWN_ENDED

processor = ParallelBatchProcessor(
    config=config,
    observers=[metrics],
)

result = await processor.process_all()

# Get detailed metrics
collected_metrics = await metrics.get_metrics()
# Returns: {
#   "items_processed": 100,
#   "items_succeeded": 95,
#   "items_failed": 5,
#   "avg_processing_time": 1.2,
#   "rate_limits_hit": 0,
#   ...
# }

# Export in different formats
json_export = await metrics.export_json()
prometheus_export = await metrics.export_prometheus()

# If you don't use `async with`, call shutdown() to clean up workers/strategies:
# processor = ParallelBatchProcessor(config=config)
# ... add work, process ...
# await processor.shutdown()
```

---

## Common Use Cases

### Structured Data Extraction

Extract structured data with automatic validation retry:

```python
from pydantic import BaseModel, Field
from async_batch_llm import PydanticAIStrategy, LLMWorkItem
from pydantic_ai import Agent

class ContactInfo(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    phone: str

agent = Agent("gemini-2.5-flash", output_type=ContactInfo)
strategy = PydanticAIStrategy(agent=agent)

async with ParallelBatchProcessor(config=config) as processor:
    for text in contact_texts:
        await processor.add_work(
            LLMWorkItem(
                item_id=text.id,
                strategy=strategy,
                prompt=f"Extract contact info: {text}",
            )
        )

    result = await processor.process_all()

# Framework automatically retries on validation errors
# Each retry can use different temperature (via custom strategy)
```

### Document Summarization

Summarize many documents in parallel:

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

agent = Agent("gemini-2.5-flash", output_type=Summary)
strategy = PydanticAIStrategy(agent=agent)

async with ParallelBatchProcessor(config=config) as processor:
    for doc in documents:
        await processor.add_work(
            LLMWorkItem(
                item_id=doc.id,
                strategy=strategy,
                prompt=f"Summarize this document:\n\n{doc.text}",
            )
        )

    result = await processor.process_all()

    # Process results
    for item_result in result.results:
        if item_result.success:
            print(f"{item_result.item_id}: {item_result.output.title}")
```

### RAG with Context Caching

Process queries against large document context with caching:

```python
from async_batch_llm import GeminiCachedModel, GeminiStrategy
from google import genai

client = genai.Client(api_key="your-api-key")

# Cache the large document context once via the explicit API; see also https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/
cached_model = GeminiCachedModel(
    model="gemini-2.0-flash",
    client=client,
    cached_content=[
        genai.types.Content(
            role="user",
            parts=[genai.types.Part(text=large_document_corpus)],
        )
    ],
    cache_ttl_seconds=3600,
)
strategy = GeminiStrategy(model=cached_model)

async with ParallelBatchProcessor(config=config) as processor:
    # Process multiple queries against the cached context
    for query in user_queries:
        await processor.add_work(
            LLMWorkItem(
                item_id=query.id,
                strategy=strategy,  # Reuse cached strategy
                prompt=query.text,
            )
        )

    result = await processor.process_all()

# Cached tokens are billed at ~10% of the usual rate, so reusing context can reduce total cost substantially
```

### Custom Post-Processing

Save results to database as they complete:

```python
from dataclasses import dataclass

@dataclass
class WorkContext:
    user_id: str
    document_id: str

async def save_result(result):
    """Save successful results to database."""
    if result.success:
        await db.save(
            user_id=result.context.user_id,
            document_id=result.context.document_id,
            summary=result.output,
        )

async with ParallelBatchProcessor(
    config=config,
    post_processor=save_result,
) as processor:
    # Add work with context
    await processor.add_work(
        LLMWorkItem(
            item_id="doc_123",
            strategy=strategy,
            prompt="Summarize...",
            context=WorkContext(user_id="user_1", document_id="doc_123"),
        )
    )

    result = await processor.process_all()
```

---

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
