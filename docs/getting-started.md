# Getting Started

This guide will help you get started with async-batch-llm.

## Installation

Install async-batch-llm with the extras you need:

```bash
# Basic installation
pip install async-batch-llm

# With PydanticAI support (recommended for structured output)
pip install 'async-batch-llm[pydantic-ai]'

# With Google Gemini support
pip install 'async-batch-llm[gemini]'

# With OpenAI support
pip install 'async-batch-llm[openai]'

# With OpenRouter support (any of OpenAI/Anthropic/Google/DeepSeek/etc.
# behind one OpenAI-compatible API)
pip install 'async-batch-llm[openrouter]'

# With DeepSeek support (direct DeepSeek API, native cache-hit tracking)
pip install 'async-batch-llm[deepseek]'

# With everything
pip install 'async-batch-llm[all]'
```

## Quickstart: the high-level API

For the common case — "run this strategy over these prompts" — reach for
`process_prompts` (collect everything) or `process_stream` (handle each result
as it finishes). Both accept bare strings (ids auto-generated) or
`(item_id, prompt)` pairs, and forward any `ParallelBatchProcessor` option as a
keyword argument.

```python
import asyncio
from async_batch_llm import llm, process_prompts, process_stream

async def main():
    strategy = llm("openai:gpt-4o-mini")  # reads OPENAI_API_KEY

    # Collect all results into a BatchResult:
    result = await process_prompts(strategy, ["Summarize A", "Summarize B"])
    print(f"{result.succeeded}/{result.total_items} succeeded")
    for r in result.successes:
        print(r.item_id, "->", r.output)

    # …or stream results as each item completes:
    async for r in process_stream(strategy, [("a", "first"), ("b", "second")]):
        print("done:", r.item_id, r.success)

asyncio.run(main())
```

`process_stream` is built on the processor's first-class streaming mode
(`start()`/`add_work()`/`finish()`/`results()`) — workers push each completed
result onto an internal queue, so results arrive in **completion order**. When
you don't pass `error_classifier=`, it's auto-selected from the strategy
(`OpenAIStrategy` → `OpenAIErrorClassifier`, `GeminiStrategy` →
`GeminiErrorClassifier`, etc.).

The `llm()` factory builds the same strategy objects as the explicit
two-object form. It accepts `"gemini:..."`, `"openai:..."`,
`"openrouter:..."`, and `"deepseek:..."` prefixes; keyword arguments forward
to the model constructor:

```python
from async_batch_llm import llm

strategy = llm("openai:gpt-4o-mini")                     # reads OPENAI_API_KEY
strategy = llm("gemini:gemini-2.5-flash")                # reads GOOGLE_API_KEY
strategy = llm("openrouter:anthropic/claude-haiku-4-5")  # reads OPENROUTER_API_KEY
strategy = llm("deepseek:deepseek-v4-flash", thinking=False, max_connections=150)
```

For custom clients, Gemini cached models, response parsers with structured
output, or providers without a prefix, use the explicit two-object
construction shown in [Built-in Strategies](#built-in-strategies) below —
that remains the "advanced construction" path.

For large sources, use a lazy iterable plus
`ProcessorConfig(max_queue_size=N)` and consume `process_stream` incrementally.
`process_prompts` collects every result and therefore is not a constant-memory
output path. See [Bounded Work and Backpressure](bounded-work.md).

The rest of this guide covers the underlying building blocks, which you use
directly when you need custom queueing, per-item context, middleware, or
observers.

## Core Concepts

### 1. Strategy Pattern

async-batch-llm uses a strategy pattern to support any LLM provider. A strategy encapsulates:

- How to call the LLM
- How to handle errors
- How to manage resources (e.g., caches)

```python
from async_batch_llm import LLMCallStrategy

class MyCustomStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
        # Call your LLM here
        response = await my_llm.generate(prompt)
        tokens = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        return response, tokens
```

### 2. Work Items

Each task is represented by an `LLMWorkItem`:

```python
from async_batch_llm import LLMWorkItem

work_item = LLMWorkItem(
    item_id="unique-id",
    strategy=my_strategy,
    prompt="Your prompt here",
    context={"metadata": "optional"}
)
```

### 3. Parallel Processing

The `ParallelBatchProcessor` manages parallel execution:

```python
from async_batch_llm import ParallelBatchProcessor, ProcessorConfig

config = ProcessorConfig(
    max_workers=5,
    timeout_per_item=30.0,
)

async with ParallelBatchProcessor(config=config) as processor:
    await processor.add_work(work_item)
    result = await processor.process_all()
```

## Built-in Strategies

### PydanticAI Strategy

For structured output with validation:

```python
from async_batch_llm import PydanticAIStrategy
from pydantic_ai import Agent
from pydantic import BaseModel

class Output(BaseModel):
    field1: str
    field2: int

agent = Agent("gemini-2.5-flash", output_type=Output)
strategy = PydanticAIStrategy(agent=agent)
```

### Gemini Strategy

Direct Gemini API calls:

```python
from async_batch_llm import GeminiModel, GeminiStrategy
from google import genai

client = genai.Client(api_key="your-key")
model = GeminiModel("gemini-2.5-flash", client)
strategy = GeminiStrategy(model, response_parser=lambda r: r.text)
```

### Gemini with Context Caching

With context caching for repeated prompts (70-90% cost savings):

```python
from async_batch_llm import GeminiCachedModel, GeminiStrategy

cached_model = GeminiCachedModel(
    "gemini-2.5-flash", client,
    cached_content=[system_instruction, context_docs],
)
strategy = GeminiStrategy(cached_model, response_parser=lambda r: r.text)
```

### OpenAI Strategy

Direct OpenAI API calls (added in v0.9.0):

```python
from async_batch_llm import OpenAIModel, OpenAIStrategy

model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-...")
strategy = OpenAIStrategy(model)
```

See [OpenAI Integration](OPENAI_INTEGRATION.md) for structured output,
caching, and error handling.

### OpenRouter Strategy

Reach Anthropic, OpenAI, Google, DeepSeek, etc. through one OpenAI-compatible
API (added in v0.9.0):

```python
from async_batch_llm import OpenRouterModel, OpenRouterStrategy

model = OpenRouterModel.from_api_key(
    "anthropic/claude-haiku-4-5",
    api_key="sk-or-...",
)
strategy = OpenRouterStrategy(model)
```

See [OpenRouter Integration](OPENROUTER_INTEGRATION.md) for the
per-upstream-provider caching matrix and the Anthropic `cache_control`
opt-in pattern.

### DeepSeek Strategy

Direct DeepSeek API access with native cache-hit token tracking (added in
v0.10.0):

```python
from async_batch_llm import DeepSeekModel, DeepSeekStrategy

model = DeepSeekModel.from_api_key(
    "deepseek-v4-flash",     # reads DEEPSEEK_API_KEY
    thinking=False,          # non-thinking: cheaper/faster for batch work
    max_connections=200,     # see the high-concurrency note below
)
strategy = DeepSeekStrategy(model)
```

DeepSeek allows **thousands of concurrent connections** — far more than most
providers — so it's a great fit for large parallel batches. To actually use
that headroom, raise `ProcessorConfig(max_workers=...)` *and* pass a matching
`max_connections` so the underlying httpx pool (default ~100) doesn't become
the bottleneck. See the [DeepSeek quickstart in the
README](https://github.com/geoff-davis/async-batch-llm#deepseek-quickstart)
for the full pattern (thinking toggle, JSON mode, connection pool,
fence-tolerant parser, and the prepaid-balance gotcha).

### Structured (JSON) output

For the OpenAI-compatible providers (OpenAI / OpenRouter / DeepSeek), request
JSON with `from_api_key(..., json_mode=True)` and parse it with the built-in
`pydantic_json_parser`, which strips markdown code fences before validating:

```python
from pydantic import BaseModel

from async_batch_llm import DeepSeekModel, DeepSeekStrategy, pydantic_json_parser


class Topic(BaseModel):
    label: str
    confidence: float


model = DeepSeekModel.from_api_key("deepseek-chat", json_mode=True)
strategy = DeepSeekStrategy(model, pydantic_json_parser(Topic))
```

## Open file limits and high concurrency

Each in-flight request typically holds a socket — an operating-system **file
descriptor** — so a high `max_workers` (together with the provider connection
pool and your app's own fds) can run into the OS **open-file limit**. The
symptom is `OSError: [Errno 24] Too many open files`, and it bites hardest on
**macOS**, whose default soft limit is only ~256.

`ParallelBatchProcessor` emits a `UserWarning` at construction when
`max_workers` is close to the current soft limit (`RLIMIT_NOFILE`). It does
**not** raise the limit for you — changing it mutates process-global state, so
that's your call. Three ways to handle it:

1. **Raise the limit for the shell**, before running:

   ```bash
   ulimit -n 8192      # raise the soft limit (must be ≤ the hard limit)
   ```

2. **Raise it in-process**, early in your program (Unix only):

   ```python
   import resource

   soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
   target = hard if hard != resource.RLIM_INFINITY else 8192
   resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, target), hard))
   ```

3. **Lower `max_workers`** to fit the available limit. For I/O-bound LLM calls
   the throughput gain flattens out well before you exhaust the fd budget, so
   capping workers is often fine.

Also size the **connection pool** to your worker count so the pool itself isn't
the bottleneck — `max_connections=` on the OpenAI-compatible `from_api_key(...)`
(see the [OpenAI integration guide](OPENAI_INTEGRATION.md#connection-pool-sizing-max_connections)),
and httpx limits via `HttpOptions` for the Gemini client.

## Next Steps

- [Production Checklist](production-checklist.md) - Worker count, connection
  pools, fd limits, timeout/retry budgets, rate-limit tuning, constant-memory streaming
- [Basic Examples](examples/basic.md) - See more usage examples
- [Custom Strategies](examples/custom-strategies.md) - Build your own strategies
- [Advanced Patterns](examples/advanced.md) - Learn advanced techniques
