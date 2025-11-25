# batch-llm

**Process thousands of LLM requests in parallel with automatic retries, rate limiting, and flexible error handling.**

Works with any LLM provider (OpenAI, Anthropic, Google, LangChain, or custom) through a simple
strategy pattern. Built on asyncio for efficient I/O-bound processing.

[![PyPI version](https://badge.fury.io/py/batch-llm.svg)](https://badge.fury.io/py/batch-llm)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://geoff-davis.github.io/batch-llm/)

**📚 [Read the Documentation](https://geoff-davis.github.io/batch-llm/)**

---

## Why batch-llm?

- ✅ **Universal** - Works with any LLM provider through a simple strategy interface
- ✅ **Reliable** - Built-in retry logic, timeout handling, and coordinated rate limiting
- ✅ **Fast** - Parallel async processing with configurable concurrency
- ✅ **Observable** - Token tracking, metrics collection, and event hooks
- ✅ **Cost-Effective** - Shared caching strategies can dramatically reduce repeated prompt costs
- ✅ **Type-Safe** - Full generic type support with Pydantic validation

---

## Quick Start

### Installation

```bash
# Basic installation
pip install batch-llm

# With PydanticAI support (recommended for structured output)
pip install 'batch-llm[pydantic-ai]'

# With Google Gemini support
pip install 'batch-llm[gemini]'

# With everything
pip install 'batch-llm[all]'

# Alternatively, using the uv workflow from this repo's Makefile:
uv venv && uv sync
```

Once dependencies are installed, run the pinned tooling via `make check-all` so your local Ruff/mypy
versions match CI (all Makefile targets call `uv run` to use the synced environment).

### Basic Example

Process a batch of documents with structured output:

```python
import asyncio
from batch_llm import (
    ParallelBatchProcessor,
    LLMWorkItem,
    ProcessorConfig,
    PydanticAIStrategy,
)
from pydantic_ai import Agent
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]

async def main():
    # Create agent and wrap in strategy
    agent = Agent("gemini-2.5-flash", result_type=Summary)
    strategy = PydanticAIStrategy(agent=agent)

    # Configure processor
    config = ProcessorConfig(max_workers=5, timeout_per_item=30.0)

    # Process items with automatic resource cleanup
    async with ParallelBatchProcessor[str, Summary, None](config=config) as processor:
        # Add work items
        for doc in ["Document 1 text...", "Document 2 text..."]:
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"doc_{hash(doc)}",
                    strategy=strategy,
                    prompt=f"Summarize: {doc}",
                )
            )

        # Process all in parallel
        result = await processor.process_all()

    print(f"Succeeded: {result.succeeded}/{result.total_items}")
    print(f"Tokens used: {result.total_input_tokens + result.total_output_tokens}")

asyncio.run(main())
```

---

## Core Features

### Any LLM Provider

Built-in strategies for common providers:

- **`PydanticAIStrategy`** - PydanticAI agents with structured output
- **`GeminiStrategy`** - Direct Google Gemini API calls
- **`GeminiCachedStrategy`** - Gemini with context caching via the explicit cache API (more predictable than implicit caching)

Create custom strategies for any provider:

```python
from batch_llm.llm_strategies import LLMCallStrategy
from batch_llm import TokenUsage

class OpenAIStrategy(LLMCallStrategy[str]):
    def __init__(self, client, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    async def execute(self, prompt: str, attempt: int, timeout: float):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        output = response.choices[0].message.content or ""
        tokens: TokenUsage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return output, tokens

# Use with batch-llm
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key="...")
strategy = OpenAIStrategy(client=client)
```

See [`examples/`](examples/) directory for OpenAI, Anthropic, LangChain, and more.

### Automatic Retries

Configure retry behavior with exponential backoff and jitter:

```python
from batch_llm.core import RetryConfig

config = ProcessorConfig(
    max_workers=5,
    timeout_per_item=30.0,
    retry=RetryConfig(
        max_attempts=3,
        initial_wait=1.0,
        exponential_base=2.0,
        jitter=True,  # Prevents thundering herd
    ),
)
```

The framework automatically retries on validation errors, network errors, and other transient failures.

### Rate Limiting

Coordinated rate limit handling across all workers:

```python
from batch_llm.core import RateLimitConfig

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
from batch_llm import GeminiCachedStrategy
from google import genai

client = genai.Client(api_key="your-api-key")

# Create one cached strategy using the explicit cache API (see https://ai.google.dev/gemini-api/docs/caching?lang=python); implicit caching exists but is harder to control
strategy = GeminiCachedStrategy(
    model="gemini-2.0-flash",
    client=client,
    cached_content=[
        genai.types.Content(
            role="user",
            parts=[genai.types.Part(text="Large document context...")]
        )
    ],
    cache_ttl_seconds=3600,  # 1 hour
    auto_renew=True,         # Automatic renewal for long pipelines
)

async with ParallelBatchProcessor(config=config) as processor:
    # Reuse the same strategy for all items
    for item in items:
        await processor.add_work(
            LLMWorkItem(
                item_id=item.id,
                strategy=strategy,  # Shared instance
                prompt=format_prompt(item),
            )
        )

    result = await processor.process_all()

# Framework calls prepare() once per shared strategy (creates cache)
# All items share the cache (90% discount on cached tokens)
# Cleanup now runs once when the processor context exits, so the Gemini cache
# stays alive (unless you call delete_cache()) across the whole batch
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
print(f"Effective cost: {result.effective_input_tokens()} tokens")
```

### Observability

Monitor processing with metrics, middleware, and event observers:

```python
from batch_llm import MetricsObserver

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
from batch_llm import PydanticAIStrategy, LLMWorkItem
from pydantic_ai import Agent

class ContactInfo(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    phone: str

agent = Agent("gemini-2.5-flash", result_type=ContactInfo)
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

agent = Agent("gemini-2.5-flash", result_type=Summary)
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
from batch_llm import GeminiCachedStrategy
from google import genai

client = genai.Client(api_key="your-api-key")

# Cache the large document context once via the explicit API; see also https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/
strategy = GeminiCachedStrategy(
    model="gemini-2.0-flash",
    client=client,
    cached_content=[
        genai.types.Content(
            role="user",
            parts=[genai.types.Part(text=large_document_corpus)]
        )
    ],
    cache_ttl_seconds=3600,
)

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

### Progressive Temperature on Retries

Increase creativity on retries to get past validation errors:

```python
from pydantic import ValidationError
from batch_llm import RetryState
from batch_llm.llm_strategies import LLMCallStrategy

class ProgressiveTempStrategy(LLMCallStrategy[str]):
    """Increase temperature only when validation keeps failing."""

    def __init__(self, client, temps=None):
        self.client = client
        self.temps = temps if temps is not None else [0.0, 0.5, 1.0]

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ):
        state = state or RetryState()
        failures = state.get("validation_failures", 0)
        temp = self.temps[min(failures, len(self.temps) - 1)]
        response = await self.client.generate(prompt, temperature=temp)
        return response.text, extract_tokens(response)

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ):
        if state and isinstance(exception, ValidationError):
            state.set("validation_failures", state.get("validation_failures", 0) + 1)
```

### Smart Model Escalation

Start with cheap models, escalate only on quality issues:

```python
from pydantic import ValidationError

class SmartModelEscalationStrategy(LLMCallStrategy[Output]):
    """Escalate to better models ONLY on validation errors."""

    MODELS = [
        "gemini-2.5-flash-lite",  # Cheapest
        "gemini-2.5-flash",       # Moderate
        "gemini-2.5-pro",         # Most capable
    ]

    def __init__(self, client):
        self.client = client

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ):
        """Only count validation errors for escalation."""
        if state is None:
            return
        if isinstance(exception, ValidationError):
            state.set("validation_failures", state.get("validation_failures", 0) + 1)
        # Network/rate limit errors don't trigger escalation

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ):
        state = state or RetryState()
        failures = state.get("validation_failures", 0)
        model_idx = min(failures, len(self.MODELS) - 1)
        model = self.MODELS[model_idx]
        response = await self.client.generate(prompt, model=model)
        return parse_output(response), extract_tokens(response)
```

**Cost Savings:**

- Validation error → Escalate to smarter model ✅
- Network error → Retry same cheap model ✅
- Rate limit error → Retry same cheap model ✅
- Most tasks succeed on attempt 1 (cheap)
- Result: **~60-80% cost reduction**

See [`examples/example_smart_model_escalation.py`](examples/example_smart_model_escalation.py) for complete implementation.

### Partial Recovery with RetryState

Save partial results and retry only failed fields:

```python
from batch_llm import RetryState

class PartialRecoveryStrategy(LLMCallStrategy[dict]):
    """Parse partial results and retry only failed fields."""

    async def execute(self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None):
        if state is None:
            state = RetryState()

        # Check for partial results from previous attempt
        partial_results = state.get("partial_results", {})
        failed_fields = state.get("failed_fields", ["name", "email", "phone", "address"])

        if attempt == 1:
            final_prompt = f"{prompt}\nExtract: {', '.join(failed_fields)}"
        else:
            # Retry only failed fields
            final_prompt = (
                f"{prompt}\nYou already got these right: {partial_results}"
                f"\nNow extract only: {', '.join(failed_fields)}"
            )

        response = await self.client.generate(final_prompt)
        result = parse_response(response)

        # Merge with partial results
        if attempt > 1:
            result = {**partial_results, **result}

        # Check for missing fields
        missing = [f for f in ["name", "email", "phone", "address"] if f not in result]
        if missing:
            # Save what we got and retry
            state.set("partial_results", {k: v for k, v in result.items()})
            state.set("failed_fields", missing)
            raise ValueError(f"Missing fields: {missing}")

        return result, extract_tokens(response)
```

**Cost Considerations:**

- Retries focus only on the fields that failed validation, so the second attempt
  usually consumes fewer tokens than the first.
- Actual savings depend on how many fields typically fail and the provider's billing model.

---

## Configuration

### ProcessorConfig

Complete configuration options:

```python
from batch_llm import ProcessorConfig
from batch_llm.core import RetryConfig, RateLimitConfig

config = ProcessorConfig(
    # Core Settings
    max_workers=5,              # Number of parallel workers
    timeout_per_item=120.0,     # Max seconds per item (including retries)

    # Retry Configuration
    retry=RetryConfig(
        max_attempts=3,          # Maximum retry attempts
        initial_wait=1.0,        # Initial retry delay (seconds)
        max_wait=60.0,           # Maximum retry delay
        exponential_base=2.0,    # Backoff multiplier
        jitter=True,             # Add random jitter
    ),

    # Rate Limit Configuration
    rate_limit=RateLimitConfig(
        cooldown_seconds=300.0,        # Cooldown after rate limit (5 min)
        backoff_multiplier=1.5,        # Increase cooldown on repeated limits
        slow_start_items=50,           # Gradual ramp-up over 50 items
        slow_start_initial_delay=2.0,  # Start slow
        slow_start_final_delay=0.1,    # Ramp to full speed
    ),

    # Progress Reporting
    progress_interval=10,              # Log progress every N items
    progress_callback_timeout=5.0,     # Timeout for progress callbacks

    # Queue Management
    max_queue_size=0,                  # Max items in queue (0 = unlimited)
)
```

### Choosing Worker Count

**Rate-Limited APIs (OpenAI, Anthropic, Gemini):**

- Start with `max_workers=5`
- Monitor `rate_limit_count` in metrics
- Reduce workers if hitting limits frequently

**Unlimited APIs (Local Models):**

- Use `max_workers=min(cpu_count() * 2, 20)`
- Cap at 20 to avoid excessive context switching

**Testing/Debugging:**

- Use `max_workers=2` for easier log reading

---

## Testing

### Three Testing Approaches

#### 1. Dry-Run Mode (No API Calls)

```python
config = ProcessorConfig(dry_run=True)  # No API calls made

async with ParallelBatchProcessor(config=config) as processor:
    await processor.add_work(work_item)
    result = await processor.process_all()  # Returns mock data
```

#### 2. Mock Strategies (Unit Tests)

```python
from batch_llm.testing import MockAgent

mock_agent = MockAgent(
    response_factory=lambda p: Summary(title="Test", key_points=["A", "B"]),
    latency=0.01,  # Simulate 10ms latency
)

strategy = PydanticAIStrategy(agent=mock_agent)
```

#### 3. Small Batch Integration Tests

```python
# Test with 5 items before processing 1000
test_items = full_dataset[:5]

config = ProcessorConfig(max_workers=2, timeout_per_item=30.0)
result = await process_batch(test_items, config)

if result.succeeded == len(test_items):
    # Now process full batch
    full_result = await process_batch(full_dataset, config)
```

---

## Performance

### Throughput

- **Sequential**: ~1 item/second (single threaded)
- **5 workers**: ~5 items/second (parallel)
- **10 workers**: ~10 items/second (parallel)

### Example: 1000 Items

- **Sequential**: ~16 minutes
- **5 workers**: ~3 minutes (5× faster)
- **10 workers**: ~1.5 minutes (10× faster)

**Note:** Actual throughput depends on LLM latency (~200-500ms per call for most APIs).

---

## Examples

Check out the [`examples/`](examples/) directory for complete working examples:

- [`example_llm_strategies.py`](examples/example_llm_strategies.py) - All built-in strategies
- [`example_openai.py`](examples/example_openai.py) - OpenAI integration
- [`example_anthropic.py`](examples/example_anthropic.py) - Anthropic Claude
- [`example_langchain.py`](examples/example_langchain.py) - LangChain & RAG
- [`example_gemini_direct.py`](examples/example_gemini_direct.py) - Direct Gemini API
- [`example_gemini_smart_retry.py`](examples/example_gemini_smart_retry.py) - Smart retry patterns
- [`example_smart_model_escalation.py`](examples/example_smart_model_escalation.py) - Cost optimization
- [`example_context_manager.py`](examples/example_context_manager.py) - Resource management

---

## Documentation

- **[Full Documentation](https://geoff-davis.github.io/batch-llm/)** - Getting started, examples, and API reference
- **[API Reference](https://geoff-davis.github.io/batch-llm/api/core/)** - Complete API documentation
- **[Migration Guides](https://geoff-davis.github.io/batch-llm/migration/v0.4/)** - Version upgrade guides

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
git clone https://github.com/geoff-davis/batch-llm.git
cd batch-llm
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

**Questions?** Open an issue on [GitHub](https://github.com/geoff-davis/batch-llm/issues) or check the [documentation](https://geoff-davis.github.io/batch-llm/).
