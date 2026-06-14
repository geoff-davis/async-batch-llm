# async-batch-llm

**An asyncio toolkit for latency-sensitive LLM workloads made of independent calls:
bulk prompt runs you want back during the current workflow, plus single-call and
service request paths that need the same retry/rate-limit behavior.**

It provides provider-agnostic execution surfaces — a bounded worker pool for bulk
runs, plus queue-less single-call and gateway helpers for request paths — with
coordinated cooldowns for rate limits, error-type-aware retries, bounded streaming,
and token/cost accounting (including failed attempts). Use it when provider batch
APIs are too slow for the job; use those batch APIs when a cheaper 24-hour
turnaround is acceptable.

Works with any LLM provider (OpenAI, Anthropic, Google, DeepSeek, OpenRouter,
PydanticAI, or custom) through a simple strategy pattern. Built on asyncio for
efficient I/O-bound processing.

[![PyPI version](https://badge.fury.io/py/async-batch-llm.svg)](https://badge.fury.io/py/async-batch-llm)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why async-batch-llm?

- ✅ **Universal** - Works with any LLM provider through a simple strategy interface
- ✅ **Reliable** - Built-in retry logic, timeout handling, and coordinated rate limiting
- ✅ **Fast** - Parallel async processing with configurable concurrency
- ✅ **Observable** - Token tracking, metrics collection, and event hooks
- ✅ **Cost-Effective** - Shared caching strategies can dramatically reduce repeated prompt costs
- ✅ **Type-Safe** - Full generic type support with Pydantic validation

---

## Installation

```bash
# Basic installation
pip install async-batch-llm

# With PydanticAI support (recommended for structured output)
pip install 'async-batch-llm[pydantic-ai]'

# With Google Gemini support
pip install 'async-batch-llm[gemini]'

# With OpenAI support
pip install 'async-batch-llm[openai]'

# With OpenRouter support
pip install 'async-batch-llm[openrouter]'

# With DeepSeek support
pip install 'async-batch-llm[deepseek]'

# With everything
pip install 'async-batch-llm[all]'
```

---

## Quick Example

```python
import asyncio
from async_batch_llm import (
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
    agent = Agent("gemini-2.5-flash", output_type=Summary)
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

## Next Steps

- [Getting Started Guide](getting-started.md) - Learn the basics
- [Examples](examples/basic.md) - See more examples
- [API Reference](api/core.md) - Full API documentation
- [Contributing](contributing.md) - Help improve async-batch-llm

---

## License

MIT License - See [LICENSE](https://github.com/geoff-davis/async-batch-llm/blob/main/LICENSE) for details.
