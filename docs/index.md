# batch-llm

**Process thousands of LLM requests in parallel with automatic retries, rate limiting, and flexible error handling.**

Works with any LLM provider (OpenAI, Anthropic, Google, LangChain, or custom) through a simple strategy pattern.
Built on asyncio for efficient I/O-bound processing.

[![PyPI version](https://badge.fury.io/py/batch-llm.svg)](https://badge.fury.io/py/batch-llm)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why batch-llm?

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

## Next Steps

- [Getting Started Guide](getting-started.md) - Learn the basics
- [Examples](examples/basic.md) - See more examples
- [API Reference](api/core.md) - Full API documentation
- [Contributing](contributing.md) - Help improve batch-llm

---

## License

MIT License - See [LICENSE](https://github.com/geoff-davis/batch-llm/blob/main/LICENSE) for details.
