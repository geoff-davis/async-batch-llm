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

# With everything
pip install 'async-batch-llm[all]'
```

## Core Concepts

### 1. Strategy Pattern

async-batch-llm uses a strategy pattern to support any LLM provider. A strategy encapsulates:

- How to call the LLM
- How to handle errors
- How to manage resources (e.g., caches)

```python
from async_batch_llm import LLMCallStrategy

class MyCustomStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt: str, attempt: int, timeout: float):
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

agent = Agent("gemini-2.5-flash", result_type=Output)
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

## Next Steps

- [Basic Examples](examples/basic.md) - See more usage examples
- [Custom Strategies](examples/custom-strategies.md) - Build your own strategies
- [Advanced Patterns](examples/advanced.md) - Learn advanced techniques
