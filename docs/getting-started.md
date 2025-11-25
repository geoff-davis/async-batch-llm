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

batch-llm uses a strategy pattern to support any LLM provider. A strategy encapsulates:

- How to call the LLM
- How to handle errors
- How to manage resources (e.g., caches)

```python
from batch_llm import LLMCallStrategy

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
from batch_llm import LLMWorkItem

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
from batch_llm import ParallelBatchProcessor, ProcessorConfig

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
from batch_llm import PydanticAIStrategy
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
from batch_llm.llm_strategies.gemini import GeminiStrategy
from google import genai

client = genai.Client(api_key="your-key")
strategy = GeminiStrategy(
    client=client,
    model="gemini-2.5-flash",
    output_type=str
)
```

### Gemini Cached Strategy

With context caching for repeated prompts:

```python
from batch_llm.llm_strategies.gemini import GeminiCachedStrategy

strategy = GeminiCachedStrategy(
    client=client,
    model="gemini-2.5-flash",
    system_instruction="Your RAG context here...",
    output_type=str
)
```

## Next Steps

- [Basic Examples](examples/basic.md) - See more usage examples
- [Custom Strategies](examples/custom-strategies.md) - Build your own strategies
- [Advanced Patterns](examples/advanced.md) - Learn advanced techniques
