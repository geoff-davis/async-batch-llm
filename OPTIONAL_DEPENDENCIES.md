# Optional Dependencies

## Overview

As of v0.4.0, `batch-llm` has made both **PydanticAI** and **google-genai** optional
dependencies. The strategy pattern makes it easy to plug into any LLM provider.

**Core dependency**: Only `pydantic>=2.0.0` is required.

## Installation Options

### Minimal Installation (Core only)

```bash
pip install async-batch-llm
# or
uv add batch-llm
```

This installs only the core dependencies:

- `pydantic>=2.0.0`

**Use case**: Custom strategies with any LLM provider (OpenAI, Anthropic, Cohere, etc.)

**What works**:

- ✅ Custom LLM strategies
- ✅ All retry/rate-limit logic
- ✅ Middleware and observers
- ✅ Full framework functionality

**What requires extras**:

- ❌ `PydanticAIStrategy` (requires `[pydantic-ai]`)
- ❌ `GeminiStrategy` / `GeminiCachedStrategy` (requires `[gemini]`)

### With PydanticAI Support

```bash
pip install 'async-batch-llm[pydantic-ai]'
# or
uv add 'async-batch-llm[pydantic-ai]'
```

**Use case**: Using PydanticAI agents with batch-llm

**Enables**:

- `PydanticAIStrategy` for wrapping PydanticAI agents
- Structured output with Pydantic models
- Works with any PydanticAI-supported provider

### With Gemini SDK

```bash
pip install 'async-batch-llm[gemini]'
# or
uv add 'async-batch-llm[gemini]'
```

**Use case**: Direct Google Gemini API integration

**Enables**:

- `GeminiStrategy` for direct Gemini API calls
- `GeminiCachedStrategy` for Gemini with context caching
- `GeminiErrorClassifier` for Gemini-specific error handling

### With Everything

```bash
pip install 'async-batch-llm[all]'
# or
uv add 'async-batch-llm[all]'
```

Includes:

- PydanticAI support
- Gemini SDK support

### For Development

```bash
pip install 'async-batch-llm[dev]'
# or
uv add 'async-batch-llm[dev]'
```

Includes:

- All dependencies above
- pytest, pytest-asyncio
- ruff, mypy

## Usage Patterns (v0.4.x)

### Pattern 1: PydanticAI Strategy (requires `pydantic-ai`)

```python
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, PydanticAIStrategy
from pydantic_ai import Agent

# Create agent
agent = Agent('openai:gpt-4', result_type=MyOutput)

# Wrap in strategy
strategy = PydanticAIStrategy(agent=agent)

# Use in work item
work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="Your prompt here",
)
```

**Required**: `batch-llm[pydantic-ai]`

### Pattern 2: Gemini Strategies (requires `gemini`)

```python
from batch_llm.llm_strategies import GeminiStrategy, GeminiCachedStrategy
from google import genai

client = genai.Client(api_key="your-api-key")

# Simple Gemini strategy
strategy = GeminiStrategy(
    model="gemini-2.5-flash",
    client=client,
    response_parser=lambda r: r.text,
)

# OR cached strategy for RAG
strategy = GeminiCachedStrategy(
    model="gemini-2.5-flash",
    client=client,
    response_parser=lambda r: r.text,
    cached_content=your_cached_content,
    cache_ttl_seconds=3600,
)

work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="Your prompt here",
)
```

**Required**: `batch-llm[gemini]`

### Pattern 3: Custom Strategy (no extra dependencies)

```python
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import LLMCallStrategy

class MyCustomStrategy(LLMCallStrategy[MyOutput]):
    """Custom strategy for any LLM provider."""

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[MyOutput, dict[str, int]]:
        # Your custom LLM API call here
        output = await call_your_llm_api(prompt)
        token_usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        return output, token_usage

# Use it
strategy = MyCustomStrategy()
work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="Your prompt here",
)
```

**Required**: Only `batch-llm` (no extras needed)

## Examples

### Direct Gemini API (no PydanticAI)

See `examples/example_llm_strategies.py` for complete Gemini examples using built-in strategies.

Install:

```bash
pip install 'async-batch-llm[gemini]'
export GOOGLE_API_KEY=your_api_key  # GEMINI_API_KEY also works
python examples/example_llm_strategies.py
```

**Full documentation**: [docs/GEMINI_INTEGRATION.md](docs/GEMINI_INTEGRATION.md)

### OpenAI Custom Strategy

See `examples/example_openai.py` for OpenAI integration.

Install:

```bash
pip install async-batch-llm openai
export OPENAI_API_KEY=your_api_key
python examples/example_openai.py
```

### Anthropic Claude Custom Strategy

See `examples/example_anthropic.py` for Anthropic integration.

Install:

```bash
pip install async-batch-llm anthropic
export ANTHROPIC_API_KEY=your_api_key
python examples/example_anthropic.py
```

### LangChain Integration

See `examples/example_langchain.py` for LangChain integration including RAG.

Install:

```bash
pip install async-batch-llm langchain langchain-openai
python examples/example_langchain.py
```

## Migration from pre-0.4.0

### From pre-0.4.0 with PydanticAI

If you were using `agent=` parameter before v0.4.0:

```python
# Legacy (<0.4.0)
work_item = LLMWorkItem(
    item_id="item_1",
    agent=agent,  # ❌ Removed in v0.4.0
    prompt="...",
)

# v0.4.0+
from batch_llm import PydanticAIStrategy

strategy = PydanticAIStrategy(agent=agent)
work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,  # ✅ v0.4.0+ API
    prompt="...",
)
```

**Migration**: Install `batch-llm[pydantic-ai]` and wrap agents in `PydanticAIStrategy`

### From pre-0.4.0 with direct_call

If you were using `direct_call=` parameter before v0.4.0:

```python
# Legacy (<0.4.0)
async def my_call(attempt: int, timeout: float):
    ...

work_item = LLMWorkItem(
    item_id="item_1",
    direct_call=my_call,  # ❌ Removed in v0.4.0
)

# v0.4.0+
from batch_llm.llm_strategies import LLMCallStrategy

class MyStrategy(LLMCallStrategy[OutputType]):
    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Your call logic here
        ...

strategy = MyStrategy()
work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,  # ✅ v0.4.0+ API
    prompt="...",
)
```

**Migration**: Create a custom strategy class

See [docs/archive/MIGRATION_V0_1.md](docs/archive/MIGRATION_V0_1.md) for complete migration guide.

## Benefits of Optional Dependencies

1. **Smaller install**: Core package is lighter without optional deps
2. **Flexibility**: Use any LLM provider with custom strategies
3. **Provider agnostic**: Not tied to specific LLM services
4. **Full control**: Implement exactly the integration you need

## Technical Implementation

Optional dependencies are conditionally imported:

```python
# In src/batch_llm/llm_strategies.py
try:
    from pydantic_ai import Agent
except ImportError:
    Agent = Any  # Fallback for type hints
```

This allows the package to:

- Work without optional deps installed
- Maintain type hints for those who do install them
- Fail gracefully with clear errors if you try to use a strategy without installing its dependencies

## Recommendation

- **For maximum flexibility**: Use core only + custom strategies
- **For PydanticAI users**: Install with `[pydantic-ai]` extra
- **For Gemini users**: Install with `[gemini]` extra
- **For quick prototypes**: Use `[all]` to get everything
- **For development**: Use `[dev]` for testing and linting tools
