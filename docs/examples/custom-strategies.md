# Custom Strategies

Learn how to create custom strategies for any LLM provider.

Two shortcuts come first. OpenAI, Gemini, OpenRouter, and DeepSeek have built-in
strategies (`llm("openai:gpt-6-luna")`, or `OpenAIStrategy(OpenAIModel(...))`), so you
don't need a custom one for them. To wrap an async client you already have, a single
function passed to [`CallableStrategy`](../callable-integration.md) is usually enough.
Subclass `LLMCallStrategy`, as below, when you need lifecycle hooks (`prepare()`,
`cleanup()`, `on_error()`) or per-attempt logic.

## Basic Custom Strategy

This raw Chat Completions strategy shows the shape of `execute()`. It is named
`RawChatStrategy` so it doesn't shadow the built-in `OpenAIStrategy`, which uses the
Responses API by default.

```python
from async_batch_llm import LLMCallStrategy

class RawChatStrategy(LLMCallStrategy[str]):
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content
        tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return output, tokens, None
```

## Resource Management

Use `prepare()` and `cleanup()` for resource lifecycle:

```python
class CachedStrategy(LLMCallStrategy[str]):
    def __init__(self, client, system_instruction: str):
        self.client = client
        self.system_instruction = system_instruction
        self.cache_name = None

    async def prepare(self):
        """Create cache before processing."""
        self.cache_name = await self.client.create_cache(
            content=self.system_instruction
        )

    async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
        # Use the cached content
        response = await self.client.generate(
            prompt=prompt,
            cache_name=self.cache_name
        )
        return response.text, response.usage, None

    async def cleanup(self):
        """Delete cache after processing."""
        if self.cache_name:
            await self.client.delete_cache(self.cache_name)
```

## Error Handling

Use `on_error()` to track failures and adjust behavior:

```python
from pydantic import ValidationError
from async_batch_llm import RetryState

class SmartRetryStrategy(LLMCallStrategy[dict]):
    def __init__(self, client):
        self.client = client

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ):
        """Track validation errors for smart escalation."""
        if state is not None and isinstance(exception, ValidationError):
            state.set("validation_failures", state.get("validation_failures", 0) + 1)

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ):
        # Use cheaper model initially, escalate only on validation errors
        failures = state.get("validation_failures", 0) if state is not None else 0
        if failures == 0:
            model = "cheap-model"
        elif failures == 1:
            model = "medium-model"
        else:
            model = "expensive-model"

        response = await self.client.generate(prompt, model=model)
        return response.output, response.tokens, None
```

## Progressive Temperature

To raise the temperature only after validation failures, count them in `RetryState`
from `on_error()`, as in the example above. Keying off `attempt` would also raise it
after a transport error or timeout. See
[Progressive Temperature on Retries](advanced.md#progressive-temperature-on-retries)
for the full pattern.

## Anthropic Example

```python
from anthropic import AsyncAnthropic

class AnthropicStrategy(LLMCallStrategy[str]):
    def __init__(self, client: AsyncAnthropic, model: str = "claude-sonnet-5-5"):
        self.client = client
        self.model = model

    async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.content[0].text
        tokens = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }

        return output, tokens, None
```

## Usage

```python
from async_batch_llm import ParallelBatchProcessor, LLMWorkItem, ProcessorConfig

async def main():
    # Use your custom strategy
    strategy = RawChatStrategy(client=openai_client, model="gpt-6-luna")

    config = ProcessorConfig(concurrency=5)

    async with ParallelBatchProcessor(config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test",
                strategy=strategy,
                prompt="Hello!"
            )
        )

        result = await processor.process_all()
```
