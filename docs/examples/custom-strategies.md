# Custom Strategies

Learn how to create custom strategies for any LLM provider.

## Basic Custom Strategy

```python
from async_batch_llm import LLMCallStrategy

class OpenAIStrategy(LLMCallStrategy[str]):
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    async def execute(self, prompt: str, attempt: int, timeout: float):
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

        return output, tokens
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

    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Use the cached content
        response = await self.client.generate(
            prompt=prompt,
            cache_name=self.cache_name
        )
        return response.text, response.usage

    async def cleanup(self):
        """Delete cache after processing."""
        if self.cache_name:
            await self.client.delete_cache(self.cache_name)
```

## Error Handling

Use `on_error()` to track failures and adjust behavior:

```python
from pydantic import ValidationError

class SmartRetryStrategy(LLMCallStrategy[dict]):
    def __init__(self, client):
        self.client = client
        self.validation_failures = 0

    async def on_error(self, exception: Exception, attempt: int):
        """Track validation errors for smart escalation."""
        if isinstance(exception, ValidationError):
            self.validation_failures += 1

    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Use cheaper model initially, escalate only on validation errors
        if self.validation_failures == 0:
            model = "cheap-model"
        elif self.validation_failures == 1:
            model = "medium-model"
        else:
            model = "expensive-model"

        response = await self.client.generate(prompt, model=model)
        return response.output, response.tokens
```

## Progressive Temperature

Increase temperature on retry for better success rates:

```python
class ProgressiveTempStrategy(LLMCallStrategy[str]):
    def __init__(self, client, temperatures=None):
        self.client = client
        self.temperatures = temperatures or [0.0, 0.5, 1.0]

    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Use progressively higher temperature on retries
        temp_index = min(attempt - 1, len(self.temperatures) - 1)
        temperature = self.temperatures[temp_index]

        response = await self.client.generate(
            prompt=prompt,
            temperature=temperature
        )

        return response.text, response.usage
```

## Anthropic Example

```python
from anthropic import AsyncAnthropic

class AnthropicStrategy(LLMCallStrategy[str]):
    def __init__(self, client: AsyncAnthropic, model: str = "claude-3-5-sonnet-20241022"):
        self.client = client
        self.model = model

    async def execute(self, prompt: str, attempt: int, timeout: float):
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

        return output, tokens
```

## Usage

```python
from async_batch_llm import ParallelBatchProcessor, LLMWorkItem, ProcessorConfig

async def main():
    # Use your custom strategy
    strategy = OpenAIStrategy(client=openai_client, model="gpt-4")

    config = ProcessorConfig(max_workers=5)

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
