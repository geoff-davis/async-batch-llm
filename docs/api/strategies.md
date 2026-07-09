# Strategies API Reference

## LLMCallStrategy

::: async_batch_llm.LLMCallStrategy

## ModelStrategy

Shared base for the provider-named strategies below; delegates to an
`LLMModel`. Use directly for a custom model you don't want a dedicated
subclass for.

::: async_batch_llm.ModelStrategy

## PydanticAIStrategy

::: async_batch_llm.PydanticAIStrategy

## Structured JSON Parsing

::: async_batch_llm.pydantic_json_parser

::: async_batch_llm.strip_code_fences

## GeminiStrategy

::: async_batch_llm.GeminiStrategy

## OpenAIStrategy

::: async_batch_llm.OpenAIStrategy

## OpenRouterStrategy

::: async_batch_llm.OpenRouterStrategy

## DeepSeekStrategy

::: async_batch_llm.DeepSeekStrategy

## Models

### GeminiModel

::: async_batch_llm.GeminiModel

### GeminiCachedModel

::: async_batch_llm.GeminiCachedModel

### OpenAICompatibleModel

::: async_batch_llm.OpenAICompatibleModel

### OpenAIModel

::: async_batch_llm.OpenAIModel

### OpenRouterModel

::: async_batch_llm.OpenRouterModel

### DeepSeekModel

::: async_batch_llm.DeepSeekModel

## Protocols

### LLMModel

::: async_batch_llm.LLMModel

### ManagedLLMModel

::: async_batch_llm.ManagedLLMModel

### LLMResponse

::: async_batch_llm.LLMResponse
