# Errors, Classification and Rate Limits

Every exception type the library exports subclasses
[`AsyncBatchLLMError`](core.md#asyncbatchllmerror) and keeps its built-in base
(`TimeoutError`, `RuntimeError` or `ValueError`), so existing `except` clauses
still match. `BatchBudgetExceeded` is documented with the budget settings on the
[core page](core.md#batchbudgetexceeded), `LLMCallError` with
[single calls](single-gateway.md#llmcallerror), and the artifact errors on the
[artifacts page](artifacts.md#artifact-errors).

For which category each built-in classifier assigns, see
[Provider classifier categories](strategies.md#provider-classifier-categories).

## Error classification

### ErrorClassifier

::: async_batch_llm.strategies.ErrorClassifier

### ErrorInfo

::: async_batch_llm.ErrorInfo

### DefaultErrorClassifier

::: async_batch_llm.DefaultErrorClassifier

### GeminiErrorClassifier

::: async_batch_llm.GeminiErrorClassifier

### OpenAIErrorClassifier

::: async_batch_llm.OpenAIErrorClassifier

### OpenRouterErrorClassifier

::: async_batch_llm.OpenRouterErrorClassifier

### PydanticAIErrorClassifier

::: async_batch_llm.PydanticAIErrorClassifier

## Rate-limit strategies

`RateLimitConfig` controls cooldown and slow-start; a `RateLimitStrategy`
decides how long each cooldown lasts.

### RateLimitStrategy

::: async_batch_llm.RateLimitStrategy

### ExponentialBackoffStrategy

::: async_batch_llm.ExponentialBackoffStrategy

### FixedDelayStrategy

::: async_batch_llm.FixedDelayStrategy

## Deadlines, aborts and interruptions

::: async_batch_llm.ItemDeadlineExceeded

::: async_batch_llm.BatchDeadlineExceeded

::: async_batch_llm.BatchAbortedError

::: async_batch_llm.BatchInterruptedError

::: async_batch_llm.FrameworkTimeoutError

::: async_batch_llm.RateLimitRetriesExceeded

## Provider responses and structured output

::: async_batch_llm.EmptyResponseError

::: async_batch_llm.ProviderResponseError

::: async_batch_llm.StructuredOutputSchemaError

::: async_batch_llm.StructuredOutputValidationError

::: async_batch_llm.TokenTrackingError

## Configuration errors

::: async_batch_llm.MiddlewareContractError

::: async_batch_llm.QuotaScopeError

::: async_batch_llm.TokenEstimationError

::: async_batch_llm.TokenEstimatorRequired

::: async_batch_llm.TokenEstimateExceedsLimit
