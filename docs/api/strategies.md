# Strategies API Reference

## CallableStrategy

Adapter for an existing asynchronous SDK, gateway client, agent, or application
service. See [Use Your Existing Async Client](../callable-integration.md) for
callback and retry-state semantics.

::: async_batch_llm.CallableStrategy

## CallOutcome

::: async_batch_llm.CallOutcome

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

#### `estimate_tokens(prompt, attempt, state)`

Optional per-strategy TPM estimator. It runs after middleware and coordinated
cooldown but before the atomic quota reservation and provider-capacity wait.
It may be synchronous or asynchronous and receives retry state so model
escalation can change the expected output. `ProcessorConfig.token_estimator`
takes precedence. The default returns `None`.

`quota_scope` identifies strategies sharing RPM, TPM, and cooldown by object
identity. It defaults to `concurrency_scope`, which identifies shared provider
capacity. Override them independently when account quota and client capacity
have different ownership. Explicit `None` uses strategy identity; unhashable
objects are supported. A raising property fails closed with non-retryable
`QuotaScopeError`, without copying the property exception message. Initial
configuration raises this error directly (including `call_result` and pool
construction); a middleware-selected strategy failure becomes a per-item
`quota_scope_error` result. No provider request or quota debit follows it.

#### `async def prepare() -> None`

Initialize resources before making LLM calls (e.g., create caches, initialize clients).

**Default:** No-op

#### `async def execute(prompt: str, attempt: int, timeout: float, state: RetryState | None = None) -> tuple[TOutput, TokenUsage]`

Execute an LLM call.

**Parameters:**

- `prompt` (str): The prompt to send to the LLM
- `attempt` (int): Which retry attempt this is (1, 2, 3, ...)
- `timeout` (float): Maximum time to wait for response (seconds)
  - Note: Timeout enforcement is handled by the framework wrapping this call in `asyncio.wait_for()`
- `state` (RetryState | None): Mutable per-work-item state provided by the framework
  so strategies can track partial progress across retries

**Returns:** Tuple of `(output, token_usage)`

- `output` (TOutput): The LLM response
- `token_usage` ([TokenUsage](core.md#tokenusage)): Token usage dict with optional keys: `input_tokens`,
  `output_tokens`, `total_tokens`, `cached_input_tokens`

**Raises:** Any exception to trigger retry (if retryable) or failure

#### `async def dry_run(prompt: str) -> tuple[TOutput, TokenUsage]`

Return mock output for dry-run mode (testing without API calls).

Called when `ProcessorConfig(dry_run=True)` is set. Override this method to provide realistic mock data for testing.

**Parameters:**

- `prompt` (str): The prompt that would have been sent to the LLM

**Returns:** Tuple of `(mock_output, mock_token_usage)`

**Default behavior:**

- Returns string `"[DRY-RUN] Mock output for prompt: {prompt[:50]}..."` as output
- Returns mock token usage: 100 input, 50 output, 150 total tokens

**Example override:**

```python
class MyStrategy(LLMCallStrategy[Output]):
    async def dry_run(self, prompt: str) -> tuple[Output, TokenUsage]:
        # Return realistic mock data
        mock_output = Output(result="Test result")
        mock_tokens: TokenUsage = {
            "input_tokens": len(prompt.split()),
            "output_tokens": 50,
            "total_tokens": len(prompt.split()) + 50,
        }
        return mock_output, mock_tokens
```

#### `async def on_error(exception: Exception, attempt: int, state: RetryState | None = None) -> None`

Handle errors that occur during execute().

Called by the framework when `execute()` raises an exception, before deciding whether to retry. This allows strategies to:

- Inspect the error type to adjust retry behavior
- Store error information for use in the next attempt
- Modify prompts based on validation errors
- Track error patterns across attempts
- Make intelligent decisions (e.g., escalate to smarter model only on validation errors)

**Parameters:**

- `exception` (Exception): The exception that was raised during `execute()`
- `attempt` (int): Which attempt number failed (1, 2, 3, ...)
- `state` (RetryState | None): Retry state that persists across attempts (v0.3.0)

**Default:** No-op

**Use Cases:**

1. **Smart Model Escalation** - Only escalate to expensive models on validation errors, not
   network errors:

   ```python
   class SmartModelEscalationStrategy(LLMCallStrategy[Output]):
       async def on_error(self, exception: Exception, attempt: int, state=None) -> None:
           if state is not None and isinstance(exception, ValidationError):
               state.set("validation_failures", state.get("validation_failures", 0) + 1)

       async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
           # Only escalate model on validation errors
           failures = state.get("validation_failures", 0) if state is not None else 0
           model_index = min(failures, len(MODELS) - 1)
           model = MODELS[model_index]
           # Make call with appropriate model...
   ```

1. **Smart Retry with Partial Parsing** - Build better retry prompts based on what failed:

   ```python
   class SmartRetryStrategy(LLMCallStrategy[Output]):
       async def on_error(self, exception: Exception, attempt: int, state=None) -> None:
           if state is not None and isinstance(exception, ValidationError):
               state.set("last_validation_error", exception)
               # last_response set in execute() before raising

       async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
           if attempt > 1 and state and state.get("last_validation_error"):
               # Build smart retry prompt with partial parsing feedback
               prompt = self._create_retry_prompt_with_partial_data(prompt, state)
           # Make call with improved prompt...
   ```

1. **Error Type Tracking** - Distinguish between different error types:

   ```python
   class ErrorTrackingStrategy(LLMCallStrategy[Output]):
       async def on_error(self, exception: Exception, attempt: int, state=None) -> None:
           if state is None:
               return
           if isinstance(exception, ValidationError):
               key = "validation_errors"
           elif isinstance(exception, ConnectionError):
               key = "network_errors"
           elif "429" in str(exception):
               key = "rate_limit_errors"
           else:
               key = "other_errors"
           state.set(key, state.get(key, 0) + 1)
   ```

**Important Notes:**

- Exceptions in `on_error()` are caught and logged by the framework - they won't crash processing
- `on_error()` is only called when `execute()` raises an exception, not on success
- The error is still propagated to the framework's retry logic after `on_error()` returns
- Share strategy/client instances across work items, and keep all item-specific mutation in
  the supplied `RetryState`. Use concurrency-safe metrics or observers for batch-wide counters.

**See Also:**

- [examples/example_smart_model_escalation.py][ex-escalation] - Complete
  smart model escalation example
- [examples/example_gemini_smart_retry.py][ex-smart-retry] - Smart retry with
  partial parsing

[ex-escalation]: https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_smart_model_escalation.py
[ex-smart-retry]: https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_gemini_smart_retry.py

#### `async def cleanup() -> None`

Clean up resources after all attempts complete (e.g., delete caches, close clients).

**Default:** No-op

**Custom Strategy Example:**

```python
from async_batch_llm import LLMCallStrategy, TokenUsage

class MyCustomStrategy(LLMCallStrategy[str]):
    async def execute(
        self, prompt: str, attempt: int, timeout: float, state=None
    ) -> tuple[str, TokenUsage]:
        # Your custom LLM API call
        response = await my_llm_api.generate(prompt)

        tokens: TokenUsage = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        }

        return response.text, tokens
```

---

### Structured JSON Parsing

`pydantic_json_parser(Model)` strips a normal outer Markdown fence and performs
strict Pydantic JSON/schema validation. Recovery is disabled by default.

```python
parser = pydantic_json_parser(
    Classification,
    recover_trailing_markdown=True,
)
strategy = OpenAIStrategy(model, parser)
```

With recovery enabled, strict parsing still runs first. On failure, the parser
uses a real JSON decoder to read exactly one complete top-level object or array,
accepts only the explicitly supported trailing closing-fence artifacts (three
backticks, with or without the observed trailing underscore), and then runs
normal Pydantic schema validation. It does not repair malformed JSON or accept
scalars, multiple JSON values, arbitrary prose, or schema-invalid data. Those
failures continue through the configured retry policy.

A recovered `WorkItemResult` exposes typed recovery properties backed by
metadata. Processor stats and `MetricsObserver` include
`structured_output_recoveries`, `structured_output_retries_avoided`, and counts
by `structured_output_recovery_reasons`.

### DeepSeek strict JSON Schema output

DeepSeek's Responses API can enforce a JSON Schema before ABL parses the
result. Pass a Pydantic model class to get that model back directly:

```python
from pydantic import BaseModel

from async_batch_llm import DeepSeekModel, DeepSeekStrategy

class Verdict(BaseModel):
    valid: bool
    reason: str

model = DeepSeekModel.from_api_key(
    "deepseek-v4-flash",
    api_surface="responses",
    response_schema=Verdict,
    thinking=False,
    max_connections=64,
)
strategy = DeepSeekStrategy(
    model,
    generation_config={"max_tokens": 256},  # mapped to max_output_tokens
)
```

A JSON Schema mapping is also accepted; the default strategy output is then
the decoded JSON value. `schema_name=` overrides the Pydantic class name or
schema `title` used in the provider request.

This mode runs through the ordinary `LLMGateway` and batch strategy path, so
capacity admission, proactive quotas, retries, timing, token/cache accounting,
and cost calculation remain active. Result metadata includes `api_surface`,
`provider_request_id`, and `response_schema` (`name`, Python/schema identity,
and canonical SHA-256). The same API/schema fields participate in inferred
JSONL/SQLite artifact identity, preventing Chat JSON-mode or different-schema
results from being replayed as compatible strict outputs.

ABL validates the surface and schema locally and passes model IDs through to
DeepSeek for validation. See the [provider Responses reference](https://api-docs.deepseek.com/api/create-response/)
for available models. The `deepseek` extra requires OpenAI SDK 1.66.0 or newer;
caller-supplied clients must expose callable `responses.create`. This does not
add multimodal work-item support or an automatic surface fallback. The explicit fallback is Chat Completions
with `json_mode=True` plus `pydantic_json_parser(...)`; it requests valid JSON
but does not enforce the schema and therefore cannot safely repair malformed
JSON inside the object. Provider/schema rejection is non-retryable under
`structured_output_schema_rejected`; an accepted response that fails local
validation is retryable under `structured_output_validation_error`.

---

### Provider classifier categories

`PydanticAIStrategy` recommends the exported `PydanticAIErrorClassifier`.
It classifies `ModelHTTPError` by HTTP status (429 rate limit, 401 authentication,
403 permission denied, other deterministic 4xx client errors, transient 5xx
server errors). `UsageLimitExceeded` is non-retryable with category
`usage_limit_exceeded`. Exact `UnexpectedModelBehavior` remains a retryable
`validation_error`; its content-filter and incomplete-tool-call subclasses
retain ordinary retry backoff.

All built-in classifiers handle framework deadlines, aborts and framework
timeouts before provider matching, along with named middleware-contract and
quota-scope errors. Terminal rate-limit retry exhaustion uses
`rate_limit_retries_exceeded` with `is_rate_limit=False`. Empty provider
responses use non-retryable `empty_response`; the exception remains a
`ValueError` subclass. Structured-output schema rejection is non-retryable
`structured_output_schema_rejected`; local output validation is retryable
`structured_output_validation_error`.

Gemini provider timeouts use `timeout`, including bare `TimeoutError()`.
Default, OpenAI and OpenRouter provider timeouts use `api_timeout`.
Gemini daily quota exhaustion uses non-retryable `quota_exhausted` only when
every reported quota violation has an explicit per-day quota ID; ambiguous and
mixed daily/minute violations remain rate limits. Gemini RPC retry-delay hints
are parsed alongside HTTP headers.

Category changes affect per-category metrics and `abort_on_error_categories`.
The string factory rejects unknown model kwargs with a list of accepted names,
and unresolved OpenAI credentials raise `ValueError`.
