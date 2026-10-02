# Advanced Patterns

## Smart Model Escalation

Save costs by starting with cheap models and escalating only on validation errors:

```python
from pydantic import ValidationError
from async_batch_llm import LLMCallStrategy, RetryState

class SmartModelEscalation(LLMCallStrategy[dict]):
    MODELS = [
        "gemini-3.5-flash-lite",  # Cheapest
        "gemini-3.5-flash",       # Medium
        "gemini-3.1-pro-preview",         # Most capable
    ]

    def __init__(self, client):
        self.client = client

    async def on_error(self, exception: Exception, attempt: int, state: RetryState | None = None):
        """Only escalate on validation errors, not network/rate limit errors."""
        if state is not None and isinstance(exception, ValidationError):
            state.set("validation_failures", state.get("validation_failures", 0) + 1)

    async def execute(self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None):
        # Network error on attempt 2? Retry with same cheap model
        # Validation error on attempt 2? Escalate to better model
        failures = state.get("validation_failures", 0) if state is not None else 0
        model_index = min(failures, len(self.MODELS) - 1)
        model = self.MODELS[model_index]

        response = await self.client.generate(prompt, model=model)
        return response.output, response.tokens, None
```

**Cost savings: 60-80% vs. always using the best model.**

## Smart Retry with Validation Feedback

Tell the LLM exactly what failed on retry. Wrapping the validation error in
`TokenTrackingError` keeps the tokens the failed call billed in the item's totals:

```python
from async_batch_llm import TokenTrackingError

class SmartRetryStrategy(LLMCallStrategy[PersonData]):
    def __init__(self, client):
        self.client = client

    async def on_error(self, exception: Exception, attempt: int, state: RetryState | None = None):
        cause = exception.__cause__ if isinstance(exception, TokenTrackingError) else exception
        if state is not None and isinstance(cause, ValidationError):
            state.set("last_validation_error", cause)

    async def execute(self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None):
        if attempt == 1:
            final_prompt = prompt
        else:
            # Create retry prompt with field-level feedback
            final_prompt = self._create_retry_prompt(prompt, state)

        response = await self.client.generate(final_prompt)
        tokens = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        try:
            output = PersonData.model_validate_json(response.text)
        except ValidationError as e:
            if state is not None:
                state.set("last_response", response.text)
            raise TokenTrackingError(str(e), token_usage=tokens) from e
        return output, tokens, None

    def _create_retry_prompt(self, original_prompt: str, state: RetryState | None) -> str:
        # Parse state.get("last_validation_error") to identify which fields failed.
        # A transport error must not overwrite this validation feedback.
        # Build prompt like: "These fields succeeded: [age]. Fix these: [name, email]"
        return retry_prompt
```

## Shared Context Caching

Dramatically reduce costs for RAG and repeated context:

```python
from async_batch_llm import GeminiCachedModel, GeminiStrategy
from google import genai
from google.genai.types import Content

async def process_with_caching():
    client = genai.Client(api_key="your-key")

    # Load large RAG context once
    with open("knowledge_base.txt") as f:
        rag_context = f.read()  # Could be 100K+ tokens

    # Model manages cache lifecycle (prepare/cleanup)
    cached_model = GeminiCachedModel(
        "gemini-3.5-flash", client,
        cached_content=[Content(parts=[{"text": rag_context}], role="user")],
    )
    strategy = GeminiStrategy(cached_model, response_parser=lambda r: r.text)

    config = ProcessorConfig(concurrency=5)

    async with ParallelBatchProcessor(config=config) as processor:
        # All 100 queries share the same cached context
        for i in range(100):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"query_{i}",
                    strategy=strategy,
                    prompt=f"Answer based on context: {questions[i]}"
                )
            )

        result = await processor.process_all()
        # Cache automatically cleaned up on exit
```

**Cost savings: ~90% for input tokens on cached content.**

## Middleware for Custom Logic

Inject custom behavior into the processing pipeline:

```python
from async_batch_llm import BaseMiddleware, LLMWorkItem, WorkItemResult

class LoggingMiddleware(BaseMiddleware):
    """Subclass BaseMiddleware to get no-op defaults for the hooks you skip.

    Return values matter: before_process must return the work item
    (returning None SKIPS the item, recording it as failed), and
    after_process must return the result.
    """

    async def before_process(self, work_item: LLMWorkItem):
        print(f"Starting {work_item.item_id}")
        return work_item

    async def after_process(self, result: WorkItemResult):
        if result.success:
            print(f"Success: {result.item_id}")
        else:
            print(f"Failed: {result.item_id} - {result.error}")
        return result

    async def on_error(self, work_item: LLMWorkItem, error: Exception):
        print(f"Error in {work_item.item_id}: {error}")
        return None  # None = use default error handling; a WorkItemResult would replace it

async def main():
    logging_middleware = LoggingMiddleware()

    async with ParallelBatchProcessor(
        config=config,
        middlewares=[logging_middleware]
    ) as processor:
        # Add work items...
        result = await processor.process_all()
```

## Custom Observers

Track custom metrics as the run progresses. Events are for live signals; for token
and cost totals, read the finished `BatchResult`, whose per-item `token_usage`
includes failed attempts (an `ITEM_COMPLETED` event reports only the final
attempt's tokens):

```python
from async_batch_llm import CachedTokenRates
from async_batch_llm.observers import BaseObserver, ProcessingEvent
from typing import Any

class ProgressCounter(BaseObserver):
    def __init__(self):
        self.completed = 0
        self.failed = 0
        self.rate_limits = 0

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        if event == ProcessingEvent.ITEM_COMPLETED:
            self.completed += 1
        elif event == ProcessingEvent.ITEM_FAILED:
            self.failed += 1
        elif event == ProcessingEvent.RATE_LIMIT_HIT:
            self.rate_limits += 1

async def main():
    counter = ProgressCounter()

    async with ParallelBatchProcessor(
        config=config,
        observers=[counter]
    ) as processor:
        # Add work items...
        result = await processor.process_all()

    print(f"Completed {counter.completed}, failed {counter.failed}, "
          f"rate limits {counter.rate_limits}")
    # Totals cover every attempt, including failed ones
    print(f"Tokens: {result.total_input_tokens} in / {result.total_output_tokens} out")
    cost = result.estimated_cost(
        input_per_mtok=0.15,  # example prices
        output_per_mtok=0.60,
        cached_token_rate=CachedTokenRates.OPENAI,
    )
    print(f"Estimated cost: ${cost:.4f}")
```

To stop a run at a spend cap, use
`GuardrailConfig(max_total_cost=..., cost_function=...)`; see
[Token and cost budgets](../guardrails.md#token-and-cost-budgets).

## Adapting Worker Count Between Batches

Processors are one-shot (`add_work()` raises after `process_all()`), and the
worker count is captured at construction — so adapt by inspecting the stats
and building the *next* processor with a different config:

```python
async def adaptive_processing(items, max_workers=10):
    config = ProcessorConfig(
        concurrency=max_workers,  # Start optimistic
        attempt_timeout=30.0
    )

    async with ParallelBatchProcessor(config=config) as processor:
        for item in items:
            await processor.add_work(item)
        result = await processor.process_all()
        stats = await processor.get_stats()

    if stats["rate_limit_count"] > 5:
        # Too many rate limits — use fewer workers for the next batch
        return result, 3
    return result, max_workers
```

(Within a single batch you don't need this: the rate-limit cooldown and
slow-start ramp already throttle all workers automatically.)

## Progressive Temperature on Retries

Increase creativity on retries to get past validation errors. Note that rate
limits don't advance the `attempt` number (they're retried at the same logical
attempt), so escalation here is driven by *validation* failures, not throttling.

```python
from pydantic import ValidationError
from async_batch_llm import RetryState
from async_batch_llm.llm_strategies import LLMCallStrategy

class ProgressiveTempStrategy(LLMCallStrategy[str]):
    """Increase temperature only when validation keeps failing."""

    def __init__(self, client, temps=None):
        self.client = client
        self.temps = temps if temps is not None else [0.0, 0.5, 1.0]

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ):
        state = state or RetryState()
        failures = state.get("validation_failures", 0)
        temp = self.temps[min(failures, len(self.temps) - 1)]
        response = await self.client.generate(prompt, temperature=temp)
        return response.text, extract_tokens(response), None

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ):
        if state and isinstance(exception, ValidationError):
            state.set("validation_failures", state.get("validation_failures", 0) + 1)
```

## Partial Recovery with RetryState

Save partial results across attempts and retry only the fields that failed —
often cheaper than re-extracting everything.

The strategy raises its own exception type to trigger the retry. The default
classifier treats built-in errors such as `ValueError`, `TypeError`, and `KeyError` as
non-retryable logic errors, so raising one of those would fail the item on its first attempt.

```python
from async_batch_llm import RetryState
from async_batch_llm.llm_strategies import LLMCallStrategy

class MissingFieldsError(Exception):
    """Retryable: deliberately not a ValueError subclass."""

class PartialRecoveryStrategy(LLMCallStrategy[dict]):
    """Parse partial results and retry only failed fields."""

    FIELDS = ["name", "email", "phone", "address"]

    def __init__(self, client):
        self.client = client

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ):
        state = state or RetryState()
        partial = state.get("partial_results", {})
        needed = state.get("failed_fields", self.FIELDS)

        if attempt == 1:
            final_prompt = f"{prompt}\nExtract: {', '.join(needed)}"
        else:
            final_prompt = (
                f"{prompt}\nYou already got these right: {partial}"
                f"\nNow extract only: {', '.join(needed)}"
            )

        response = await self.client.generate(final_prompt)
        result = parse_response(response)
        if attempt > 1:
            result = {**partial, **result}

        missing = [f for f in self.FIELDS if f not in result]
        if missing:
            state.set("partial_results", dict(result))
            state.set("failed_fields", missing)
            raise MissingFieldsError(f"Missing fields: {missing}")

        return result, extract_tokens(response), None
```

Retries focus only on the fields that failed validation, so the follow-up
attempt usually consumes fewer tokens than the first. See
[`examples/example_smart_model_escalation.py`](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_smart_model_escalation.py)
and `examples/example_gemini_smart_retry.py` for complete, runnable versions.
