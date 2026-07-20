# Advanced Patterns

## Smart Model Escalation

Save costs by starting with cheap models and escalating only on validation errors:

```python
from pydantic import ValidationError
from async_batch_llm import LLMCallStrategy, RetryState

class SmartModelEscalation(LLMCallStrategy[dict]):
    MODELS = [
        "gemini-2.5-flash-lite",  # Cheapest
        "gemini-2.5-flash",       # Medium
        "gemini-2.5-pro",         # Most capable
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
        return response.output, response.tokens
```

**Cost savings: 60-80% vs. always using the best model.**

## Smart Retry with Validation Feedback

Tell the LLM exactly what failed on retry:

```python
class SmartRetryStrategy(LLMCallStrategy[PersonData]):
    def __init__(self, client):
        self.client = client

    async def on_error(self, exception: Exception, attempt: int, state: RetryState | None = None):
        if state is not None and isinstance(exception, ValidationError):
            state.set("last_validation_error", exception)

    async def execute(self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None):
        if attempt == 1:
            final_prompt = prompt
        else:
            # Create retry prompt with field-level feedback
            final_prompt = self._create_retry_prompt(prompt, state)

        try:
            response = await self.client.generate(final_prompt)
            output = PersonData.model_validate_json(response.text)
            return output, tokens
        except ValidationError as e:
            if state is not None:
                state.set("last_response", response.text)
            raise

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
        "gemini-2.5-flash", client,
        cached_content=[Content(parts=[{"text": rag_context}], role="user")],
    )
    strategy = GeminiStrategy(cached_model, response_parser=lambda r: r.text)

    config = ProcessorConfig(max_workers=5)

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

Track custom metrics:

```python
from async_batch_llm.observers import BaseObserver, ProcessingEvent
from async_batch_llm import LLMWorkItem, WorkItemResult
from typing import Any

class CostTracker(BaseObserver):
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        if event == ProcessingEvent.ITEM_COMPLETED:
            # The ITEM_COMPLETED payload carries the item's total tokens as an int
            total = data.get("tokens", 0)
            self.total_tokens += total
            self.total_cost += total * 0.00001  # Example rate

async def main():
    cost_tracker = CostTracker()

    async with ParallelBatchProcessor(
        config=config,
        observers=[cost_tracker]
    ) as processor:
        # Add work items...
        result = await processor.process_all()

        print(f"Total tokens: {cost_tracker.total_tokens}")
        print(f"Estimated cost: ${cost_tracker.total_cost:.4f}")
```

## Adapting Worker Count Between Batches

Processors are one-shot (`add_work()` raises after `process_all()`), and the
worker count is captured at construction — so adapt by inspecting the stats
and building the *next* processor with a different config:

```python
async def adaptive_processing(items, max_workers=10):
    config = ProcessorConfig(
        max_workers=max_workers,  # Start optimistic
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
        return response.text, extract_tokens(response)

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ):
        if state and isinstance(exception, ValidationError):
            state.set("validation_failures", state.get("validation_failures", 0) + 1)
```

## Partial Recovery with RetryState

Save partial results across attempts and retry only the fields that failed —
often cheaper than re-extracting everything.

```python
from async_batch_llm import RetryState
from async_batch_llm.llm_strategies import LLMCallStrategy

class PartialRecoveryStrategy(LLMCallStrategy[dict]):
    """Parse partial results and retry only failed fields."""

    FIELDS = ["name", "email", "phone", "address"]

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
            raise ValueError(f"Missing fields: {missing}")

        return result, extract_tokens(response)
```

Retries focus only on the fields that failed validation, so the follow-up
attempt usually consumes fewer tokens than the first. See
[`examples/example_smart_model_escalation.py`](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_smart_model_escalation.py)
and `examples/example_gemini_smart_retry.py` for complete, runnable versions.
