# Basic Usage Examples

The first two examples use the high-level `process_prompts()` API, which covers most
batches. The rest drive `ParallelBatchProcessor` directly for full control over
queueing and lifecycle.

## Simple Batch Processing

Process multiple prompts in parallel:

```python
import asyncio
from async_batch_llm import llm, process_prompts

async def main():
    prompts = ["What is Python?", "What is async/await?", "What is asyncio?"]
    batch = await process_prompts(llm("openai:gpt-4o-mini"), prompts, concurrency=5)

    for result in batch.results:  # completion order
        if result.success:
            print(f"{result.item_id}: {result.output}")
        else:
            print(f"{result.item_id}: Failed - {result.error}")

asyncio.run(main())
```

## Context Passing

Pass `(item_id, prompt, context)` triples to carry application data into each result:

```python
from dataclasses import dataclass

from async_batch_llm import llm, process_prompts

@dataclass
class FileContext:
    filepath: str
    original_content: str

async def process_with_context():
    files = [("file1.py", "content1"), ("file2.py", "content2")]
    batch = await process_prompts(
        llm("openai:gpt-4o-mini"),
        [
            (path, f"Summarize: {content}", FileContext(path, content))
            for path, content in files
        ],
    )

    for result in batch.results:
        if result.success and result.context:
            print(f"File: {result.context.filepath}")
            print(f"Summary: {result.output}")
```

## Full control: ParallelBatchProcessor

The remaining examples build work items and drive the processor yourself. They use a
PydanticAI agent (`pip install 'async-batch-llm[pydantic-ai,gemini]'`).
`GoogleModel` reads `GOOGLE_API_KEY` when it is constructed. Pass a model object,
as here, rather than a bare model name: pydantic-ai 2.x rejects bare names, and the
provider prefixes differ between the 1.x and 2.x lines.

```python
import asyncio
from async_batch_llm import (
    ParallelBatchProcessor,
    LLMWorkItem,
    ProcessorConfig,
    PydanticAIStrategy,
)
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

async def main():
    agent = Agent(GoogleModel("gemini-2.5-flash"), output_type=str)
    strategy = PydanticAIStrategy(agent=agent)

    config = ProcessorConfig(concurrency=5)

    async with ParallelBatchProcessor(config=config) as processor:
        prompts = [
            "What is Python?",
            "What is async/await?",
            "What is asyncio?"
        ]

        for i, prompt in enumerate(prompts):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"item_{i}",
                    strategy=strategy,
                    prompt=prompt
                )
            )

        result = await processor.process_all()

        for work_result in result.results:
            if work_result.success:
                print(f"{work_result.item_id}: {work_result.output}")
            else:
                print(f"{work_result.item_id}: Failed - {work_result.error}")

asyncio.run(main())
```

## Structured Output

Use Pydantic models for validated output:

```python
from pydantic import BaseModel

class CodeReview(BaseModel):
    issues: list[str]
    suggestions: list[str]
    rating: int

async def review_code():
    agent = Agent(GoogleModel("gemini-2.5-flash"), output_type=CodeReview)
    strategy = PydanticAIStrategy(agent=agent)

    config = ProcessorConfig(concurrency=3)

    async with ParallelBatchProcessor(config=config) as processor:
        code_snippets = ["def foo(): pass", "def bar(): return 42"]

        for snippet in code_snippets:
            await processor.add_work(
                LLMWorkItem(
                    item_id=snippet[:20],
                    strategy=strategy,
                    prompt=f"Review this code:\n{snippet}"
                )
            )

        result = await processor.process_all()

        for work_result in result.results:
            if work_result.success:
                review = work_result.output
                print(f"Rating: {review.rating}/10")
                print(f"Issues: {review.issues}")
```

## Post-Processing

Use post-processors to handle results as they complete:

```python
async def save_result(result):
    """Called for each completed work item."""
    if result.success:
        # Save to database, file, etc.
        await save_to_db(result.item_id, result.output)
        print(f"Saved {result.item_id}")

async def process_with_post_processor():
    agent = Agent(GoogleModel("gemini-2.5-flash"), output_type=str)
    strategy = PydanticAIStrategy(agent=agent)

    config = ProcessorConfig(concurrency=5)

    async with ParallelBatchProcessor(
        config=config,
        post_processor=save_result  # Called for each result
    ) as processor:
        # Add work items...
        result = await processor.process_all()
```

## Metrics Collection

Track metrics using observers:

```python
from async_batch_llm.observers import MetricsObserver

async def process_with_metrics():
    metrics = MetricsObserver()

    agent = Agent(GoogleModel("gemini-2.5-flash"), output_type=str)
    strategy = PydanticAIStrategy(agent=agent)

    config = ProcessorConfig(concurrency=5)

    async with ParallelBatchProcessor(
        config=config,
        observers=[metrics]
    ) as processor:
        # Add work items...
        result = await processor.process_all()

        # Get collected metrics
        collected_metrics = await metrics.get_metrics()
        print(f"Items processed: {collected_metrics['items_processed']}")
        print(f"Succeeded: {collected_metrics['items_succeeded']}")
        print(f"Failed: {collected_metrics['items_failed']}")
```
