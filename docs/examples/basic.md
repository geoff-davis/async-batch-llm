# Basic Usage Examples

## Simple Batch Processing

Process multiple prompts in parallel:

```python
import asyncio
from async_batch_llm import (
    ParallelBatchProcessor,
    LLMWorkItem,
    ProcessorConfig,
    PydanticAIStrategy,
)
from pydantic_ai import Agent

async def main():
    agent = Agent("gemini-2.5-flash", result_type=str)
    strategy = PydanticAIStrategy(agent=agent)

    config = ProcessorConfig(max_workers=5)

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
    agent = Agent("gemini-2.5-flash", result_type=CodeReview)
    strategy = PydanticAIStrategy(agent=agent)

    config = ProcessorConfig(max_workers=3)

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

## Context Passing

Pass context through the processing pipeline:

```python
from dataclasses import dataclass

@dataclass
class FileContext:
    filepath: str
    original_content: str

async def process_with_context():
    agent = Agent("gemini-2.5-flash", result_type=str)
    strategy = PydanticAIStrategy(agent=agent)

    config = ProcessorConfig(max_workers=5)

    async with ParallelBatchProcessor[str, str, FileContext](config=config) as processor:
        files = [
            ("file1.py", "content1"),
            ("file2.py", "content2"),
        ]

        for filepath, content in files:
            await processor.add_work(
                LLMWorkItem(
                    item_id=filepath,
                    strategy=strategy,
                    prompt=f"Summarize: {content}",
                    context=FileContext(filepath=filepath, original_content=content)
                )
            )

        result = await processor.process_all()

        for work_result in result.results:
            if work_result.success and work_result.context:
                print(f"File: {work_result.context.filepath}")
                print(f"Summary: {work_result.output}")
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
    agent = Agent("gemini-2.5-flash", result_type=str)
    strategy = PydanticAIStrategy(agent=agent)

    config = ProcessorConfig(max_workers=5)

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

    agent = Agent("gemini-2.5-flash", result_type=str)
    strategy = PydanticAIStrategy(agent=agent)

    config = ProcessorConfig(max_workers=5)

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
