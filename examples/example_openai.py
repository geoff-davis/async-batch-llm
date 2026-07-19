"""Example of using async-batch-llm with OpenAI via the built-in OpenAIModel.

This example uses the first-class ``OpenAIModel`` and ``OpenAIStrategy``
classes (added in v0.9.0). For a custom-strategy reference (writing your own
``LLMCallStrategy`` subclass against the raw OpenAI SDK) see
``examples/example_llm_strategies.py``.

## Installation

```bash
pip install 'async-batch-llm[openai]'
```

## Setup

```bash
export OPENAI_API_KEY=sk-...
```

## Features demonstrated

1. ``OpenAIModel.from_api_key`` convenience constructor.
2. ``OpenAIStrategy`` with the default text-passthrough parser.
3. ``OpenAIStrategy`` with a structured-output ``response_parser``.
4. ``OpenAIErrorClassifier`` wired into the processor.
5. Cached prompt token tracking — OpenAI's automatic prompt cache surfaces
   ``cached_input_tokens`` for prompts longer than ~1024 tokens.
"""

import asyncio
import json
import os

from pydantic import BaseModel

from async_batch_llm import (
    LLMWorkItem,
    OpenAIErrorClassifier,
    OpenAIModel,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
)


class SentimentOutput(BaseModel):
    """Structured output for sentiment classification."""

    sentiment: str  # "positive" | "negative" | "neutral"
    confidence: float


async def example_simple_text() -> None:
    """Basic text generation across a small batch."""
    print("\n=== Example 1: Simple text generation ===\n")

    model = OpenAIModel.from_api_key(
        "gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    strategy = OpenAIStrategy(model)
    config = ProcessorConfig(max_workers=3, attempt_timeout=30.0)

    async with ParallelBatchProcessor[None, str, None](
        config=config,
        error_classifier=OpenAIErrorClassifier(),
    ) as processor:
        for i, q in enumerate(
            [
                "What is the capital of France?",
                "Explain quantum computing in one sentence.",
                "Largest planet in the solar system?",
            ]
        ):
            await processor.add_work(LLMWorkItem(item_id=f"q{i}", strategy=strategy, prompt=q))
        result = await processor.process_all()

    for r in result.results:
        if r.success:
            print(f"{r.item_id}: {r.output}")
    print(
        f"\nTokens: input={result.total_input_tokens} "
        f"output={result.total_output_tokens} "
        f"cached={result.total_cached_tokens}"
    )


async def example_structured_output() -> None:
    """Structured output via a custom response_parser.

    We ask the model to return JSON in a fixed shape, then parse the
    response text into a Pydantic model. For OpenAI specifically, you can
    also use ``client.chat.completions.parse(...)`` (response_format=...) by
    writing a thin custom strategy — left to the reader to keep this example
    portable across OpenAI-compatible providers.
    """
    print("\n=== Example 2: Structured output ===\n")

    model = OpenAIModel.from_api_key(
        "gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        system_instruction=(
            "You classify user-supplied text as positive, negative, or "
            "neutral. Respond with ONLY a JSON object of the shape "
            '{"sentiment": "positive"|"negative"|"neutral", '
            '"confidence": <0.0-1.0>}. No preamble, no code fence.'
        ),
        # OpenAI JSON mode reduces stray prose; not strictly required.
        extra_body={"response_format": {"type": "json_object"}},
    )

    strategy = OpenAIStrategy(
        model,
        response_parser=lambda r: SentimentOutput.model_validate_json(r.text),
    )
    config = ProcessorConfig(max_workers=2, attempt_timeout=30.0)

    reviews = [
        "This product is amazing! Highly recommend.",
        "Not worth the money. Very disappointed.",
        "It's okay. Does the job.",
    ]

    async with ParallelBatchProcessor[None, SentimentOutput, dict](
        config=config,
        error_classifier=OpenAIErrorClassifier(),
    ) as processor:
        for i, text in enumerate(reviews):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"review_{i}",
                    strategy=strategy,
                    prompt=text,
                    context={"original": text},
                )
            )
        result = await processor.process_all()

    for r in result.results:
        if r.success and r.output is not None:
            print(
                f"{r.item_id}: sentiment={r.output.sentiment} "
                f"confidence={r.output.confidence:.2f} "
                f"text={json.dumps(r.context['original'])[:60]}..."
            )
        else:
            print(f"{r.item_id} FAILED: {r.error}")


async def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("Set OPENAI_API_KEY before running this example.")
        return
    await example_simple_text()
    await example_structured_output()


if __name__ == "__main__":
    asyncio.run(main())
