"""Grounded Gemini batches: web-search citations through the framework.

Requests Gemini grounding (the ``google_search`` tool) via
``generation_config`` and reads the citations back through the typed views —
``result.grounding.sources`` / ``.queries`` — with no custom strategy, no
extractor, and no reaching into the raw SDK response. Grounding lands in
``WorkItemResult.metadata['grounding']`` by default whenever the response
carries it (v0.16.0, issue #52 Phase 2).

## Installation

```bash
pip install 'async-batch-llm[gemini]'
# or
uv add 'async-batch-llm[gemini]'
```

## Setup

Set your API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

Get an API key from: https://aistudio.google.com/apikey

## Features Demonstrated

1. Requesting grounding via GeminiStrategy(generation_config={"tools": [...]})
2. Typed access to citations: result.grounding.sources / .queries
3. The plain-dict fallback: result.metadata["grounding"] (JSON-serializable)
4. Mixing grounded answers with normal batch processing (retries, concurrency)

## References

- Grounding with Google Search: https://ai.google.dev/gemini-api/docs/google-search
- Typed auxiliary output docs: docs/API.md
- Gemini integration guide: docs/GEMINI_INTEGRATION.md
"""

import asyncio
import json
import os

from google import genai
from google.genai import types

from async_batch_llm import (
    GeminiModel,
    GeminiStrategy,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


async def main():
    """Process a small batch of questions that benefit from live web search."""

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable not set")
        print("Get your API key from: https://aistudio.google.com/apikey")
        print("Then run: export GOOGLE_API_KEY=your_key_here  # GEMINI_API_KEY also works")
        return

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Questions where a grounded (web-searched) answer beats model memory.
    questions = [
        "Who won the most recent FIFA World Cup, and what was the final score?",
        "What is the current version of Python, and when was it released?",
        "What were the headline announcements at the most recent Google I/O?",
    ]

    # Request the google_search tool on every call. Grounding metadata is
    # extracted into metadata['grounding'] automatically — nothing else to wire.
    model = GeminiModel("gemini-2.5-flash", client)
    strategy = GeminiStrategy(
        model,
        generation_config={"tools": [types.Tool(google_search=types.GoogleSearch())]},
    )

    config = ProcessorConfig(max_workers=3, attempt_timeout=60.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        for i, question in enumerate(questions):
            await processor.add_work(
                LLMWorkItem(item_id=f"question_{i}", strategy=strategy, prompt=question)
            )
        result = await processor.process_all()

    print(f"✓ Completed: {result.succeeded}/{result.total_items} successful\n")

    for item_result in result.results:
        print("=" * 60)
        if not item_result.success:
            print(f"{item_result.item_id}: FAILED - {item_result.error}")
            continue

        print(f"{item_result.item_id}: {item_result.output}\n")

        # Typed access — result.grounding is a Grounding | None view parsed
        # from metadata['grounding'] on each read.
        grounding = item_result.grounding
        if grounding is None:
            print("  (no grounding metadata — the model answered from memory)")
            continue

        if grounding.queries:
            print(f"  Searches the model ran: {', '.join(grounding.queries)}")
        for source in grounding.sources:
            print(f"  Source: {source.title or '(untitled)'} — {source.uri}")

        # The underlying dict stays JSON-serializable — persist it as-is.
        _ = json.dumps(item_result.metadata)


if __name__ == "__main__":
    # Note: Requires GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable
    # export GOOGLE_API_KEY=your_api_key_here
    asyncio.run(main())
