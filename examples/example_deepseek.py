"""Example of using async-batch-llm with DeepSeek via the built-in DeepSeekModel.

``DeepSeekModel`` is an ``OpenAICompatibleModel`` pointed at DeepSeek's API
(``https://api.deepseek.com``). It differs from ``OpenAIModel`` in one
important way: DeepSeek reports its automatic context-cache hits at the top
level of the usage object (``prompt_cache_hit_tokens``) rather than under
OpenAI's nested ``prompt_tokens_details.cached_tokens`` — so ``DeepSeekModel``
overrides token extraction to surface them in ``cached_input_tokens``.

## Installation

```bash
pip install 'async-batch-llm[deepseek]'
```

## Setup

```bash
export DEEPSEEK_API_KEY=sk-...
```

## Features demonstrated

1. ``DeepSeekModel.from_api_key`` (reads ``DEEPSEEK_API_KEY``).
2. ``thinking=False`` to force non-thinking mode (cheaper/faster for batch
   classification; V4 models default to thinking).
3. ``max_connections`` to size the httpx pool to ``max_workers`` so high
   concurrency isn't bottlenecked at httpx's ~100 default.
4. ``DeepSeekStrategy`` with the default text-passthrough parser.
5. Native cache-hit token tracking + provider-aware billing via
   ``CachedTokenRates.DEEPSEEK``.
6. ``OpenAIErrorClassifier`` (DeepSeek is OpenAI-compatible, so the OpenAI
   classifier handles its errors — including 402 Insufficient Balance as a
   clear non-retryable error).
"""

import asyncio
import os

from async_batch_llm import (
    CachedTokenRates,
    DeepSeekModel,
    DeepSeekStrategy,
    LLMWorkItem,
    OpenAIErrorClassifier,
    ParallelBatchProcessor,
    ProcessorConfig,
)


async def main() -> None:
    if "DEEPSEEK_API_KEY" not in os.environ:
        print("Set DEEPSEEK_API_KEY before running this example.")
        return

    model = DeepSeekModel.from_api_key(
        "deepseek-chat",
        thinking=False,  # non-thinking: cheaper and faster for this workload
        max_connections=3,  # match max_workers below (scale both together)
    )
    strategy = DeepSeekStrategy(model)
    config = ProcessorConfig(max_workers=3, attempt_timeout=60.0)

    questions = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Largest planet in the solar system?",
    ]

    async with ParallelBatchProcessor[None, str, None](
        config=config,
        error_classifier=OpenAIErrorClassifier(),
    ) as processor:
        for i, q in enumerate(questions):
            await processor.add_work(LLMWorkItem(item_id=f"q{i}", strategy=strategy, prompt=q))
        result = await processor.process_all()

    for r in result.results:
        if r.success:
            print(f"{r.item_id}: {r.output}")
        else:
            print(f"{r.item_id} FAILED: {r.error}")

    print(
        f"\nTokens: input={result.total_input_tokens} "
        f"output={result.total_output_tokens} "
        f"cached={result.total_cached_tokens}"
    )
    # DeepSeek cache reads cost 10% of normal — use the matching rate.
    print(f"Billable input tokens: {result.effective_input_tokens(CachedTokenRates.DEEPSEEK)}")


if __name__ == "__main__":
    asyncio.run(main())
