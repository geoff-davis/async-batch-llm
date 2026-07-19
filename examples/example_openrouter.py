"""Example of using async-batch-llm with OpenRouter.

OpenRouter (https://openrouter.ai) exposes a unified OpenAI-compatible API
that fronts many upstream providers — Anthropic, OpenAI, Google, DeepSeek,
Mistral, etc. This example uses the built-in ``OpenRouterModel`` and
``OpenRouterStrategy`` classes (added in v0.9.0) to:

1. Run a cross-provider batch (Anthropic + OpenAI in one go).
2. Demonstrate Anthropic prompt caching via OpenRouter, which is **opt-in**:
   you have to attach ``cache_control: {"type": "ephemeral"}`` markers to
   the message blocks you want cached.

## Installation

```bash
pip install 'async-batch-llm[openrouter]'
```

## Setup

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Get a key at https://openrouter.ai/keys.

## Caching matrix (per upstream provider)

- **OpenAI** — automatic for prompts >~1024 tokens.
- **Gemini (implicit)** — automatic.
- **DeepSeek** — automatic.
- **Anthropic** — opt-in via ``cache_control`` markers (shown below).
- **Gemini explicit ``CachedContent``** — not exposed via OpenRouter; use
  ``GeminiCachedModel`` if you need it.
"""

import asyncio
import os

from async_batch_llm import (
    LLMWorkItem,
    OpenRouterErrorClassifier,
    OpenRouterModel,
    OpenRouterStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
)


async def example_cross_provider_batch() -> None:
    """Run one batch where each item uses a different upstream provider."""
    print("\n=== Example 1: Cross-provider batch ===\n")

    config = ProcessorConfig(max_workers=2, attempt_timeout=60.0)

    api_key = os.environ["OPENROUTER_API_KEY"]
    anthropic_model = OpenRouterModel.from_api_key(
        "anthropic/claude-haiku-4-5",
        api_key=api_key,
        title="async-batch-llm example",
    )
    openai_model = OpenRouterModel.from_api_key(
        "openai/gpt-4o-mini",
        api_key=api_key,
        title="async-batch-llm example",
    )

    items = [
        ("anthropic", anthropic_model, "What is Anthropic best known for?"),
        ("openai", openai_model, "Name three programming languages."),
    ]

    async with ParallelBatchProcessor[None, str, None](
        config=config,
        error_classifier=OpenRouterErrorClassifier(),
    ) as processor:
        for item_id, model, prompt in items:
            strategy = OpenRouterStrategy(model)
            await processor.add_work(LLMWorkItem(item_id=item_id, strategy=strategy, prompt=prompt))
        result = await processor.process_all()

    for r in result.results:
        if r.success:
            print(f"[{r.item_id}] {r.output}")
        else:
            print(f"[{r.item_id}] FAILED: {r.error}")


async def example_anthropic_prompt_caching() -> None:
    """Demonstrate Anthropic prompt caching via OpenRouter — explicit opt-in.

    Anthropic caches per-message-block when the block carries
    ``cache_control: {"type": "ephemeral"}``. To send those structured
    blocks through the framework, write a small ``LLMCallStrategy`` that
    builds the list of message dicts inside ``execute()`` and hands them
    straight to ``model.generate(prompt=messages_list)`` — bypassing the
    ``LLMWorkItem.prompt: str`` constraint.

    The first call writes the cache; subsequent calls within the TTL
    (~5 minutes by default) hit it and ``cached_input_tokens`` reports the
    saving.
    """
    print("\n=== Example 2: Anthropic prompt caching via OpenRouter ===\n")

    # A long shared system block. Anthropic requires ~1024 tokens minimum
    # for caching to be worth it; we pad here purely for demonstration.
    long_system = "You are a helpful assistant specialized in concise factual answers. " + (
        "This is reference context that should be cached on the second call. " * 80
    )

    from async_batch_llm.base import LLMResponse, RetryState, TokenUsage
    from async_batch_llm.llm_strategies import LLMCallStrategy

    class CachedAnthropicStrategy(LLMCallStrategy[str]):
        """Build a system+user message list with cache_control on the system block."""

        def __init__(self, model: OpenRouterModel, system_text: str) -> None:
            self.model = model
            self.system_text = system_text

        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> tuple[str, TokenUsage]:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": prompt},
            ]
            response: LLMResponse = await self.model.generate(messages)
            return response.text, response.token_usage

    model = OpenRouterModel.from_api_key(
        "anthropic/claude-haiku-4-5",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    strategy = CachedAnthropicStrategy(model, system_text=long_system)
    config = ProcessorConfig(max_workers=1, attempt_timeout=60.0)

    # Run two questions back-to-back: the second should hit the cache.
    for round_idx, q in enumerate(["What's 2 + 2?", "What's the capital of Japan?"]):
        async with ParallelBatchProcessor[None, str, None](
            config=config,
            error_classifier=OpenRouterErrorClassifier(),
        ) as processor:
            await processor.add_work(
                LLMWorkItem(item_id=f"round_{round_idx}", strategy=strategy, prompt=q)
            )
            result = await processor.process_all()

        cached = result.total_cached_tokens
        total_in = result.total_input_tokens
        ratio = (cached / total_in * 100.0) if total_in else 0.0
        print(
            f"Round {round_idx + 1}: input={total_in} "
            f"cached={cached} ({ratio:.1f}% cache hit) "
            f"output={result.total_output_tokens}"
        )


async def main() -> None:
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Set OPENROUTER_API_KEY before running this example.")
        return
    await example_cross_provider_batch()
    await example_anthropic_prompt_caching()


if __name__ == "__main__":
    asyncio.run(main())
