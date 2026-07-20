"""Example: one resilient LLM call without the batch machinery.

`call()` / `call_result()` run a single prompt through the same resilience pipeline
the batch processor uses — error-type-aware retries, the coordinated rate-limit
cooldown, and token accounting — but with no queue, workers, or result stream.

Reach for these when you have *one* prompt (or a few unrelated ones) and want the
operational layer without standing up a ParallelBatchProcessor. For many calls
that should share a cooldown or a prepared/cached strategy, use LLMCallPool or
process_prompts instead.

Runs against MockAgent, so no API key is needed:

    python examples/example_single_call.py
"""

import asyncio

from pydantic import BaseModel

from async_batch_llm import (
    LLMCallError,
    ProcessorConfig,
    PydanticAIStrategy,
    RateLimitConfig,
    RetryConfig,
    call,
    call_result,
)
from async_batch_llm.testing import MockAgent


class Summary(BaseModel):
    """Structured output for the demo."""

    headline: str
    sentiment: str


def _strategy(**mock_kwargs) -> PydanticAIStrategy:
    """A PydanticAI strategy backed by MockAgent (stands in for a real model)."""
    agent = MockAgent(
        response_factory=lambda prompt: Summary(
            headline=f"Re: {prompt[:40]}", sentiment="positive"
        ),
        latency=0.02,
        **mock_kwargs,
    )
    return PydanticAIStrategy(agent=agent)


async def main() -> None:
    # 1. The happy path: call() returns the output directly, or raises.
    summary = await call(_strategy(), "Quarterly numbers are up across every region.")
    print(f"call() -> {summary.headline!r} ({summary.sentiment})")

    # 2. call_result() returns the full WorkItemResult — useful when you want token
    #    accounting or want to branch on failure without exception handling.
    result = await call_result(_strategy(), "Summarize the all-hands recap.")
    print(
        f"call_result() -> success={result.success}, "
        f"tokens={result.token_usage.get('total_tokens')}"
    )

    # 3. Resilience still applies to a single call. This mock raises a 429 on the
    #    first attempt; the coordinated cooldown waits it out and the retry wins.
    #    (cooldown_seconds is short here so the demo runs fast — the default is
    #    300s, sized for real providers.)
    cfg = ProcessorConfig(
        rate_limit=RateLimitConfig(cooldown_seconds=0.1),
        retry=RetryConfig(max_attempts=3, max_rate_limit_retries=5),
    )
    recovered = await call(_strategy(rate_limit_on_call=1), "Try me once.", config=cfg)
    print(f"recovered after a 429 -> {recovered.headline!r}")

    # 4. When a call fails for good, call_result() returns the failed
    #    WorkItemResult — inspect .error / .exception / .token_usage without
    #    exception handling.
    failed = await call_result(
        _strategy(failure_rate=1.0),
        "This one always fails.",
        config=ProcessorConfig(retry=RetryConfig(max_attempts=2)),
    )
    print(
        f"call_result() on failure -> success={failed.success}, "
        f"error={failed.error!r}, exception={type(failed.exception).__name__}"
    )

    # call() instead re-raises the provider's own exception, preserving its type.
    # (LLMCallError is the fallback only when a failed result has no preserved
    # exception — e.g. the pool's max_pending / submit_timeout rejections.)
    try:
        await call(
            _strategy(failure_rate=1.0),
            "This one always fails.",
            config=ProcessorConfig(retry=RetryConfig(max_attempts=2)),
        )
    except LLMCallError as exc:
        print(f"call() raised LLMCallError -> {exc.result.error}")
    except Exception as exc:
        print(f"call() re-raised the provider exception -> {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
