"""Example: an in-process shared call pool for the request path.

LLMCallPool is a long-lived object you create once (e.g. at app startup) and call
`submit()` on from any number of concurrent handlers. A semaphore caps global
concurrency against the provider, and a single shared rate-limit coordinator
gives every caller one coordinated cooldown: when one request hits a 429, all
callers briefly pause, then slow-start back up — instead of a thundering herd
each retrying independently.

There is no queue, no worker pool, and no background dispatcher. Each caller's
coroutine runs the request itself under the semaphore, so a cancelled caller (a
disconnected client) simply frees its slot.

Runs against MockAgent, so no API key is needed:

    python examples/example_gateway.py
"""

import asyncio

from pydantic import BaseModel

from async_batch_llm import (
    LLMCallPool,
    ProcessorConfig,
    PydanticAIStrategy,
    RateLimitConfig,
    RetryConfig,
)
from async_batch_llm.testing import MockAgent


class Reply(BaseModel):
    """Structured output for the demo."""

    text: str


def build_pool() -> LLMCallPool[Reply]:
    """Construct the shared call pool once, at startup.

    max_workers is the global concurrency budget against the provider. The mock
    raises a single 429 (on the 3rd call) so the shared cooldown is visible; a
    short cooldown keeps the demo fast.
    """
    agent = MockAgent(
        response_factory=lambda prompt: Reply(text=f"answer to: {prompt}"),
        latency=0.05,
        rate_limit_on_call=3,
    )
    strategy = PydanticAIStrategy(agent=agent)
    config = ProcessorConfig(
        max_workers=4,
        rate_limit=RateLimitConfig(cooldown_seconds=0.1),
        retry=RetryConfig(max_attempts=3, max_rate_limit_retries=5),
    )
    return LLMCallPool(strategy, config=config)


async def handle_request(pool: LLMCallPool[Reply], n: int) -> None:
    """Simulate one incoming request handler calling the shared pool."""
    reply = await pool.submit(f"request #{n}")
    print(f"  request #{n} -> {reply.text!r}")


async def main() -> None:
    # Manage the pool's lifetime with `async with` (calls aclose() on exit,
    # which runs the strategy's cleanup()). In a web app this maps to a lifespan
    # handler that stashes the pool on app state.
    async with build_pool() as pool:
        print("Dispatching 8 concurrent requests through the shared call pool...")
        await asyncio.gather(*(handle_request(pool, n) for n in range(8)))

        # submit_result() returns the full WorkItemResult instead of raising, so you
        # can read token usage or branch on failure.
        result = await pool.submit_result("one more, please")
        print(
            f"\nsubmit_result() -> success={result.success}, "
            f"tokens={result.token_usage.get('total_tokens')}"
        )

    # After the context exits the pool is closed and rejects further work.
    try:
        await pool.submit("too late")
    except RuntimeError as exc:
        print(f"submit after close -> {exc}")

    await demo_load_shedding()


async def demo_load_shedding() -> None:
    """The two opt-in knobs that bound the request path under load.

    `max_pending` caps in-flight requests (running + waiting) and rejects
    over-cap submits instantly; `submit_timeout` bounds per-caller latency. Both
    return a failed WorkItemResult rather than raising (or raise via submit()).
    """
    print("\nLoad-shedding knobs:")
    slow = MockAgent(response_factory=lambda p: Reply(text=f"answer to: {p}"), latency=0.3)
    strategy = PydanticAIStrategy(agent=slow)

    # Admission cap: max_workers=1 + max_pending=0 → at most 1 in flight.
    async with LLMCallPool(strategy, config=ProcessorConfig(max_workers=1), max_pending=0) as pool:
        held = asyncio.create_task(pool.submit_result("holds the slot"))
        await asyncio.sleep(0.05)  # let it become in-flight
        rejected = await pool.submit_result("over the cap")
        print(f"  over-cap submit -> success={rejected.success}, error={rejected.error!r}")
        await held

    # Latency budget: the call takes ~0.3s but the budget is 0.05s.
    async with LLMCallPool(
        strategy, config=ProcessorConfig(max_workers=2), submit_timeout=0.05
    ) as pool:
        timed_out = await pool.submit_result("too slow for the budget")
        print(f"  timed-out submit -> success={timed_out.success}, error={timed_out.error!r}")


if __name__ == "__main__":
    asyncio.run(main())
