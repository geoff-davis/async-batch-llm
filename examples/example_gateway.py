"""Example: a shared, rate-limited gateway for the request path.

LLMGateway is a long-lived object you create once (e.g. at app startup) and call
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
    LLMGateway,
    ProcessorConfig,
    PydanticAIStrategy,
    RateLimitConfig,
    RetryConfig,
)
from async_batch_llm.testing import MockAgent


class Reply(BaseModel):
    """Structured output for the demo."""

    text: str


def build_gateway() -> LLMGateway[Reply]:
    """Construct the shared gateway once, at startup.

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
    return LLMGateway(strategy, config=config)


async def handle_request(gateway: LLMGateway[Reply], n: int) -> None:
    """Simulate one incoming request handler calling the shared gateway."""
    reply = await gateway.submit(f"request #{n}")
    print(f"  request #{n} -> {reply.text!r}")


async def main() -> None:
    # Manage the gateway's lifetime with `async with` (calls aclose() on exit,
    # which runs the strategy's cleanup()). In a web app this maps to a lifespan
    # handler that stashes the gateway on app state.
    async with build_gateway() as gateway:
        print("Dispatching 8 concurrent requests through the gateway...")
        await asyncio.gather(*(handle_request(gateway, n) for n in range(8)))

        # try_submit() returns the full WorkItemResult instead of raising, so you
        # can read token usage or branch on failure.
        result = await gateway.try_submit("one more, please")
        print(
            f"\ntry_submit() -> success={result.success}, "
            f"tokens={result.token_usage.get('total_tokens')}"
        )

    # After the context exits the gateway is closed and rejects further work.
    try:
        await gateway.submit("too late")
    except RuntimeError as exc:
        print(f"submit after close -> {exc}")


if __name__ == "__main__":
    asyncio.run(main())
