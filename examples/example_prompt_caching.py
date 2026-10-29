"""Prompt-cache monitoring: per-group stats, the hit-rate floor, and a saved tracker.

A fake client stands in for a provider. Each prompt names its cache group, and the
client reports cached input tokens the way a provider's usage does. Halfway through,
the "classify" job's prefix changes and its cache stops hitting, which the
hit-rate floor catches.

Run without credentials or network access:

    uv run python examples/example_prompt_caching.py
"""

from __future__ import annotations

import asyncio
import json

from async_batch_llm import (
    CacheGroupTracker,
    CallableStrategy,
    CallOutcome,
    ErrorCategory,
    GuardrailConfig,
    LLMCallPool,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryState,
)

PREFIX_TOKENS = 2000  # the static part of each prompt
ITEM_TOKENS = 100  # the per-item part


class FakeCachingClient:
    """Reports usage like a provider with prefix caching."""

    def __init__(self, broken_after: int | None = None) -> None:
        self.calls = 0
        self.broken_after = broken_after

    async def __call__(
        self, prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        self.calls += 1
        group, _, item = prompt.partition(":")
        # The first call of a group writes the cache; later calls read it, until
        # the "classify" prefix changes and every call writes it again.
        broken = group == "classify" and self.broken_after is not None and int(item) >= self.broken_after
        first = int(item) == 0
        cached = 0 if first or broken else PREFIX_TOKENS
        usage = {
            "input_tokens": PREFIX_TOKENS + ITEM_TOKENS,
            "output_tokens": 20,
            "total_tokens": PREFIX_TOKENS + ITEM_TOKENS + 20,
            "cached_input_tokens": cached,
            "cache_write_input_tokens": PREFIX_TOKENS if cached == 0 else 0,
        }
        # A built-in model stamps cache_group itself; a custom client sets it here.
        metadata = {"cache_group": group}
        if broken:
            metadata["cache_miss_reason"] = "input_changed"
        return CallOutcome(output=f"answer for {prompt}", token_usage=usage, metadata=metadata)


def _print_groups(stats: dict) -> None:
    for entry in stats["cache_groups"]:
        window = entry["window_hit_rate"]
        print(
            f"  {entry['group']!s:10} calls={entry['measured_calls']:3} "
            f"hit_rate={entry['hit_rate']:.0%} "
            f"window={'-' if window is None else f'{window:.0%}'} "
            f"last_miss={entry['last_miss_reason']}"
        )


async def processor_with_a_floor() -> None:
    print("1. A processor run with a 70% floor over each group's last 5 calls")
    config = ProcessorConfig(
        max_workers=1,  # one worker keeps the demo's order deterministic
        guardrails=GuardrailConfig(min_cache_hit_rate=0.7, cache_hit_window=5),
    )
    strategy = CallableStrategy(FakeCachingClient(broken_after=8))
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        for i in range(20):
            for group in ("summarize", "classify"):
                await processor.add_work(
                    LLMWorkItem(item_id=f"{group}-{i}", strategy=strategy, prompt=f"{group}:{i}")
                )
        result = await processor.process_all()
        stats = await processor.get_stats()
    termination = result.termination
    print(f"  stopped: {termination.kind}, category {termination.error_category}")
    assert termination.error_category == ErrorCategory.CACHE_HIT_RATE_BELOW_FLOOR
    print(f"  reason: {termination.reason}")
    print(f"  {result.succeeded} succeeded before the stop; the rest ended batch_aborted")
    _print_groups(stats)


async def pool_with_a_saved_tracker() -> str:
    print("\n2. A call pool whose statistics survive a restart")
    config = ProcessorConfig(max_workers=4, guardrails=GuardrailConfig(cache_hit_window=5))
    strategy = CallableStrategy(FakeCachingClient())
    async with LLMCallPool(strategy, config=config) as pool:
        for i in range(6):
            await pool.submit_result(f"summarize:{i}")
        saved = json.dumps(pool.cache_tracker.to_dict())
    print(f"  saved {len(saved)} bytes of tracker state")

    # A later process: continue from the saved state.
    restored = CacheGroupTracker.from_dict(json.loads(saved))
    async with LLMCallPool(strategy, config=config, cache_tracker=restored) as pool:
        for i in range(6, 9):
            await pool.submit_result(f"summarize:{i}")
        _print_groups(pool.cache_stats())
        breaches = pool.cache_tracker.below_floor(0.7)
        print(f"  below a 70% floor: {[b.message for b in breaches] or 'none'}")
    return saved


async def rebuild_from_results() -> None:
    print("\n3. Rebuilding statistics from saved results (no tracker state)")
    config = ProcessorConfig(max_workers=1)
    strategy = CallableStrategy(FakeCachingClient(broken_after=2))
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        for i in range(6):
            await processor.add_work(
                LLMWorkItem(item_id=f"classify-{i}", strategy=strategy, prompt=f"classify:{i}")
            )
        result = await processor.process_all()

    tracker = CacheGroupTracker(window=3)
    for item in result.results:  # in a real job, rows loaded from your own storage
        tracker.record_result(item)
    _print_groups(tracker.stats())
    for breach in tracker.below_floor(0.7):
        print(f"  {breach.message}")


async def main() -> None:
    await processor_with_a_floor()
    await pool_with_a_saved_tracker()
    await rebuild_from_results()


if __name__ == "__main__":
    asyncio.run(main())
