"""Worker-loop overhead benchmark (no network).

Isolates *framework* per-item overhead from API latency: the strategy just
``asyncio.sleep(~5ms)`` and returns. With ``max_workers`` workers and a 5ms
"call", the theoretical ceiling is ``workers / latency`` items/sec; the gap
between that ceiling and the measured rate is the worker-loop overhead this
benchmark exists to track.

Defaults: 10,000 items, 100 workers, 5ms latency → ceiling ≈ 20,000 items/sec.

Run:
    uv run python examples/benchmark_worker_overhead.py

Two sections:

  A. Latency-bound (5ms latency, 100 workers) — realistic throughput, and the
     concurrent_post_processing win when a post-processor is slow.
  B. Overhead-bound micro-benchmark (~0 latency) — isolates the *per-item
     framework CPU* (logging, event emit, lock churn) that items #1–#3 target.
     This is the run where the demote-to-debug, no-observer-payload, fast
     built-in emit, and single-lock changes actually move the needle; the
     5ms section is dominated by the sleep and hides them.
"""

import asyncio
import logging
import time

from async_batch_llm import (
    LLMWorkItem,
    MetricsObserver,
    ParallelBatchProcessor,
    ProcessorConfig,
)
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy

LATENCY_S = 0.005
NUM_ITEMS = 10_000
NUM_WORKERS = 100

# Only exercise the concurrent-post-processing scenario when this build has it
# (lets the same script run against an older "before" tree for A/B comparison).
_SUPPORTS_CONCURRENT = "concurrent_post_processing" in ProcessorConfig.__dataclass_fields__


class SleepStrategy(LLMCallStrategy[str]):
    """Mock strategy: fixed latency, no network, trivial token payload."""

    def __init__(self, latency: float = LATENCY_S) -> None:
        self.latency = latency

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        if self.latency:
            await asyncio.sleep(self.latency)
        return "ok", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None


async def run_scenario(
    name: str,
    *,
    workers: int = NUM_WORKERS,
    latency: float = LATENCY_S,
    observers=None,
    post_processor=None,
    concurrent: bool = False,
) -> float:
    config_kwargs = dict(
        max_workers=workers,
        timeout_per_item=30.0,
        # Raise the interval so the periodic INFO progress line doesn't fire
        # hundreds of times and skew the measurement.
        progress_interval=NUM_ITEMS + 1,
    )
    if _SUPPORTS_CONCURRENT:
        config_kwargs["concurrent_post_processing"] = concurrent
    config = ProcessorConfig(**config_kwargs)

    strategy = SleepStrategy(latency)
    async with ParallelBatchProcessor(
        config=config,
        observers=observers or [],
        post_processor=post_processor,
    ) as processor:
        for i in range(NUM_ITEMS):
            await processor.add_work(LLMWorkItem(item_id=str(i), strategy=strategy, prompt="x"))
        start = time.perf_counter()
        result = await processor.process_all()
        elapsed = time.perf_counter() - start

    items_per_sec = result.total_items / elapsed
    print(f"  {name:46s} {elapsed:7.3f}s  {items_per_sec:11,.0f} items/sec")
    return items_per_sec


async def _slow_post_processor(result) -> None:
    await asyncio.sleep(0.002)


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    print(
        f"A. latency-bound: items={NUM_ITEMS:,}  workers={NUM_WORKERS}  "
        f"latency={LATENCY_S * 1000:.0f}ms  ceiling ≈ {NUM_WORKERS / LATENCY_S:,.0f} items/sec"
    )
    await run_scenario("no observers, inline post-proc (default)")
    await run_scenario("MetricsObserver (fast built-in)", observers=[MetricsObserver()])
    await run_scenario("slow (~2ms) post-proc, inline", post_processor=_slow_post_processor)
    if _SUPPORTS_CONCURRENT:
        await run_scenario(
            "slow (~2ms) post-proc, concurrent",
            post_processor=_slow_post_processor,
            concurrent=True,
        )
    else:
        print("  (concurrent_post_processing not available in this build)")

    # Overhead-bound: ~0 latency, a single worker, so per-item framework CPU is
    # the whole cost. Best-of-3 to damp scheduler noise.
    print(f"\nB. overhead-bound micro: items={NUM_ITEMS:,}  workers=1  latency=0 (best of 3)")
    for name, with_observer in [("no observers", False), ("MetricsObserver", True)]:
        runs = [
            await run_scenario(
                f"{name} (run {r + 1})",
                workers=1,
                latency=0.0,
                observers=[MetricsObserver()] if with_observer else None,
            )
            for r in range(3)
        ]
        print(f"  -> best {name:40s} {max(runs):11,.0f} items/sec")


if __name__ == "__main__":
    asyncio.run(main())
