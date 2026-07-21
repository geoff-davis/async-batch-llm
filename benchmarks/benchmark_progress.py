"""Opt-in microbenchmark for progress dispatch overhead.

Run with::

    uv run python benchmarks/benchmark_progress.py --items 10000

The output is descriptive JSON; this script intentionally has no pass/fail
threshold because scheduler and terminal performance vary by machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import async_batch_llm.streaming as streaming
from async_batch_llm import LLMCallStrategy, process_prompts


class _ImmediateStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        return prompt, {}, None


class _NoopBar:
    def __init__(self, *, total: int, **_: Any) -> None:
        self.total = total
        self.n = 0

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        pass


async def _timed(run: Callable[[], Awaitable[object]]) -> float:
    started = time.perf_counter()
    await run()
    return time.perf_counter() - started


async def benchmark(items: int) -> dict[str, float | int]:
    prompts = [str(index) for index in range(items)]
    strategy = _ImmediateStrategy()

    disabled = await _timed(lambda: process_prompts(strategy, prompts, concurrency=50))

    reporter_class = streaming._ProgressReporter
    streaming._ProgressReporter = lambda interval: reporter_class(  # type: ignore[misc,assignment]
        interval,
        bar_factory=_NoopBar,
    )
    try:
        bundled = await _timed(
            lambda: process_prompts(strategy, prompts, concurrency=50, progress=True)
        )
    finally:
        streaming._ProgressReporter = reporter_class

    user_calls = 0

    def user_callback(*_: object) -> None:
        nonlocal user_calls
        user_calls += 1

    custom = await _timed(
        lambda: process_prompts(strategy, prompts, concurrency=50, progress=user_callback)
    )
    return {
        "items": items,
        "progress_disabled_seconds": disabled,
        "bundled_progress_seconds": bundled,
        "per_item_user_callback_seconds": custom,
        "user_callback_calls": user_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(benchmark(args.items)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
