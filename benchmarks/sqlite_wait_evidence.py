"""Opt-in SQLite waiting comparison; alternatives never change production defaults."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import async_batch_llm.sqlite_artifacts as storage
from async_batch_llm import (
    ArtifactIdentity,
    CallableStrategy,
    CallOutcome,
    LLMWorkItem,
    ResumePolicy,
    SqliteArtifactStore,
    WorkItemResult,
)

ORIGINAL = storage._await_without_cancelling


async def callback_wait(future: asyncio.Future[Any]) -> Any:
    if future.done():
        return future.result()
    waiter = asyncio.get_running_loop().create_future()

    def completed(_: asyncio.Future[Any]) -> None:
        if not waiter.done():
            waiter.set_result(None)

    future.add_done_callback(completed)
    try:
        await waiter
    finally:
        future.remove_done_callback(completed)
    return future.result()


async def shield_wait(future: asyncio.Future[Any]) -> Any:
    return await asyncio.shield(future)


async def wait_wait(future: asyncio.Future[Any]) -> Any:
    await asyncio.wait([future])
    return future.result()


VARIANTS = {
    "current": ORIGINAL,
    "callback": callback_wait,
    "shield": shield_wait,
    "wait": wait_wait,
}


async def races(variant: str, count: int) -> dict[str, Any]:
    """Native futures: completed, queued completion, executor completion, cancellation."""
    wait = VARIANTS[variant]
    loop = asyncio.get_running_loop()
    diagnostics = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: diagnostics.append(context["message"]))
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i in range(count):
            future = loop.create_future()
            if i % 3 == 0:
                future.set_result(i)
            elif i % 3 == 1:
                loop.call_soon(future.set_result, i)
            else:
                future = loop.run_in_executor(executor, lambda value=i: value)
            assert await asyncio.wait_for(wait(future), 2) == i
        for fail in (False, True):
            future = loop.create_future()
            for _ in range(3):
                caller = asyncio.create_task(wait(future))
                await asyncio.sleep(0)
                caller.cancel()
                try:
                    await caller
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("caller cancellation lost")
                assert not future.done()
            error = ValueError("owned failure")
            if fail:
                future.set_exception(error)
                try:
                    await wait(future)
                except ValueError as exc:
                    assert exc is error
                else:
                    raise AssertionError("owned failure lost")
            else:
                future.set_result(42)
                assert await wait(future) == 42
    await asyncio.sleep(0)
    loop.set_exception_handler(previous_handler)
    return {
        "variant": variant,
        "completions": count,
        "repeated_cancellation_cases": 2,
        "loop_diagnostics": diagnostics,
    }


async def lookup_trial(variant: str, count: int) -> dict[str, Any]:
    storage._await_without_cancelling = VARIANTS[variant]
    loop = asyncio.get_running_loop()
    original_call_later = loop.call_later
    timers = 0
    heartbeat_lag: list[float] = []
    running = True

    def call_later(delay, callback, *args, **kwargs):
        nonlocal timers
        if delay == storage._FUTURE_WAKEUP_BACKUP_SECONDS:
            timers += 1
        return original_call_later(delay, callback, *args, **kwargs)

    async def heartbeat():
        while running:
            started = loop.time()
            await asyncio.sleep(0.001)
            heartbeat_lag.append(max(0, loop.time() - started - 0.001))

    with tempfile.TemporaryDirectory(prefix="abl-e-wait-") as directory:

        async def invoke(prompt):
            return CallOutcome(output=prompt)

        strategy = CallableStrategy(invoke)
        store = SqliteArtifactStore(
            Path(directory) / "run.sqlite",
            identity=ArtifactIdentity(provider="evidence", model="fake"),
        )
        item = LLMWorkItem("one", strategy, "prompt")
        try:
            key = await store.prepare_item(item)
            await store.append(item, key, WorkItemResult(item_id="one", success=True, output="ok"))
            for _ in range(20):
                await store.lookup(item, key, ResumePolicy.REUSE_SUCCESSES)
            loop.call_later = call_later
            monitor = asyncio.create_task(heartbeat())
            started = time.perf_counter()
            for _ in range(count):
                result = await store.lookup(item, key, ResumePolicy.REUSE_SUCCESSES)
                assert result is not None and result.output == "ok"
            elapsed = time.perf_counter() - started
            running = False
            await monitor
        finally:
            loop.call_later = original_call_later
            await store.close()
            storage._await_without_cancelling = ORIGINAL
    return {
        "variant": variant,
        "lookups": count,
        "seconds": elapsed,
        "backup_timers": timers,
        "heartbeat_samples": len(heartbeat_lag),
        "heartbeat_max_lag_seconds": max(heartbeat_lag, default=0),
    }


async def main(args):
    results = {
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "races": [],
        "lookups": [],
    }
    variants = list(VARIANTS)
    for variant in variants:
        results["races"].append(await races(variant, args.count))
    for trial in range(args.trials):
        for variant in variants[trial % 4 :] + variants[: trial % 4]:
            row = await lookup_trial(variant, args.count)
            row["trial"] = trial
            results["lookups"].append(row)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
