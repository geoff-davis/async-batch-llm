"""Isolate the gzip write path: stdlib gzip (sync) vs aiogzip queue (async).

No API calls and no LLM latency — this answers, cleanly, "is the aiogzip
single-consumer queue writer slower than a plain blocking gzip write for small
records, and where (if anywhere) does it cross over to faster?"

Lines are pre-serialized so we time *only* compression + write + async
machinery, not JSON encoding. Each case is run several times; we report medians.

Run (aiogzip pulled in just for this run, no permanent install):

    uv run --with aiogzip python examples/bench_gzip_write.py
"""

import asyncio
import gzip
import json
import statistics
import tempfile
import time
from pathlib import Path


class StreamingGzipWriter:
    """Same shape as the demo's writer, but takes pre-serialized lines."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task = None

    async def __aenter__(self) -> "StreamingGzipWriter":
        self._task = asyncio.create_task(self._consume())
        return self

    async def _consume(self) -> None:
        from aiogzip import AsyncGzipFile

        async with AsyncGzipFile(str(self.path), "wt") as out:
            while True:
                line = await self._queue.get()
                if line is None:
                    break
                await out.write(line)

    async def write(self, line: str) -> None:
        await self._queue.put(line)

    async def __aexit__(self, *exc) -> None:
        await self._queue.put(None)
        if self._task is not None:
            await self._task


def make_lines(count: int, size: int) -> list[str]:
    pad = "x" * size
    return [json.dumps({"i": i, "d": pad}) + "\n" for i in range(count)]


def sync_write(path: Path, lines: list[str]) -> float:
    start = time.perf_counter()
    with gzip.open(path, "wt") as f:
        for line in lines:
            f.write(line)
    return time.perf_counter() - start


async def async_write(path: Path, lines: list[str]) -> float:
    start = time.perf_counter()
    async with StreamingGzipWriter(path) as w:
        for line in lines:
            await w.write(line)
    return time.perf_counter() - start


async def main() -> None:
    # (record count, payload bytes per record)
    cases = [
        (10_000, 50),
        (10_000, 1_024),
        (2_000, 65_536),
        (500, 524_288),  # 512 KiB — past aiogzip's 256 KiB offload threshold
    ]
    trials = 5

    print(f"{'records':>9}{'rec bytes':>11}{'sync ms':>10}{'async ms':>11}{'async/sync':>12}")
    print("-" * 53)
    for count, size in cases:
        lines = make_lines(count, size)
        with tempfile.TemporaryDirectory() as d:
            sync_runs = [sync_write(Path(d) / "s.gz", lines) * 1000 for _ in range(trials)]
            async_runs = [await async_write(Path(d) / "a.gz", lines) * 1000 for _ in range(trials)]
        sync_ms = statistics.median(sync_runs)
        async_ms = statistics.median(async_runs)
        ratio = async_ms / sync_ms if sync_ms else 0.0
        print(f"{count:>9}{size:>11}{sync_ms:>10.1f}{async_ms:>11.1f}{ratio:>11.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
