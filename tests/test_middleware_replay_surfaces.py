"""Replay through the high-level APIs with middleware registered.

`tests/test_middleware_artifacts.py` drives `ParallelBatchProcessor`
directly. These cover the composition users actually write:
`process_prompts` / `process_stream` with an artifact store, a resume
policy, and middleware — where preprocessing must still run once per item
on a replay-only run while no provider call is made.
"""

from __future__ import annotations

from typing import Any

import pytest

from async_batch_llm import (
    ArtifactIOError,
    JsonlArtifactStore,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    ResumePolicy,
    SqliteArtifactStore,
    process_prompts,
    process_stream,
)
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.middleware import BaseMiddleware

pytestmark = pytest.mark.asyncio

PROMPTS = ["alpha", "beta", "gamma"]
_TOKENS = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}


class CountingStrategy(LLMCallStrategy[str]):
    """Counts provider calls and pool-resize requests."""

    max_concurrency = 8

    def __init__(self) -> None:
        self.calls = 0
        self.resizes = 0

    async def request_concurrency(self, concurrency: int) -> None:
        self.resizes += 1

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: Any = None
    ) -> tuple[str, dict[str, int]]:
        del attempt, timeout, state
        self.calls += 1
        return f"out:{prompt}", _TOKENS


class Upper(BaseMiddleware):
    """Transforms the request and marks the result."""

    def __init__(self) -> None:
        self.before = 0
        self.after = 0

    async def before_process(self, work_item):  # type: ignore[no-untyped-def]
        self.before += 1
        work_item.prompt = work_item.prompt.upper()
        return work_item

    async def after_process(self, result):  # type: ignore[no-untyped-def]
        self.after += 1
        result.output += "!"
        return result


@pytest.fixture(params=[JsonlArtifactStore, SqliteArtifactStore], ids=["jsonl", "sqlite"])
def store_factory(request, tmp_path):
    def create():
        return request.param(tmp_path / "artifacts")

    return create


async def _run(surface: str, strategy, middleware, store):  # type: ignore[no-untyped-def]
    kwargs = {
        "artifact_store": store,
        "resume": ResumePolicy.REUSE_SUCCESSES,
        "middlewares": [middleware],
        "config": ProcessorConfig(max_workers=2, concurrency=2, attempt_timeout=5.0),
    }
    if surface == "prompts":
        return (await process_prompts(strategy, PROMPTS, **kwargs)).results
    return [result async for result in process_stream(strategy, PROMPTS, **kwargs)]


@pytest.mark.parametrize("surface", ["prompts", "stream"])
async def test_high_level_replay_preprocesses_without_provider_calls(surface, store_factory):
    """A replay-only second run still preprocesses every item, calls no
    provider, and does not rerun ``after_process`` over stored output."""
    first_strategy, first_middleware = CountingStrategy(), Upper()
    first = await _run(surface, first_strategy, first_middleware, store_factory())

    assert len(first) == len(PROMPTS)
    assert all(result.success for result in first)
    assert first_strategy.calls == len(PROMPTS)
    assert first_middleware.before == len(PROMPTS)
    assert first_middleware.after == len(PROMPTS)
    # The effective (transformed) request reached the provider.
    assert sorted(result.output for result in first) == [
        f"out:{prompt.upper()}!" for prompt in sorted(PROMPTS)
    ]

    second_strategy, second_middleware = CountingStrategy(), Upper()
    second = await _run(surface, second_strategy, second_middleware, store_factory())

    assert all(result.replayed_from_artifact for result in second)
    assert second_strategy.calls == 0, "replay must not reach the provider"
    assert second_middleware.before == len(PROMPTS), "preprocessing runs on replayed items"
    assert second_middleware.after == 0, "after_process must not rerun on replay"
    # Stored output already carries the first run's after_process marker.
    assert sorted(result.output for result in second) == sorted(result.output for result in first)


async def test_lookup_failure_propagates_and_reaches_no_provider(store_factory):
    """A store read failure is not an audit outcome: it terminates the batch
    rather than falling through to execution."""

    class UnreadableStore:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def lookup(self, work_item, key, policy):
            raise ArtifactIOError("index unreadable")

    strategy = CountingStrategy()
    processor = ParallelBatchProcessor(
        config=ProcessorConfig(max_workers=1, attempt_timeout=5.0),
        artifact_store=UnreadableStore(store_factory()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    with pytest.raises(ArtifactIOError):
        async with processor:
            await processor.add_work(LLMWorkItem("only", strategy, "prompt"))
            await processor.process_all()
    assert strategy.calls == 0


async def test_artifact_preparation_error_writes_no_record(store_factory, tmp_path):
    """An item whose identity cannot be prepared fails on its own and leaves
    nothing behind for a later resume to find."""
    store = store_factory()
    processor = ParallelBatchProcessor(
        config=ProcessorConfig(max_workers=1, attempt_timeout=5.0),
        artifact_store=store,
        resume=ResumePolicy.REUSE_SUCCESSES,
    )

    class NoIdentity(CountingStrategy):
        # Declaring no identity is the documented "cannot infer" signal.
        artifact_identity = None

    async with processor:
        await processor.add_work(LLMWorkItem("only", NoIdentity(), "prompt"))
        batch = await processor.process_all()

    assert batch.results[0].error_category == "artifact_preparation_error"
    assert not (tmp_path / "artifacts").exists(), (
        "a failed preparation must not create or checkpoint into an artifact"
    )


@pytest.mark.parametrize("surface", ["prompts", "stream"])
async def test_replay_only_run_still_configures_the_effective_strategy(surface, store_factory):
    """Capacity/pool configuration is bound to the effective strategy even
    when every item replays and nothing is executed."""
    await _run(surface, CountingStrategy(), Upper(), store_factory())

    replaying = CountingStrategy()
    results = await _run(surface, replaying, Upper(), store_factory())

    assert all(result.replayed_from_artifact for result in results)
    assert replaying.calls == 0
    assert replaying.resizes == 1, "the effective strategy was never configured"
