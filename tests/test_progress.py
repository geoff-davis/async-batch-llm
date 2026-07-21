"""Tests for coalesced bundled progress and per-item user callbacks."""

import asyncio
import logging

import pytest

import async_batch_llm.streaming as streaming_module
from async_batch_llm import (
    AbortMode,
    GuardrailConfig,
    LLMCallStrategy,
    ProcessorConfig,
    RetryConfig,
    process_prompts,
    process_stream,
)
from async_batch_llm.streaming import _ProgressReporter


class EchoStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        tokens = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return prompt, tokens, None


class FakeBar:
    instances: list["FakeBar"] = []

    def __init__(self, *, total=None, unit=None, desc=None):
        self.total = total
        self.n = 0
        self.refreshes = 0
        self.closed = False
        type(self).instances.append(self)

    def refresh(self):
        self.refreshes += 1

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_tqdm(monkeypatch):
    """Install a fake tqdm.auto module so the bar path runs deterministically.

    A fresh Bar class per test keeps `instances` isolated, so a straggler
    callback thread from a previous test can't pollute this test's count.
    """
    import sys
    import types

    class Bar(FakeBar):
        instances: list[FakeBar] = []

    auto = types.ModuleType("tqdm.auto")
    auto.tqdm = Bar
    tqdm_pkg = types.ModuleType("tqdm")
    tqdm_pkg.auto = auto
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_pkg)
    monkeypatch.setitem(sys.modules, "tqdm.auto", auto)
    return Bar


class TestProgressTrue:
    async def test_tqdm_bar_updates_and_closes(self, fake_tqdm):
        batch = await process_prompts(EchoStrategy(), ["a", "b", "c"], progress=True)
        assert batch.succeeded == 3
        assert len(fake_tqdm.instances) == 1
        bar = fake_tqdm.instances[0]
        assert bar.n == 3
        assert bar.total == 3
        assert bar.closed

    async def test_streaming_progress(self, fake_tqdm):
        seen = []
        async for result in process_stream(EchoStrategy(), ["a", "b"], progress=True):
            seen.append(result)
        assert len(seen) == 2
        assert fake_tqdm.instances[0].closed

    async def test_logging_fallback_without_tqdm(self, monkeypatch, caplog):
        # Force the ImportError path regardless of the installed environment.
        import builtins

        real_import = builtins.__import__

        def no_tqdm(name, *args, **kwargs):
            if name.startswith("tqdm"):
                raise ImportError("no tqdm")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_tqdm)
        with caplog.at_level(logging.INFO, logger=streaming_module.__name__):
            batch = await process_prompts(EchoStrategy(), ["a", "b"], progress=True)
        assert batch.succeeded == 2
        assert any(
            "falling back to coalesced interval logging" in r.message for r in caplog.records
        )
        assert any("items completed" in r.message for r in caplog.records)

    async def test_progress_false_installs_nothing(self):
        batch = await process_prompts(EchoStrategy(), ["a"])
        assert batch.succeeded == 1


class TestCustomCallable:
    async def test_callable_receives_updates(self):
        calls = []

        def reporter(completed, total, item_id):
            calls.append((completed, total, item_id))

        batch = await process_prompts(EchoStrategy(), ["a", "b"], progress=reporter)
        assert batch.succeeded == 2
        assert len(calls) == 2
        # Callbacks run as detached tasks and may arrive out of order; every
        # completion count is delivered exactly once.
        assert sorted(c[0] for c in calls) == [1, 2]

    async def test_async_callable_receives_every_update(self):
        calls = []

        async def reporter(completed, total, item_id):
            calls.append((completed, total, item_id))

        batch = await process_prompts(EchoStrategy(), ["a", "b", "c"], progress=reporter)
        assert batch.succeeded == 3
        assert sorted(c[0] for c in calls) == [1, 2, 3]

    async def test_sync_callable_keeps_thread_dispatch(self, monkeypatch):
        real_to_thread = asyncio.to_thread
        thread_dispatches = 0

        async def tracking_to_thread(func, /, *args, **kwargs):
            nonlocal thread_dispatches
            thread_dispatches += 1
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", tracking_to_thread)
        await process_prompts(EchoStrategy(), ["a", "b", "c"], progress=lambda *_: None)
        assert thread_dispatches == 3

    async def test_bundled_reporter_avoids_thread_dispatch(self, monkeypatch, fake_tqdm):
        real_to_thread = asyncio.to_thread
        thread_dispatches = 0

        async def tracking_to_thread(func, /, *args, **kwargs):
            nonlocal thread_dispatches
            thread_dispatches += 1
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", tracking_to_thread)
        await process_prompts(EchoStrategy(), ["a", "b", "c"], progress=True)
        assert thread_dispatches == 0

    async def test_conflict_with_progress_callback_raises(self):
        def cb(completed, total, item_id):
            pass

        with pytest.raises(ValueError, match="not both"):
            await process_prompts(EchoStrategy(), ["a"], progress=True, progress_callback=cb)


class TestReporterUnit:
    async def test_first_update_coalesces_intermediate_and_forces_final(self):
        clock = FakeClock()
        bar = FakeBar(total=0)
        reporter = _ProgressReporter(
            0.1,
            clock=clock,
            bar_factory=lambda **_: bar,
        )

        await reporter(1, 4, "a")
        await reporter(2, 4, "b")
        await reporter(3, 4, "c")
        assert bar.refreshes == 1
        assert (bar.n, bar.total) == (1, 4)

        clock.advance(0.1)
        await reporter(3, 5, "c")
        assert bar.refreshes == 2
        assert (bar.n, bar.total) == (3, 5)

        await reporter.aclose(completed=4, total=5)
        assert bar.refreshes == 3
        assert (bar.n, bar.total) == (4, 5)
        assert bar.closed

    async def test_growing_total_does_not_treat_temporary_equality_as_final(self):
        clock = FakeClock()
        bar = FakeBar(total=0)
        reporter = _ProgressReporter(1.0, clock=clock, bar_factory=lambda **_: bar)

        await reporter(1, 1, "first")
        await reporter(1, 3, "producer-added-more")
        assert bar.refreshes == 1
        assert (bar.n, bar.total) == (1, 1)
        await reporter.aclose(completed=3, total=3)
        assert bar.refreshes == 2
        assert (bar.n, bar.total) == (3, 3)

    async def test_out_of_order_concurrent_updates_stay_monotonic(self):
        clock = FakeClock()
        bar = FakeBar(total=0)
        reporter = _ProgressReporter(0.1, clock=clock, bar_factory=lambda **_: bar)
        await asyncio.gather(
            reporter(3, 4, "c"),
            reporter(1, 2, "a"),
            reporter(2, 3, "b"),
        )
        clock.advance(0.1)
        await reporter(2, 2, "late")
        assert (bar.n, bar.total) == (3, 4)
        await reporter.aclose()

    async def test_close_is_idempotent_and_drops_late_updates(self):
        bar = FakeBar(total=0)
        reporter = _ProgressReporter(bar_factory=lambda **_: bar)
        await reporter(1, 2, "a")
        await reporter.aclose(completed=1, total=2)
        refreshes = bar.refreshes
        await reporter.aclose(completed=2, total=2)
        await reporter(2, 2, "b")
        assert bar.refreshes == refreshes
        assert bar.closed

    async def test_logging_fallback_is_time_coalesced(self):
        clock = FakeClock()
        messages: list[tuple[int, int]] = []
        reporter = _ProgressReporter(
            0.01,
            clock=clock,
            log_progress=lambda completed, total: messages.append((completed, total)),
        )
        reporter._tqdm_failed = True
        reporter._bar_factory_resolved = True

        for i in range(1, 1_001):
            await reporter(i, 1_000, "item")
        assert messages == [(1, 1_000)]
        clock.advance(1.0)
        await reporter(1_000, 1_000, "item")
        await reporter.aclose()
        assert messages == [(1, 1_000), (1_000, 1_000), (1_000, 1_000)]

    async def test_one_million_updates_have_bounded_render_count(self):
        clock = FakeClock()
        bar = FakeBar(total=0)
        reporter = _ProgressReporter(0.1, clock=clock, bar_factory=lambda **_: bar)
        for completed in range(1, 1_000_001):
            await reporter(completed, 1_000_000, "item")
        await reporter.aclose(completed=1_000_000, total=1_000_000)
        assert bar.refreshes == 2


class TestFinalization:
    async def test_failure_still_forces_exact_final_state(self, fake_tqdm):
        class FailingStrategy(EchoStrategy):
            async def execute(self, prompt, attempt, timeout, state=None):
                if prompt == "bad":
                    raise ValueError("bad item")
                return await super().execute(prompt, attempt, timeout, state)

        batch = await process_prompts(
            FailingStrategy(),
            ["good", "bad"],
            config=ProcessorConfig(retry=RetryConfig(max_attempts=1)),
            progress=True,
        )
        assert batch.total_items == 2
        bar = fake_tqdm.instances[0]
        assert (bar.n, bar.total) == (2, 2)
        assert bar.closed

    async def test_producer_error_closes_with_exact_accepted_count(self, fake_tqdm):
        async def source():
            yield "a"
            yield "b"
            raise RuntimeError("producer failed")

        with pytest.raises(RuntimeError, match="producer failed"):
            async for _ in process_stream(EchoStrategy(), source(), progress=True):
                pass
        bar = fake_tqdm.instances[0]
        assert (bar.n, bar.total) == (2, 2)
        assert bar.closed

    async def test_early_stream_exit_closes_reporter(self, fake_tqdm):
        stream = process_stream(EchoStrategy(), ["a", "b", "c"], progress=True)
        async for _ in stream:
            break
        await stream.aclose()
        assert len(fake_tqdm.instances) == 1
        assert fake_tqdm.instances[0].closed

    async def test_batch_deadline_forces_final_state(self, fake_tqdm):
        class SlowStrategy(EchoStrategy):
            async def execute(self, prompt, attempt, timeout, state=None):
                await asyncio.sleep(60)
                return await super().execute(prompt, attempt, timeout, state)

        batch = await process_prompts(
            SlowStrategy(),
            ["a", "b"],
            config=ProcessorConfig(
                concurrency=2,
                guardrails=GuardrailConfig(
                    batch_timeout=0.01,
                    abort_mode=AbortMode.CANCEL_ACTIVE,
                ),
            ),
            progress=True,
        )
        assert batch.termination.kind == "batch_timeout"
        bar = fake_tqdm.instances[0]
        assert (bar.n, bar.total) == (2, 2)
        assert bar.closed

    async def test_external_cancellation_closes_reporter(self, fake_tqdm):
        started = asyncio.Event()

        class BlockingStrategy(EchoStrategy):
            async def execute(self, prompt, attempt, timeout, state=None):
                started.set()
                await asyncio.Event().wait()
                return await super().execute(prompt, attempt, timeout, state)

        task = asyncio.create_task(
            process_prompts(BlockingStrategy(), ["a", "b"], concurrency=1, progress=True)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(fake_tqdm.instances) == 1
        assert fake_tqdm.instances[0].closed


@pytest.mark.parametrize("value", [True, False, 0, -1, float("inf"), float("nan"), "0.1"])
def test_progress_refresh_interval_validation(value):
    with pytest.raises(ValueError, match="progress_refresh_interval_seconds"):
        ProcessorConfig(progress_refresh_interval_seconds=value)
