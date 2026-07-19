"""Tests for progress=True on process_prompts/process_stream (issue #100)."""

import logging

import pytest

import async_batch_llm.streaming as streaming_module
from async_batch_llm import LLMCallStrategy, process_prompts, process_stream
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
        assert any("falling back to interval logging" in r.message for r in caplog.records)
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

    async def test_conflict_with_progress_callback_raises(self):
        def cb(completed, total, item_id):
            pass

        with pytest.raises(ValueError, match="not both"):
            await process_prompts(EchoStrategy(), ["a"], progress=True, progress_callback=cb)


class TestReporterUnit:
    def test_fallback_logs_every_ten_and_on_completion(self, monkeypatch, caplog):
        reporter = _ProgressReporter()
        reporter._tqdm_failed = True
        with caplog.at_level(logging.INFO, logger=streaming_module.__name__):
            for i in range(1, 26):
                reporter(i, 25, f"item_{i}")
        messages = [r.message for r in caplog.records]
        assert "progress: 10/25 items completed" in messages
        assert "progress: 20/25 items completed" in messages
        assert "progress: 25/25 items completed" in messages
        # No per-item spam.
        assert len(messages) == 3

    def test_bar_total_follows_growth(self, fake_tqdm):
        reporter = _ProgressReporter()
        reporter(1, 2, "a")
        reporter(2, 4, "b")
        bar = fake_tqdm.instances[0]
        assert bar.total == 4
        assert bar.n == 2
        reporter.close()
        assert bar.closed

    def test_concurrent_threads_create_exactly_one_bar(self, fake_tqdm):
        # Sync callbacks run via asyncio.to_thread, so the reporter is hit
        # from many threads at once; the lazy bar creation must not race.
        from concurrent.futures import ThreadPoolExecutor

        reporter = _ProgressReporter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: reporter(i, 100, f"item_{i}"), range(1, 101)))
        assert len(fake_tqdm.instances) == 1
        assert fake_tqdm.instances[0].n == 100

    def test_out_of_order_updates_stay_monotonic(self, fake_tqdm):
        reporter = _ProgressReporter()
        reporter(3, 3, "c")
        reporter(2, 3, "b")  # late arrival must not move the bar backwards
        assert fake_tqdm.instances[0].n == 3

    def test_late_callback_after_close_creates_no_new_bar(self, fake_tqdm):
        # A cancelled asyncio.to_thread callback doesn't stop its thread, so
        # an update can land after close(); it must be dropped, not re-open
        # a fresh bar.
        reporter = _ProgressReporter()
        reporter(1, 2, "a")
        reporter.close()
        reporter(2, 2, "b")
        assert len(fake_tqdm.instances) == 1
        assert fake_tqdm.instances[0].closed
