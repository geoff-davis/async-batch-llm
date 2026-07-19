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
        FakeBar.instances.append(self)

    def refresh(self):
        self.refreshes += 1

    def close(self):
        self.closed = True


@pytest.fixture
def fake_tqdm(monkeypatch):
    """Install a fake tqdm.auto module so the bar path runs deterministically."""
    import sys
    import types

    FakeBar.instances = []
    auto = types.ModuleType("tqdm.auto")
    auto.tqdm = FakeBar
    tqdm_pkg = types.ModuleType("tqdm")
    tqdm_pkg.auto = auto
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_pkg)
    monkeypatch.setitem(sys.modules, "tqdm.auto", auto)
    return FakeBar


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
        assert calls[-1][0] == 2

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
