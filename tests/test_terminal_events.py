"""One final-result event per published item, after middleware and persistence."""

import asyncio
from collections import Counter
from unittest.mock import AsyncMock, MagicMock

import pytest

from async_batch_llm import (
    ArtifactIOError,
    GuardrailConfig,
    JsonlArtifactStore,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    ResumePolicy,
    RetryConfig,
    SqliteArtifactStore,
    WorkItemResult,
)
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.middleware import BaseMiddleware
from async_batch_llm.observers import BaseObserver, MetricsObserver, ProcessingEvent

TERMINALS = {
    ProcessingEvent.ITEM_COMPLETED,
    ProcessingEvent.ITEM_FAILED,
    ProcessingEvent.ITEM_REPLAYED,
}


class Strategy(LLMCallStrategy):
    async def execute(self, prompt, attempt, timeout, state=None):
        if prompt in {"failure", "recover", "recover_failure"}:
            raise ValueError("permanent")
        if prompt == "retry":
            raise ConnectionError("temporary")
        if prompt in {"deadline", "timeout", "batch_timeout"}:
            await asyncio.sleep(10)
        return "ok", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class Middleware(BaseMiddleware):
    async def before_process(self, item):
        return None if item.prompt == "filtered" else item

    async def after_process(self, result):
        if result.item_id == "flip":
            return WorkItemResult(
                item_id=result.item_id,
                success=False,
                error="Changed: rejected",
                error_category="custom",
            )
        return result

    async def on_error(self, item, error):
        if item.prompt.startswith("recover"):
            return WorkItemResult(
                item_id=item.item_id,
                success=item.prompt == "recover",
                output="recovered",
                error="Recovery: failed",
                error_category="custom",
            )
        return None


class Recorder(BaseObserver):
    def __init__(self):
        self.events = []

    async def on_event(self, event, data):
        self.events.append((event, data))


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["batch", "stream"])
@pytest.mark.parametrize(
    "case",
    [
        "success",
        "failure",
        "retry",
        "deadline",
        "timeout",
        "filtered",
        "recover",
        "recover_failure",
        "flip",
        "abort",
        "batch_timeout",
    ],
)
async def test_terminal_metrics_match_final_results(case, surface):
    metrics, recorder = MetricsObserver(), Recorder()
    guard = GuardrailConfig(
        total_timeout_per_item=0.015 if case == "deadline" else None,
        batch_timeout=0.015 if case == "batch_timeout" else None,
        abort_on_error_categories=frozenset({"logic_error"}) if case == "abort" else frozenset(),
    )
    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=0.015 if case == "timeout" else 1,
        guardrails=guard,
        retry=RetryConfig(max_attempts=2, initial_wait=0.001, max_wait=0.001, jitter=False),
    )
    prompts = ["failure", "success", "success"] if case == "abort" else [case]
    async with ParallelBatchProcessor(
        config=config, middlewares=[Middleware()], observers=[metrics, recorder]
    ) as p:
        # Stage items before start so a fast abort cannot reject the test's submissions.
        for i, prompt in enumerate(prompts):
            await p.add_work(
                LLMWorkItem(
                    item_id="flip" if case == "flip" else str(i), prompt=prompt, strategy=Strategy()
                )
            )
        if surface == "batch":
            results = (await p.process_all()).results
        else:
            p.start()
            await p.finish()
            results = [r async for r in p.results()]
        stats = await p.get_stats()
    measured = await metrics.get_metrics()
    assert [measured[f"items_{key}"] for key in ("processed", "succeeded", "failed")] == [
        stats[key] for key in ("processed", "succeeded", "failed")
    ]
    assert measured["error_counts"] == stats["error_counts"]
    terminals = [(e, d) for e, d in recorder.events if e in TERMINALS]
    assert Counter(d["item_id"] for _, d in terminals) == Counter(r.item_id for r in results)
    for result in results:
        event = next(e for e, d in terminals if d["item_id"] == result.item_id)
        assert event is (
            ProcessingEvent.ITEM_COMPLETED if result.success else ProcessingEvent.ITEM_FAILED
        )
    if case == "abort":
        events = [e for e, _ in recorder.events]
        assert events.index(ProcessingEvent.ITEM_FAILED) < events.index(
            ProcessingEvent.BATCH_ABORTED
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("store_type", [JsonlArtifactStore, SqliteArtifactStore])
@pytest.mark.parametrize("replay", [False, True])
async def test_checkpoint_precedes_terminal_and_replay_stays_single(tmp_path, store_type, replay):
    path = tmp_path / "artifact"

    async def run(resume):
        store = store_type(path)
        metrics, recorder = MetricsObserver(), Recorder()
        original = store.append
        written = []

        async def append(*args):
            await original(*args)
            written.append(args[-1].item_id)

        store.append = append
        original_event = recorder.on_event

        async def observe(event, data):
            if event in TERMINALS and event is not ProcessingEvent.ITEM_REPLAYED:
                assert data["item_id"] in written
            await original_event(event, data)

        recorder.on_event = observe
        async with ParallelBatchProcessor(
            artifact_store=store, resume=resume, observers=[metrics, recorder]
        ) as p:
            for prompt in ["success", "failure"]:
                await p.add_work(LLMWorkItem(item_id=prompt, prompt=prompt, strategy=Strategy()))
            result = await p.process_all()
            stats = await p.get_stats()
        return result, stats, await metrics.get_metrics(), recorder.events

    first = await run(ResumePolicy.NONE)
    result, stats, metrics, events = await run(ResumePolicy.REUSE_ALL) if replay else first
    assert metrics["items_processed"] == stats["processed"] == 2
    assert metrics["items_succeeded"] == metrics["items_failed"] == 1
    assert metrics["error_counts"] == stats["error_counts"]
    terminals = [e for e, _ in events if e in TERMINALS]
    assert len(terminals) == 2
    assert all(e is ProcessingEvent.ITEM_REPLAYED for e in terminals) == replay
    assert all(r.replayed_from_artifact for r in result.results) == replay


@pytest.mark.asyncio
@pytest.mark.parametrize("store_type", [JsonlArtifactStore, SqliteArtifactStore])
async def test_failed_checkpoint_has_no_terminal_event(tmp_path, store_type):
    store, recorder, metrics = store_type(tmp_path / "artifact"), Recorder(), MetricsObserver()

    async def fail(*args):
        raise ArtifactIOError("write failed")

    store.append = fail
    async with ParallelBatchProcessor(artifact_store=store, observers=[recorder, metrics]) as p:
        await p.add_work(LLMWorkItem(item_id="x", prompt="success", strategy=Strategy()))
        with pytest.raises(ArtifactIOError, match="write failed"):
            await p.process_all()
        assert (await p.get_stats())["processed"] == 0
    assert not any(e in TERMINALS for e, _ in recorder.events)
    assert (await metrics.get_metrics())["items_processed"] == 0


@pytest.mark.asyncio
async def test_duck_observer_and_caller_list_snapshot():
    class Duck:
        def __init__(self):
            self.events = []

        async def on_event(self, event, data):
            self.events.append(event)

    observer = Duck()
    supplied = [observer]
    async with ParallelBatchProcessor(observers=supplied) as p:
        supplied.clear()
        await p.add_work(LLMWorkItem(item_id="x", prompt="success", strategy=Strategy()))
        await p.process_all()
    assert ProcessingEvent.ITEM_COMPLETED in observer.events


@pytest.mark.parametrize("observer", [object(), type("Bad", (), {"on_event": 3})()])
def test_invalid_observer_rejected_at_construction(observer):
    with pytest.raises(TypeError, match="callable on_event"):
        ParallelBatchProcessor(observers=[observer])


@pytest.mark.asyncio
async def test_mock_observer_uses_guarded_dispatch(monkeypatch):
    from async_batch_llm._internal import event_dispatcher

    observer = MagicMock()
    observer.on_event = AsyncMock()
    wait = AsyncMock(wraps=asyncio.wait_for)
    monkeypatch.setattr(event_dispatcher.asyncio, "wait_for", wait)
    dispatcher = event_dispatcher.EventDispatcher([observer], [])
    await dispatcher.emit(ProcessingEvent.ITEM_COMPLETED)
    assert wait.await_count == 1
    assert observer.on_event.await_count == 1


@pytest.mark.asyncio
async def test_external_cancellation_before_result_has_no_terminal_event():
    entered = asyncio.Event()

    class Blocked(Strategy):
        async def execute(self, *args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    recorder, metrics = Recorder(), MetricsObserver()
    async with ParallelBatchProcessor(observers=[recorder, metrics]) as p:
        await p.add_work(LLMWorkItem(item_id="x", prompt="x", strategy=Blocked()))
        task = asyncio.create_task(p.process_all())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await p.get_stats())["processed"] == 0
    assert not any(e in TERMINALS for e, _ in recorder.events)
    assert (await metrics.get_metrics())["items_processed"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["after", "error"])
@pytest.mark.parametrize("error_text", [None, ": detail"])
async def test_failed_middleware_result_error_count_matches_text(hook, error_text):
    class NoErrorText(BaseMiddleware):
        async def after_process(self, result):
            return WorkItemResult(item_id=result.item_id, success=False, error=error_text)

        async def on_error(self, item, error):
            return WorkItemResult(item_id=item.item_id, success=False, error=error_text)

    metrics = MetricsObserver()
    async with ParallelBatchProcessor(middlewares=[NoErrorText()], observers=[metrics]) as p:
        await p.add_work(
            LLMWorkItem(
                item_id="x", prompt="success" if hook == "after" else "failure", strategy=Strategy()
            )
        )
        result = await p.process_all()
        stats = await p.get_stats()
    measured = await metrics.get_metrics()
    assert not result.results[0].success
    assert measured["items_failed"] == stats["failed"] == 1
    assert measured["error_counts"] == stats["error_counts"] == ({"": 1} if error_text else {})


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", [None, "populated"])
async def test_observer_appended_to_processor_is_delivered_events(initial):
    observer = Recorder()
    supplied = [Recorder()] if initial else None
    async with ParallelBatchProcessor(observers=supplied) as p:
        p.observers.append(observer)
        await p.add_work(LLMWorkItem(item_id="x", prompt="success", strategy=Strategy()))
        await p.process_all()
    assert any(event is ProcessingEvent.ITEM_COMPLETED for event, _ in observer.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("store_type", [JsonlArtifactStore, SqliteArtifactStore])
@pytest.mark.parametrize("surface", ["batch", "stream"])
async def test_serialization_fallback_emits_final_failure_before_abort(
    store_type, surface, tmp_path
):
    class Unserializable(Strategy):
        async def execute(self, prompt, attempt, timeout, state=None):
            return object(), {"total_tokens": 2}

    metrics, recorder = MetricsObserver(), Recorder()
    config = ProcessorConfig(
        guardrails=GuardrailConfig(
            abort_on_error_categories=frozenset({"artifact_serialization_error"})
        )
    )
    async with ParallelBatchProcessor(
        artifact_store=store_type(tmp_path / "artifact"),
        config=config,
        observers=[metrics, recorder],
    ) as p:
        await p.add_work(LLMWorkItem(item_id="bad", prompt="bad", strategy=Unserializable()))
        if surface == "batch":
            results = (await p.process_all()).results
        else:
            p.start()
            await p.finish()
            results = [r async for r in p.results()]
        stats = await p.get_stats()
    assert not results[0].success
    assert results[0].error_category == "artifact_serialization_error"
    measured = await metrics.get_metrics()
    assert measured["items_failed"] == stats["failed"] == 1
    assert measured["items_succeeded"] == stats["succeeded"] == 0
    assert measured["error_counts"] == stats["error_counts"] == {"ArtifactSerializationError": 1}
    assert [(e, d["error_category"]) for e, d in recorder.events if e in TERMINALS] == [
        (ProcessingEvent.ITEM_FAILED, "artifact_serialization_error")
    ]
    kinds = [e for e, _ in recorder.events]
    assert kinds.index(ProcessingEvent.ITEM_FAILED) < kinds.index(ProcessingEvent.BATCH_ABORTED)
