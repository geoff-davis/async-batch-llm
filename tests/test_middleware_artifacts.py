"""Current-run middleware policy precedes replay on both artifact backends."""

from __future__ import annotations

import asyncio
import sys
import time
import warnings
from dataclasses import InitVar, dataclass, replace
from types import SimpleNamespace

import pytest

from async_batch_llm import (
    ArtifactError,
    ArtifactFormatError,
    ArtifactIOError,
    GuardrailConfig,
    ItemDeadlineExceeded,
    JsonlArtifactStore,
    LLMCallStrategy,
    LLMWorkItem,
    MiddlewareContractError,
    ParallelBatchProcessor,
    ProcessorConfig,
    ResumePolicy,
    RetryConfig,
    SqliteArtifactStore,
    WorkItemResult,
    process_prompts,
    process_stream,
)
from async_batch_llm.artifacts import infer_artifact_identity
from async_batch_llm.middleware import BaseMiddleware
from async_batch_llm.observers import BaseObserver, ProcessingEvent


class Strategy(LLMCallStrategy[str]):
    def __init__(self, model="model-a", *, failures=0):
        self.model = SimpleNamespace(_model=model)
        self.failures = failures
        self.calls = []
        self.states = []
        self.preparations = 0

    async def prepare(self):
        self.preparations += 1

    async def execute(self, prompt, attempt, timeout, state=None):
        self.calls.append(prompt)
        self.states.append(state)
        if len(self.calls) <= self.failures:
            raise ConnectionError("temporary provider failure")
        return prompt.upper(), {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}, None


class Transform(BaseMiddleware):
    def __init__(self, transform=lambda item: item):
        self.transform = transform
        self.before = 0
        self.after = 0
        self.errors = 0

    async def before_process(self, item):
        self.before += 1
        return self.transform(item)

    async def after_process(self, result):
        self.after += 1
        result.output += "!"
        return result

    async def on_error(self, item, error):
        self.errors += 1


@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.asyncio
async def test_work_item_initvar_and_side_effects_are_not_repeated(middleware):
    constructed = []

    @dataclass
    class Item(LLMWorkItem):
        label: InitVar[str] = "initial"

        def __post_init__(self, label):
            super().__post_init__()
            constructed.append(label)

    strategy = Strategy()
    item = Item("item", strategy, "prompt", label="original")
    replacement = Item("item", strategy, "changed", label="replacement")
    results = await run(
        None, strategy, Transform(lambda _: replacement) if middleware else [], items=[item]
    )
    assert results[0].success
    assert strategy.calls == ["changed" if middleware else "prompt"]
    assert constructed == ["original", "replacement"]


@pytest.mark.parametrize("hook", ["middleware", "processor"])
@pytest.mark.asyncio
async def test_preprocessing_preserves_subclass_validation(hook):
    class Item(LLMWorkItem):
        def _validate_fields(self):
            super()._validate_fields()
            if self.prompt != "valid":
                raise ValueError("subclass requires valid prompt")

    strategy = Strategy()
    with pytest.raises(ValueError, match="subclass requires"):
        Item("item", strategy, "invalid")

    def invalidate(item):
        item.prompt = "invalid"
        return item

    class Processor(ParallelBatchProcessor):
        async def _run_middlewares_before(self, item):
            return invalidate(item)

    processor_type = Processor if hook == "processor" else ParallelBatchProcessor
    async with processor_type(
        config=ProcessorConfig(max_workers=1),
        middlewares=[Transform(invalidate)] if hook == "middleware" else [],
    ) as processor:
        await processor.add_work(Item("item", strategy, "valid"))
        result = (await processor.process_all()).results[0]
    assert result.error_category == "middleware_contract_error"
    assert "subclass requires" in result.error
    assert strategy.calls == []


@pytest.mark.parametrize("stored_result", [None, {}, [], 1, "invalid"])
def test_top_level_exclusion_category_wins_over_result_shape(stored_result):
    from async_batch_llm._internal.artifact_codec import is_non_replayable_record

    assert is_non_replayable_record({"error_category": "batch_aborted", "result": stored_result})
    assert not is_non_replayable_record({"result": stored_result})


@pytest.mark.asyncio
async def test_strategy_configuration_runs_once_with_concurrent_workers():
    setup_started = asyncio.Event()
    release_setup = asyncio.Event()
    preprocessed = 0

    class ConfiguredStrategy(Strategy):
        configurations = 0

        async def request_concurrency(self, concurrency):
            self.configurations += 1
            setup_started.set()
            await release_setup.wait()

    class Middleware(BaseMiddleware):
        async def before_process(self, item):
            nonlocal preprocessed
            preprocessed += 1
            return item

    strategy = ConfiguredStrategy()
    async with ParallelBatchProcessor(
        config=ProcessorConfig(concurrency=4), middlewares=[Middleware()]
    ) as processor:
        for index in range(8):
            await processor.add_work(LLMWorkItem(str(index), strategy, "prompt"))
        task = asyncio.create_task(processor.process_all())
        try:
            await setup_started.wait()
            while preprocessed < 4:
                await asyncio.sleep(0)
            # Let every worker reach configuration while the first hook waits.
            for _ in range(5):
                await asyncio.sleep(0)
            assert strategy.configurations == 1
        finally:
            release_setup.set()
        batch = await task
    assert len(batch.results) == 8 and all(result.success for result in batch.results)
    assert strategy.configurations == 1


@pytest.mark.parametrize("bad_input", ["identity", "context"])
@pytest.mark.asyncio
async def test_invalid_artifact_input_fails_only_its_item(store_factory, bad_input):
    strategy = Strategy()
    other = Strategy("different")
    middleware = Transform()
    results = await run(
        store_factory(),
        strategy,
        middleware,
        items=[
            LLMWorkItem("first", strategy, "first"),
            LLMWorkItem(
                "bad",
                other if bad_input == "identity" else strategy,
                "bad",
                object() if bad_input == "context" else None,
            ),
            LLMWorkItem("last", strategy, "last"),
        ],
    )
    assert [result.success for result in results] == [True, False, True]
    assert results[1].error_category == "artifact_preparation_error"
    assert strategy.calls == ["first", "last"] and other.calls == []
    assert middleware.errors == 0


@pytest.mark.parametrize("guard", ["item", "abort", "batch"])
@pytest.mark.asyncio
async def test_completed_lookup_wins_guardrail_and_preserves_stored_success(store_factory, guard):
    from async_batch_llm._internal.guardrails import AbortCause

    class ReplayStrategy(Strategy):
        configurations = 0

        async def request_concurrency(self, concurrency):
            self.configurations += 1

    await run(store_factory(), Strategy(), Transform())
    store = store_factory()
    lookup = store.lookup
    strategy = ReplayStrategy()
    async with ParallelBatchProcessor(
        config=ProcessorConfig(
            concurrency=1,
            guardrails=GuardrailConfig(total_timeout_per_item=0.2 if guard == "item" else None),
        ),
        artifact_store=store,
        resume=ResumePolicy.REUSE_ALL,
        middlewares=[Transform()],
    ) as processor:

        async def lookup_and_trip(item, key, policy):
            result = await lookup(item, key, policy)
            assert result is not None and result.success
            if guard == "item":
                # Finish the lookup in the same event-loop turn as expiry.
                time.sleep(0.21)
            else:
                await processor._abort_controller.trip(
                    AbortCause(
                        kind="batch_timeout" if guard == "batch" else "fail_fast", reason="test"
                    )
                )
            return result

        store.lookup = lookup_and_trip
        await processor.add_work(LLMWorkItem("item", strategy, "prompt"))
        result = (await processor.process_all()).results[0]
        assert result.success and result.replayed_from_artifact
        assert (
            processor._rate_limit_coord is processor._admission_registry.resolve(strategy).cooldown
        )
    assert strategy.calls == [] and strategy.configurations == 1
    reader = store_factory()
    try:
        assert len([result async for result in reader.iter_results()]) == 1
    finally:
        await reader.close()
    resumed = (await run(store_factory(), strategy, Transform()))[0]
    assert resumed.success and resumed.replayed_from_artifact and strategy.calls == []


@pytest.mark.parametrize(
    "category", ["batch_aborted", "batch_deadline_exceeded", "framework_total_item_timeout"]
)
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.asyncio
async def test_guardrail_audit_records_do_not_mask_success(
    store_factory, category, legacy, monkeypatch
):
    from async_batch_llm._internal import artifact_codec

    strategy = Strategy()
    await run(store_factory(), strategy, Transform())
    store = store_factory()
    item = LLMWorkItem("item", strategy, "prompt")
    try:
        key = await store.prepare_item(item)
        with monkeypatch.context() as patch:
            if legacy:
                patch.setattr(artifact_codec, "NON_REPLAYABLE_CATEGORIES", ())
            await store.append(
                item,
                key,
                WorkItemResult("item", success=False, error="stopped", error_category=category),
            )
    finally:
        await store.close()
    resumed = (await run(store_factory(), strategy, Transform()))[0]
    assert resumed.success and resumed.replayed_from_artifact
    assert strategy.calls == ["prompt"]


@pytest.mark.asyncio
async def test_before_override_controls_effective_request_without_registered_middleware():
    original = Strategy()
    effective = Strategy()

    class Processor(ParallelBatchProcessor):
        before_calls = 0

        async def _run_middlewares_before(self, item):
            self.before_calls += 1
            return replace(
                await super()._run_middlewares_before(item), strategy=effective, prompt="new"
            )

    async with Processor(config=ProcessorConfig(max_workers=1)) as processor:
        await processor.add_work(LLMWorkItem("item", original, "old"))
        result = (await processor.process_all()).results[0]
        assert processor.before_calls == 1 and result.output == "NEW"
    assert effective.calls == ["new"] and original.calls == []


@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.asyncio
async def test_capacity_warning_points_to_submission(middleware):
    class SmallStrategy(Strategy):
        @property
        def max_concurrency(self):
            return 1

    strategy = SmallStrategy()
    with pytest.warns(UserWarning, match="max_workers=2 exceeds") as warnings:
        async with ParallelBatchProcessor(
            config=ProcessorConfig(max_workers=2),
            middlewares=[Transform()] if middleware else [],
        ) as processor:
            await processor.add_work(LLMWorkItem("item", strategy, "prompt"))
            await processor.process_all()
    assert warnings[0].filename == __file__


@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.parametrize("filter_action", ["default", "ignore"])
@pytest.mark.asyncio
async def test_capacity_warnings_preserve_registry_and_module_filters(middleware, filter_action):
    class SmallStrategy(Strategy):
        @property
        def max_concurrency(self):
            return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default", UserWarning)
        if filter_action == "ignore":
            warnings.filterwarnings("ignore", category=UserWarning, module=rf"^{__name__}$")
        async with ParallelBatchProcessor(
            config=ProcessorConfig(max_workers=2),
            middlewares=[Transform()] if middleware else [],
        ) as processor:
            for index in range(6):
                await processor.add_work(LLMWorkItem(str(index), SmallStrategy(), "prompt"))
            await processor.process_all()
    assert len(caught) == (1 if filter_action == "default" else 0)


@pytest.mark.parametrize("surface", ["prompts", "stream"])
@pytest.mark.parametrize("ignore", [False, True])
@pytest.mark.asyncio
async def test_streaming_capacity_warning_uses_consumer_module(surface, ignore):
    class SmallStrategy(Strategy):
        @property
        def max_concurrency(self):
            return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default", UserWarning)
        if ignore:
            warnings.filterwarnings("ignore", category=UserWarning, module=rf"^{__name__}$")
        for _ in range(2):
            kwargs = {"config": ProcessorConfig(max_workers=2), "middlewares": [Transform()]}
            if surface == "prompts":
                await process_prompts(SmallStrategy(), ["prompt"], **kwargs)
            else:
                async for _ in process_stream(SmallStrategy(), ["prompt"], **kwargs):
                    pass
    assert len(caught) == (0 if ignore else 1)
    if caught:
        assert caught[0].filename == __file__


@pytest.mark.parametrize("surface", ["batch", "stream"])
@pytest.mark.asyncio
async def test_submission_without_warning_does_not_create_registry(surface, monkeypatch):
    from async_batch_llm import streaming

    monkeypatch.delattr(sys.modules[__name__], "__warningregistry__", raising=False)
    monkeypatch.delattr(asyncio.base_events, "__warningregistry__", raising=False)

    def unexpected_capture():
        raise AssertionError("unnecessary per-item frame walk")

    if surface == "batch":
        monkeypatch.setattr(
            "async_batch_llm.parallel.capture_capacity_warning_source", unexpected_capture
        )
        await run(None, Strategy(), [])
    else:
        captures = []
        capture = streaming.capture_capacity_warning_source

        def capture_once():
            captures.append(True)
            return capture()

        monkeypatch.setattr(streaming, "capture_capacity_warning_source", capture_once)
        await process_prompts(Strategy(), ["prompt"] * 3)
        assert len(captures) == 1
    assert not hasattr(sys.modules[__name__], "__warningregistry__")
    assert not hasattr(asyncio.base_events, "__warningregistry__")


@pytest.mark.parametrize(
    ("phase", "prepare_delay"), [("prepare", 0.0), ("append", 0.0), ("append", 0.1)]
)
@pytest.mark.parametrize("error_type", [ArtifactIOError, ArtifactFormatError])
@pytest.mark.asyncio
async def test_item_timeout_checkpoint_failure_still_propagates(
    store_factory, phase, prepare_delay, error_type
):
    class DeadlineStrategy(Strategy):
        async def execute(self, *args, **kwargs):
            self.calls.append("deadline")
            # This test checks checkpoint-error policy, not timer scheduling.
            # Enter execution after real store preparation before timing out.
            raise ItemDeadlineExceeded("test execution deadline")

    class HangingMiddleware(BaseMiddleware):
        async def before_process(self, item):
            await asyncio.Event().wait()

    strategy = DeadlineStrategy()
    store = store_factory()
    prepare = store.prepare_item

    async def slow_prepare(item):
        await asyncio.sleep(prepare_delay)
        return await prepare(item)

    async def fail(*args):
        if phase == "append":
            assert strategy.calls == ["deadline"]
            assert args[-1].error_category == "framework_total_item_timeout"
        raise error_type("timeout checkpoint failed")

    if phase == "prepare":
        store.prepare_item = fail
    else:
        store.prepare_item = slow_prepare
        store.append = fail
    with pytest.raises(error_type, match="timeout checkpoint failed"):
        await run(
            store,
            strategy,
            HangingMiddleware() if phase == "prepare" else [],
            config=ProcessorConfig(
                max_workers=1,
                guardrails=GuardrailConfig(
                    total_timeout_per_item=0.05 if phase == "prepare" else None
                ),
            ),
        )


@pytest.mark.parametrize("phase", ["prepare", "append"])
@pytest.mark.parametrize("error_type", [ArtifactError, ArtifactIOError, ArtifactFormatError])
@pytest.mark.parametrize("abort_kind", ["fail_fast", "batch_timeout"])
@pytest.mark.asyncio
async def test_audit_artifact_failure_preserves_controlled_stop(
    store_factory, phase, error_type, abort_kind, caplog
):
    from async_batch_llm._internal.guardrails import AbortCause

    strategy = Strategy()
    store = store_factory()
    prepare = store.prepare_item
    append = store.append
    async with ParallelBatchProcessor(
        config=ProcessorConfig(max_workers=1), artifact_store=store
    ) as processor:

        async def prepare_or_fail(item):
            if phase == "prepare" and processor._abort_controller.aborted:
                raise error_type("audit failed")
            return await prepare(item)

        async def append_or_fail(item, key, result):
            if phase == "append" and processor._abort_controller.aborted:
                raise error_type("audit failed")
            await append(item, key, result)
            if item.item_id == "1":
                await processor._trip_abort(AbortCause(kind=abort_kind, reason="test"))

        store.prepare_item = prepare_or_fail
        store.append = append_or_fail
        for index in range(5):
            await processor.add_work(LLMWorkItem(str(index), strategy, "prompt"))
        batch = await processor.process_all()
        assert processor._queue._unfinished_tasks == 0
    assert [result.success for result in batch.results] == [True, True, False, False, False]
    category = "batch_aborted" if abort_kind == "fail_fast" else "batch_deadline_exceeded"
    assert all(result.error_category == category for result in batch.results[2:])
    assert batch.termination.kind == abort_kind
    assert len(strategy.calls) == 2
    assert (
        sum("Cannot checkpoint terminal item" in record.message for record in caplog.records) == 3
    )


@pytest.mark.asyncio
async def test_configuration_rechecks_strategy_identity_and_releases_entries():
    class ConfiguredStrategy(Strategy):
        configurations = 0

        async def request_concurrency(self, concurrency):
            self.configurations += 1

    first, second = ConfiguredStrategy(), ConfiguredStrategy()
    async with ParallelBatchProcessor(config=ProcessorConfig(concurrency=1)) as processor:
        await processor.add_work(LLMWorkItem("first", first, "prompt"))
        # Deterministically simulate a stale entry under a reused object ID.
        processor._strategy_configurations[id(second)] = processor._strategy_configurations[
            id(first)
        ]
        await processor.add_work(LLMWorkItem("second", second, "prompt"))
        await processor.process_all()
        assert first.configurations == second.configurations == 1
    assert processor._strategy_configurations == {}


@pytest.mark.asyncio
async def test_queued_aborts_are_audited_and_run_on_resume(store_factory):
    from async_batch_llm._internal.guardrails import AbortCause

    strategy = Strategy()
    store = store_factory()
    async with ParallelBatchProcessor(
        config=ProcessorConfig(max_workers=1), artifact_store=store
    ) as processor:
        for index in range(3):
            await processor.add_work(LLMWorkItem(str(index), strategy, "prompt"))
        await processor._trip_abort(AbortCause(kind="fail_fast", reason="test"))
        batch = await processor.process_all()
    assert len(batch.results) == 3
    assert all(result.error_category == "batch_aborted" for result in batch.results)
    assert strategy.calls == []
    reader = store_factory()
    try:
        assert len([result async for result in reader.iter_results()]) == 3
    finally:
        await reader.close()
    resumed = await run(
        store_factory(),
        strategy,
        [],
        items=[LLMWorkItem(str(index), strategy, "prompt") for index in range(3)],
    )
    assert all(result.success and not result.replayed_from_artifact for result in resumed)
    assert len(strategy.calls) == 3


@pytest.fixture(params=[JsonlArtifactStore, SqliteArtifactStore], ids=["jsonl", "sqlite"])
def store_factory(request, tmp_path):
    def create(**kwargs):
        return request.param(tmp_path / "artifacts", **kwargs)

    return create


async def run(store, strategy, middleware, *, context=None, config=None, items=None):
    async with ParallelBatchProcessor(
        config=config or ProcessorConfig(max_workers=1),
        artifact_store=store,
        resume=ResumePolicy.REUSE_ALL,
        middlewares=middleware if isinstance(middleware, list) else [middleware],
    ) as processor:
        for item in items or [LLMWorkItem("item", strategy, "prompt", context)]:
            await processor.add_work(item)
        batch = await processor.process_all()
        assert processor._queue._unfinished_tasks == 0
        return batch.results


@pytest.mark.asyncio
async def test_current_filter_prevents_old_success_replay(store_factory):
    first = await run(store_factory(), Strategy(), Transform())
    assert first[0].success
    strategy = Strategy()
    current = {"run": "current"}
    middleware = Transform(lambda item: None)
    store = store_factory()
    result = (await run(store, strategy, middleware, context=current))[0]
    assert result.error_category == "middleware_filtered"
    assert result.context is current
    assert result.submission_index == 0
    assert not result.replayed_from_artifact
    assert strategy.preparations == 0 and strategy.calls == []
    assert middleware.before == 1 and middleware.after == 0
    # Filtering never even opens the store or pins an inferred strategy.
    assert store.identity is None


@pytest.mark.asyncio
async def test_old_filter_cannot_bypass_current_allow(store_factory):
    strategy = Strategy()
    filtered = (await run(store_factory(), strategy, Transform(lambda item: None)))[0]
    assert filtered.error_category == "middleware_filtered"
    result = (await run(store_factory(), strategy, Transform()))[0]
    assert result.success and not result.replayed_from_artifact
    assert strategy.calls == ["prompt"]


@pytest.mark.asyncio
async def test_explicitly_appended_filter_record_is_not_replay_eligible(store_factory):
    strategy = Strategy()
    item = LLMWorkItem("item", strategy, "prompt")
    store = store_factory()
    try:
        key = await store.prepare_item(item)
        await store.append(
            item,
            key,
            WorkItemResult(
                item_id="item",
                success=False,
                error="Skipped by middleware",
                error_category="middleware_filtered",
            ),
        )
        assert await store.lookup(item, key, ResumePolicy.REUSE_ALL) is None
    finally:
        await store.close()
    result = (await run(store_factory(), strategy, Transform()))[0]
    assert result.success and not result.replayed_from_artifact


@pytest.mark.parametrize("changed", ["prompt", "context", "model", "strategy"])
@pytest.mark.asyncio
async def test_replay_identity_follows_effective_request(store_factory, changed):
    await run(store_factory(), Strategy(), Transform(), context={"version": 1})
    effective = Strategy("model-b" if changed == "model" else "model-a")
    if changed == "strategy":

        class OtherStrategy(Strategy):
            pass

        effective = OtherStrategy()
        # Custom strategies without a model infer provider from their class.
        del effective.model
    original = Strategy()
    middleware = Transform(
        lambda item: replace(
            item,
            prompt="changed" if changed == "prompt" else item.prompt,
            context={"version": 2} if changed == "context" else item.context,
            strategy=effective,
            submission_index=999,
            _artifact_key=object(),
        )
    )
    store = store_factory(include_prompt=True, include_context=True)
    result = (await run(store, original, middleware, context={"version": 1}))[0]
    assert result.success and not result.replayed_from_artifact
    assert result.submission_index == 0
    assert original.preparations == 0 and original.calls == []
    assert effective.preparations == 1 and len(effective.calls) == 1
    assert store.identity == infer_artifact_identity(effective)
    # Reopen against that same effective request: the newly written key must match.
    replay = Strategy()
    replay_middleware = Transform(
        lambda item: replace(
            item,
            prompt="changed" if changed == "prompt" else item.prompt,
            context={"version": 2} if changed == "context" else item.context,
            strategy=effective,
        )
    )
    second = (await run(store_factory(), replay, replay_middleware, context={"version": 1}))[0]
    assert second.replayed_from_artifact
    assert second.output == result.output
    assert len(effective.calls) == 1 and replay.preparations == 0
    assert replay_middleware.after == 0


@pytest.mark.asyncio
async def test_replay_uses_current_effective_context_index_and_stored_final_result(store_factory):
    strategy = Strategy()
    before = Transform(lambda item: replace(item, context={"run": "old"}))
    first = (await run(store_factory(context_in_identity=False), strategy, before))[0]
    assert before.before == before.after == 1
    current = {"run": "current"}
    effective = Strategy()
    after = Transform(
        lambda item: replace(item, context=current, strategy=effective, submission_index=9)
    )
    store = store_factory(context_in_identity=False)
    second = (await run(store, Strategy(), after))[0]
    assert second.replayed_from_artifact and second.context is current
    assert second.submission_index == 0
    assert second.output == "PROMPT!" and second.token_usage == first.token_usage
    assert second.timing == first.timing
    assert effective.preparations == 0 and effective.calls == []
    assert after.before == 1 and after.after == 0
    reader = store_factory()
    try:
        assert len([result async for result in reader.iter_results()]) == 1
    finally:
        await reader.close()


@pytest.mark.parametrize("invalid", ["item_id", "prompt", "strategy", "object", "inplace_id"])
@pytest.mark.asyncio
async def test_invalid_replacement_is_terminal_before_provider_work(invalid):
    strategy = Strategy()

    def bad(item):
        if invalid == "object":
            return object()
        replacement = item if invalid == "inplace_id" else replace(item)
        setattr(
            replacement,
            "item_id" if invalid == "inplace_id" else invalid,
            "different" if "id" in invalid else None,
        )
        return replacement

    middleware = Transform(bad)
    result = (await run(None, strategy, middleware))[0]
    assert isinstance(result.exception, MiddlewareContractError)
    assert result.error_category == "middleware_contract_error"
    assert result.item_id == "item" and result.submission_index == 0
    assert strategy.preparations == 0 and strategy.calls == []
    assert middleware.before == 1 and middleware.after == middleware.errors == 0


@pytest.mark.asyncio
async def test_before_once_and_shared_retry_state_with_effective_strategy():
    original = Strategy()
    effective = Strategy(failures=2)
    middleware = Transform(
        lambda item: replace(item, strategy=effective, prompt=item.prompt + "-once")
    )
    started = []

    class Observer(BaseObserver):
        async def on_event(self, event, data):
            if event is ProcessingEvent.ITEM_STARTED:
                started.append(data["item_id"])

    async with ParallelBatchProcessor(
        config=ProcessorConfig(
            max_workers=1,
            retry=RetryConfig(
                max_attempts=3,
                initial_wait=0.001,
                max_wait=0.001,
                jitter=False,
            ),
        ),
        middlewares=[middleware],
        observers=[Observer()],
    ) as processor:
        await processor.add_work(LLMWorkItem("item", original, "prompt"))
        result = (await processor.process_all()).results[0]
    assert result.success
    assert middleware.before == middleware.after == 1 and middleware.errors == 0
    assert effective.calls == ["prompt-once"] * 3
    assert len({id(state) for state in effective.states}) == 1
    assert effective.preparations == 1 and original.preparations == 0
    assert started == ["item"]


@pytest.mark.asyncio
async def test_filter_after_replacement_keeps_effective_context():
    current = {"policy": "current"}
    result = (
        await run(
            None,
            Strategy(),
            [
                Transform(lambda item: replace(item, context=current, submission_index=42)),
                Transform(lambda item: None),
            ],
        )
    )[0]
    assert result.context is current and result.submission_index == 0
    assert result.error_category == "middleware_filtered"


@pytest.mark.parametrize("guard", ["item", "batch"])
@pytest.mark.asyncio
async def test_preprocessing_deadline_or_abort_is_audit_only(store_factory, guard):
    cancelled = asyncio.Event()

    class Hanging(BaseMiddleware):
        async def before_process(self, item):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    strategy = Strategy()
    store = store_factory()
    config = ProcessorConfig(
        max_workers=1,
        guardrails=GuardrailConfig(
            total_timeout_per_item=0.02 if guard == "item" else None,
            batch_timeout=0.02 if guard == "batch" else None,
        ),
    )
    result = (await run(store, strategy, Hanging(), config=config))[0]
    assert result.error_category == (
        "framework_total_item_timeout" if guard == "item" else "batch_deadline_exceeded"
    )
    assert cancelled.is_set() and strategy.calls == [] and strategy.preparations == 0
    reader = store_factory()
    try:
        records = [entry async for entry in reader.iter_results()]
        assert len(records) == 1 and records[0].error_category == result.error_category
    finally:
        await reader.close()
    resumed = (await run(store_factory(), strategy, Transform()))[0]
    assert resumed.success and not resumed.replayed_from_artifact


@pytest.mark.asyncio
async def test_ordinary_preprocessing_exceptions_remain_fail_open():
    class Broken(BaseMiddleware):
        async def before_process(self, item):
            raise RuntimeError("callback bug")

    following = Transform(lambda item: replace(item, prompt="effective"))
    strategy = Strategy()
    result = (await run(None, strategy, [Broken(), following]))[0]
    assert result.success and strategy.calls == ["effective"]
    assert following.before == following.after == 1


@pytest.mark.asyncio
async def test_effective_audit_fields_and_exact_key_precede_publication(store_factory, monkeypatch):
    import importlib

    from async_batch_llm._internal.artifact_codec import fingerprint_work_item

    store = store_factory(include_prompt=True, include_context=True)
    records = []
    keys = []
    module = importlib.import_module(type(store).__module__)
    build = module.build_item_record

    def capture(**kwargs):
        assert kwargs["prepared_item"] is keys[0]
        record = build(**kwargs)
        records.append(record)
        return record

    monkeypatch.setattr(module, "build_item_record", capture)
    prepare = store.prepare_item

    async def prepare_key(item):
        key = await prepare(item)
        keys.append(key)
        return key

    monkeypatch.setattr(store, "prepare_item", prepare_key)
    effective = Strategy("effective-model")
    replacement = LLMWorkItem("item", effective, "effective prompt", {"effective": True})

    class Rewrite(Transform):
        async def after_process(self, result):
            # A retained replacement object's legacy key is not authoritative.
            replacement._artifact_key = object()
            return await super().after_process(result)

    published = []

    async def postprocess(result):
        assert len(records) == 1
        published.append(result)

    async with ParallelBatchProcessor(
        config=ProcessorConfig(max_workers=1),
        artifact_store=store,
        middlewares=[Rewrite(lambda item: replacement)],
        post_processor=postprocess,
    ) as processor:
        await processor.add_work(LLMWorkItem("item", Strategy(), "original"))
        batch = await processor.process_all()
    assert len(published) == 1 and batch.succeeded == 1
    record = records[0]
    expected = fingerprint_work_item(
        replacement,
        context_in_identity=True,
        encoder=None,
        context_fingerprinter=None,
    )
    assert record["prompt_fingerprint"] == expected.prompt_fingerprint
    assert record["context_fingerprint"] == expected.context_fingerprint
    assert record["input_fingerprint"] == expected.input_fingerprint
    assert record["raw_prompt"] == "effective prompt"
    assert record["raw_context"] == {"effective": True}
    assert record["strategy_class"].endswith(".Strategy")
    assert record["provider"] == infer_artifact_identity(effective).provider
    assert record["model"] == "effective-model"
    assert record["result"]["output"] == "EFFECTIVE PROMPT!"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.asyncio
async def test_identity_is_independent_of_preprocessing_completion_order(store_factory, reverse):
    second_preprocessed = asyncio.Event()
    effective = Strategy("effective-model")
    first_id = "b" if reverse else "a"

    class Reorder(BaseMiddleware):
        async def before_process(self, item):
            if item.item_id == first_id:
                await second_preprocessed.wait()
            else:
                second_preprocessed.set()
            return replace(item, strategy=effective, prompt="effective:" + item.prompt)

    originals = [Strategy("original-a"), Strategy("original-b")]
    results = await run(
        store_factory(),
        originals[0],
        Reorder(),
        config=ProcessorConfig(max_workers=2),
        items=[LLMWorkItem("a", originals[0], "a"), LLMWorkItem("b", originals[1], "b")],
    )
    assert all(result.success for result in results)
    assert {r.item_id: r.submission_index for r in results} == {"a": 0, "b": 1}
    assert sorted(effective.calls) == ["effective:a", "effective:b"]
    assert effective.preparations == 1
    assert all(original.preparations == 0 and original.calls == [] for original in originals)


@pytest.mark.asyncio
async def test_replacement_selects_classifier_and_quota_scope():
    from async_batch_llm import ErrorClassifier, ErrorInfo

    scope = object()

    class RetryNothing(ErrorClassifier):
        def classify(self, exception):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="effective_category",
            )

    class Effective(Strategy):
        @property
        def quota_scope(self):
            return scope

        def recommended_error_classifier(self):
            return RetryNothing()

    original = Strategy()
    effective = Effective(failures=10)
    middleware = Transform(lambda item: replace(item, strategy=effective))
    async with ParallelBatchProcessor(
        config=ProcessorConfig(max_workers=1, max_requests_per_minute=60),
        middlewares=[middleware],
    ) as processor:
        await processor.add_work(LLMWorkItem("item", original, "prompt"))
        result = (await processor.process_all()).results[0]
        assert processor._admission_registry.entry_count == 1
        assert processor._admission_registry.states[0].scope is scope
    assert result.error_category == "effective_category"
    assert effective.calls == ["prompt"] and original.calls == []
    assert middleware.errors == 1 and middleware.after == 0


@pytest.mark.parametrize("phase", ["prepare", "lookup"])
@pytest.mark.asyncio
async def test_deadline_during_artifact_io_starts_no_provider(phase):
    cancelled = asyncio.Event()
    key = object()

    class Store:
        async def prepare_item(self, item):
            if phase == "prepare":
                await self.hang()
            return key

        async def lookup(self, item, prepared, policy):
            await self.hang()

        async def hang(self):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def append(self, item, prepared, result):
            assert phase == "lookup" and prepared is key
            assert result.error_category == "framework_total_item_timeout"

        async def close(self):
            pass

    strategy = Strategy()
    result = (
        await run(
            Store(),
            strategy,
            Transform(),
            config=ProcessorConfig(
                max_workers=1,
                guardrails=GuardrailConfig(total_timeout_per_item=0.02),
            ),
        )
    )[0]
    assert result.error_category == "framework_total_item_timeout"
    assert cancelled.is_set() and strategy.calls == [] and strategy.preparations == 0


@pytest.mark.asyncio
async def test_existing_retry_override_signature_preserves_once_only_preprocessing():
    seen = []

    class Processor(ParallelBatchProcessor):
        async def _process_item_with_retries(self, item, worker_id, deadline=None):
            seen.append(item.prompt)
            return await super()._process_item_with_retries(item, worker_id, deadline)

    middleware = Transform(lambda item: replace(item, prompt=item.prompt + "-once"))
    strategy = Strategy()
    async with Processor(middlewares=[middleware]) as processor:
        await processor.add_work(LLMWorkItem("item", strategy, "prompt"))
        result = (await processor.process_all()).results[0]
    assert result.success
    assert seen == strategy.calls == ["prompt-once"]
    assert middleware.before == middleware.after == 1


@pytest.mark.asyncio
async def test_custom_store_none_token_is_checkpointed():
    written = []

    class Store:
        async def prepare_item(self, item):
            return None

        async def lookup(self, item, prepared, policy):
            assert prepared is None

        async def append(self, item, prepared, result):
            assert prepared is None
            written.append(result)

        async def close(self):
            pass

    result = (await run(Store(), Strategy(), Transform()))[0]
    assert result.success and written == [result]


@pytest.mark.parametrize("guard", ["item", "batch"])
@pytest.mark.asyncio
async def test_interrupted_mutation_preserves_accepted_identity(guard):
    class MutateAndBlock(BaseMiddleware):
        async def before_process(self, item):
            item.item_id = "changed"
            item.submission_index = 999
            item.context = {"current": True}
            await asyncio.Event().wait()

    strategy = Strategy()
    item = LLMWorkItem("accepted", strategy, "prompt")
    result = (
        await run(
            None,
            strategy,
            MutateAndBlock(),
            items=[item],
            config=ProcessorConfig(
                guardrails=GuardrailConfig(
                    total_timeout_per_item=0.02 if guard == "item" else None,
                    batch_timeout=0.02 if guard == "batch" else None,
                ),
            ),
        )
    )[0]
    assert not result.success and result.item_id == item.item_id == "accepted"
    assert result.submission_index == item.submission_index == 0
    assert result.context == {"current": True} and strategy.calls == []


@pytest.mark.asyncio
async def test_retry_override_can_tighten_prepared_deadline():
    import time

    class Processor(ParallelBatchProcessor):
        async def _process_item_with_retries(self, item, worker_id, deadline=None):
            return await super()._process_item_with_retries(
                item,
                worker_id,
                time.perf_counter() - 1,
            )

    strategy = Strategy()
    middleware = Transform()
    async with Processor(middlewares=[middleware]) as processor:
        await processor.add_work(LLMWorkItem("item", strategy, "prompt"))
        result = (await processor.process_all()).results[0]
    assert result.error_category == "framework_total_item_timeout"
    assert strategy.calls == [] and strategy.preparations == 0
    assert middleware.before == 1 and middleware.after == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier_success", [False, True])
async def test_legacy_filter_artifact_cannot_replay_after_policy_allows(
    store_factory, earlier_success
):
    strategy = Strategy()
    item = LLMWorkItem("item", strategy, "prompt")
    store = store_factory()
    try:
        key = await store.prepare_item(item)
        if earlier_success:
            await store.append(item, key, WorkItemResult("item", success=True, output="stored"))
        # v0.23.0 filter records had no category and were marked replayable.
        await store.append(
            item,
            key,
            WorkItemResult(
                item_id="item",
                success=False,
                error="Skipped by middleware",
            ),
        )
    finally:
        await store.close()
    result = (await run(store_factory(), strategy, Transform()))[0]
    assert result.success
    assert result.replayed_from_artifact is earlier_success
    assert result.output == ("stored" if earlier_success else "PROMPT!")
    assert strategy.calls == ([] if earlier_success else ["prompt"])
