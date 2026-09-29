"""Configuration failures are run-local: they never replay from artifacts (#178)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from async_batch_llm import (
    ArtifactIOError,
    JsonlArtifactStore,
    LLMCallStrategy,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    ResumePolicy,
    SqliteArtifactStore,
    TokenEstimate,
    WorkItemResult,
)
from async_batch_llm._internal.artifact_codec import (
    BEST_EFFORT_AUDIT_CATEGORIES,
    CONFIGURATION_FAILURE_CATEGORIES,
    GUARDRAIL_AUDIT_CATEGORIES,
    NON_REPLAYABLE_CATEGORIES,
)
from async_batch_llm.middleware import BaseMiddleware

CATEGORIES = (
    "token_estimator_required",
    "token_estimation_error",
    "token_estimate_exceeds_limit",
    "quota_scope_error",
)


class Strategy(LLMCallStrategy[str]):
    """One class for both runs, so the inferred artifact identity is unchanged."""

    def __init__(self, *, broken_scope: bool = False, fail: Exception | None = None):
        self.model = SimpleNamespace(_model="model-a")
        self.broken_scope = broken_scope
        self.fail = fail
        self.calls: list[str] = []

    @property
    def quota_scope(self) -> object:
        if self.broken_scope:
            raise RuntimeError("broken quota scope")
        return self

    async def execute(self, prompt, attempt, timeout, state=None):
        self.calls.append(prompt)
        if self.fail is not None:
            raise self.fail
        return prompt.upper(), {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}, None


class SwapToBrokenScope(BaseMiddleware):
    """A submitted strategy with a broken scope fails at add_work and is never
    checkpointed; a middleware replacement is how quota_scope_error reaches a store."""

    async def before_process(self, item):
        item.strategy = Strategy(broken_scope=True)
        return item


def _estimate(prompt: str, **kwargs: Any) -> TokenEstimate:
    return TokenEstimate(10)


def _raising_estimator(prompt: str, **kwargs: Any) -> TokenEstimate:
    raise RuntimeError("estimator bug")


def _too_large(prompt: str, **kwargs: Any) -> TokenEstimate:
    return TokenEstimate(1_001)


FIXED = ProcessorConfig(max_workers=1, max_tokens_per_minute=1_000, token_estimator=_estimate)
BROKEN: dict[str, tuple[ProcessorConfig, list[BaseMiddleware]]] = {
    "token_estimator_required": (
        ProcessorConfig(max_workers=1, max_tokens_per_minute=1_000),
        [],
    ),
    "token_estimation_error": (
        ProcessorConfig(
            max_workers=1, max_tokens_per_minute=1_000, token_estimator=_raising_estimator
        ),
        [],
    ),
    "token_estimate_exceeds_limit": (
        ProcessorConfig(max_workers=1, max_tokens_per_minute=1_000, token_estimator=_too_large),
        [],
    ),
    "quota_scope_error": (FIXED, [SwapToBrokenScope()]),
}


@pytest.fixture(params=[JsonlArtifactStore, SqliteArtifactStore], ids=["jsonl", "sqlite"])
def store_factory(request, tmp_path):
    def create():
        return request.param(tmp_path / "artifacts")

    return create


async def _run(store, strategy, config, middlewares=(), *, item_id="item") -> WorkItemResult:
    async with ParallelBatchProcessor(
        config=config,
        artifact_store=store,
        resume=ResumePolicy.REUSE_ALL,
        middlewares=list(middlewares),
    ) as processor:
        await processor.add_work(LLMWorkItem(item_id, strategy, "prompt"))
        return (await processor.process_all()).results[0]


def test_configuration_categories_are_non_replayable_but_not_audit_only():
    assert set(CONFIGURATION_FAILURE_CATEGORIES) == set(CATEGORIES)
    assert set(CATEGORIES) <= set(NON_REPLAYABLE_CATEGORIES)
    # Replay eligibility must not grant permission to swallow checkpoint errors.
    assert not set(CATEGORIES) & set(GUARDRAIL_AUDIT_CATEGORIES)
    assert not set(CATEGORIES) & set(BEST_EFFORT_AUDIT_CATEGORIES)


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.asyncio
async def test_fixed_configuration_reexecutes_instead_of_replaying(store_factory, category):
    config, middlewares = BROKEN[category]
    await _run(store_factory(), Strategy(), FIXED, item_id="seed")  # store exists
    broken = Strategy()
    first = await _run(store_factory(), broken, config, middlewares)
    assert not first.success and first.error_category == category
    assert broken.calls == []

    store = store_factory()
    item = LLMWorkItem("item", Strategy(), "prompt")
    try:
        stored = [r.error_category async for r in store.iter_results() if r.item_id == "item"]
        key = await store.prepare_item(item)
        lookup = await store.lookup(item, key, ResumePolicy.REUSE_ALL)
    finally:
        await store.close()
    # quota_scope_error fails before the artifact key is prepared, so it is never
    # checkpointed; the estimator failures are, and must not replay.
    assert stored == ([] if category == "quota_scope_error" else [category])
    assert lookup is None

    fixed = Strategy()
    resumed = await _run(store_factory(), fixed, FIXED)
    assert resumed.success and not resumed.replayed_from_artifact
    assert fixed.calls == ["prompt"]


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.asyncio
async def test_configuration_record_does_not_mask_older_success(
    store_factory, category, legacy, monkeypatch
):
    """A v0.26 record (marked replay-eligible) is excluded; the older success replays."""
    from async_batch_llm._internal import artifact_codec

    strategy = Strategy()
    assert (await _run(store_factory(), strategy, FIXED)).success
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
                WorkItemResult("item", success=False, error="config", error_category=category),
            )
    finally:
        await store.close()
    resumed = await _run(store_factory(), strategy, FIXED)
    assert resumed.success and resumed.replayed_from_artifact
    assert strategy.calls == ["prompt"]


@pytest.mark.asyncio
async def test_ordinary_failure_still_replays_under_reuse_all(store_factory):
    failing = Strategy(fail=ValueError("bad output"))
    first = await _run(store_factory(), failing, FIXED)
    assert not first.success and first.error_category not in NON_REPLAYABLE_CATEGORIES

    again = Strategy()
    resumed = await _run(store_factory(), again, FIXED)
    assert not resumed.success and resumed.replayed_from_artifact
    assert again.calls == []


@pytest.mark.parametrize("category", CATEGORIES[:3])
@pytest.mark.asyncio
async def test_configuration_failure_checkpoint_error_propagates(store_factory, category):
    """Non-replayable is not audit-only: a failed append is not swallowed."""
    config, middlewares = BROKEN[category]
    store = store_factory()
    appended: list[str | None] = []

    async def fail(item, key, result):
        appended.append(result.error_category)
        raise ArtifactIOError("config checkpoint failed")

    store.append = fail
    with pytest.raises(ArtifactIOError, match="config checkpoint failed"):
        await _run(store, Strategy(), config, middlewares)
    assert appended == [category]
