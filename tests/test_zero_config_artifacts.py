"""Tests for zero-config checkpointing (issue #99).

JsonlArtifactStore("run.jsonl") with no ArtifactIdentity: provider/model are
inferred from the strategy at run start, remaining identity fields default
to "unversioned", and prompt/context participation in the compatibility
fingerprint still protects against silent reuse across changed inputs.
"""

import pytest

from async_batch_llm import (
    ArtifactError,
    JsonlArtifactStore,
    LLMCallStrategy,
    OpenAIModel,
    OpenAIStrategy,
    ResumePolicy,
    process_prompts,
)
from async_batch_llm.artifacts import infer_artifact_identity


class FakeModel:
    def __init__(self, model_id: str):
        self._model = model_id


class CountingStrategy(LLMCallStrategy[str]):
    """Deterministic strategy with a model attribute for identity inference."""

    def __init__(self, model_id: str = "fake-model-1"):
        self.model = FakeModel(model_id)
        self.calls = 0

    async def execute(self, prompt, attempt, timeout, state=None):
        self.calls += 1
        tokens = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return f"out:{prompt}", tokens, None


class TestIdentityInference:
    def test_builtin_openai_strategy(self):
        strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test"))
        identity = infer_artifact_identity(strategy)
        assert identity.provider == "openai"
        assert identity.model == "gpt-4o-mini"
        assert identity.prompt_version == "unversioned"
        assert identity.parser_version == "unversioned"
        assert identity.application_version == "unversioned"

    def test_inference_is_deterministic(self):
        a = infer_artifact_identity(CountingStrategy("m"))
        b = infer_artifact_identity(CountingStrategy("m"))
        assert a == b

    def test_custom_model_class_uses_class_name(self):
        identity = infer_artifact_identity(CountingStrategy("my-model"))
        assert identity.provider == "FakeModel"
        assert identity.model == "my-model"

    def test_strategy_without_model_falls_back_to_strategy_name(self):
        class Bare(LLMCallStrategy[str]):
            async def execute(self, prompt, attempt, timeout, state=None):
                return "x", {}, None

        identity = infer_artifact_identity(Bare())
        assert identity.provider == "Bare"
        assert identity.model == "unknown"


class TestMinimalForm:
    async def test_checkpoints_and_resumes(self, tmp_path):
        path = tmp_path / "run.jsonl"

        first = CountingStrategy()
        batch1 = await process_prompts(
            first,
            ["alpha", "beta"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert batch1.succeeded == 2
        assert first.calls == 2

        second = CountingStrategy()
        batch2 = await process_prompts(
            second,
            ["alpha", "beta"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert batch2.succeeded == 2
        assert second.calls == 0  # everything replayed
        assert all(r.replayed_from_artifact for r in batch2.results)

    async def test_changed_model_invalidates_reuse(self, tmp_path):
        path = tmp_path / "run.jsonl"
        await process_prompts(
            CountingStrategy("model-a"),
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )

        changed = CountingStrategy("model-b")
        batch = await process_prompts(
            changed,
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert changed.calls == 1  # identity changed -> no replay
        assert not batch.results[0].replayed_from_artifact

    async def test_changed_prompt_invalidates_reuse(self, tmp_path):
        path = tmp_path / "run.jsonl"
        await process_prompts(
            CountingStrategy(),
            [("id_1", "original prompt")],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )

        changed = CountingStrategy()
        batch = await process_prompts(
            changed,
            [("id_1", "different prompt")],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert changed.calls == 1
        assert not batch.results[0].replayed_from_artifact

    async def test_original_model_records_still_replay_after_switch(self, tmp_path):
        path = tmp_path / "run.jsonl"
        await process_prompts(
            CountingStrategy("model-a"),
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        await process_prompts(
            CountingStrategy("model-b"),
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )

        back = CountingStrategy("model-a")
        batch = await process_prompts(
            back,
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert back.calls == 0
        assert batch.results[0].replayed_from_artifact

    async def test_iter_results_reads_zero_config_artifact(self, tmp_path):
        path = tmp_path / "run.jsonl"
        await process_prompts(
            CountingStrategy(),
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )

        audit = JsonlArtifactStore(path)
        seen = [result async for result in audit.iter_results()]
        await audit.close()
        assert len(seen) == 1
        assert seen[0].output == "out:alpha"


class TestUnresolvedIdentityGuards:
    async def test_new_artifact_without_identity_or_items_raises(self, tmp_path):
        store = JsonlArtifactStore(tmp_path / "new.jsonl")
        with pytest.raises(ArtifactError, match="without a\\s+resolved identity"):
            _ = [r async for r in store.iter_results()]
        await store.close()
