"""Artifacts and results written by published releases still load and replay.

The fixtures under tests/fixtures/v0_*/ were written by that release (see
scripts/write_legacy_fixtures.py and each directory's VERSION file) with an
explicit ArtifactIdentity, two successes and one non-retryable failure.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from async_batch_llm import (
    ArtifactIdentity,
    BatchResult,
    JsonlArtifactStore,
    LLMCallStrategy,
    ResumePolicy,
    SqliteArtifactStore,
    process_prompts,
)

FIXTURES = Path(__file__).parent / "fixtures"
IDENTITY = ArtifactIdentity(provider="fixture", model="fixture-model", prompt_version="1")
PROMPTS = [("ok1", "alpha"), ("ok2", "beta"), ("bad", "gamma")]
RELEASES = {"v0_18": "0.18.0", "v0_21": "0.21.0", "v0_24_1": "0.24.1", "v0_26": "0.26.0"}
STORES = [
    (release, name, store)
    for release in RELEASES
    for name, store in (
        ("artifacts.jsonl", JsonlArtifactStore),
        ("artifacts.sqlite", SqliteArtifactStore),
    )
    if (FIXTURES / release / name).exists()
]


class NewStrategy(LLMCallStrategy[str]):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, prompt, attempt, timeout, state=None):
        self.calls.append(prompt)
        return f"new:{prompt}", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None


def test_every_release_has_its_fixtures():
    # SQLite stores exist from v0.21; v0.18 wrote JSONL only.
    counts = {release: [r for r, _, _ in STORES].count(release) for release in RELEASES}
    assert counts == {"v0_18": 1, "v0_21": 2, "v0_24_1": 2, "v0_26": 2}
    for release, version in RELEASES.items():
        assert (FIXTURES / release / "VERSION").read_text().strip() == version


async def _resume(path: Path, store_type, resume: ResumePolicy):
    strategy = NewStrategy()
    batch = await process_prompts(
        strategy,
        PROMPTS,
        artifact_store=store_type(path, identity=IDENTITY),
        resume=resume,
        preserve_order=True,
    )
    return strategy, batch


@pytest.mark.parametrize(("release", "name", "store_type"), STORES)
@pytest.mark.asyncio
async def test_reuse_all_replays_every_stored_result(tmp_path, release, name, store_type):
    path = tmp_path / name
    shutil.copy(FIXTURES / release / name, path)
    strategy, batch = await _resume(path, store_type, ResumePolicy.REUSE_ALL)
    assert strategy.calls == []
    assert [r.replayed_from_artifact for r in batch.results] == [True, True, True]
    assert [r.output for r in batch.results] == ["old:alpha", "old:beta", None]
    assert [r.success for r in batch.results] == [True, True, False]
    assert batch.results[2].error_category == "logic_error"
    assert batch.results[0].metadata.get("fixture") is True
    assert batch.results[0].token_usage["total_tokens"] == 5


@pytest.mark.parametrize(("release", "name", "store_type"), STORES)
@pytest.mark.asyncio
async def test_reuse_successes_reruns_the_failure_and_appends(tmp_path, release, name, store_type):
    path = tmp_path / name
    shutil.copy(FIXTURES / release / name, path)
    strategy, batch = await _resume(path, store_type, ResumePolicy.REUSE_SUCCESSES)
    assert strategy.calls == ["gamma"]
    assert [r.output for r in batch.results] == ["old:alpha", "old:beta", "new:gamma"]
    assert [r.replayed_from_artifact for r in batch.results] == [True, True, False]

    # The old file accepted the new record; a later run replays it.
    again, rerun = await _resume(path, store_type, ResumePolicy.REUSE_SUCCESSES)
    assert again.calls == []
    assert [r.output for r in rerun.results] == ["old:alpha", "old:beta", "new:gamma"]


@pytest.mark.parametrize("release", RELEASES)
@pytest.mark.parametrize("name", ["batch_result.json", "batch_result.jsonl"])
def test_serialized_batch_result_loads(release, name):
    path = FIXTURES / release / name
    batch = (
        BatchResult.from_json(path.read_text())
        if name.endswith(".json")
        else BatchResult.from_jsonl(path)
    )
    assert batch.total_items == 3 and batch.succeeded == 2 and batch.failed == 1
    by_id = batch.by_id()
    assert by_id["ok1"].output == "old:alpha"
    assert by_id["bad"].error_category == "logic_error"
    assert batch.total_input_tokens + batch.total_output_tokens == 10
