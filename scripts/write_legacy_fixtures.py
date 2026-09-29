"""Write artifact and result fixtures with a published async-batch-llm release.

Run once per release, with that release installed, for example:

    uv run --no-project --python 3.12 --with async-batch-llm==0.18.0 \
        python scripts/write_legacy_fixtures.py tests/fixtures/v0_18

The current test suite (tests/test_legacy_fixtures.py) checks that these files
still load, replay and accept new records. Do not regenerate fixtures for a
release after it is committed: the point is to keep what that release wrote.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import async_batch_llm as abl

IDENTITY = {"provider": "fixture", "model": "fixture-model", "prompt_version": "1"}
PROMPTS = [("ok1", "alpha"), ("ok2", "beta"), ("bad", "gamma")]


class FixtureStrategy(abl.LLMCallStrategy):
    async def execute(self, prompt, attempt, timeout, state=None):
        if prompt == "gamma":
            raise ValueError("fixture failure")
        tokens = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        return f"old:{prompt}", tokens, {"fixture": True}


async def run(store) -> abl.BatchResult:
    return await abl.process_prompts(
        FixtureStrategy(),
        PROMPTS,
        artifact_store=store,
        resume=abl.ResumePolicy.REUSE_ALL,
        preserve_order=True,
    )


async def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    identity = abl.ArtifactIdentity(**IDENTITY)
    batch = await run(abl.JsonlArtifactStore(out / "artifacts.jsonl", identity=identity))
    if hasattr(abl, "SqliteArtifactStore"):
        await run(abl.SqliteArtifactStore(out / "artifacts.sqlite", identity=identity))
    (out / "batch_result.json").write_text(batch.to_json())
    batch.to_jsonl(out / "batch_result.jsonl")
    (out / "VERSION").write_text(abl.__version__ + "\n")
    print(abl.__version__, sorted(p.name for p in out.iterdir()))


asyncio.run(main(Path(sys.argv[1])))
