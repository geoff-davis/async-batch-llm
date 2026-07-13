"""Resumable production batch with deadlines and fail-fast guardrails.

Requires ``pip install 'async-batch-llm[openai]'`` and ``OPENAI_API_KEY``.
Run the example again with the same inputs to replay the checkpointed successes
without another provider call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from async_batch_llm import (
    AbortMode,
    ArtifactIdentity,
    GuardrailConfig,
    JsonlArtifactStore,
    OpenAIModel,
    OpenAIStrategy,
    ProcessorConfig,
    ResumePolicy,
    process_prompts,
)


async def main() -> None:
    model_name = "gpt-4o-mini"
    strategy = OpenAIStrategy(OpenAIModel.from_api_key(model_name))
    prompts = [
        ("invoice-001", "Extract the vendor and total from: Example invoice A"),
        ("invoice-002", "Extract the vendor and total from: Example invoice B"),
    ]
    store = JsonlArtifactStore(
        "runs/invoice-extraction.jsonl",
        identity=ArtifactIdentity(
            provider="openai",
            model=model_name,
            prompt_version="invoice-v4",
            parser_version="invoice-schema-v2",
            application_version="billing-pipeline-v7",
        ),
    )
    config = ProcessorConfig(
        max_workers=20,
        timeout_per_item=30,
        guardrails=GuardrailConfig(
            total_timeout_per_item=180,
            batch_timeout=3600,
            abort_on_error_categories=frozenset({"authentication", "insufficient_balance"}),
            abort_mode=AbortMode.DRAIN_ACTIVE,
        ),
    )

    result = await process_prompts(
        strategy,
        prompts,
        config=config,
        artifact_store=store,
        resume=ResumePolicy.REUSE_SUCCESSES,
        preserve_order=True,
    )

    print(
        f"termination={result.termination.kind} succeeded={result.succeeded} failed={result.failed}"
    )
    for item in result.results:
        replay = "replayed" if item.replayed_from_artifact else "executed"
        print(item.item_id, replay, item.output if item.success else item.error)
    Path("summary.json").write_text(result.to_json(), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
