"""Execute the network-free embedded-application example in CI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from examples.example_callable_application import run_demo


@pytest.mark.asyncio
async def test_callable_application_example(tmp_path: Path) -> None:
    before = {task for task in asyncio.all_tasks() if not task.done()}
    report = await run_demo(tmp_path / "application.jsonl", verbose=False)
    await asyncio.sleep(0)
    after = {task for task in asyncio.all_tasks() if not task.done()}

    assert report.first_client_calls == 7
    assert report.resumed_client_calls == 0
    assert len(report.first_results) == 5
    assert len(report.resumed_results) == 5
    assert all(result.replayed_from_artifact for result in report.resumed_results)
    assert report.artifact_item_records == 5
    assert set(report.checkpoint_verified_ids) == {f"doc-{index}" for index in range(5)}

    by_id = {result.item_id: result for result in report.first_results}
    # Failed billed parse (5) + successful retry (6).
    assert by_id["doc-1"].token_usage["total_tokens"] == 11
    assert by_id["doc-3"].token_usage["total_tokens"] == 11
    assert report.feedback_by_document["doc-1"] == [
        None,
        "For doc-1, return one JSON object with document_id and label.",
    ]
    assert report.feedback_by_document["doc-3"] == [
        None,
        "For doc-3, return one JSON object with document_id and label.",
    ]
    assert report.first_results[0].item_id != "doc-0"
    assert report.config.max_queue_size == 4
    assert report.config.max_result_queue_size == 2
    assert report.first_client_prepared == 1
    assert report.first_client_closed == 1
    assert after - before == set()
