"""Tests for BatchResult.outputs()/summary() and wall-time stamping (issue #96)."""

import pytest

from async_batch_llm import (
    AttemptTiming,
    BatchResult,
    BatchTermination,
    LLMCallStrategy,
    WorkItemResult,
    WorkItemTiming,
    process_prompts,
)


def make_result(
    item_id,
    *,
    success=True,
    tokens=(100, 10, 20),
    category=None,
    tries=1,
    replayed=False,
    admission=0.0,
    execution=0.5,
    index=0,
):
    attempts = [
        AttemptTiming(
            attempt=i + 1,
            try_number=i + 1,
            admission_wait_seconds=admission / tries,
            execution_seconds=execution / tries,
            success=success and i == tries - 1,
        )
        for i in range(tries)
    ]
    return WorkItemResult(
        item_id=item_id,
        success=success,
        output=f"output-{item_id}" if success else None,
        error=None if success else "boom",
        token_usage={
            "input_tokens": tokens[0],
            "cached_input_tokens": tokens[1],
            "output_tokens": tokens[2],
            "total_tokens": tokens[0] + tokens[2],
        },
        timing=WorkItemTiming(total_seconds=execution, attempts=attempts),
        admission_wait_seconds=admission,
        error_category=category,
        replayed_from_artifact=replayed,
        submission_index=index,
    )


class TestOutputs:
    def test_outputs_yields_successful_outputs_in_order(self):
        batch = BatchResult(
            results=[
                make_result("a", index=0),
                make_result("b", success=False, category="validation", index=1),
                make_result("c", index=2),
            ]
        )
        assert list(batch.outputs()) == ["output-a", "output-c"]

    def test_outputs_with_ids(self):
        batch = BatchResult(results=[make_result("a"), make_result("b", success=False)])
        assert list(batch.outputs(with_ids=True)) == [("a", "output-a")]

    def test_outputs_is_lazy(self):
        batch = BatchResult(results=[make_result("a")])
        iterator = batch.outputs()
        assert iter(iterator) is iterator  # generator, not a list

    def test_empty_batch(self):
        batch = BatchResult(results=[])
        assert list(batch.outputs()) == []


class TestSummary:
    def make_batch(self):
        return BatchResult(
            results=[
                make_result("ok_1", admission=0.2, execution=1.0, tries=1, index=0),
                make_result("ok_2", admission=0.4, execution=2.0, tries=3, index=1),
                make_result(
                    "rate_limited",
                    success=False,
                    category="rate_limit",
                    tries=2,
                    index=2,
                ),
                make_result("bad", success=False, category="validation", index=3),
                make_result("mystery", success=False, index=4),
                make_result("old", replayed=True, tokens=(500, 100, 50), tries=0, index=5),
            ],
            termination=BatchTermination(
                kind="fail_fast",
                reason="too many validation failures",
                error_category="validation",
                triggering_item_id="bad",
            ),
            wall_time_seconds=12.34,
        )

    def test_summary_is_complete_report(self):
        text = self.make_batch().summary()
        assert "6 total" in text
        assert "3 succeeded" in text
        assert "3 failed" in text
        assert "(1 replayed from artifact)" in text
        # Termination metadata surfaces for guardrail stops.
        assert "fail_fast" in text
        assert "too many validation failures" in text
        assert "category=validation" in text
        assert "item=bad" in text
        # Retries: ok_2 has 2 extra tries, rate_limited has 1.
        assert "3 extra attempt(s) across 2 item(s)" in text
        # Failure grouping, sorted by category.
        assert "rate_limit" in text
        assert "validation" in text
        assert "uncategorized" in text
        assert "mystery" in text
        # Percentile lines exist.
        assert "admission wait" in text
        assert "p50" in text and "p95" in text and "p99" in text
        assert "execution" in text
        assert "12.3s" in text

    def test_replay_tokens_reported_separately(self):
        text = self.make_batch().summary()
        # Current-run tokens (5 items × 100/10/20) exclude the replayed
        # item's 500/100/50.
        assert "Tokens:    in 500 (cached 50) · out 100" in text
        assert "Replayed:  in 500 (cached 100) · out 50" in text

    def test_wall_time_unavailable(self):
        batch = BatchResult(results=[make_result("a")])
        assert "Wall time: n/a" in batch.summary()

    def test_completed_termination_has_no_detail_suffix(self):
        batch = BatchResult(results=[make_result("a")])
        assert "Stopped:   completed\n" in batch.summary() + "\n"

    def test_summary_on_empty_batch(self):
        text = BatchResult(results=[]).summary()
        assert "0 total" in text

    def test_summary_is_printable(self, capsys):
        print(self.make_batch().summary())
        captured = capsys.readouterr()
        assert "Batch summary" in captured.out


class TestSummaryAfterRoundTrip:
    def test_dict_round_trip(self):
        batch = TestSummary().make_batch()
        restored = BatchResult.from_dict(batch.to_dict())
        assert restored.wall_time_seconds == pytest.approx(12.34)
        assert restored.summary() == batch.summary()

    def test_json_round_trip(self):
        batch = TestSummary().make_batch()
        restored = BatchResult.from_json(batch.to_json())
        assert restored.summary() == batch.summary()

    def test_jsonl_round_trip(self, tmp_path):
        batch = TestSummary().make_batch()
        path = tmp_path / "run.jsonl"
        batch.to_jsonl(path)
        restored = BatchResult.from_jsonl(path)
        assert restored.wall_time_seconds == pytest.approx(12.34)
        assert restored.summary() == batch.summary()

    def test_jsonl_round_trip_empty_batch(self, tmp_path):
        batch = BatchResult(results=[], wall_time_seconds=1.5)
        path = tmp_path / "empty.jsonl"
        batch.to_jsonl(path)
        restored = BatchResult.from_jsonl(path)
        assert restored.wall_time_seconds == pytest.approx(1.5)

    def test_legacy_records_without_wall_time_decode_to_none(self):
        batch = BatchResult(results=[make_result("a")])
        data = batch.to_dict()
        del data["wall_time_seconds"]
        restored = BatchResult.from_dict(data)
        assert restored.wall_time_seconds is None
        assert "Wall time: n/a" in restored.summary()


class TestWallTimeStamping:
    async def test_process_prompts_stamps_wall_time(self):
        class EchoStrategy(LLMCallStrategy[str]):
            async def execute(self, prompt, attempt, timeout, state=None):
                tokens = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
                return prompt.upper(), tokens, None

        batch = await process_prompts(EchoStrategy(), ["a", "b"])
        assert batch.wall_time_seconds is not None
        assert batch.wall_time_seconds >= 0.0
        assert list(batch.outputs()) == sorted(["A", "B"]) or set(batch.outputs()) == {"A", "B"}
        assert "2 total" in batch.summary()

    def test_in_input_order_preserves_wall_time(self):
        batch = BatchResult(
            results=[make_result("b", index=1), make_result("a", index=0)],
            wall_time_seconds=3.0,
        )
        ordered = batch.in_input_order()
        assert ordered.wall_time_seconds == 3.0
