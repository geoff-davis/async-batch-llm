"""Tests for BatchResult ergonomics: successes/failures/by_id/estimated_cost."""

import pytest

from async_batch_llm import BatchResult, CachedTokenRates, WorkItemResult


def _result(item_id: str, *, success: bool, inp: int, out: int, cached: int = 0):
    return WorkItemResult(
        item_id=item_id,
        success=success,
        output="ok" if success else None,
        error=None if success else "boom",
        token_usage={
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
            "cached_input_tokens": cached,
        },
    )


def test_successes_and_failures_partition():
    batch = BatchResult(
        results=[
            _result("a", success=True, inp=10, out=5),
            _result("b", success=False, inp=10, out=0),
            _result("c", success=True, inp=10, out=5),
        ]
    )
    assert [r.item_id for r in batch.successes] == ["a", "c"]
    assert [r.item_id for r in batch.failures] == ["b"]
    # Preserves completion order within each group.
    assert batch.succeeded == 2
    assert batch.failed == 1


def test_by_id_lookup():
    batch = BatchResult(
        results=[
            _result("a", success=True, inp=1, out=1),
            _result("b", success=False, inp=1, out=0),
        ]
    )
    by_id = batch.by_id()
    assert set(by_id) == {"a", "b"}
    assert by_id["a"].output == "ok"
    assert by_id["b"].error == "boom"


def test_by_id_last_wins_on_duplicate():
    first = _result("dup", success=False, inp=1, out=0)
    second = _result("dup", success=True, inp=1, out=1)
    batch = BatchResult(results=[first, second])
    assert batch.by_id()["dup"] is second


def test_estimated_cost_no_cache():
    batch = BatchResult(results=[_result("a", success=True, inp=1_000_000, out=500_000)])
    # 1M input @ $2/M + 0.5M output @ $6/M = 2.0 + 3.0
    cost = batch.estimated_cost(input_per_mtok=2.0, output_per_mtok=6.0)
    assert cost == pytest.approx(5.0)


def test_estimated_cost_applies_cache_discount():
    batch = BatchResult(results=[_result("a", success=True, inp=1000, out=500, cached=800)])
    # effective input = 1000 - int(800 * 0.9) = 1000 - 720 = 280
    # cost = 280/1e6 * 10 + 500/1e6 * 20 = 0.0028 + 0.01 = 0.0128
    cost = batch.estimated_cost(10.0, 20.0, cached_token_rate=CachedTokenRates.GEMINI)
    assert cost == pytest.approx(0.0128)


def test_estimated_cost_warns_without_rate_when_cached_present():
    batch = BatchResult(results=[_result("a", success=True, inp=1000, out=500, cached=800)])
    with pytest.warns(UserWarning, match="cached_token_rate"):
        batch.estimated_cost(10.0, 20.0)
