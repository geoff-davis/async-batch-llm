"""Public CacheGroupTracker: recording, floor check, save/restore, pool wiring (#247)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from async_batch_llm import (
    CacheFloorBreach,
    CacheGroupTracker,
    GuardrailConfig,
    LLMCallPool,
    ProcessorConfig,
    WorkItemResult,
)
from async_batch_llm._internal.cache_groups import MAX_CACHE_GROUPS, CacheHitTracker

from .test_cache_groups import _FAST_RETRY, _NO_COOLDOWN, _Cached, _prompts


def _usage(input_tokens: int, cached: int = 0, written: int = 0) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": 1,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": written,
    }


def _entry(tracker: CacheGroupTracker, label: str | None) -> dict[str, Any]:
    return next(e for e in tracker.stats()["cache_groups"] if e["group"] == label)


def _roundtrip(tracker: CacheGroupTracker) -> CacheGroupTracker:
    return CacheGroupTracker.from_dict(json.loads(json.dumps(tracker.to_dict())))


# ── Recording ────────────────────────────────────────────────────────────────


def test_records_measured_and_unmeasured_calls() -> None:
    tracker = CacheGroupTracker(window=2)
    tracker.record("g", _usage(100, 40, 10))
    tracker.record("g", _usage(100, 60))
    tracker.record("g", None)
    tracker.record("g", {"output_tokens": 5})
    tracker.record("g", _usage(0))
    entry = _entry(tracker, "g")
    assert entry == {
        "group": "g",
        "measured_calls": 2,
        "unmeasured_calls": 3,
        "input_tokens": 200,
        "cached_input_tokens": 100,
        "cache_write_input_tokens": 10,
        "hit_rate": 0.5,
        "last_miss_reason": None,
        "window_hit_rate": 0.5,
    }


def test_groups_use_the_metadata_rules() -> None:
    tracker = CacheGroupTracker()
    for value in (None, "", 3):
        tracker.record(value, _usage(10))
    assert _entry(tracker, None)["measured_calls"] == 3
    tracker.record("x" * 200, _usage(10))
    assert [e["group"] for e in tracker.stats()["cache_groups"]][1].startswith("sha256:")


def test_miss_reasons_are_kept_and_cut() -> None:
    tracker = CacheGroupTracker()
    tracker.record("g", _usage(10), miss_reason="input_changed")
    tracker.record("g", _usage(10, 10), miss_reason="")
    tracker.record("h", _usage(10), miss_reason="r" * 100)
    assert _entry(tracker, "g")["last_miss_reason"] == "input_changed"
    assert _entry(tracker, "h")["last_miss_reason"] == "r" * 64


def test_usage_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        CacheGroupTracker().record("g", [("input_tokens", 10)])  # type: ignore[arg-type]


@pytest.mark.parametrize("window", [0, -1, True, 2.0, None])
def test_window_must_be_a_positive_int(window: Any) -> None:
    with pytest.raises(ValueError, match="window"):
        CacheGroupTracker(window=window)


def test_reset_forgets_everything() -> None:
    tracker = CacheGroupTracker()
    tracker.record("g", _usage(10))
    tracker.reset()
    assert tracker.stats() == {"cache_groups": [], "cache_groups_overflow": None}


def test_overflow_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    tracker = CacheGroupTracker()
    with caplog.at_level(logging.WARNING):
        for i in range(MAX_CACHE_GROUPS + 3):
            tracker.record(f"g{i}", _usage(10, 5))
    assert len(tracker.stats()["cache_groups"]) == MAX_CACHE_GROUPS
    assert tracker.stats()["cache_groups_overflow"]["measured_calls"] == 3
    assert sum("cache_groups_overflow" in r.message for r in caplog.records) == 1


# ── Floor check ──────────────────────────────────────────────────────────────


def test_below_floor_waits_for_a_full_window() -> None:
    tracker = CacheGroupTracker(window=3)
    tracker.record("g", _usage(100))
    tracker.record("g", _usage(100))
    assert tracker.below_floor(0.5) == []
    tracker.record("g", _usage(100, 100))
    [breach] = tracker.below_floor(0.5)
    assert breach == CacheFloorBreach(
        group="g", window_hit_rate=pytest.approx(1 / 3), floor=0.5, window=3
    )
    assert tracker.below_floor(1 / 3) == []  # inclusive


def test_below_floor_lists_lowest_first_and_overflow_last() -> None:
    tracker = CacheGroupTracker(window=1)
    tracker.record("mid", _usage(100, 40), miss_reason="input_changed")
    tracker.record("low", _usage(100, 10))
    tracker.record(None, _usage(100, 20))
    tracker.record("ok", _usage(100, 90))
    breaches = tracker.below_floor(0.5)
    assert [b.group for b in breaches] == ["low", None, "mid"]
    assert breaches[2].last_miss_reason == "input_changed"
    assert "(ungrouped)" in breaches[1].message
    for i in range(MAX_CACHE_GROUPS):
        tracker.record(f"x{i}", _usage(100, 100))
    last = tracker.below_floor(0.5)[-1]
    assert last.overflow and last.group is None and last.window_hit_rate is None
    assert "More than 64 cache groups" in last.message


@pytest.mark.parametrize("floor", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_floor_validation(floor: Any) -> None:
    with pytest.raises(ValueError, match="floor"):
        CacheGroupTracker().below_floor(floor)


def test_processor_floor_and_public_check_agree() -> None:
    """The processor's stop reason is the public breach's message, from the same counts."""
    tripped: list[Any] = []

    async def announce(cause: Any) -> None:  # pragma: no cover - never awaited here
        pass

    internal = CacheHitTracker(
        GuardrailConfig(min_cache_hit_rate=0.6, cache_hit_window=2),
        trip_now=lambda cause: tripped.append(cause) or True,
        announce=announce,
    )
    public = CacheGroupTracker(window=2)
    calls = [(_usage(100, 100), None), (_usage(100, 10), "input_changed"), (_usage(100, 20), None)]
    cause = None
    for usage, reason in calls:
        metadata = {"cache_group": "g", "cache_miss_reason": reason}
        recorded = internal.record(item_id="i", metadata=metadata, usage=usage, known=True)
        cause = cause or recorded
        public.record("g", usage, miss_reason=reason)
    assert cause is not None
    assert internal.stats() == public.stats()
    # The internal floor tripped at the second call (rate 0.55); recheck at its state.
    replay = CacheGroupTracker(window=2)
    for usage, reason in calls[:2]:
        replay.record("g", usage, miss_reason=reason)
    assert cause.reason == replay.below_floor(0.6)[0].message


# ── Save and restore ─────────────────────────────────────────────────────────


def test_round_trip_restores_exact_state_and_window() -> None:
    tracker = CacheGroupTracker(window=3)
    for cached in (100, 100, 0, 100):
        tracker.record("g", _usage(100, cached, 5), miss_reason="input_changed")
    tracker.record(None, None)
    restored = _roundtrip(tracker)
    assert restored.window == 3
    assert restored.stats() == tracker.stats()
    assert restored.to_dict() == tracker.to_dict()
    # The restored window keeps sliding exactly as the original's does.
    for t in (tracker, restored):
        t.record("g", _usage(100, 0))
        t.record("g", _usage(100, 0))
    assert restored.stats() == tracker.stats()
    assert restored.below_floor(0.5) == tracker.below_floor(0.5) != []


def test_round_trip_keeps_overflow_without_warning_again(caplog: pytest.LogCaptureFixture) -> None:
    tracker = CacheGroupTracker()
    for i in range(MAX_CACHE_GROUPS + 1):
        tracker.record(f"g{i}", _usage(10, 1))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        restored = _roundtrip(tracker)
        restored.record("another", _usage(10))
    assert not caplog.records
    assert restored.stats()["cache_groups_overflow"]["measured_calls"] == 2
    assert restored.below_floor(0.0)[-1].overflow


def _state(**changes: Any) -> dict[str, Any]:
    tracker = CacheGroupTracker(window=2)
    tracker.record("g", _usage(10, 5))
    state = tracker.to_dict()
    group_changes = changes.pop("group", None)
    if group_changes:
        state["groups"][0].update(group_changes)
    state.update(changes)
    return state


@pytest.mark.parametrize(
    ("state", "match"),
    [
        ([], "mapping"),
        (_state(format="other"), "not a CacheGroupTracker"),
        (_state(version=2), "version"),
        (_state(window=0), "window"),
        (_state(groups={}), "groups"),
        (_state(group={"group": ""}), "group must be"),
        (_state(group={"group": "x" * 129}), "group must be"),
        (_state(group={"measured_calls": -1}), "measured_calls"),
        (_state(group={"input_tokens": True}), "input_tokens"),
        (_state(group={"last_miss_reason": "r" * 65}), "last_miss_reason"),
        (_state(group={"recent": [[1, 0], [1, 0], [1, 0]]}), "last 1 measured"),
        (_state(group={"recent": [[10, 5], [10, 5]]}), "last 1 measured"),
        (_state(group={"recent": []}), "last 1 measured"),
        (_state(group={"recent": [[9, 5]]}), "doesn't match input_tokens"),
        (_state(group={"recent": [[10, 4]]}), "doesn't match cached_input_tokens"),
        (
            _state(group={"measured_calls": 0, "unmeasured_calls": 1, "recent": []}),
            "no measured calls",
        ),
        (
            _state(
                group={
                    "measured_calls": 0,
                    "unmeasured_calls": 1,
                    "cached_input_tokens": 0,
                    "recent": [],
                }
            ),
            "no measured calls",
        ),
        (_state(group={"measured_calls": 0, "unmeasured_calls": 0, "recent": []}), "at least one"),
        (_state(group={"input_tokens": 1, "measured_calls": 2}), "less than measured_calls"),
        (_state(group={"recent": [[0, 0]]}), "recent entries"),
        (_state(group={"recent": [[1, -1]]}), "recent entries"),
        (_state(group={"recent": [(1, 0)]}), "recent entries"),
        (_state(overflow={"measured_calls": 0}), "overflow is only possible"),
    ],
)
def test_from_dict_rejects_bad_state(state: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CacheGroupTracker.from_dict(state)


def _two_call_state() -> dict[str, Any]:
    """Window 2, two uncached 100-token calls: below a 0.8 floor."""
    tracker = CacheGroupTracker(window=2)
    tracker.record("g", _usage(100))
    tracker.record("g", _usage(100))
    assert tracker.below_floor(0.8)
    return tracker.to_dict()


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        # A shortened window would silently restart the group's warm-up.
        ({"recent": []}, "last 2 measured"),
        ({"recent": [[100, 0]]}, "last 2 measured"),
        ({"input_tokens": 1}, "less than measured_calls"),
        ({"input_tokens": 150}, "doesn't match input_tokens"),
        ({"input_tokens": 300}, "doesn't match input_tokens"),
        ({"cached_input_tokens": 5}, "doesn't match cached_input_tokens"),
        ({"measured_calls": 0, "unmeasured_calls": 1, "recent": []}, "no measured calls"),
    ],
)
def test_from_dict_rejects_states_record_cannot_produce(
    changes: dict[str, Any], match: str
) -> None:
    state = _two_call_state()
    state["groups"][0].update(changes)
    with pytest.raises(ValueError, match=match):
        CacheGroupTracker.from_dict(state)


def test_partial_window_totals_may_exceed_the_window() -> None:
    tracker = CacheGroupTracker(window=2)
    for cached in (0, 50, 100):
        tracker.record("g", _usage(100, cached))
    state = tracker.to_dict()
    assert CacheGroupTracker.from_dict(state).stats() == tracker.stats()
    # The call that left the window reported at least one input token.
    for total, valid in ((199, False), (200, False), (201, True), (300, True)):
        state["groups"][0]["input_tokens"] = total
        if valid:
            CacheGroupTracker.from_dict(state)
        else:
            with pytest.raises(ValueError, match="doesn't match input_tokens"):
                CacheGroupTracker.from_dict(state)


def _overflow_state(**overflow: Any) -> dict[str, Any]:
    tracker = CacheGroupTracker()
    for i in range(MAX_CACHE_GROUPS + 1):
        tracker.record(f"g{i}", _usage(10, 1))
    state = tracker.to_dict()
    state["overflow"].update(overflow)
    return state


@pytest.mark.parametrize(
    ("overflow", "match"),
    [
        ({"measured_calls": 0, "unmeasured_calls": 1}, "no measured calls"),
        (
            {
                "measured_calls": 0,
                "unmeasured_calls": 1,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
            },
            "no measured calls",
        ),
        (
            {"measured_calls": 0, "cached_input_tokens": 0, "cache_write_input_tokens": 0},
            "at least one",
        ),
        ({"input_tokens": 0}, "less than measured_calls"),
    ],
)
def test_from_dict_checks_overflow_consistency(overflow: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CacheGroupTracker.from_dict(_overflow_state(**overflow))


def test_from_dict_rejects_duplicate_groups() -> None:
    state = _state()
    state["groups"].append(dict(state["groups"][0]))
    with pytest.raises(ValueError, match="more than once"):
        CacheGroupTracker.from_dict(state)


# ── Rebuilding from saved results ────────────────────────────────────────────


def test_record_result_counts_successes_only() -> None:
    tracker = CacheGroupTracker()
    ok = WorkItemResult(
        item_id="a",
        success=True,
        token_usage={
            "input_tokens": 100,
            "output_tokens": 1,
            "total_tokens": 101,
            "cached_input_tokens": 80,
        },
        metadata={"cache_group": "g", "cache_miss_reason": "input_changed"},
    )
    failed = WorkItemResult(
        item_id="b",
        success=False,
        token_usage={"input_tokens": 100, "output_tokens": 0, "total_tokens": 100},
        metadata={"cache_group": "g"},
    )
    no_metadata = WorkItemResult(item_id="c", success=True)  # default usage: zero input
    for result in (ok, failed, no_metadata):
        tracker.record_result(result)
    entry = _entry(tracker, "g")
    assert (entry["measured_calls"], entry["cached_input_tokens"]) == (1, 80)
    assert entry["last_miss_reason"] == "input_changed"
    assert _entry(tracker, None)["unmeasured_calls"] == 1


# ── Pool wiring ──────────────────────────────────────────────────────────────


def _pool_config(window: int = 3) -> ProcessorConfig:
    return ProcessorConfig(
        max_workers=2,
        retry=_FAST_RETRY,
        rate_limit=_NO_COOLDOWN,
        guardrails=GuardrailConfig(cache_hit_window=window),
    )


@pytest.mark.asyncio
async def test_pool_exposes_its_tracker_and_continues_a_restored_one() -> None:
    async with LLMCallPool(_Cached(), config=_pool_config()) as pool:
        assert pool.cache_tracker.window == 3
        await asyncio.gather(*(pool.submit_result(p) for p in _prompts("a", 90, 100, 2)))
        assert pool.cache_tracker.stats() == pool.cache_stats()
        saved = json.dumps(pool.cache_tracker.to_dict())

    restored = CacheGroupTracker.from_dict(json.loads(saved))
    async with LLMCallPool(
        _Cached(), config=_pool_config(window=10), cache_tracker=restored
    ) as pool:
        assert pool.cache_tracker is restored
        await pool.submit_result(_prompts("a", 0, 100, 1, start=2)[0])
        entry = pool.cache_stats()["cache_groups"][0]
        # Three calls across both pools, and the restored tracker's window (3) applies.
        assert entry["measured_calls"] == 3
        assert entry["window_hit_rate"] == pytest.approx(180 / 300)
        assert pool.cache_tracker.below_floor(0.7)[0].group == "a"
        pool.reset_cache_stats()
        assert restored.stats()["cache_groups"] == []


def test_pool_rejects_a_non_tracker() -> None:
    with pytest.raises(TypeError, match="CacheGroupTracker"):
        LLMCallPool(_Cached(), cache_tracker={"window": 3})  # type: ignore[arg-type]
