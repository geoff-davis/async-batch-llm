"""Total token/cost budget guardrail (issue #183)."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import math
import warnings
from typing import Any

import pytest

from async_batch_llm import (
    AbortMode,
    AttemptUsage,
    BatchBudgetExceeded,
    GuardrailConfig,
    JsonlArtifactStore,
    LLMCallPool,
    LLMCallStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    RateLimitConfig,
    ResumePolicy,
    RetryConfig,
    SqliteArtifactStore,
    TokenEstimate,
    call,
    call_result,
    process_prompts,
    process_stream,
)
from async_batch_llm.base import LLMWorkItem, RetryState, TokenUsage
from async_batch_llm.observers import BaseObserver, ProcessingEvent
from async_batch_llm.serialization import batch_result_from_dict, batch_result_to_dict


def _usage(total: int) -> TokenUsage:
    output = min(total, 1)
    return {"input_tokens": total - output, "output_tokens": output, "total_tokens": total}


class _Fixed(LLMCallStrategy[str]):
    """Succeeds with a fixed usage per prompt (default 3 tokens)."""

    def __init__(self, tokens: dict[str, int] | None = None, delay: float = 0.0) -> None:
        self.tokens = tokens or {}
        self.delay = delay
        self.calls: list[str] = []
        self.cancelled = 0

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls.append(prompt)
        delay = self.delay if prompt != "fast" else 0.0
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return prompt, _usage(self.tokens.get(prompt, 3)), None


class _FailWithUsage(Exception):
    def __init__(self, total: int) -> None:
        super().__init__("transient")
        self._failed_token_usage = dict(_usage(total))


class _AlwaysFails(LLMCallStrategy[str]):
    def __init__(self, total: int = 3) -> None:
        self.total = total
        self.calls = 0

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls += 1
        raise _FailWithUsage(self.total)


class _Recorder(BaseObserver):
    def __init__(self) -> None:
        self.events: list[tuple[ProcessingEvent, dict[str, Any]]] = []

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        self.events.append((event, dict(data)))


_FAST_RETRY = RetryConfig(max_attempts=5, initial_wait=0.001, max_wait=0.001, jitter=False)
_NO_COOLDOWN = RateLimitConfig(
    cooldown_seconds=0,
    max_cooldown_seconds=0,
    slow_start_items=0,
    slow_start_initial_delay=0,
    slow_start_final_delay=0,
    backoff_multiplier=1,
)


def _config(workers: int = 1, **guardrails: Any) -> ProcessorConfig:
    return ProcessorConfig(
        max_workers=workers,
        retry=_FAST_RETRY,
        rate_limit=_NO_COOLDOWN,
        guardrails=GuardrailConfig(**guardrails),
    )


async def _run_processor(
    strategy: LLMCallStrategy[str],
    prompts: list[str],
    config: ProcessorConfig,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    async with ParallelBatchProcessor[str, str, None](config=config, **kwargs) as processor:
        for prompt in prompts:
            await processor.add_work(LLMWorkItem(item_id=prompt, strategy=strategy, prompt=prompt))
        result = await processor.process_all()
        stats = await processor.get_stats()
    return result, stats


# ── Token cap ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_crossing_attempt_overshoot_single_worker() -> None:
    """D183-1: cap 10, one call reporting 100 overshoots by 90 with nothing else in flight."""
    strategy = _Fixed({"a": 100})
    result, stats = await _run_processor(strategy, ["a", "b", "c"], _config(max_total_tokens=10))

    by_id = {item.item_id: item for item in result.results}
    assert strategy.calls == ["a"]
    assert by_id["a"].success
    assert by_id["b"].error_category == "batch_budget_exceeded"
    assert by_id["c"].error_category == "batch_budget_exceeded"
    assert isinstance(by_id["b"].exception, BatchBudgetExceeded)
    assert result.termination.kind == "budget_exceeded"
    assert result.termination.error_category == "batch_budget_exceeded"
    assert result.termination.triggering_item_id == "a"
    assert "100 of 10 tokens" in (result.termination.reason or "")
    assert stats["budget_tokens_used"] == 100
    assert stats["aborted"] == 2


@pytest.mark.asyncio
async def test_exact_cap_trips_and_below_cap_does_not() -> None:
    strategy = _Fixed()
    result, stats = await _run_processor(strategy, ["a", "b", "c"], _config(max_total_tokens=6))
    assert strategy.calls == ["a", "b"]
    assert result.termination.kind == "budget_exceeded"
    assert stats["budget_tokens_used"] == 6

    strategy = _Fixed()
    result, stats = await _run_processor(strategy, ["a", "b", "c"], _config(max_total_tokens=10))
    assert result.succeeded == 3
    assert result.termination.kind == "completed"
    assert stats["budget_tokens_used"] == 9


@pytest.mark.asyncio
async def test_failed_attempts_and_retries_count_and_stop_the_retry() -> None:
    strategy = _AlwaysFails(total=3)
    result, stats = await _run_processor(strategy, ["a"], _config(max_total_tokens=6))

    (item,) = result.results
    assert strategy.calls == 2  # the second attempt crosses the cap; no third starts
    assert item.error_category == "batch_budget_exceeded"
    # The original provider failure stays reachable for diagnosis.
    assert isinstance(item.exception.__context__, _FailWithUsage)
    assert item.token_usage["total_tokens"] == 6
    assert stats["budget_tokens_used"] == 6


@pytest.mark.asyncio
async def test_unknown_and_zero_usage() -> None:
    class NoUsage(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            return prompt, {}, None

    result, stats = await _run_processor(NoUsage(), ["a", "b"], _config(max_total_tokens=1))
    assert result.succeeded == 2
    assert stats["budget_tokens_used"] == 0
    assert stats["budget_unknown_usage_attempts"] == 2

    zero = _Fixed({"a": 0, "b": 0})
    result, stats = await _run_processor(zero, ["a", "b"], _config(max_total_tokens=1))
    assert result.succeeded == 2
    assert stats["budget_unknown_usage_attempts"] == 0


@pytest.mark.asyncio
async def test_dry_run_is_not_counted() -> None:
    config = _config(max_total_tokens=1)
    config.dry_run = True
    result, stats = await _run_processor(_Fixed(), ["a", "b"], config)
    assert result.total_items == 2
    assert result.termination.kind == "completed"
    assert stats["budget_tokens_used"] == 0


@pytest.mark.asyncio
async def test_config_reuse_gets_a_fresh_budget() -> None:
    config = _config(max_total_tokens=6)
    for _ in range(2):
        strategy = _Fixed()
        result, stats = await _run_processor(strategy, ["a", "b", "c"], config)
        assert strategy.calls == ["a", "b"]
        assert stats["budget_tokens_used"] == 6


# ── Abort modes and event order ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_active_counts_draining_call_and_cancel_active_cancels_it() -> None:
    drain = _Fixed({"fast": 100}, delay=0.05)
    result, stats = await _run_processor(
        drain,
        ["fast", "slow"],
        _config(workers=2, max_total_tokens=10, abort_mode=AbortMode.DRAIN_ACTIVE),
    )
    by_id = {item.item_id: item for item in result.results}
    assert by_id["fast"].success and by_id["slow"].success
    assert stats["budget_tokens_used"] == 103  # draining usage still recorded, no second trip

    cancel = _Fixed({"fast": 100}, delay=1.0)
    result, stats = await asyncio.wait_for(
        _run_processor(
            cancel,
            ["fast", "slow"],
            _config(workers=2, max_total_tokens=10, abort_mode=AbortMode.CANCEL_ACTIVE),
        ),
        timeout=0.5,
    )
    by_id = {item.item_id: item for item in result.results}
    assert by_id["fast"].success
    assert by_id["slow"].error_category == "batch_budget_exceeded"
    assert cancel.cancelled == 1


@pytest.mark.asyncio
async def test_simultaneous_crossings_trip_once() -> None:
    release = asyncio.Event()
    started = 0

    class Together(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            nonlocal started
            started += 1
            if started == 3:
                release.set()
            await release.wait()
            return prompt, _usage(50), None

    recorder = _Recorder()
    result, stats = await _run_processor(
        Together(),
        ["a", "b", "c", "d"],
        _config(workers=3, max_total_tokens=10),
        observers=[recorder],
    )
    aborted = [data for event, data in recorder.events if event is ProcessingEvent.BATCH_ABORTED]
    assert len(aborted) == 1
    assert aborted[0]["kind"] == "budget_exceeded"
    assert stats["budget_tokens_used"] == 150  # at most max_workers started attempts
    assert {item.item_id for item in result.results if item.success} == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_budget_event_order_announces_before_crossing_item_completes() -> None:
    recorder = _Recorder()
    await _run_processor(
        _Fixed({"a": 100}), ["a", "b"], _config(max_total_tokens=10), observers=[recorder]
    )
    order = [
        (event, data.get("item_id"))
        for event, data in recorder.events
        if event in {ProcessingEvent.BATCH_ABORTED, ProcessingEvent.ITEM_COMPLETED}
    ]
    assert order[:2] == [
        (ProcessingEvent.BATCH_ABORTED, None),
        (ProcessingEvent.ITEM_COMPLETED, "a"),
    ]


@pytest.mark.asyncio
async def test_first_cause_wins_against_fail_fast() -> None:
    class Auth(Exception):
        pass

    class Classifier:
        def classify(self, exception):
            from async_batch_llm import ErrorInfo

            return ErrorInfo(False, False, False, "authentication")

    class FailsWithUsage(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            error = Auth("nope")
            error._failed_token_usage = dict(_usage(100))  # type: ignore[attr-defined]
            raise error

    result, _ = await _run_processor(
        FailsWithUsage(),
        ["a", "b"],
        _config(max_total_tokens=10, abort_on_error_categories=frozenset({"authentication"})),
        error_classifier=Classifier(),
    )
    # The budget trips inside the attempt, before the terminal failure could fail fast.
    assert result.termination.kind == "budget_exceeded"


# ── Cost function ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cost_cap_and_attempt_usage_snapshot() -> None:
    seen: list[AttemptUsage] = []

    def cost(usage: AttemptUsage) -> float:
        seen.append(usage)
        with pytest.raises(TypeError):
            usage.usage["total_tokens"] = 0  # type: ignore[index]
        return usage.usage["total_tokens"] * 0.5

    strategy = _Fixed()
    result, stats = await _run_processor(
        strategy, ["a", "b", "c"], _config(max_total_cost=3.0, cost_function=cost)
    )
    assert strategy.calls == ["a", "b"]
    assert result.termination.kind == "budget_exceeded"
    assert "Cost budget reached" in (result.termination.reason or "")
    assert stats["budget_cost_used"] == 3.0
    assert stats["budget_cost_complete"] is True
    assert seen[0].item_id == "a" and seen[0].attempt == 1 and seen[0].try_number == 1
    assert seen[0].strategy is strategy and seen[0].success
    # Mutating the snapshot is impossible, and results keep their own usage.
    assert result.results[0].token_usage["total_tokens"] == 3


@pytest.mark.asyncio
async def test_cost_tracking_without_cap() -> None:
    result, stats = await _run_processor(
        _Fixed(), ["a", "b"], _config(cost_function=lambda usage: 0.25)
    )
    assert result.termination.kind == "completed"
    assert stats["budget_cost_used"] == 0.5


@pytest.mark.asyncio
async def test_simultaneous_token_and_cost_crossing_reports_tokens() -> None:
    result, _ = await _run_processor(
        _Fixed(),
        ["a", "b"],
        _config(max_total_tokens=3, max_total_cost=1.0, cost_function=lambda usage: 1.0),
    )
    assert "Token budget reached" in (result.termination.reason or "")


@pytest.mark.asyncio
async def test_cost_function_failure_fails_closed(caplog: pytest.LogCaptureFixture) -> None:
    def cost(usage: AttemptUsage) -> float:
        raise RuntimeError("secret pricing detail")

    strategy = _Fixed()
    result, stats = await _run_processor(
        strategy, ["a", "b"], _config(max_total_cost=100.0, cost_function=cost)
    )
    assert strategy.calls == ["a"]
    assert result.results[0].success
    assert result.termination.kind == "budget_exceeded"
    assert "cost function failed (RuntimeError)" in (result.termination.reason or "")
    assert stats["budget_tokens_used"] == 3
    assert stats["budget_cost_complete"] is False
    assert "secret pricing detail" not in caplog.text


@pytest.mark.parametrize(
    ("cost", "cap"),
    [(10**1000, 1.0), (10**1000, None), (1e308, None), (1e308, 1.7e308)],
    ids=["huge-int-capped", "huge-int-tracking", "sum-overflow-tracking", "sum-overflow-capped"],
)
@pytest.mark.asyncio
async def test_cost_overflow_fails_closed_without_retrying(cost: float, cap: float | None) -> None:
    """B183-1: conversion or running-sum overflow stops the run; no provider retries."""
    guardrails: dict[str, Any] = {"cost_function": lambda usage: cost}
    if cap is not None:
        guardrails["max_total_cost"] = cap
    strategy = _Fixed()
    result, stats = await _run_processor(strategy, ["a", "b", "c"], _config(**guardrails))

    assert result.termination.kind == "budget_exceeded"
    assert "invalid value" in (result.termination.reason or "")
    assert stats["budget_cost_complete"] is False
    assert math.isfinite(stats["budget_cost_used"])
    assert stats["budget_tokens_used"] == 3 * len(strategy.calls)
    # The first call succeeded or the second one tripped; nothing was retried.
    assert len(strategy.calls) == len(set(strategy.calls)) <= 2
    assert result.results[0].success


async def _coroutine_cost() -> float:
    return 1.0


@pytest.mark.parametrize(
    "value",
    [True, math.nan, math.inf, -1.0, "1", None],
    ids=["bool", "nan", "inf", "negative", "str", "none"],
)
@pytest.mark.asyncio
async def test_cost_function_invalid_return_fails_closed(value: object) -> None:
    result, stats = await _run_processor(
        _Fixed(), ["a", "b"], _config(cost_function=lambda usage: value)
    )
    assert result.termination.kind == "budget_exceeded"
    assert "invalid value" in (result.termination.reason or "")
    assert stats["budget_cost_complete"] is False


@pytest.mark.asyncio
async def test_cost_function_returning_awaitable_fails_closed_without_warning() -> None:
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result, _ = await _run_processor(
            _Fixed(), ["a", "b"], _config(cost_function=lambda usage: _coroutine_cost())
        )
        gc.collect()
    assert result.termination.kind == "budget_exceeded"
    assert "invalid value" in (result.termination.reason or "")
    assert not [r for r in records if "never awaited" in str(r.message)]


# ── Reconciliation ordering (Codex constraint 1) ─────────────────────────────


def _tpm_config(**guardrails: Any) -> ProcessorConfig:
    return ProcessorConfig(
        max_workers=1,
        max_tokens_per_minute=1_000,
        token_estimator=lambda prompt, **kwargs: TokenEstimate(20),
        retry=_FAST_RETRY,
        rate_limit=_NO_COOLDOWN,
        guardrails=GuardrailConfig(**guardrails),
    )


def _reconciled_known(recorder: _Recorder) -> list[dict[str, Any]]:
    return [
        data
        for event, data in recorder.events
        if event is ProcessingEvent.QUOTA_RECONCILED and data["known_usage"]
    ]


@pytest.mark.asyncio
async def test_cost_callback_cancellation_still_reconciles_known_usage() -> None:
    def cost(usage: AttemptUsage) -> float:
        raise asyncio.CancelledError

    recorder = _Recorder()
    with pytest.raises(BaseException):  # noqa: B017 - propagation is the contract
        await asyncio.wait_for(
            _run_processor(
                _Fixed(),
                ["a"],
                _tpm_config(cost_function=cost),
                observers=[recorder],
            ),
            timeout=2,
        )
    reconciled = _reconciled_known(recorder)
    assert [data["reported_tokens"] for data in reconciled] == [3]


@pytest.mark.asyncio
async def test_cancellation_in_abort_observer_still_reconciles_known_usage() -> None:
    class CancelOnAbort(_Recorder):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
            await super().on_event(event, data)
            if event is ProcessingEvent.BATCH_ABORTED:
                raise asyncio.CancelledError

    recorder = CancelOnAbort()
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(
            _run_processor(
                _Fixed({"a": 100}),
                ["a", "b"],
                _tpm_config(max_total_tokens=10),
                observers=[recorder],
            ),
            timeout=2,
        )
    events = [event for event, _ in recorder.events]
    reconciled = _reconciled_known(recorder)
    assert [data["reported_tokens"] for data in reconciled] == [100]
    # Reconciliation happened before the announcement that was cancelled.
    assert events.index(ProcessingEvent.QUOTA_RECONCILED) < events.index(
        ProcessingEvent.BATCH_ABORTED
    )


# ── Surfaces ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_prompts_and_process_stream_enforce_the_budget() -> None:
    batch = await process_prompts(
        _Fixed({"a": 100}), ["a", "b"], config=_config(max_total_tokens=10), preserve_order=True
    )
    assert batch.results[1].error_category == "batch_budget_exceeded"
    assert batch.termination.kind == "budget_exceeded"

    streamed = [
        item
        async for item in process_stream(
            _Fixed({"a": 100}), ["a", "b"], config=_config(max_total_tokens=10)
        )
    ]
    assert sorted(item.error_category or "ok" for item in streamed) == [
        "batch_budget_exceeded",
        "ok",
    ]


@pytest.mark.parametrize(
    "guardrails",
    [
        {"max_total_tokens": 10},
        {"max_total_cost": 1.0, "cost_function": lambda usage: 0.0},
        {"cost_function": lambda usage: 0.0},
    ],
    ids=["tokens", "cost", "tracking"],
)
@pytest.mark.asyncio
async def test_single_call_and_pool_reject_budgets(guardrails: dict[str, Any]) -> None:
    config = _config(**guardrails)
    with pytest.raises(ValueError, match="supported only by batch processor runs"):
        await call(_Fixed(), "a", config=config)
    with pytest.raises(ValueError, match="supported only by batch processor runs"):
        await call_result(_Fixed(), "a", config=config)
    with pytest.raises(ValueError, match="supported only by batch processor runs"):
        LLMCallPool(_Fixed(), config=config)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_total_tokens": 0}, "max_total_tokens"),
        ({"max_total_tokens": -5}, "max_total_tokens"),
        ({"max_total_tokens": True}, "max_total_tokens"),
        ({"max_total_tokens": 10.0}, "max_total_tokens"),
        ({"max_total_cost": 1.0}, "requires cost_function"),
        ({"max_total_cost": 0.0, "cost_function": abs}, "max_total_cost"),
        ({"max_total_cost": math.nan, "cost_function": abs}, "max_total_cost"),
        ({"max_total_cost": math.inf, "cost_function": abs}, "max_total_cost"),
        ({"max_total_cost": True, "cost_function": abs}, "max_total_cost"),
        ({"cost_function": 3}, "cost_function must be callable"),
    ],
)
def test_budget_config_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GuardrailConfig(**kwargs)


# ── Artifacts, replay and serialization ──────────────────────────────────────


@pytest.mark.parametrize("store_type", [JsonlArtifactStore, SqliteArtifactStore])
@pytest.mark.asyncio
async def test_budget_audit_records_never_replay_and_replays_are_free(
    tmp_path: Any, store_type: Any
) -> None:
    path = tmp_path / "budget.artifact"
    first = _Fixed({"a": 100})
    batch = await process_prompts(
        first,
        ["a", "b"],
        config=_config(max_total_tokens=10),
        artifact_store=store_type(path),
    )
    assert batch.termination.kind == "budget_exceeded"

    # The audit record for "b" was written but is never replayed; the replayed
    # success for "a" costs nothing, so a small budget is enough to finish "b".
    second = _Fixed({"a": 100})
    config = _config(max_total_tokens=5)
    batch = await process_prompts(
        second,
        ["a", "b"],
        config=config,
        artifact_store=store_type(path),
        resume=ResumePolicy.REUSE_ALL,
        preserve_order=True,
    )
    assert second.calls == ["b"]
    assert batch.results[0].replayed_from_artifact
    assert batch.results[1].success
    assert batch.termination.kind == "completed"


def test_budget_termination_round_trips() -> None:
    from async_batch_llm import BatchResult, BatchTermination

    result: BatchResult[str, None] = BatchResult(
        results=[],
        termination=BatchTermination(
            kind="budget_exceeded",
            reason="Token budget reached: 12 of 10 tokens used",
            error_category="batch_budget_exceeded",
            triggering_item_id="a",
        ),
    )
    restored = batch_result_from_dict(batch_result_to_dict(result))
    assert restored.termination == result.termination


@pytest.mark.asyncio
async def test_metrics_observer_matches_stats_and_crossing_item_is_postprocessed() -> None:
    from async_batch_llm import MetricsObserver

    metrics = MetricsObserver()
    processed: list[str] = []

    async def post(result: Any) -> None:
        processed.append(result.item_id)

    result, stats = await _run_processor(
        _Fixed({"a": 100}),
        ["a", "b", "c"],
        _config(max_total_tokens=10),
        observers=[metrics],
        post_processor=post,
    )
    collected = await metrics.get_metrics()
    assert stats["aborted"] == 2
    assert collected["items_aborted"] == stats["aborted"]
    assert "a" in processed  # the crossing success is published and post-processed


@pytest.mark.parametrize("store_type", [JsonlArtifactStore, SqliteArtifactStore])
@pytest.mark.asyncio
async def test_crossing_item_checkpoint_failure_stays_fatal(tmp_path: Any, store_type: Any) -> None:
    from async_batch_llm import ArtifactIOError

    store = store_type(tmp_path / "fatal.artifact")
    append = store.append

    async def fail_success_checkpoint(item: Any, key: Any, result: Any) -> None:
        if result.success:
            raise ArtifactIOError("disk full")
        await append(item, key, result)

    store.append = fail_success_checkpoint
    strategy = _Fixed({"a": 100})
    processor = ParallelBatchProcessor[str, str, None](
        config=_config(max_total_tokens=10), artifact_store=store
    )
    try:
        for prompt in ["a", "b"]:
            await processor.add_work(LLMWorkItem(item_id=prompt, strategy=strategy, prompt=prompt))
        with pytest.raises(ArtifactIOError, match="disk full"):
            await processor.process_all()
        # Existing policy: a fatal artifact error overwrites every abort cause.
        assert processor.termination.kind == "artifact_error"
    finally:
        await processor.cleanup()
