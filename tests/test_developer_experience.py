"""User-facing diagnostics and validation across execution surfaces."""

from __future__ import annotations

import logging
import traceback

import pytest

from async_batch_llm import CallableStrategy, call, call_result, process_prompts


async def user_function_with_bug(prompt, *, attempt, timeout, state):
    raise ValueError("systematic user bug")


@pytest.mark.asyncio
async def test_batch_logs_user_frame_before_detaching(caplog):
    with caplog.at_level(logging.ERROR):
        batch = await process_prompts(CallableStrategy(user_function_with_bug), ["one"])
    assert "user_function_with_bug" in caplog.text
    assert batch.results[0].exception.__traceback__ is None


@pytest.mark.asyncio
async def test_call_preserves_user_frame():
    with pytest.raises(ValueError) as caught:
        await call(CallableStrategy(user_function_with_bug), "one")
    assert "user_function_with_bug" in {
        frame.name for frame in traceback.extract_tb(caught.value.__traceback__)
    }


@pytest.mark.asyncio
async def test_call_result_preserves_user_frame():
    result = await call_result(CallableStrategy(user_function_with_bug), "one")
    assert "user_function_with_bug" in {
        frame.name for frame in traceback.extract_tb(result.exception.__traceback__)
    }


def test_callable_rejects_signature_before_execution():
    async def missing_keywords(prompt):
        return prompt

    with pytest.raises(TypeError, match="attempt"):
        CallableStrategy(missing_keywords)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["batch", "stream"])
async def test_repeated_bug_has_one_diagnostic_and_count_summary(caplog, surface):
    with caplog.at_level(logging.ERROR):
        strategy = CallableStrategy(user_function_with_bug)
        if surface == "stream":
            batch = await process_prompts(strategy, [str(i) for i in range(200)])
        else:
            from async_batch_llm import LLMWorkItem, ParallelBatchProcessor

            async with ParallelBatchProcessor() as processor:
                for i in range(200):
                    await processor.add_work(LLMWorkItem(str(i), strategy, str(i)))
                batch = await processor.process_all()
    diagnostics = [record for record in caplog.records if record.exc_info is not None]
    assert len(diagnostics) == 1
    summaries = [r for r in caplog.records if "200 items failed with the same error" in r.message]
    assert len(summaries) == 1
    assert "200 items failed with the same error" in batch.summary()
    assert batch.failed == 200
    from async_batch_llm import BatchResult

    assert (
        "200 items failed with the same error" in BatchResult.from_json(batch.to_json()).summary()
    )


@pytest.mark.parametrize("field", ["latency", "failure_rate"])
@pytest.mark.parametrize("value", [True, False, "1", None, [1], float("nan")])
def test_fake_strategy_rejects_non_numbers(field, value):
    from async_batch_llm.testing import FakeStrategy

    # Same convention as numeric config fields: bool and non-numbers are rejected.
    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        FakeStrategy("ok", **{field: value}, seed=0)


@pytest.mark.asyncio
async def test_fake_strategy_schedule_and_usage():
    from async_batch_llm import ProcessorConfig
    from async_batch_llm.core import RateLimitConfig, RetryConfig
    from async_batch_llm.testing import FakeRateLimitError, FakeStrategy

    strategy = FakeStrategy(
        lambda prompt: prompt.upper(),
        failure_schedule=[FakeRateLimitError(), None],
        token_usage={"prompt_tokens": 2, "completion_tokens": 3},
    )
    result = await call_result(
        strategy,
        "hi",
        config=ProcessorConfig(
            retry=RetryConfig(max_attempts=1),
            rate_limit=RateLimitConfig(cooldown_seconds=0, slow_start_items=0),
        ),
    )
    assert result.output == "HI"
    assert result.token_usage["total_tokens"] == 5
    assert strategy.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["text", b"bytes", {"id": "prompt"}])
async def test_rejects_ambiguous_prompt_sources(source):
    with pytest.raises(TypeError, match="prompt"):
        await process_prompts(CallableStrategy(user_function_with_bug), source)


@pytest.mark.asyncio
async def test_integer_prompt_warning_once_per_call():
    from async_batch_llm.testing import FakeStrategy

    with pytest.warns(DeprecationWarning, match="Integer prompt") as warnings:
        batch = await process_prompts(FakeStrategy(lambda prompt: prompt), [1, 2, 3])
    assert len(warnings) == 1
    assert [result.output for result in batch.in_input_order().results] == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_duplicate_ids_warn_once():
    from async_batch_llm.testing import FakeStrategy

    with pytest.warns(UserWarning, match="Duplicate") as warnings:
        await process_prompts(FakeStrategy("ok"), [("id", "one"), ("id", "two"), ("id", "three")])
    assert len(warnings) == 1


def test_processor_config_keyword_suggestion():
    from async_batch_llm import ProcessorConfig

    with pytest.raises(TypeError, match="Did you mean 'max_workers'"):
        ProcessorConfig(max_worker=2)


@pytest.mark.asyncio
async def test_openai_usage_aliases_are_normalized():
    from async_batch_llm import CallOutcome

    async def invoke(prompt, **kwargs):
        return CallOutcome("ok", {"prompt_tokens": 7, "completion_tokens": 3})

    result = await call_result(CallableStrategy(invoke), "one")
    assert result.success
    assert result.token_usage == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }


@pytest.mark.asyncio
async def test_classifier_bug_has_one_user_traceback(caplog):
    from async_batch_llm.strategies import ErrorClassifier

    class BrokenClassifier(ErrorClassifier):
        def classify(self, exception):
            raise KeyError("classifier bug")

    with caplog.at_level(logging.ERROR):
        batch = await process_prompts(
            CallableStrategy(user_function_with_bug, error_classifier=BrokenClassifier()),
            [str(i) for i in range(3)],
        )
    diagnostics = [r for r in caplog.records if r.exc_info]
    assert len(diagnostics) == 1
    assert "classifier bug" in caplog.text
    assert all(r.error_category == "classifier_error" for r in batch.results)


def test_context_manager_retains_output_type(tmp_path):
    import subprocess
    import sys

    snippet = tmp_path / "typing_case.py"
    snippet.write_text("""
from typing_extensions import assert_type
from async_batch_llm import ParallelBatchProcessor, LLMWorkItem, CallableStrategy, CallOutcome
class Output: pass
async def invoke(prompt: str, **kwargs: object) -> CallOutcome[Output]:
    return CallOutcome(Output())
async def run() -> None:
    async with ParallelBatchProcessor[str, Output, None]() as processor:
        assert_type(processor, ParallelBatchProcessor[str, Output, None])
        await processor.add_work(LLMWorkItem(item_id="x", prompt="x", strategy=CallableStrategy(invoke)))
        result = await processor.process_all()
        assert_type(result.results[0].output, Output | None)
""")
    completed = subprocess.run(
        [sys.executable, "-m", "mypy", str(snippet), "--follow-imports=silent"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_processor_config_wrapper_preserves_dataclass_contracts():
    import copy
    import dataclasses
    import inspect
    import pickle

    from async_batch_llm import ProcessorConfig

    signature = inspect.signature(ProcessorConfig)
    assert "max_workers" in signature.parameters
    assert "attempt_timeout" in signature.parameters
    assert all(p.kind != p.VAR_KEYWORD for p in signature.parameters.values())
    original = ProcessorConfig(max_workers=3, attempt_timeout=9)
    for restored in (
        copy.copy(original),
        copy.deepcopy(original),
        pickle.loads(pickle.dumps(original)),
    ):
        assert restored == original
    replaced = dataclasses.replace(original, max_workers=4)
    assert replaced.max_workers == 4
    assert replaced.attempt_timeout == 9

    class Specialized(ProcessorConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    assert Specialized(max_workers=7).max_workers == 7

    class TypeFailure(ProcessorConfig):
        def __post_init__(self):
            raise TypeError("original validation TypeError")

    class ValueFailure(ProcessorConfig):
        def __post_init__(self):
            raise ValueError("original validation ValueError")

    with pytest.raises(TypeError, match="^original validation TypeError$"):
        TypeFailure(max_workers=3)
    with pytest.raises(ValueError, match="^original validation ValueError$"):
        ValueFailure(max_workers=3)


@pytest.mark.parametrize(
    "exception,category,retryable,expected",
    [
        (ValueError("bug"), "logic_error", False, True),
        (RuntimeError("unknown"), "unknown", True, True),
        (ConnectionError("network"), "connection_error", True, False),
        (TimeoutError("provider"), "api_timeout", True, False),
        (ValueError("custom"), "custom", False, True),
    ],
)
def test_terminal_traceback_policy(exception, category, retryable, expected):
    from async_batch_llm._internal.error_logging import terminal_traceback
    from async_batch_llm.strategies import ErrorInfo

    info = ErrorInfo(retryable, False, False, category)
    assert (terminal_traceback(exception, info) is not None) is expected


@pytest.mark.asyncio
async def test_pool_retains_user_frame():
    from async_batch_llm import LLMCallPool

    async with LLMCallPool(CallableStrategy(user_function_with_bug)) as pool:
        with pytest.raises(ValueError) as caught:
            await pool.submit("one")
    assert "user_function_with_bug" in {
        frame.name for frame in traceback.extract_tb(caught.value.__traceback__)
    }


def test_uninspectable_callable_keeps_runtime_validation(monkeypatch):
    import async_batch_llm.callable_strategy as module

    def unavailable(callback):
        raise ValueError("signature unavailable")

    monkeypatch.setattr(module.inspect, "signature", unavailable)
    assert isinstance(CallableStrategy(user_function_with_bug), CallableStrategy)


@pytest.mark.asyncio
async def test_mixed_usage_naming_is_rejected():
    from async_batch_llm import CallOutcome

    async def invoke(prompt, **kwargs):
        return CallOutcome("ok", {"input_tokens": 1, "completion_tokens": 2})

    result = await call_result(CallableStrategy(invoke), "one")
    assert not result.success
    assert "cannot mix" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["work_item", "call", "prompts", "stream", "pool"])
async def test_invalid_strategy_has_wrapper_hint(surface):
    from async_batch_llm import LLMCallPool, LLMWorkItem, process_stream

    with pytest.raises(TypeError, match="CallableStrategy"):
        if surface == "work_item":
            LLMWorkItem("id", object(), "prompt")
        elif surface == "call":
            await call(object(), "prompt")
        elif surface == "prompts":
            await process_prompts(object(), ["prompt"])
        elif surface == "stream":
            async for _ in process_stream(object(), ["prompt"]):
                pass
        else:
            LLMCallPool(object())


def test_terminal_log_cache_bounds_many_unique_long_failures():
    from async_batch_llm._internal.error_logging import TerminalFailureLogs
    from async_batch_llm.strategies import ErrorInfo

    target = logging.Logger("quiet-terminal-cache", level=logging.CRITICAL)
    logs = TerminalFailureLogs()
    info = ErrorInfo(False, False, False, "logic_error")
    for index in range(20_000):
        error = ValueError(f"{index}:" + "x" * 4000)
        logs.log(target, error, info, "terminal failure")
    assert len(logs._counts) <= 1000
    assert all(len(key[2]) <= 200 for key in logs._counts)
    assert sum(len(key[2]) for key in logs._counts) <= 200_000


def test_terminal_log_overflow_stays_visible_and_tracked_keys_still_deduplicate(caplog):
    from async_batch_llm._internal.error_logging import TerminalFailureLogs
    from async_batch_llm.strategies import ErrorInfo

    target = logging.getLogger("terminal-cache-overflow")
    logs = TerminalFailureLogs()
    info = ErrorInfo(False, False, False, "logic_error")
    with caplog.at_level(logging.DEBUG, logger=target.name):
        for index in range(1000):
            logs.log(target, ValueError(str(index)), info, "fill cache")
        caplog.clear()
        for _ in range(2):
            logs.log(target, ValueError("overflow"), info, "untracked error")
        logs.log(target, ValueError("0"), info, "tracked duplicate")
        logs.summarize(target)
        logs.summarize(target)
    assert len(logs._counts) == 1000
    assert [r.levelno for r in caplog.records] == [
        logging.ERROR,
        logging.ERROR,
        logging.DEBUG,
        logging.ERROR,
    ]
    assert "2 items failed with the same error: ValueError: 0" in caplog.records[-1].message
    assert all(key[2] != "overflow" for key in logs._counts)


def test_terminal_log_key_uses_bounded_message_prefix(caplog):
    from async_batch_llm._internal.error_logging import TerminalFailureLogs
    from async_batch_llm.strategies import ErrorInfo

    logs = TerminalFailureLogs()
    target = logging.getLogger("terminal-cache-prefix")
    info = ErrorInfo(False, False, False, "logic_error")
    with caplog.at_level(logging.DEBUG, logger=target.name):
        for suffix in ("first", "second"):
            logs.log(target, ValueError("x" * 200 + suffix), info, "long error")
    assert list(logs._counts.values()) == [2]
    assert [r.levelno for r in caplog.records] == [logging.ERROR, logging.DEBUG]
