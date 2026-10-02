"""v0.28 deprecations: each warns once, from the caller's code, removed in 1.0."""

from __future__ import annotations

import dataclasses
import importlib
import subprocess
import sys
import warnings

import pytest

from async_batch_llm import (
    GuardrailConfig,
    LLMCallPool,
    ParallelBatchProcessor,
    ProcessorConfig,
    RateLimitConfig,
    RetryConfig,
    StartupRampConfig,
    process_prompts,
)
from async_batch_llm.testing import FakeStrategy


def _deprecations(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [w for w in caught if issubclass(w.category, DeprecationWarning)]


# Each config with a valid first positional value for its first field.
POSITIONAL = [
    (ProcessorConfig, 7, "max_workers"),
    (RetryConfig, 4, "max_attempts"),
    (RateLimitConfig, 30.0, "cooldown_seconds"),
    (StartupRampConfig, 2, "initial_concurrency"),
    (GuardrailConfig, 10.0, "total_timeout_per_item"),
]


@pytest.mark.parametrize(("cls", "value", "field"), POSITIONAL)
def test_positional_config_arguments_warn_once_from_caller(cls, value, field):
    config, caught = _deprecations(lambda: cls(value))
    assert getattr(config, field) == value
    assert len(caught) == 1
    message = str(caught[0].message)
    assert f"positional arguments to {cls.__name__}()" in message
    assert f"{cls.__name__}({field}=...)" in message
    assert "keyword-only" in message
    assert caught[0].filename == __file__


@pytest.mark.parametrize(("cls", "value", "field"), POSITIONAL)
def test_keyword_construction_and_replace_stay_silent(cls, value, field):
    config, caught = _deprecations(lambda: cls(**{field: value}))
    assert caught == []
    _, caught = _deprecations(lambda: dataclasses.replace(config))
    assert caught == []
    _, caught = _deprecations(cls)
    assert caught == []


def test_processor_config_keyword_typos_still_get_suggestions():
    with pytest.raises(TypeError, match="Did you mean 'max_workers'"):
        ProcessorConfig(max_worker=3)  # type: ignore[call-arg]


def test_enable_detailed_logging_true_warns_from_caller_and_keeps_its_value():
    config, caught = _deprecations(lambda: ProcessorConfig(enable_detailed_logging=True))
    assert len(caught) == 1
    assert "enable_detailed_logging=True) has no effect" in str(caught[0].message)
    assert "removed in 1.0" in str(caught[0].message)
    assert caught[0].filename == __file__
    # Deprecated, not changed: the value stays visible until 1.0 removes the field.
    assert config.enable_detailed_logging is True
    assert dataclasses.asdict(config)["enable_detailed_logging"] is True
    # A user's own copy is a new construction with True, so it warns again.
    copy, caught = _deprecations(lambda: dataclasses.replace(config, max_workers=2))
    assert len(caught) == 1
    assert copy.enable_detailed_logging is True


@pytest.mark.asyncio
async def test_library_copies_of_a_user_config_do_not_repeat_the_warning():
    # process_prompts(concurrency=...) and ParallelBatchProcessor's legacy
    # parameters both copy the caller's config with dataclasses.replace().
    config, caught = _deprecations(lambda: ProcessorConfig(enable_detailed_logging=True))
    assert len(caught) == 1

    async def run():
        result = await process_prompts(FakeStrategy("ok"), ["a"], config=config, concurrency=2)
        with pytest.warns(DeprecationWarning) as legacy:
            processor = ParallelBatchProcessor(config=config, max_workers=3)
        await processor.cleanup()
        return result, legacy

    with warnings.catch_warnings(record=True) as all_caught:
        warnings.simplefilter("always")
        result, legacy = await run()
    assert result.succeeded == 1
    for w in [*all_caught, *legacy]:
        assert "enable_detailed_logging" not in str(w.message)
    assert config.enable_detailed_logging is True


def test_enable_detailed_logging_false_is_silent():
    _, caught = _deprecations(lambda: ProcessorConfig(enable_detailed_logging=False))
    assert caught == []


def test_timeout_per_item_alias_warns_from_caller():
    # Before 0.28 this warning pointed at the keyword-checking wrapper in
    # _internal/input_validation.py instead of the caller.
    config, caught = _deprecations(lambda: ProcessorConfig(timeout_per_item=45.0))
    assert config.attempt_timeout == 45.0
    assert len(caught) == 1
    assert "timeout_per_item" in str(caught[0].message)
    assert caught[0].filename == __file__


def test_call_pool_module_and_gateway_shim():
    call_pool = importlib.import_module("async_batch_llm.call_pool")
    gateway = importlib.import_module("async_batch_llm.gateway")
    assert call_pool.LLMCallPool is LLMCallPool
    assert gateway.LLMCallPool is LLMCallPool
    assert call_pool.__all__ == ["LLMCallPool"]
    # The deprecated alias lives only on the old module path (and the package root).
    with pytest.raises(AttributeError):
        call_pool.LLMGateway  # noqa: B018


def test_importing_old_module_path_is_silent():
    code = (
        "import warnings\n"
        "warnings.simplefilter('error', DeprecationWarning)\n"
        "warnings.filterwarnings('ignore', category=DeprecationWarning, module='google')\n"
        "from async_batch_llm.gateway import LLMCallPool\n"
        "from async_batch_llm.call_pool import LLMCallPool as Current\n"
        "assert LLMCallPool is Current\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
