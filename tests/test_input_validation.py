"""Input validation tests for LLMWorkItem and ProcessorConfig.

These tests lock in helpful error messages at construction time so that
invalid inputs fail fast with user-friendly errors, rather than surfacing
deep inside the worker loop.
"""

import pytest

from async_batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RateLimitConfig,
    RetryConfig,
)
from async_batch_llm.llm_strategies import LLMCallStrategy


class _DummyStrategy(LLMCallStrategy[str]):
    """Minimal valid strategy for constructing work items in tests."""

    async def execute(self, prompt, attempt, timeout, state=None):  # type: ignore[override]
        return "ok", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


# ─── LLMWorkItem validation ───────────────────────────────────────────


class TestLLMWorkItemValidation:
    def test_strategy_none_rejected(self):
        with pytest.raises(ValueError, match="strategy"):
            LLMWorkItem(item_id="a", strategy=None, prompt="hi")  # type: ignore[arg-type]

    def test_prompt_wrong_type_rejected(self):
        with pytest.raises(TypeError, match="prompt"):
            LLMWorkItem(item_id="a", strategy=_DummyStrategy(), prompt=123)  # type: ignore[arg-type]

    def test_empty_prompt_allowed(self):
        # Empty prompt is the default and sometimes legitimate — must not raise.
        item = LLMWorkItem(item_id="a", strategy=_DummyStrategy())
        assert item.prompt == ""

    def test_valid_construction(self):
        item = LLMWorkItem(item_id="a", strategy=_DummyStrategy(), prompt="hello")
        assert item.item_id == "a"
        assert item.prompt == "hello"

    def test_existing_item_id_validation_preserved(self):
        # Regression — empty / whitespace item_id should still be rejected.
        with pytest.raises(ValueError, match="item_id"):
            LLMWorkItem(item_id="", strategy=_DummyStrategy())
        with pytest.raises(ValueError, match="item_id"):
            LLMWorkItem(item_id="   ", strategy=_DummyStrategy())

    @pytest.mark.asyncio
    async def test_same_work_item_instance_cannot_be_submitted_twice(self):
        processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig(max_workers=1))
        item = LLMWorkItem(item_id="same", strategy=_DummyStrategy(), prompt="hello")

        await processor.add_work(item)
        with pytest.raises(ValueError, match="same LLMWorkItem instance"):
            await processor.add_work(item)

        result = await processor.process_all()
        await processor.cleanup()
        assert result.total_items == 1
        assert result.results[0].submission_index == 0


# ─── ProcessorConfig validation ────────────────────────────────────────


class TestProcessorConfigValidation:
    def test_max_workers_zero_rejected(self):
        with pytest.raises(ValueError, match="max_workers"):
            ProcessorConfig(max_workers=0)

    def test_max_workers_negative_rejected(self):
        with pytest.raises(ValueError, match="max_workers"):
            ProcessorConfig(max_workers=-1)

    def test_attempt_timeout_zero_rejected(self):
        with pytest.raises(ValueError, match="attempt_timeout"):
            ProcessorConfig(attempt_timeout=0)

    def test_attempt_timeout_negative_rejected(self):
        with pytest.raises(ValueError, match="attempt_timeout"):
            ProcessorConfig(attempt_timeout=-5.0)

    def test_post_processor_timeout_zero_rejected(self):
        with pytest.raises(ValueError, match="post_processor_timeout"):
            ProcessorConfig(post_processor_timeout=0)

    def test_post_processor_timeout_negative_rejected(self):
        with pytest.raises(ValueError, match="post_processor_timeout"):
            ProcessorConfig(post_processor_timeout=-1.0)

    def test_valid_defaults_construct_cleanly(self):
        config = ProcessorConfig()
        assert config.max_workers == 5
        assert config.post_processor_timeout == 90.0

    def test_valid_custom_values_construct_cleanly(self):
        config = ProcessorConfig(
            max_workers=10,
            attempt_timeout=60.0,
            post_processor_timeout=30.0,
        )
        assert config.max_workers == 10
        assert config.post_processor_timeout == 30.0


class TestNestedConfigValidation:
    """Spot-check nested RetryConfig / RateLimitConfig already validate."""

    def test_retry_max_attempts_zero_rejected(self):
        with pytest.raises(ValueError, match="max_attempts"):
            RetryConfig(max_attempts=0)

    def test_rate_limit_cooldown_negative_rejected(self):
        with pytest.raises(ValueError, match="cooldown_seconds"):
            RateLimitConfig(cooldown_seconds=-1.0)
