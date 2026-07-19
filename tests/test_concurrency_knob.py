"""Tests for the unified concurrency= knob (issue #97)."""

import dataclasses
import warnings

import pytest

from async_batch_llm import (
    LLMCallStrategy,
    OpenAIModel,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    process_prompts,
)


class EchoStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        tokens = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return prompt, tokens, None


class TestConfigDerivation:
    def test_concurrency_sizes_workers_and_admission(self):
        config = ProcessorConfig(concurrency=12)
        assert config.max_workers == 12
        assert config.max_provider_concurrency == 12
        assert config.concurrency == 12

    def test_defaults_without_concurrency_unchanged(self):
        config = ProcessorConfig()
        assert config.max_workers == 5
        assert config.max_provider_concurrency is None
        assert config.concurrency is None

    def test_explicit_max_workers_wins(self):
        config = ProcessorConfig(max_workers=3, concurrency=12)
        assert config.max_workers == 3
        assert config.max_provider_concurrency == 12

    def test_explicit_admission_wins(self):
        config = ProcessorConfig(concurrency=12, max_provider_concurrency=4)
        assert config.max_workers == 12
        assert config.max_provider_concurrency == 4

    def test_concurrency_validated(self):
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            ProcessorConfig(concurrency=0)

    def test_dataclasses_replace_round_trip(self):
        config = ProcessorConfig(concurrency=12)
        clone = dataclasses.replace(config)
        assert clone.max_workers == 12
        assert clone.max_provider_concurrency == 12

    def test_positional_first_arg_is_still_max_workers(self):
        config = ProcessorConfig(7)
        assert config.max_workers == 7


class TestModelPoolResize:
    async def test_factory_default_pool_is_resized(self):
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test")
        assert model.max_concurrency is None
        assert await model.request_concurrency(37) is True
        assert model.max_concurrency == 37
        await model.cleanup()

    async def test_explicit_max_connections_refuses_resize(self):
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test", max_connections=8)
        assert await model.request_concurrency(37) is False
        assert model.max_concurrency == 8
        await model.cleanup()

    async def test_caller_supplied_client_refuses_resize(self):
        from unittest.mock import MagicMock

        model = OpenAIModel("gpt-4o-mini", MagicMock())
        assert await model.request_concurrency(37) is False
        assert model.max_concurrency is None

    async def test_resize_rejects_invalid_value(self):
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test")
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            await model.request_concurrency(0)
        await model.cleanup()

    async def test_strategy_forwards_to_model(self):
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test")
        strategy = OpenAIStrategy(model)
        assert await strategy.request_concurrency(21) is True
        assert strategy.max_concurrency == 21
        await model.cleanup()

    async def test_strategy_without_model_hook_returns_false(self):
        strategy = EchoStrategy()
        assert not hasattr(strategy, "model")
        # Custom strategies simply have no hook; the processor treats the
        # absence as "nothing to size".
        assert getattr(strategy, "request_concurrency", None) is None


class TestProcessorWiring:
    async def test_add_work_resizes_factory_model_without_warning(self, fast_retry):
        from async_batch_llm import LLMWorkItem

        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test")
        strategy = OpenAIStrategy(model)
        config = ProcessorConfig(concurrency=9, retry=fast_retry)
        processor = ParallelBatchProcessor(config=config)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            await processor.add_work(LLMWorkItem(item_id="x", strategy=strategy, prompt="hello"))
        assert model.max_concurrency == 9
        await model.cleanup()

    async def test_add_work_warns_on_real_contradiction(self, fast_retry):
        from async_batch_llm import LLMWorkItem

        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test", max_connections=2)
        strategy = OpenAIStrategy(model)
        config = ProcessorConfig(concurrency=9, retry=fast_retry)
        processor = ParallelBatchProcessor(config=config)
        with pytest.warns(UserWarning, match="max_concurrency=2"):
            await processor.add_work(LLMWorkItem(item_id="x", strategy=strategy, prompt="hello"))
        assert model.max_concurrency == 2
        await model.cleanup()


class TestShorthand:
    async def test_process_prompts_concurrency_shorthand(self):
        batch = await process_prompts(EchoStrategy(), ["a", "b"], concurrency=7)
        assert batch.succeeded == 2

    async def test_shorthand_with_config_derives(self):
        config = ProcessorConfig(attempt_timeout=30.0)
        batch = await process_prompts(EchoStrategy(), ["a"], config=config, concurrency=7)
        assert batch.succeeded == 1
        # The caller's config object is untouched.
        assert config.concurrency is None
        assert config.max_workers == 5

    async def test_shorthand_conflict_raises(self):
        config = ProcessorConfig(concurrency=3)
        with pytest.raises(ValueError, match="Conflicting concurrency"):
            await process_prompts(EchoStrategy(), ["a"], config=config, concurrency=7)

    async def test_shorthand_matching_config_value_is_fine(self):
        config = ProcessorConfig(concurrency=3)
        batch = await process_prompts(EchoStrategy(), ["a"], config=config, concurrency=3)
        assert batch.succeeded == 1
