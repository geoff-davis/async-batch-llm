"""Tests for metadata pass-through from strategy.execute() to WorkItemResult.

Added in v0.10.0 alongside the 3-tuple ``LLMCallStrategy.execute()``
contract. Verifies:

- Built-in strategies surface ``LLMResponse.metadata`` into
  ``WorkItemResult.metadata``.
- Custom strategies returning the legacy 2-tuple still work via the compat
  shim and get ``metadata=None``.
- The ``_unpack_strategy_result`` helper rejects malformed shapes.
- ``gemini_safety_ratings`` continues to be populated for backward
  compatibility (deprecated; will be removed once the metadata path is the
  sole source).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from async_batch_llm import (
    LLMWorkItem,
    OpenAIStrategy,
    OpenRouterStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
)
from async_batch_llm.base import (
    LLMResponse,
    RetryState,
    TokenUsage,
    WorkItemResult,
    _unpack_strategy_result,
)
from async_batch_llm.llm_strategies import LLMCallStrategy


def _mock_model_with_metadata(metadata: dict | None) -> AsyncMock:
    m = AsyncMock()
    m.generate = AsyncMock(
        return_value=LLMResponse(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            metadata=metadata,
        )
    )
    return m


# ── _unpack_strategy_result ───────────────────────────────────────────


class TestUnpackStrategyResult:
    def test_three_tuple_passes_through(self):
        out, tokens, meta = _unpack_strategy_result(("hi", {"total_tokens": 5}, {"x": 1}))
        assert out == "hi"
        assert tokens == {"total_tokens": 5}
        assert meta == {"x": 1}

    def test_three_tuple_with_none_metadata(self):
        out, tokens, meta = _unpack_strategy_result(("hi", {"total_tokens": 5}, None))
        assert meta is None

    def test_two_tuple_compat_shim_returns_none_metadata(self):
        out, tokens, meta = _unpack_strategy_result(("hi", {"total_tokens": 5}))
        assert out == "hi"
        assert tokens == {"total_tokens": 5}
        assert meta is None

    def test_non_tuple_rejected(self):
        with pytest.raises(ValueError, match="must return a tuple"):
            _unpack_strategy_result("not a tuple")
        with pytest.raises(ValueError, match="must return a tuple"):
            _unpack_strategy_result({"a": 1})

    def test_wrong_arity_rejected(self):
        with pytest.raises(ValueError, match="2- or 3-tuple"):
            _unpack_strategy_result(("a",))
        with pytest.raises(ValueError, match="2- or 3-tuple"):
            _unpack_strategy_result(("a", "b", "c", "d"))


# ── Built-in strategies forward LLMResponse.metadata ─────────────────


class TestBuiltinStrategiesForwardMetadata:
    @pytest.mark.asyncio
    async def test_openai_strategy_forwards_metadata(self):
        meta = {"finish_reason": "stop", "model": "gpt-4o-mini"}
        model = _mock_model_with_metadata(meta)
        strategy = OpenAIStrategy(model)

        out, tokens, returned_meta = await strategy.execute("p", 1, 10.0)

        assert out == "hello"
        assert returned_meta == meta

    @pytest.mark.asyncio
    async def test_openrouter_strategy_forwards_provider_metadata(self):
        meta = {
            "finish_reason": "stop",
            "model": "anthropic/claude-haiku-4-5",
            "provider": "Anthropic",
        }
        model = _mock_model_with_metadata(meta)
        strategy = OpenRouterStrategy(model)

        out, tokens, returned_meta = await strategy.execute("p", 1, 10.0)

        assert returned_meta == meta
        assert returned_meta["provider"] == "Anthropic"

    @pytest.mark.asyncio
    async def test_metadata_is_none_when_model_has_none(self):
        model = _mock_model_with_metadata(None)
        strategy = OpenAIStrategy(model)

        _, _, meta = await strategy.execute("p", 1, 10.0)

        assert meta is None


# ── End-to-end: WorkItemResult.metadata is populated ─────────────────


class _ThreeTupleStrategy(LLMCallStrategy[str]):
    """Test helper: returns the configured metadata in a 3-tuple."""

    def __init__(self, metadata: dict | None) -> None:
        self.metadata = metadata

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[str, TokenUsage, dict | None]:
        return (
            f"out:{prompt}",
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            self.metadata,
        )


class _TwoTupleLegacyStrategy(LLMCallStrategy[str]):
    """Test helper: returns the legacy 2-tuple shape (no metadata)."""

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[str, TokenUsage]:
        return (
            f"legacy:{prompt}",
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )


class TestWorkItemResultMetadata:
    @pytest.mark.asyncio
    async def test_three_tuple_strategy_populates_workitem_metadata(self):
        strategy = _ThreeTupleStrategy(metadata={"provider": "test", "model": "x"})
        config = ProcessorConfig(max_workers=1, timeout_per_item=2.0)

        async with ParallelBatchProcessor[str, str, None](config=config) as processor:
            await processor.add_work(LLMWorkItem(item_id="i1", strategy=strategy, prompt="hi"))
            result = await processor.process_all()

        assert result.results[0].success
        assert result.results[0].metadata == {"provider": "test", "model": "x"}

    @pytest.mark.asyncio
    async def test_two_tuple_legacy_strategy_yields_none_metadata(self):
        """Pre-v0.10.0 strategies still work; metadata defaults to None."""
        strategy = _TwoTupleLegacyStrategy()
        config = ProcessorConfig(max_workers=1, timeout_per_item=2.0)

        async with ParallelBatchProcessor[str, str, None](config=config) as processor:
            await processor.add_work(LLMWorkItem(item_id="i1", strategy=strategy, prompt="hi"))
            result = await processor.process_all()

        assert result.results[0].success
        assert result.results[0].output == "legacy:hi"
        assert result.results[0].metadata is None

    @pytest.mark.asyncio
    async def test_three_tuple_with_none_metadata(self):
        strategy = _ThreeTupleStrategy(metadata=None)
        config = ProcessorConfig(max_workers=1, timeout_per_item=2.0)

        async with ParallelBatchProcessor[str, str, None](config=config) as processor:
            await processor.add_work(LLMWorkItem(item_id="i1", strategy=strategy, prompt="hi"))
            result = await processor.process_all()

        assert result.results[0].metadata is None


# ── Backward-compat: gemini_safety_ratings populated from metadata ───


class TestGeminiSafetyRatingsBackcompat:
    """The legacy ``gemini_safety_ratings`` field is populated from
    ``metadata['safety_ratings']`` so existing user code keeps working
    until that field is removed."""

    def test_backfill_from_metadata(self):
        result = WorkItemResult(
            item_id="x",
            success=True,
            metadata={"safety_ratings": {"HARM_HATE": "LOW"}},
        )
        assert result.gemini_safety_ratings == {"HARM_HATE": "LOW"}

    def test_explicit_field_takes_precedence(self):
        result = WorkItemResult(
            item_id="x",
            success=True,
            metadata={"safety_ratings": {"HARM_HATE": "HIGH"}},
            gemini_safety_ratings={"HARM_HATE": "LOW"},
        )
        # Caller-provided explicit value should not be overwritten.
        assert result.gemini_safety_ratings == {"HARM_HATE": "LOW"}

    def test_no_metadata_no_ratings(self):
        result = WorkItemResult(item_id="x", success=True)
        assert result.gemini_safety_ratings is None
        assert result.metadata is None

    def test_metadata_without_safety_ratings(self):
        result = WorkItemResult(
            item_id="x",
            success=True,
            metadata={"finish_reason": "stop"},
        )
        assert result.gemini_safety_ratings is None
        assert result.metadata == {"finish_reason": "stop"}
