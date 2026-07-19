"""Concurrency-capacity metadata and mismatch diagnostics."""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from async_batch_llm import (
    LLMCallStrategy,
    LLMGateway,
    LLMWorkItem,
    OpenAIModel,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    TokenUsage,
)


class _CapacityStrategy(LLMCallStrategy[str]):
    def __init__(self, capacity: int | None) -> None:
        self._capacity = capacity

    @property
    def max_concurrency(self) -> int | None:
        return self._capacity

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: Any = None
    ) -> tuple[str, TokenUsage, None]:
        return prompt, {"total_tokens": 0}, None


def test_owned_openai_model_records_configured_capacity() -> None:
    with (
        patch("async_batch_llm.models.AsyncOpenAI"),
        patch("httpx.Limits"),
        patch("httpx.AsyncClient"),
    ):
        model = OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test", max_connections=32)

    assert model.max_concurrency == 32
    assert OpenAIStrategy(model).max_concurrency == 32


def test_user_supplied_client_capacity_is_unknown() -> None:
    model = OpenAIModel("gpt-4o-mini", MagicMock())

    assert model.max_concurrency is None
    assert OpenAIStrategy(model).max_concurrency is None


@pytest.mark.asyncio
async def test_processor_warns_once_when_workers_exceed_strategy_capacity() -> None:
    strategy = _CapacityStrategy(capacity=2)
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig(max_workers=3))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await processor.add_work(LLMWorkItem(item_id="a", strategy=strategy, prompt="a"))
        await processor.add_work(LLMWorkItem(item_id="b", strategy=strategy, prompt="b"))

    capacity_warnings = [w for w in caught if "max_concurrency=2" in str(w.message)]
    assert len(capacity_warnings) == 1
    assert "before attempt_timeout starts" in str(capacity_warnings[0].message)
    await processor.cleanup()


@pytest.mark.asyncio
async def test_processor_does_not_warn_for_unknown_or_sufficient_capacity() -> None:
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig(max_workers=3))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await processor.add_work(
            LLMWorkItem(item_id="unknown", strategy=_CapacityStrategy(None), prompt="a")
        )
        await processor.add_work(
            LLMWorkItem(item_id="equal", strategy=_CapacityStrategy(3), prompt="b")
        )
    await processor.cleanup()


def test_gateway_warns_when_workers_exceed_strategy_capacity() -> None:
    with pytest.warns(UserWarning, match=r"LLMGateway max_workers=4.*max_concurrency=2"):
        LLMGateway(_CapacityStrategy(2), config=ProcessorConfig(max_workers=4))
