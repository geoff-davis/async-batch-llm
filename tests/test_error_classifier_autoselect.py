"""Tests for auto-selecting the error classifier from work-item strategies."""

import logging

import pytest

from async_batch_llm import (
    DeepSeekStrategy,
    GeminiStrategy,
    LLMWorkItem,
    OpenAIStrategy,
    OpenRouterStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    process_prompts,
)
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.classifiers.gemini import GeminiErrorClassifier
from async_batch_llm.classifiers.openai import OpenAIErrorClassifier
from async_batch_llm.classifiers.openrouter import OpenRouterErrorClassifier
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.strategies.errors import DefaultErrorClassifier


class _NoPreferenceStrategy(LLMCallStrategy[str]):
    """Custom strategy that abstains (recommended_error_classifier -> None)."""

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        return prompt, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None


class _GeminiRecommendingStrategy(_NoPreferenceStrategy):
    def recommended_error_classifier(self):
        return GeminiErrorClassifier()


# ── Per-strategy recommendations ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (OpenAIStrategy(object()), OpenAIErrorClassifier),
        (DeepSeekStrategy(object()), OpenAIErrorClassifier),
        (OpenRouterStrategy(object()), OpenRouterErrorClassifier),
        (GeminiStrategy(object()), GeminiErrorClassifier),
        (_NoPreferenceStrategy(), type(None)),
    ],
)
def test_strategy_recommendations(strategy, expected):
    recommended = strategy.recommended_error_classifier()
    assert isinstance(recommended, expected) or (expected is type(None) and recommended is None)


# ── Processor resolution ────────────────────────────────────────────────────


async def _add(processor, *strategies):
    for i, strat in enumerate(strategies):
        await processor.add_work(LLMWorkItem(item_id=f"i{i}", strategy=strat, prompt="x"))


@pytest.mark.asyncio
async def test_autoselect_single_provider():
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig())
    await _add(processor, OpenAIStrategy(object()), OpenAIStrategy(object()))
    processor._resolve_error_classifier()
    assert isinstance(processor.error_classifier, OpenAIErrorClassifier)


@pytest.mark.asyncio
async def test_autoselect_abstaining_strategy_keeps_default():
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig())
    await _add(processor, _NoPreferenceStrategy(), _NoPreferenceStrategy())
    processor._resolve_error_classifier()
    assert isinstance(processor.error_classifier, DefaultErrorClassifier)


@pytest.mark.asyncio
async def test_autoselect_concrete_recommendation_wins_over_abstainer():
    # One strategy abstains (None), one recommends Gemini -> Gemini chosen.
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig())
    await _add(processor, _NoPreferenceStrategy(), _GeminiRecommendingStrategy())
    processor._resolve_error_classifier()
    assert isinstance(processor.error_classifier, GeminiErrorClassifier)


@pytest.mark.asyncio
async def test_autoselect_mixed_providers_falls_back_with_warning(caplog):
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig())
    await _add(processor, OpenAIStrategy(object()), GeminiStrategy(object()))
    with caplog.at_level(logging.WARNING, logger="async_batch_llm.parallel"):
        processor._resolve_error_classifier()
    assert isinstance(processor.error_classifier, DefaultErrorClassifier)
    assert any("mixed error classifiers" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_explicit_classifier_is_never_overridden():
    explicit = OpenRouterErrorClassifier()
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(), error_classifier=explicit
    )
    # Even with Gemini-recommending items, the explicit one wins.
    await _add(processor, GeminiStrategy(object()))
    processor._resolve_error_classifier()
    assert processor.error_classifier is explicit


@pytest.mark.asyncio
async def test_autoselect_resolved_during_real_batch():
    """End-to-end: resolution happens at batch start, before workers process."""
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig(max_workers=2))
    await _add(processor, _GeminiRecommendingStrategy(), _GeminiRecommendingStrategy())
    await processor.process_all()
    assert isinstance(processor.error_classifier, GeminiErrorClassifier)


@pytest.mark.asyncio
async def test_autoselect_via_process_prompts():
    """process_prompts should also benefit from auto-selection."""
    result = await process_prompts(
        GeminiStrategy(object()),
        ["a", "b"],
        config=ProcessorConfig(max_workers=2, dry_run=True),
    )
    # dry_run avoids touching the dummy model; we only care the batch ran.
    assert result.total_items == 2
