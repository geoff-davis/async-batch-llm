"""Tests for structured-output parsing and conservative recovery."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, RootModel, ValidationError

from async_batch_llm import (
    LLMResponse,
    LLMWorkItem,
    MetricsObserver,
    ModelStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryConfig,
    pydantic_json_parser,
    strip_code_fences,
)


class TestStripCodeFences:
    def test_plain_json_unchanged(self):
        assert strip_code_fences('{"a": 1}') == '{"a": 1}'

    def test_strips_json_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert strip_code_fences(text) == '{"a": 1}'

    def test_strips_bare_fence(self):
        text = '```\n{"a": 1}\n```'
        assert strip_code_fences(text) == '{"a": 1}'

    def test_strips_uppercase_lang_tag(self):
        text = '```JSON\n{"a": 1}\n```'
        assert strip_code_fences(text) == '{"a": 1}'

    def test_handles_surrounding_whitespace(self):
        text = '  \n```json\n{"a": 1}\n```\n  '
        assert strip_code_fences(text) == '{"a": 1}'

    def test_no_trailing_fence_still_strips_opener(self):
        # Truncated/streamed output missing the closing fence.
        text = '```json\n{"a": 1}'
        assert strip_code_fences(text) == '{"a": 1}'

    def test_interior_backticks_preserved(self):
        text = '```json\n{"code": "a ``` b"}\n```'
        # The outer fence is removed; the inner content is preserved up to the
        # last closing fence.
        assert '"code"' in strip_code_fences(text)

    def test_lone_fence_line_returned_as_is(self):
        assert strip_code_fences("```json") == "```json"


class _Classification(BaseModel):
    label: str
    confidence: float


class _ClassificationList(RootModel[list[_Classification]]):
    pass


class _Scalar(RootModel[int]):
    pass


class TestPydanticJsonParser:
    def _resp(self, text: str) -> LLMResponse:
        return LLMResponse(text=text, input_tokens=1, output_tokens=1, total_tokens=2)

    def test_parses_plain_json(self):
        parser = pydantic_json_parser(_Classification)
        out = parser(self._resp('{"label": "spam", "confidence": 0.9}'))
        assert out == _Classification(label="spam", confidence=0.9)

    def test_parses_fenced_json(self):
        parser = pydantic_json_parser(_Classification)
        out = parser(self._resp('```json\n{"label": "ham", "confidence": 0.1}\n```'))
        assert out.label == "ham"
        assert out.confidence == 0.1

    def test_invalid_json_raises_validation_error(self):
        parser = pydantic_json_parser(_Classification)
        with pytest.raises(ValidationError):
            parser(self._resp('{"label": "spam"}'))  # missing confidence

    def test_trailing_fence_recovery_is_opt_in(self):
        parser = pydantic_json_parser(_Classification)
        with pytest.raises(ValidationError):
            parser(self._resp('{"label": "spam", "confidence": 0.9}\n```'))

    @pytest.mark.parametrize("artifact", ["```", "```_"])
    def test_recovers_only_allowed_trailing_fence_artifacts(self, artifact: str):
        response = self._resp(f'{{"label": "spam", "confidence": 0.9}}\n{artifact}')
        response.metadata = {"model": "test-model"}
        parser = pydantic_json_parser(_Classification, recover_trailing_markdown=True)

        out = parser(response)

        assert out == _Classification(label="spam", confidence=0.9)
        assert response.metadata == {
            "model": "test-model",
            "structured_output_recovered": True,
            "structured_output_recovery_reason": "trailing_markdown_fence",
            "structured_output_retries_avoided": 1,
        }
        assert response.structured_output_recovered
        assert response.structured_output_recovery_reason == "trailing_markdown_fence"
        assert response.structured_output_retries_avoided == 1

    def test_recovers_object_inside_unmatched_opening_fence(self):
        parser = pydantic_json_parser(_Classification, recover_trailing_markdown=True)
        response = self._resp('```json\n{"label": "ham", "confidence": 0.1}\n```_')

        assert parser(response).label == "ham"
        assert response.structured_output_recovered

    def test_recovers_top_level_array_for_root_model(self):
        parser = pydantic_json_parser(_ClassificationList, recover_trailing_markdown=True)
        response = self._resp('[{"label": "ham", "confidence": 0.1}]\n```')

        assert parser(response).root == [_Classification(label="ham", confidence=0.1)]

    @pytest.mark.parametrize(
        "text",
        [
            '{"label": "spam", "confidence": 0.9} trailing prose',
            '{"label": "spam", "confidence": 0.9} {"second": true}',
            '{"label": "spam", "confidence": 0.9',
            '{"label": "spam"}\n```',
            '{"label": "spam", "confidence": NaN}\n```',
            '{"label": "spam", "label": "ham", "confidence": 0.9}\n```',
        ],
    )
    def test_ambiguous_malformed_or_schema_invalid_output_is_not_recovered(self, text: str):
        parser = pydantic_json_parser(_Classification, recover_trailing_markdown=True)
        response = self._resp(text)

        with pytest.raises(ValidationError):
            parser(response)
        assert not response.structured_output_recovered

    def test_scalar_json_is_not_recovered(self):
        parser = pydantic_json_parser(_Scalar, recover_trailing_markdown=True)
        with pytest.raises(ValidationError):
            parser(self._resp("1\n```"))


class _RecoveryModel:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text='{"label": "spam", "confidence": 0.9}\n```',
            input_tokens=4,
            output_tokens=5,
            total_tokens=9,
            metadata={"model": "recovery-test"},
        )


@pytest.mark.asyncio
async def test_recovery_avoids_retry_and_updates_batch_and_observer_metrics():
    model = _RecoveryModel()
    strategy = ModelStrategy(
        model,
        pydantic_json_parser(_Classification, recover_trailing_markdown=True),
    )
    observer = MetricsObserver()
    processor = ParallelBatchProcessor[str, _Classification, None](
        config=ProcessorConfig(retry=RetryConfig(max_attempts=2)),
        observers=[observer],
    )
    await processor.add_work(LLMWorkItem(item_id="recover", strategy=strategy, prompt="classify"))

    batch = await processor.process_all()
    stats = await processor.get_stats()
    metrics = await observer.get_metrics()
    prometheus = await observer.export_prometheus()
    await processor.cleanup()

    result = batch.results[0]
    assert result.success
    assert model.calls == 1
    assert result.token_usage == {
        "input_tokens": 4,
        "output_tokens": 5,
        "total_tokens": 9,
    }
    assert result.metadata is not None
    assert result.metadata["model"] == "recovery-test"
    assert result.structured_output_recovered
    assert result.structured_output_recovery_reason == "trailing_markdown_fence"
    assert result.structured_output_retries_avoided == 1
    assert stats["structured_output_recoveries"] == 1
    assert stats["structured_output_retries_avoided"] == 1
    assert stats["structured_output_recovery_reasons"] == {"trailing_markdown_fence": 1}
    assert metrics["structured_output_recoveries"] == 1
    assert metrics["structured_output_retries_avoided"] == 1
    assert metrics["structured_output_recovery_reasons"] == {"trailing_markdown_fence": 1}
    assert "async_batch_llm_structured_output_recoveries 1" in prometheus
    assert "async_batch_llm_structured_output_retries_avoided 1" in prometheus
    assert 'reason="trailing_markdown_fence"} 1' in prometheus
