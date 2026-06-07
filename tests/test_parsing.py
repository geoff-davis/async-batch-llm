"""Tests for the structured-output parsing helpers (issue #26)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from async_batch_llm import LLMResponse, pydantic_json_parser, strip_code_fences


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
