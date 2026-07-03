"""Tests for the typed provider-output views (issue #52 Phase 2).

Covers the lenient ``from_metadata`` parsers, the lazy view properties on
``LLMResponse``/``WorkItemResult``, and — critically — that mixing the views
in leaves the dataclasses' fields/init/repr/eq byte-identical to before.
"""

from __future__ import annotations

import dataclasses
import json

from async_batch_llm import Grounding, GroundingSource, ToolCall
from async_batch_llm.base import LLMResponse, WorkItemResult

WELL_FORMED_METADATA = {
    "finish_reason": "stop",
    "grounding": {
        "sources": [
            {"uri": "https://example.com/a", "title": "A"},
            {"uri": "https://example.com/b", "title": None},
        ],
        "queries": ["what is a?"],
        "supports": [{"text": "a", "start_index": 0, "end_index": 1, "chunk_indices": [0]}],
    },
    "reasoning": "step 1 ... step 2",
    "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": '{"q": "a"}'}],
    "logprobs": {"content": [{"token": "a", "logprob": -0.1}]},
}


class TestFromMetadataLeniency:
    """Malformed metadata yields None / drops entries — never raises."""

    def test_grounding_source_non_dict(self):
        assert GroundingSource.from_metadata(None) is None
        assert GroundingSource.from_metadata("https://a") is None
        assert GroundingSource.from_metadata([{"uri": "https://a"}]) is None

    def test_grounding_source_requires_str_uri(self):
        assert GroundingSource.from_metadata({}) is None
        assert GroundingSource.from_metadata({"uri": 3}) is None
        assert GroundingSource.from_metadata({"uri": ""}) is None

    def test_grounding_source_wrong_typed_optionals_dropped(self):
        source = GroundingSource.from_metadata({"uri": "https://a", "title": 7, "snippet": ["x"]})
        assert source == GroundingSource(uri="https://a", title=None, snippet=None)

    def test_grounding_non_dict_or_empty(self):
        assert Grounding.from_metadata(None) is None
        assert Grounding.from_metadata("grounded") is None
        assert Grounding.from_metadata({}) is None

    def test_grounding_wrong_typed_containers(self):
        # Non-list containers and unparseable entries are dropped; when
        # everything drops, the whole view is None.
        assert Grounding.from_metadata({"sources": {"a": 1}, "queries": "q"}) is None
        parsed = Grounding.from_metadata(
            {"sources": [{"uri": "https://a"}, "junk", {"uri": 3}], "queries": ["q", 5]}
        )
        assert parsed is not None
        assert [s.uri for s in parsed.sources] == ["https://a"]
        assert parsed.queries == ["q"]
        assert parsed.supports == []

    def test_tool_call_non_dict(self):
        assert ToolCall.from_metadata(None) is None
        assert ToolCall.from_metadata("lookup") is None

    def test_tool_call_requires_str_name(self):
        assert ToolCall.from_metadata({}) is None
        assert ToolCall.from_metadata({"name": None}) is None
        assert ToolCall.from_metadata({"name": ""}) is None

    def test_tool_call_wrong_typed_fields_coerced(self):
        call = ToolCall.from_metadata({"id": 7, "name": "f", "arguments": {"a": 1}})
        assert call == ToolCall(id=None, name="f", arguments="")


class TestViewProperties:
    """The lazy views on LLMResponse and WorkItemResult."""

    def _result(self, metadata) -> WorkItemResult:
        return WorkItemResult(item_id="x", success=True, metadata=metadata)

    def test_metadata_none_all_views_none(self):
        result = self._result(None)
        assert result.grounding is None
        assert result.reasoning is None
        assert result.tool_calls is None
        assert result.logprobs is None

    def test_keys_absent_all_views_none(self):
        result = self._result({"finish_reason": "stop"})
        assert result.grounding is None
        assert result.reasoning is None
        assert result.tool_calls is None
        assert result.logprobs is None

    def test_well_formed_metadata_typed_access(self):
        result = self._result(dict(WELL_FORMED_METADATA))
        assert result.grounding is not None
        assert result.grounding.sources[0] == GroundingSource(
            uri="https://example.com/a", title="A"
        )
        assert result.grounding.queries == ["what is a?"]
        assert result.grounding.supports[0]["chunk_indices"] == [0]
        assert result.reasoning == "step 1 ... step 2"
        assert result.tool_calls == [ToolCall(id="call_1", name="lookup", arguments='{"q": "a"}')]
        assert result.logprobs == {"content": [{"token": "a", "logprob": -0.1}]}

    def test_llm_response_has_the_same_views(self):
        response = LLMResponse(
            text="t",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            metadata=dict(WELL_FORMED_METADATA),
        )
        assert response.grounding is not None
        assert response.grounding.sources[0].uri == "https://example.com/a"
        assert response.reasoning == "step 1 ... step 2"
        assert response.tool_calls is not None and response.tool_calls[0].name == "lookup"

    def test_views_are_not_cached(self):
        result = self._result({"reasoning": "first"})
        assert result.reasoning == "first"
        result.metadata["reasoning"] = "second"
        assert result.reasoning == "second"
        del result.metadata["reasoning"]
        assert result.reasoning is None

    def test_reasoning_wrong_type_is_none(self):
        assert self._result({"reasoning": 42}).reasoning is None
        assert self._result({"reasoning": ""}).reasoning is None

    def test_tool_calls_nothing_parses_is_none(self):
        assert self._result({"tool_calls": "call f"}).tool_calls is None
        assert self._result({"tool_calls": ["junk", {"id": "1"}]}).tool_calls is None

    def test_metadata_stays_json_serializable(self):
        # The contract shapes are plain dicts — persisting metadata must work.
        json.dumps(WELL_FORMED_METADATA)


class TestDataclassIntegrity:
    """Mixing the views in must not perturb the dataclasses."""

    def test_no_new_fields(self):
        for cls in (WorkItemResult, LLMResponse):
            names = {f.name for f in dataclasses.fields(cls)}
            assert not names & {"grounding", "reasoning", "tool_calls", "logprobs"}, cls

    def test_asdict_repr_eq_unchanged(self):
        a = WorkItemResult(item_id="x", success=True, metadata=dict(WELL_FORMED_METADATA))
        b = WorkItemResult(item_id="x", success=True, metadata=dict(WELL_FORMED_METADATA))
        assert a == b
        assert set(dataclasses.asdict(a)) == {f.name for f in dataclasses.fields(WorkItemResult)}
        assert "grounding=" not in repr(a)

    def test_generic_subscription_still_works(self):
        # PEP 696 defaults: both full and partial subscription.
        assert WorkItemResult[str, None](item_id="x", success=True).grounding is None
        assert WorkItemResult[str](item_id="y", success=True).reasoning is None
