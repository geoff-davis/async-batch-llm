"""Typed views over the reserved provider-output keys in ``metadata``.

**Experimental.** This surface is new and hasn't seen production use yet:
the reserved-key dict shapes and the view APIs may change in a future
minor release while they stabilize. The underlying ``metadata`` dict
channel itself is stable — pin exact shapes at your own risk, or read the
dicts defensively.

Provider-specific structured output — Gemini grounding, reasoning traces,
tool calls, logprobs — travels through the framework as plain dicts under
**reserved keys** of the ``metadata`` channel (``LLMResponse.metadata`` →
``WorkItemResult.metadata``): ``'grounding'``, ``'reasoning'``,
``'tool_calls'``, ``'logprobs'``. Those dict shapes are documented in
``docs/API.md`` and stay JSON-serializable so results can be persisted
as-is.

This module provides the typed, provider-agnostic *read* surface over that
contract: small frozen dataclasses plus :class:`ProviderOutputViews`, a
mixin inherited by ``LLMResponse`` and ``WorkItemResult`` whose lazy
read-only properties (``.grounding``, ``.reasoning``, ``.tool_calls``,
``.logprobs``) parse the metadata dict on access. Nothing is stored twice:
the dict remains the single source of truth, and the views re-read it on
every access.

Parsing is deliberately lenient: malformed or wrong-typed metadata yields
``None`` (or drops the bad entry) rather than raising, so a provider-shape
drift can never break result handling.

Added in v0.16.0 (issue #52 Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


@dataclass(frozen=True)
class GroundingSource:
    """One web source backing a grounded response.

    Attributes:
        uri: Source URL (always present; entries without one are dropped).
        title: Human-readable page title, when the provider supplied one.
        snippet: Excerpt from the source, when the provider supplied one.
    """

    uri: str
    title: str | None = None
    snippet: str | None = None

    @classmethod
    def from_metadata(cls, data: Any) -> GroundingSource | None:
        """Parse one ``metadata['grounding']['sources']`` entry; lenient, never raises."""
        if not isinstance(data, dict):
            return None
        uri = data.get("uri")
        if not isinstance(uri, str) or not uri:
            return None
        title = data.get("title")
        snippet = data.get("snippet")
        return cls(
            uri=uri,
            title=title if isinstance(title, str) else None,
            snippet=snippet if isinstance(snippet, str) else None,
        )


@dataclass(frozen=True)
class Grounding:
    """Web-grounding data from a grounded call (e.g. Gemini ``google_search``).

    Attributes:
        sources: The web sources the answer was grounded in.
        queries: Search queries the model issued (``web_search_queries``).
        supports: Answer-span → source-index links, as plain dicts
            (``{"text", "start_index", "end_index", "chunk_indices"}``).
            Kept untyped for now.
    """

    sources: list[GroundingSource] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    supports: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_metadata(cls, data: Any) -> Grounding | None:
        """Parse a ``metadata['grounding']`` dict; lenient, never raises."""
        if not isinstance(data, dict) or not data:
            return None
        raw_sources = data.get("sources")
        sources = [
            parsed
            for entry in (raw_sources if isinstance(raw_sources, list) else ())
            if (parsed := GroundingSource.from_metadata(entry)) is not None
        ]
        raw_queries = data.get("queries")
        queries = [
            q for q in (raw_queries if isinstance(raw_queries, list) else ()) if isinstance(q, str)
        ]
        raw_supports = data.get("supports")
        supports = [
            s
            for s in (raw_supports if isinstance(raw_supports, list) else ())
            if isinstance(s, dict)
        ]
        if not sources and not queries and not supports:
            return None
        return cls(sources=sources, queries=queries, supports=supports)


@dataclass(frozen=True)
class ToolCall:
    """One tool/function call the model requested. Visibility only — the
    framework never executes tools; feed these to your own dispatch loop.

    Attributes:
        id: Provider call id, when supplied.
        name: Tool/function name (always present; entries without one are dropped).
        arguments: The raw JSON-string arguments, deliberately unparsed —
            parse with ``json.loads`` (and validate) yourself.
    """

    id: str | None
    name: str
    arguments: str

    @classmethod
    def from_metadata(cls, data: Any) -> ToolCall | None:
        """Parse one ``metadata['tool_calls']`` entry; lenient, never raises."""
        if not isinstance(data, dict):
            return None
        name = data.get("name")
        if not isinstance(name, str) or not name:
            return None
        call_id = data.get("id")
        arguments = data.get("arguments")
        return cls(
            id=call_id if isinstance(call_id, str) else None,
            name=name,
            arguments=arguments if isinstance(arguments, str) else "",
        )


class ProviderOutputViews:
    """Lazy read-only typed views over the ``metadata`` dict channel.

    **Experimental** — see the module docstring.

    Mixed into ``LLMResponse`` and ``WorkItemResult``. Each property parses
    the reserved key from ``self.metadata`` on every access — no caching, no
    stored fields — so the dict stays the single source of truth and the
    inheriting dataclasses' ``fields()``/``__init__``/``repr``/``eq`` are
    untouched.
    """

    __slots__ = ()

    if TYPE_CHECKING:
        # Provided by the inheriting dataclass; annotation lives under
        # TYPE_CHECKING only so it never becomes a dataclass field.
        metadata: dict[str, Any] | None

    @property
    def grounding(self) -> Grounding | None:
        """Typed view of ``metadata['grounding']``, or None when absent/malformed."""
        md = self.metadata
        if not md:
            return None
        return Grounding.from_metadata(md.get("grounding"))

    @property
    def reasoning(self) -> str | None:
        """Typed view of ``metadata['reasoning']`` (the model's reasoning/thinking
        trace, e.g. DeepSeek ``reasoning_content``), or None when absent."""
        md = self.metadata
        if not md:
            return None
        reasoning = md.get("reasoning")
        return reasoning if isinstance(reasoning, str) and reasoning else None

    @property
    def tool_calls(self) -> list[ToolCall] | None:
        """Typed view of ``metadata['tool_calls']``, or None when absent or when
        no entry parses. Visibility only — the framework never executes tools."""
        md = self.metadata
        if not md:
            return None
        raw = md.get("tool_calls")
        if not isinstance(raw, list):
            return None
        calls = [parsed for entry in raw if (parsed := ToolCall.from_metadata(entry)) is not None]
        return calls or None

    @property
    def logprobs(self) -> Any | None:
        """``metadata['logprobs']`` verbatim (provider shapes vary too much to
        type honestly), or None when absent."""
        md = self.metadata
        if not md:
            return None
        return md.get("logprobs")
