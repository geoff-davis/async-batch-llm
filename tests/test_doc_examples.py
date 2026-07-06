"""Validate code snippets in the documentation against the real API.

Guards against the doc-drift bug classes found in the 2026-07 docs audit:

1. Fenced ``python`` blocks that no longer parse.
2. ``from async_batch_llm ... import X`` naming symbols that don't exist.
3. Doc classes overriding framework hooks (``LLMCallStrategy.execute``,
   ``Middleware.after_process``, ...) with the wrong parameter count —
   the framework calls hooks positionally, so a stale signature in a
   copied snippet fails at runtime.

Signatures are read from the live classes with ``inspect``, so these
tests track the source automatically.

Opt-outs: put ``<!-- doc-snippet: skip -->`` on its own line directly
above a fence to exclude that block. Historical docs (``docs/archive/``,
migration guides) are syntax-checked only — they intentionally show old
APIs.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.middleware.base import Middleware
from async_batch_llm.observers.base import ProcessorObserver

REPO_ROOT = Path(__file__).parent.parent

DOC_FILES = sorted(
    path
    for path in {
        REPO_ROOT / "README.md",
        REPO_ROOT / "CLAUDE.md",
        REPO_ROOT / "examples" / "README.md",
        *(REPO_ROOT / "docs").rglob("*.md"),
    }
    # Frozen planning/review docs — not user-facing, not checked at all.
    if not any(part in str(path) for part in ("docs/archive/", "docs/internal/"))
)

# Historical docs show old APIs on purpose: parse-check only.
HISTORICAL_PARTS = ("docs/migration/",)
HISTORICAL_NAMES = re.compile(r"MIGRATION_V\d")

SKIP_MARKER = "<!-- doc-snippet: skip -->"

FENCE_RE = re.compile(r"^```python\s*$")
FENCE_END_RE = re.compile(r"^```\s*$")

# Framework hook contracts: {base-class name fragment: (base class, hook names)}.
# A doc class subclassing a base (matched by name in its bases list) must
# declare these hooks with the same positional parameter count as the real
# method — the framework invokes them positionally.
HOOK_BASES: dict[str, tuple[type, tuple[str, ...]]] = {
    "LLMCallStrategy": (LLMCallStrategy, ("execute", "on_error", "prepare", "cleanup", "dry_run")),
    "Middleware": (Middleware, ("before_process", "after_process", "on_error")),
    "BaseMiddleware": (Middleware, ("before_process", "after_process", "on_error")),
    "ProcessorObserver": (ProcessorObserver, ("on_event",)),
    "BaseObserver": (ProcessorObserver, ("on_event",)),
}


@dataclass
class Snippet:
    path: Path
    line: int  # 1-indexed line of the opening fence
    source: str

    @property
    def label(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}:{self.line}"

    @property
    def historical(self) -> bool:
        rel = str(self.path.relative_to(REPO_ROOT))
        return any(part in rel for part in HISTORICAL_PARTS) or bool(HISTORICAL_NAMES.search(rel))


def extract_snippets(path: Path) -> list[Snippet]:
    snippets: list[Snippet] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        if FENCE_RE.match(lines[i]):
            skipped = i > 0 and lines[i - 1].strip() == SKIP_MARKER
            start = i + 1
            j = start
            while j < len(lines) and not FENCE_END_RE.match(lines[j]):
                j += 1
            if not skipped:
                snippets.append(Snippet(path, i + 1, "\n".join(lines[start:j])))
            i = j
        i += 1
    return snippets


ALL_SNIPPETS = [s for path in DOC_FILES for s in extract_snippets(path)]
CURRENT_SNIPPETS = [s for s in ALL_SNIPPETS if not s.historical]


def _fragment_fixups(source: str) -> str:
    """Rewrite common doc-fragment idioms into parseable Python.

    Only used as a fallback when the raw snippet doesn't parse, so these
    can't corrupt valid code.
    """
    # API-reference signature displays: a def with no body, ending either
    # on its own dedented ")" line or with ")" at the end of the def line.
    source = re.sub(r"^(\s*)\)\s*$", r"\1): ...", source, flags=re.MULTILINE)
    source = re.sub(r"^(\s*(?:async )?def .+\))\s*$", r"\1: ...", source, flags=re.MULTILINE)
    # Elided-arguments idiom: call(kwarg=1, ...)
    source = re.sub(r",\s*\.\.\.\s*\)", ")", source)
    return source


def parse_snippet(source: str) -> ast.AST | None:
    """Parse a doc snippet, tolerating top-level await and common
    documentation fragments (see _fragment_fixups)."""
    for candidate in (source, _fragment_fixups(source)):
        try:
            return compile(
                candidate,
                "<doc-snippet>",
                "exec",
                flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )
        except SyntaxError:
            continue
    return None


def real_positional_names(cls: type, hook: str) -> list[str]:
    func = inspect.getattr_static(cls, hook)
    if isinstance(func, (staticmethod, classmethod)):
        func = func.__func__
    params = inspect.signature(func).parameters.values()
    return [p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


def doc_positional_names(node: ast.AsyncFunctionDef | ast.FunctionDef) -> list[str]:
    return [a.arg for a in (*node.args.posonlyargs, *node.args.args)]


def matched_hook_bases(node: ast.ClassDef) -> list[tuple[type, tuple[str, ...]]]:
    bases: list[tuple[type, tuple[str, ...]]] = []
    for base in node.bases:
        text = ast.unparse(base)
        for fragment, contract in HOOK_BASES.items():
            if fragment in text:
                bases.append(contract)
                break
    return bases


@pytest.mark.parametrize("snippet", ALL_SNIPPETS, ids=lambda s: s.label)
def test_doc_snippet_parses(snippet: Snippet) -> None:
    """Every fenced python block in the docs must be valid Python."""
    assert parse_snippet(snippet.source) is not None, (
        f"{snippet.label}: python code block does not parse. Fix the snippet, "
        f"or mark an intentional fragment with {SKIP_MARKER!r} on the line "
        "above the fence."
    )


@pytest.mark.parametrize("snippet", CURRENT_SNIPPETS, ids=lambda s: s.label)
def test_doc_imports_resolve(snippet: Snippet) -> None:
    """Every async_batch_llm import in a doc snippet must exist."""
    tree = parse_snippet(snippet.source)
    if tree is None:  # reported by test_doc_snippet_parses
        pytest.skip("snippet does not parse")

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        # Exact-prefix match: importing executes module code, so only ever
        # import this package itself (not e.g. an async_batch_llm_x typosquat).
        module_name = node.module or ""
        if module_name != "async_batch_llm" and not module_name.startswith("async_batch_llm."):
            continue
        module = importlib.import_module(module_name)
        for alias in node.names:
            assert hasattr(module, alias.name), (
                f"{snippet.label}: `from {node.module} import {alias.name}` — "
                f"{alias.name!r} does not exist in {node.module}"
            )


@pytest.mark.parametrize("snippet", CURRENT_SNIPPETS, ids=lambda s: s.label)
def test_doc_hook_signatures_match_framework(snippet: Snippet) -> None:
    """Doc classes overriding framework hooks must match the live signatures.

    The framework calls these hooks positionally, so the positional
    parameter count must match exactly (names are compared leniently and
    extra keyword-only params are fine).
    """
    tree = parse_snippet(snippet.source)
    if tree is None:
        pytest.skip("snippet does not parse")

    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base_cls, hooks in matched_hook_bases(node):
            for item in node.body:
                if not isinstance(item, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    continue
                if item.name in hooks:
                    expected = real_positional_names(base_cls, item.name)
                    actual = doc_positional_names(item)
                    if len(actual) != len(expected):
                        problems.append(
                            f"{node.name}.{item.name}({', '.join(actual)}) has "
                            f"{len(actual)} positional params; the framework "
                            f"calls it with {len(expected)}: "
                            f"({', '.join(expected)})"
                        )
                elif base_cls is Middleware and re.match(r"(on|before|after)_", item.name):
                    problems.append(
                        f"{node.name}.{item.name} looks like a middleware hook, "
                        f"but {base_cls.__name__} has no such hook "
                        f"(real hooks: {', '.join(hooks)})"
                    )

    assert not problems, f"{snippet.label}:\n  " + "\n  ".join(problems)


# --- self-tests: prove the checker catches the bug classes it exists for ---

BAD_EXECUTE = """
from async_batch_llm.llm_strategies import LLMCallStrategy

class Broken(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout):  # missing state
        ...
"""

BAD_MIDDLEWARE = """
from async_batch_llm import BaseMiddleware

class Broken(BaseMiddleware):
    async def after_process(self, work_item, result):  # extra param
        return result

    async def on_retry(self, work_item, attempt, error):  # phantom hook
        ...
"""


def test_checker_catches_stale_execute_signature() -> None:
    tree = parse_snippet(BAD_EXECUTE)
    assert tree is not None
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    (contract,) = matched_hook_bases(node)
    method = node.body[0]
    assert isinstance(method, ast.AsyncFunctionDef)
    expected = real_positional_names(contract[0], "execute")
    assert len(doc_positional_names(method)) != len(expected)


def test_checker_catches_middleware_drift() -> None:
    tree = parse_snippet(BAD_MIDDLEWARE)
    assert tree is not None
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    (contract,) = matched_hook_bases(node)
    after = node.body[0]
    assert isinstance(after, ast.AsyncFunctionDef)
    assert len(doc_positional_names(after)) != len(
        real_positional_names(contract[0], "after_process")
    )
    hook_names = contract[1]
    assert "on_retry" not in hook_names


def test_docs_were_discovered() -> None:
    """Guard against the glob silently matching nothing."""
    assert len(DOC_FILES) > 10
    assert len(ALL_SNIPPETS) > 50
    assert any(s.path.name == "API.md" for s in ALL_SNIPPETS)


# --- PyPI MathJax dollar-sign guard -----------------------------------

# pypi.org runs MathJax over the rendered project description with $...$
# enabled as inline math, so two bare dollar signs in one text run get
# typeset as TeX (and a "%" between them starts a TeX comment that eats
# the rest of the text). GitHub's math heuristics don't fire on currency,
# which is how the v0.16.0 "A Sense of Scale" bullet shipped garbled to
# PyPI while looking fine on GitHub. MathJax only pairs delimiters inside
# a single DOM text node, so any element boundary between two dollars —
# bold, italics, a code span — defuses the pair. The check mirrors that:
# within each markdown block, drop code, split on emphasis markers, and
# flag any remaining text run holding two or more "$".

PYPI_DESCRIPTION = REPO_ROOT / "README.md"

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")
_CODE_SPAN_RE = re.compile(r"`[^`]*`")
_EMPHASIS_SPLIT_RE = re.compile(r"\*\*|\*|__|_(?=\s)|(?<=\s)_")


def _mathjax_dollar_traps(markdown: str) -> list[tuple[int, str]]:
    """Return (line, text) for blocks MathJax would misrender on PyPI."""
    blocks: list[tuple[int, list[str]]] = []
    in_fence = False
    for lineno, line in enumerate(markdown.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        # Blank lines and new list items start a fresh DOM text container.
        if in_fence or not line.strip():
            blocks.append((lineno, []))
            continue
        if _LIST_ITEM_RE.match(line) or not blocks or not blocks[-1][1]:
            blocks.append((lineno, [line]))
        else:
            blocks[-1][1].append(line)

    traps = []
    for lineno, lines in blocks:
        text = _CODE_SPAN_RE.sub(" ", " ".join(lines))
        if any(run.count("$") >= 2 for run in _EMPHASIS_SPLIT_RE.split(text)):
            traps.append((lineno, text.strip()))
    return traps


def test_readme_has_no_mathjax_dollar_traps() -> None:
    traps = _mathjax_dollar_traps(PYPI_DESCRIPTION.read_text(encoding="utf-8"))
    assert not traps, (
        "README paragraphs with two bare '$' in one text run render as TeX "
        "math on PyPI. Break the pair with bold/code around the amounts "
        f"(see this test's header comment): {traps}"
    )


def test_dollar_trap_checker_catches_known_bad() -> None:
    garbled = "DeepSeek completed for $0.054 at 97.0% accuracy; Gemini cost $0.433."
    assert _mathjax_dollar_traps(garbled)
    # An element boundary (bold) between the dollars defuses the pair.
    assert not _mathjax_dollar_traps(
        "DeepSeek completed for **$0.054** at 97.0% accuracy; Gemini cost **$0.433**."
    )
    # Code spans and fenced blocks are skipped by MathJax entirely.
    assert not _mathjax_dollar_traps("Costs `$0.10` each, `$10.00` total.")
    assert not _mathjax_dollar_traps("```text\n$1 and $2\n```")
    # Separate list items are separate DOM containers.
    assert not _mathjax_dollar_traps("- costs $0.10\n- costs $0.03")
