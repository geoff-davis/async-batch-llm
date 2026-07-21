"""Lightweight v0.20 onboarding and release-document integrity checks."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
NOTEBOOK = ROOT / "notebooks" / "async_batch_llm_quickstart.ipynb"
TERMINAL_ASSET = ROOT / "docs" / "assets" / "v0.20-quickstart.gif"


def test_notebook_is_valid_safe_json_with_parseable_python() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "sk-" not in source
    assert "AIza" not in source
    assert not re.search(r"(?i)(?:api_key|token|secret)\s*=\s*['\"][^.'\"]{8,}", source)

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        python_source = "\n".join(
            line
            for line in "".join(cell.get("source", [])).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        compile(
            python_source,
            str(NOTEBOOK),
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )


def test_terminal_asset_is_portable_and_small() -> None:
    assert TERMINAL_ASSET.read_bytes().startswith((b"GIF87a", b"GIF89a"))
    assert TERMINAL_ASSET.stat().st_size < 2 * 1024 * 1024

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "https://raw.githubusercontent.com/geoff-davis/async-batch-llm/main/"
        "docs/assets/v0.20-quickstart.gif"
    ) in readme
    assert (
        "https://colab.research.google.com/github/geoff-davis/async-batch-llm/blob/main/"
        "notebooks/async_batch_llm_quickstart.ipynb"
    ) in readme
    assert not re.search(r"!\[[^]]*]\((?!https://)", readme)


def test_primary_onboarding_uses_high_level_api() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_home = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    getting_started = (ROOT / "docs" / "getting-started.md").read_text(encoding="utf-8")
    for content in (readme, docs_home, getting_started):
        assert 'llm("openai:gpt-4o-mini")' in content
        assert "process_prompts" in content
        assert "concurrency=" in content
        assert "summary()" in content
    assert "progress=True" in readme
    assert "ParallelBatchProcessor(" not in docs_home
    assert "PydanticAIStrategy(" not in docs_home
