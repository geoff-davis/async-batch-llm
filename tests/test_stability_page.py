"""docs/stability.md classifies every public export exactly once."""

from __future__ import annotations

import re
from pathlib import Path

import async_batch_llm

PAGE = Path(__file__).parents[1] / "docs" / "stability.md"


def test_every_export_is_classified_once():
    rows = [line for line in PAGE.read_text().splitlines() if line.startswith("| `")]
    names: list[str] = []
    for row in rows:
        first_cell = row.split("|")[1]
        names.extend(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", first_cell))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert duplicates == []
    assert sorted(set(names)) == sorted(async_batch_llm.__all__)
