"""scripts/npm_audit.py: which npm advisories fail the security check."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "npm_audit.py"
_spec = importlib.util.spec_from_file_location("npm_audit", _PATH)
assert _spec is not None and _spec.loader is not None
na = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = na  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(na)


def advisory(ghsa, name, severity):
    return {
        "source": 1,
        "name": name,
        "title": f"{name} issue",
        "url": f"https://github.com/advisories/{ghsa}",
        "severity": severity,
        "range": "<=1.0.0",
    }


# Shaped like the real report on 2026-10-09: braces carries the advisory, and the
# packages above it in the tree name it by string.
REPORT = {
    "auditReportVersion": 2,
    "vulnerabilities": {
        "braces": {"severity": "high", "via": [advisory("GHSA-aaaa", "braces", "high")]},
        "micromatch": {"severity": "high", "via": ["braces"]},
        "smol-toml": {
            "severity": "moderate",
            "via": [advisory("GHSA-bbbb", "smol-toml", "moderate")],
        },
        "katex": {"severity": "low", "via": [advisory("GHSA-cccc", "katex", "low")]},
        "markdownlint-cli2": {"severity": "high", "via": ["micromatch", "smol-toml"]},
    },
    "metadata": {},
}


def ids(items):
    return [a.id for a in items]


def test_collects_each_advisory_once_and_skips_string_vias():
    found = na.advisories(REPORT)
    assert ids(found) == ["GHSA-aaaa", "GHSA-bbbb", "GHSA-cccc"]
    assert {a.package for a in found} == {"braces", "smol-toml", "katex"}


def test_level_threshold_and_allowlist():
    failing, accepted, stale = na.evaluate(na.advisories(REPORT), "moderate", {"GHSA-aaaa": "why"})
    assert ids(failing) == ["GHSA-bbbb"]  # low katex is below the level
    assert ids(accepted) == ["GHSA-aaaa"]
    assert stale == []


def test_allowlisted_only_report_passes():
    report = {
        "vulnerabilities": {k: v for k, v in REPORT["vulnerabilities"].items() if k != "smol-toml"}
    }
    failing, _, _ = na.evaluate(na.advisories(report), "moderate", {"GHSA-aaaa": "why"})
    assert failing == []


def test_stale_allowlist_entry_is_reported():
    _, _, stale = na.evaluate(
        na.advisories({"vulnerabilities": {}}), "moderate", {"GHSA-aaaa": "why"}
    )
    assert stale == ["GHSA-aaaa"]


def test_unknown_severity_fails_closed():
    report = {"vulnerabilities": {"x": {"via": [advisory("GHSA-dddd", "x", "severe")]}}}
    failing, _, _ = na.evaluate(na.advisories(report), "critical", {})
    assert ids(failing) == ["GHSA-dddd"]


@pytest.mark.parametrize("report", [{"error": {"code": "ENOLOCK"}}, {"metadata": {}}])
def test_malformed_report_raises(report):
    with pytest.raises(ValueError):
        na.advisories(report)


def test_allowlist_entries_have_reasons():
    assert all(reason.strip() for reason in na.ALLOWED.values())
    assert all(key.startswith("GHSA-") for key in na.ALLOWED)
