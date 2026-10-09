"""Run ``npm audit`` and fail on advisories at or above a severity, minus an allowlist.

``npm audit`` has no way to accept a single advisory, so an advisory without a
patched release (or one we've judged not to apply) would keep the CI check red
and hide new findings behind it. This wraps ``npm audit --json`` and fails on
every advisory at ``--audit-level`` or above that isn't in ``ALLOWED``.

    python scripts/npm_audit.py                       # default level: moderate
    python scripts/npm_audit.py --audit-level high

The npm packages here are development-only Markdown tooling; nothing in
``package.json`` ships with the Python package. Keep ``ALLOWED`` short, give
each entry a reason, and remove it once a fixed release can be installed.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

SEVERITIES = ("info", "low", "moderate", "high", "critical")

# Advisory ID -> why it's accepted. Removing an entry is the normal way to retire it.
ALLOWED: dict[str, str] = {
    "GHSA-vfj7-8cjw-p6xm": (
        "braces stack exhaustion on deeply nested patterns; no patched release exists "
        "(braces <= 3.0.3, published 2026-09-18). Reached only through markdownlint-cli2's "
        "file globbing, which expands patterns we write in the Makefile and prek config."
    ),
}


@dataclass(frozen=True)
class Advisory:
    id: str
    package: str
    severity: str
    title: str
    url: str


def advisories(report: dict[str, Any]) -> list[Advisory]:
    """The distinct advisories in an ``npm audit --json`` report.

    Packages that are vulnerable only through a dependency list that dependency's
    name as a string in ``via``; the advisory itself appears as a dict on the
    package it affects, so only the dicts are collected.
    """
    if "error" in report or "vulnerabilities" not in report:
        raise ValueError(f"unexpected npm audit output: {json.dumps(report)[:500]}")
    found: dict[str, Advisory] = {}
    for package, entry in report["vulnerabilities"].items():
        for via in entry.get("via", []):
            if not isinstance(via, dict):
                continue
            url = via.get("url", "")
            advisory_id = url.rstrip("/").rsplit("/", 1)[-1] or str(via.get("source"))
            found.setdefault(
                advisory_id,
                Advisory(
                    id=advisory_id,
                    package=via.get("name", package),
                    severity=via.get("severity", "critical"),
                    title=via.get("title", ""),
                    url=url,
                ),
            )
    return sorted(found.values(), key=lambda a: a.id)


def at_or_above(severity: str, level: str) -> bool:
    # An unknown severity counts as critical, so a format change fails closed.
    rank = SEVERITIES.index(severity) if severity in SEVERITIES else len(SEVERITIES)
    return rank >= SEVERITIES.index(level)


def evaluate(
    found: list[Advisory], level: str, allowed: dict[str, str]
) -> tuple[list[Advisory], list[Advisory], list[str]]:
    """Split advisories into (failing, accepted) and list allowlist entries no longer reported."""
    relevant = [a for a in found if at_or_above(a.severity, level)]
    failing = [a for a in relevant if a.id not in allowed]
    accepted = [a for a in relevant if a.id in allowed]
    reported = {a.id for a in found}
    stale = sorted(i for i in allowed if i not in reported)
    return failing, accepted, stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--audit-level", choices=SEVERITIES, default="moderate")
    args = parser.parse_args(argv)

    proc = subprocess.run(["npm", "audit", "--json"], capture_output=True, text=True)
    try:
        found = advisories(json.loads(proc.stdout))
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"npm audit failed (exit {proc.returncode}): {exc}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return 2

    failing, accepted, stale = evaluate(found, args.audit_level, ALLOWED)
    for a in accepted:
        print(f"accepted: {a.id} {a.severity} {a.package}: {a.title}\n  reason: {ALLOWED[a.id]}")
    for advisory_id in stale:
        print(
            f"::warning::{advisory_id} is in ALLOWED but npm audit no longer reports it; "
            "remove it from scripts/npm_audit.py"
        )
    for a in failing:
        print(f"::error::{a.id} {a.severity} {a.package}: {a.title} {a.url}")
    if failing:
        print(f"{len(failing)} advisory(ies) at or above {args.audit_level}; see `npm audit`.")
        return 1
    print(
        f"No unaccepted advisories at or above {args.audit_level} ({len(found)} reported in total)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
