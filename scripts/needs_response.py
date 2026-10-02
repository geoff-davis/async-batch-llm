"""List open issues and PRs whose latest human activity isn't from a maintainer.

An item needs a response when the most recent comment, review or opening post
by a person (bots ignored) came from someone other than a maintainer. That
covers new reports nobody has answered and follow-ups after an answer.

    python scripts/needs_response.py                  # print the list
    python scripts/needs_response.py --update-digest  # CI: maintain the digest issue

``--update-digest`` keeps one open issue labeled ``needs-response-digest``:
it opens one (or comments on the open one) mentioning the maintainer while
anything is unanswered, and closes it once nothing is. Uses the ``gh`` CLI;
set ``GH_REPO`` to choose the repository. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any

MAINTAINERS = frozenset({"geoff-davis", "geoff-keksi-ai"})
MENTION = "@geoff-davis"
# gh reports comment authors without an is_bot flag, and GitHub Actions comments
# (for example the coverage comment) come from plain "github-actions".
KNOWN_BOTS = frozenset({"github-actions", "dependabot", "copilot"})
DIGEST_LABEL = "needs-response-digest"
DIGEST_TITLE = "Needs response: unanswered issues and PRs"


@dataclass(frozen=True)
class Pending:
    kind: str  # "issue" or "PR"
    number: int
    title: str
    url: str
    who: str  # author of the latest human activity
    since: str  # its ISO timestamp


def _is_bot(author: dict[str, Any] | None) -> bool:
    if not author:
        return True  # deleted ("ghost") accounts: nothing to answer
    login = author.get("login") or ""
    return (
        bool(author.get("is_bot"))
        or login in KNOWN_BOTS
        or login.endswith("[bot]")
        or login.startswith("app/")
    )


def latest_human_activity(item: dict[str, Any]) -> tuple[str, str] | None:
    """(login, timestamp) of the newest opening post, comment or review by a person."""
    events = [(item.get("createdAt") or "", item.get("author"))]
    events += [(c.get("createdAt") or "", c.get("author")) for c in item.get("comments") or []]
    events += [(r.get("submittedAt") or "", r.get("author")) for r in item.get("reviews") or []]
    human = [(when, author["login"]) for when, author in events if not _is_bot(author)]
    if not human:
        return None
    when, login = max(
        human, key=lambda event: datetime.fromisoformat(event[0].replace("Z", "+00:00"))
    )
    return login, when


def needs_response(
    items: list[dict[str, Any]], kind: str, maintainers=MAINTAINERS
) -> list[Pending]:
    pending = []
    for item in items:
        if any(label.get("name") == DIGEST_LABEL for label in item.get("labels") or []):
            continue
        latest = latest_human_activity(item)
        if latest is None or latest[0] in maintainers:
            continue
        pending.append(Pending(kind, item["number"], item["title"], item["url"], *latest))
    return sorted(pending, key=lambda p: p.since)


def digest_action(pending: list[Pending], digest_number: int | None) -> str:
    """One of "create", "comment", "close" or "none"."""
    if pending:
        return "comment" if digest_number is not None else "create"
    return "close" if digest_number is not None else "none"


_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]<>#|~])")


def _inert(text: str) -> str:
    """Show user-controlled text literally, without Markdown or @mentions.

    Only the maintainer should be notified: a zero-width space after ``@`` keeps
    GitHub from treating ``@user`` or ``@org/team`` in a title as a mention.
    """
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text).replace("@", "@\u200b")


def render(pending: list[Pending]) -> str:
    lines = [f"{MENTION}, {len(pending)} open item(s) are waiting for a maintainer reply:", ""]
    for p in pending:
        lines.append(
            f"- {p.kind} #{p.number}: {_inert(p.title)} (last from {_inert(p.who)}, {p.since[:10]})"
        )
    return "\n".join(lines)


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def _fetch(kind: str) -> list[dict[str, Any]]:
    fields = "number,title,url,author,createdAt,comments,labels"
    if kind == "pr":
        fields += ",reviews"
    return json.loads(_gh(kind, "list", "--state", "open", "--limit", "500", "--json", fields))


def collect() -> list[Pending]:
    pending = needs_response(_fetch("issue"), "issue") + needs_response(_fetch("pr"), "PR")
    return sorted(pending, key=lambda p: p.since)


def update_digest(pending: list[Pending]) -> str:
    found = json.loads(
        _gh("issue", "list", "--state", "open", "--label", DIGEST_LABEL, "--json", "number")
    )
    digest = found[0]["number"] if found else None
    action = digest_action(pending, digest)
    if action == "create":
        _gh(
            "label",
            "create",
            DIGEST_LABEL,
            "--force",
            "--color",
            "D93F0B",
            "--description",
            "Weekly list of unanswered issues and PRs",
        )
        _gh(
            "issue",
            "create",
            "--title",
            DIGEST_TITLE,
            "--label",
            DIGEST_LABEL,
            "--body",
            render(pending),
        )
    elif action == "comment":
        _gh("issue", "comment", str(digest), "--body", render(pending))
    elif action == "close":
        _gh("issue", "close", str(digest), "--comment", "Everything open has a maintainer reply.")
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update-digest", action="store_true")
    args = parser.parse_args()
    pending = collect()
    print(render(pending) if pending else "Nothing waiting for a maintainer reply.")
    if args.update_digest:
        print(f"digest: {update_digest(pending)} ({os.environ.get('GH_REPO', 'current repo')})")


if __name__ == "__main__":
    main()
