"""scripts/needs_response.py: which open items count as waiting for a reply."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "needs_response.py"
_spec = importlib.util.spec_from_file_location("needs_response", _PATH)
assert _spec is not None and _spec.loader is not None
nr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = nr  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(nr)

USER = {"login": "someone", "is_bot": False}
MAINT = {"login": "geoff-davis", "is_bot": False}
BOT = {"login": "app/dependabot", "is_bot": True}


def item(number=1, author=USER, created="2026-09-07T01:00:00Z", comments=(), reviews=(), labels=()):
    return {
        "number": number,
        "title": f"item {number}",
        "url": f"https://example.test/{number}",
        "author": author,
        "createdAt": created,
        "comments": [{"author": a, "createdAt": t} for a, t in comments],
        "reviews": [{"author": a, "submittedAt": t} for a, t in reviews],
        "labels": [{"name": n} for n in labels],
    }


def numbers(items, kind="issue"):
    return [p.number for p in nr.needs_response(items, kind)]


def test_unanswered_outside_issue_is_pending():
    pending = nr.needs_response([item()], "issue")
    assert [(p.number, p.who, p.since) for p in pending] == [(1, "someone", "2026-09-07T01:00:00Z")]


def test_maintainer_reply_answers_it():
    assert numbers([item(comments=[(MAINT, "2026-10-02T00:00:00Z")])]) == []


def test_follow_up_after_an_answer_is_pending_again():
    comments = [(MAINT, "2026-10-02T00:00:00Z"), (USER, "2026-10-03T00:00:00Z")]
    assert numbers([item(comments=comments)]) == [1]


def test_bots_are_ignored_as_authors_and_commenters():
    assert numbers([item(author=BOT)], "PR") == []
    # A bot comment after the outside report doesn't count as an answer.
    assert numbers([item(comments=[(BOT, "2026-10-02T00:00:00Z")])]) == [1]
    # gh gives comment authors no is_bot flag; the coverage comment is "github-actions".
    github_actions = {"login": "github-actions"}
    assert numbers([item(author=BOT, comments=[(github_actions, "2026-10-02T00:00:00Z")])]) == []
    # A deleted account has nothing to answer.
    assert numbers([item(author=None)]) == []


def test_maintainer_items_without_outside_activity_are_not_pending():
    assert numbers([item(author=MAINT)]) == []
    assert numbers([item(author={"login": "geoff-keksi-ai"})]) == []
    assert numbers([item(author=MAINT, comments=[(USER, "2026-10-03T00:00:00Z")])]) == [1]


def test_pr_review_by_maintainer_answers_it():
    pr = item(reviews=[(MAINT, "2026-10-02T00:00:00Z")])
    assert numbers([pr], "PR") == []


def test_digest_issue_itself_is_skipped():
    assert numbers([item(labels=[nr.DIGEST_LABEL])]) == []


def test_pending_sorted_oldest_first():
    items = [item(2, created="2026-09-20T00:00:00Z"), item(1, created="2026-09-01T00:00:00Z")]
    assert numbers(items) == [1, 2]


@pytest.mark.parametrize(
    ("has_pending", "digest", "action"),
    [(True, None, "create"), (True, 7, "comment"), (False, 7, "close"), (False, None, "none")],
)
def test_digest_action(has_pending, digest, action):
    pending = nr.needs_response([item()], "issue") if has_pending else []
    assert nr.digest_action(pending, digest) == action


def test_render_mentions_the_maintainer_and_lists_items():
    text = nr.render(nr.needs_response([item(156)], "issue"))
    assert text.startswith(f"{nr.MENTION}, 1 open item(s)")
    assert "- issue #156: item 156 (last from someone, 2026-09-07)" in text


def test_render_mentions_only_the_maintainer():
    # Titles and logins are user-controlled; neither may notify anyone else.
    hostile = item(5, author={"login": "outside-user"})
    hostile["title"] = "Ping @octocat and @org/team about *this*"
    text = nr.render(nr.needs_response([hostile], "issue"))
    assert re.findall(r"@[A-Za-z0-9]", text) == ["@g"]  # only @geoff-davis
    assert "outside-user" in text and "@\u200boctocat" in text and "@\u200borg/team" in text
    assert "\\*this\\*" in text  # Markdown shown literally


@pytest.mark.parametrize(
    ("has_pending", "open_digests", "expected"),
    [
        (True, [], [("label", "create"), ("issue", "create")]),
        (True, [{"number": 7}], [("issue", "comment", "7")]),
        (False, [{"number": 7}], [("issue", "close", "7")]),
        (False, [], []),
    ],
)
def test_update_digest_runs_the_matching_gh_commands(
    monkeypatch, has_pending, open_digests, expected
):
    calls = []

    def fake_gh(*args):
        calls.append(args)
        if args[:2] == ("issue", "list"):
            assert ("--label", nr.DIGEST_LABEL) == args[4:6]
            return json.dumps(open_digests)
        return ""

    monkeypatch.setattr(nr, "_gh", fake_gh)
    pending = nr.needs_response([item(156)], "issue") if has_pending else []
    nr.update_digest(pending)
    writes = [call for call in calls if call[:2] != ("issue", "list")]
    assert len(writes) == len(expected)
    assert [call[: len(want)] for call, want in zip(writes, expected, strict=True)] == expected
    if has_pending:
        assert any(nr.MENTION in arg for arg in writes[-1])
