# Repository Guidelines

## Tooling Prerequisites

Python workflows run through `uv`; install it first, then sync the environment with `uv sync`. Markdown linting
depends on Node tooling—install Node 18+ and add `markdownlint-cli2` as a dev dependency
(`npm install --save-dev markdownlint-cli2`). Run the Make targets via `npx` so the locally pinned binary is used.

## Project Structure & Module Organization

Source lives in `src/async_batch_llm`, with core orchestration under `core/`, reusable strategy interfaces in
`strategies/`, parallel scheduling in `parallel.py`, and middleware/observers grouped by folder. Shared test
fixtures are in `src/async_batch_llm/testing`. End-to-end and regression suites sit in `tests/`, while runnable client
snippets are in `examples/`. Reference material, including architecture notes, is in `docs/`. The `Makefile` and
`pyproject.toml` define tooling defaults—review them before adjusting project-wide settings.

## Build, Test, and Development Commands

Use `uv` to ensure the pinned virtual environment: `uv sync` installs dependencies. Core workflows are wrapped in
make targets: `make lint` (Ruff checks), `make format` (Ruff formatting), `make typecheck` (mypy), and
`make test-fast` (pytest excluding `slow`). Run `make check-all` for the standard local gate or `make ci` to mirror
GitHub Actions. When debugging a single test, call `uv run pytest tests/test_retry_logic.py -k partial_name`.

## Coding Style & Naming Conventions

Python code targets 3.10 with a Ruff-enforced 100-character soft limit. Prefer type-hinted, dataclass-friendly APIs
and keep async flows explicit. Modules and packages use snake_case; concrete strategy classes use PascalCase suffixed
with `Strategy` or `Classifier`. Observers and middleware should expose verbs describing side effects (e.g.,
`LoggingObserver`). Run `make format` and `make lint` before submitting to keep imports sorted and styles consistent.

## Testing Guidelines

Pytest is configured via `pyproject.toml`, discovering files matching `test_*.py` and skipping `@pytest.mark.slow` by
default. Add new coverage under `tests/` mirroring the target module path, and prefer descriptive test names like
`test_strategy_handles_token_limits`. Integration fixtures live in `src/async_batch_llm/testing`; reuse rather than
duplicating helpers. For scenarios that hit remote APIs, guard them with `slow` or a dedicated marker so they stay
opt-in.

## Commit & Pull Request Guidelines

Commits follow a concise, imperative summary (e.g., `Add on_error retry callback`) with focused scope; group related
changes and document breaking behavior in `CHANGELOG.md` when relevant. Pull requests should link any tracked issues,
outline behavioral changes, list new commands or flags, and include screenshots for UI- or docs-heavy updates where
clarity helps reviewers. Confirm `make ci` succeeds locally before requesting review to reduce turnarounds.

## Review Fixes & Handoffs

For bug fixes and shared-policy changes, provide evidence that lets another session verify the change without
rebuilding the investigation. Scale the handoff to the change; documentation-only edits do not need regression tests.

- Identify the base revision and the exact review target: a commit or an immutable patch/snapshot. If a regression
  arose in uncommitted work, also identify the pre-fix snapshot; a test against `HEAD` alone cannot establish that
  regression. Commit only when authorized. Use isolated checkouts or snapshots for comparisons; never stash, reset,
  or otherwise alter another session's working tree to perform a review.
- Map each finding to the changed behavior, implementation location, and exact regression test or probe command.
  Show the same behavioral check failing before the fix and passing afterward, with revisions/snapshots and relevant
  environment details. Import errors, missing dependencies, and missing new APIs are not fail-first proof. If a
  before-run is unavailable or the finding is structural, state that limitation instead of claiming reproduction.
- Before changing a category list, exception-swallowing rule, or replay predicate, enumerate its members and callers.
  Check the resulting behavior in every affected path, including success, ordinary failure, per-item timeout, batch
  abort/deadline, and persistence failure where applicable. Distinct policies may need distinct named sets; do not
  equate replay eligibility with permission to swallow checkpoint errors. Include the policy sweep in the handoff.
- Exercise both JSONL and SQLite for replay, lookup, or record-eligibility changes. Reuse reviewer-supplied probes
  and existing fixtures, folding useful cases into the nearest regression suite rather than duplicating them.
- State validation commands, results, interpreter versions, deliberate omissions, and accepted tradeoffs precisely.
  A passing full suite does not replace finding-specific evidence. Reviewers should reuse that evidence and run
  focused checks; repeat full suites when new changes or unresolved concerns warrant it. Verify Python 3.10 and a
  current supported interpreter when cancellation or asyncio semantics may differ.
- If review is delegated, include the exact target, settled decisions, known limitations, and out-of-scope items
  so subsequent rounds do not rediscover accepted tradeoffs. Verify candidate findings before presenting them as
  confirmed regressions.
