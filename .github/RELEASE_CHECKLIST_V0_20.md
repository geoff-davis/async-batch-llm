# v0.20 Release Checklist

Working checklist for GitHub issue #107. v0.19.0 was not published; its planned
work is included in v0.20.0. Check release actions only after the candidate is
reviewed and the corresponding external result is verified.

## Code and tests

- [x] Formatting passes.
- [x] Ruff lint passes.
- [x] mypy and ty pass.
- [x] Fast tests pass.
- [x] Full tests and coverage gate pass.
- [x] Strict documentation build passes.
- [x] Supported Python 3.10–3.14 matrix passes.
- [x] No pending-task warnings.
- [x] Progress tests are deterministic and non-flaky.
- [x] Callable, retry-state, artifact, and streaming regressions pass.

## Documentation

- [x] README quick start uses the high-level API.
- [x] Documentation home and getting-started path are current.
- [x] Comparison page is linked and includes Bespoke Curator.
- [x] Troubleshooting and FAQ page is linked.
- [x] v0.20 migration targets v0.18.x.
- [x] API navigation covers callable integration and `LLMCallPool`.
- [x] Terminal asset exists, is legible, portable, and below 2 MB.
- [x] No-key Colab notebook validates and runs.
- [x] PyPI README links and images are absolute/portable.
- [x] No stale feature claim says v0.19 was published.
- [x] Result-handoff documentation covers `max_result_queue_size`.

## Packaging

- [x] `pyproject.toml` version is `0.20.0`.
- [x] Maintainer supplied the 2026-07-20 release date; changelog finalized.
- [x] Intended `v0.20.0` tag agrees with package metadata.
- [x] Wheel and sdist build.
- [x] Wheel installs in a clean environment.
- [x] Public import smoke test passes from the installed wheel.
- [x] `py.typed` is present.
- [x] Optional extras metadata is correct.
- [x] Long description renders successfully.
- [x] Terminal image and Colab link survive package rendering.
- [x] Distribution contents are reviewed; no development files or secrets leak.

## Release — maintainer only

- [ ] Create the `v0.20.0` tag.
- [ ] Push the tag.
- [ ] Verify the publish workflow.
- [ ] Verify PyPI metadata and project page.
- [ ] Verify PyPI attestations.
- [ ] Create the GitHub release from the reviewed draft.
- [ ] Verify documentation deployment.

## Promotion — optional, maintainer only

- [ ] Review factual announcement copy.
- [ ] Post only after package, release, and docs URLs are verified.
