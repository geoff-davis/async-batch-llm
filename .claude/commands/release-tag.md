# Release Tag

Tag a merged release, confirm the PyPI publish, then create the GitHub Release. Run this after a `/release-prep` PR
has been merged.

`.github/workflows/publish.yml` (triggered by the tag push) runs the test gate, checks the tag against
`pyproject.toml`, runs `make package-check`, and publishes to PyPI. It does **not** create the GitHub Release; this
command does, and only after the publish is confirmed.

## Steps

### 1. Verify merge

- Confirm we are on the `main` branch. If not, switch to it.
- Run `git pull` to get the latest.
- Read the current version from `pyproject.toml` (`version = "..."`).
- Verify no tag `v<version>` exists yet, locally or on the remote. `git tag -l` exits 0 even with no match, so check
  the **output**, not the exit code: both `git tag -l "v<version>"` and
  `git ls-remote --tags origin "refs/tags/v<version>"` must print nothing.
- If a tag already exists for this version, tell the user and stop.

### 2. Confirm with user

- Show the user the version that will be tagged and ask for confirmation before proceeding.

### 3. Tag and push

- Run `git tag v<version>` on the current HEAD.
- Run `git push origin v<version>`.
- Tell the user the tag has been pushed and that the PyPI publish workflow has been triggered.

### 4. Wait for the publish

- Find the run for the tag: `gh run list --workflow publish.yml --branch v<version> --limit 1`
  (retry for a minute or so if it hasn't appeared yet).
- Watch it to completion: `gh run watch <run-id> --exit-status`.
- If the run fails, stop. Report the failing job and step (`gh run view <run-id> --log-failed`) and do **not** create
  the GitHub Release. The tag stays pushed; how to recover (re-running the workflow, or deleting and re-pushing the
  tag after a fix) is the user's call.
- Confirm the version is on PyPI: `curl -sf https://pypi.org/pypi/async-batch-llm/<version>/json` must succeed.
  PyPI's JSON API can lag the upload briefly; retry for a minute or two before giving up.

### 5. Create GitHub Release

Only after step 4 confirmed both the green run and the PyPI version.

- Read the `[<version>]` section from `CHANGELOG.md` to use as release notes.
- Inline any reference-style links (e.g. `[#52]` → `[#52](https://github.com/...)`)
  using their definitions in `CHANGELOG.md` — the definitions are not part of
  the pasted notes, so un-inlined refs render as literal `[#52]`
  text on GitHub (this bit the v0.15.0 release notes).
- Rewrite relative links (e.g. `docs/migration/v0.27.md`) to absolute
  `https://github.com/geoff-davis/async-batch-llm/blob/v<version>/...` URLs; relative links break in a release body.
- Run `gh release create v<version> --title "v<version>" --latest --notes "<changelog section>"`.
- Confirm it exists with `gh release view v<version>`.

### 6. Clean up and report

- Delete the local release branch if it still exists (`release/v<version>`).
- Report completion only now, with the publish run URL, the PyPI URL
  (`https://pypi.org/project/async-batch-llm/<version>/`), and the GitHub Release URL.
- Remind the user of the post-release legacy-fixture step (CONTRIBUTING.md, "Release Process"): in a follow-up PR,
  run `scripts/write_legacy_fixtures.py` with the new release installed and add it to `RELEASES` in
  `tests/test_legacy_fixtures.py`.
