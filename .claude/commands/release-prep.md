# Release Prep

Prepare a new release of async-batch-llm. This skill handles changelog
generation, version bumping, and creating the release PR.

## Arguments

- `$ARGUMENTS` — optional version string (e.g., `0.7.0`). If omitted, infer from changelog categories.
  Pass it explicitly for a major release such as `1.0.0`; the heuristic below never suggests one.

## Steps

### 0. Pre-flight checks

- Run `git fetch origin main` to ensure we have the latest remote state.
- Check for uncommitted changes. If there are any, stop and tell the user
  to commit or stash them first (branch switching may lose work).
- Check that the previous release has legacy fixtures: `tests/fixtures/vX_Y/` exists and `RELEASES` in
  `tests/test_legacy_fixtures.py` lists it. If not, tell the user and offer to add them first, in their own PR:

  ```bash
  uv run --no-project --python 3.12 --with async-batch-llm==X.Y.Z \
      python scripts/write_legacy_fixtures.py tests/fixtures/vX_Y
  ```

  then add `"vX_Y": "X.Y.Z"` to `RELEASES` and to the expected counts in `test_every_release_has_its_fixtures`.
  Never regenerate a fixture directory that is already committed.

### 1. Generate changelog entries

- Run `git describe --tags --abbrev=0 origin/main` to find the latest tag.
- Run `git log <latest-tag>..origin/main --oneline` to get commits since that tag.
- Read `CHANGELOG.md` and check the `[Unreleased]` section.
- If `[Unreleased]` is empty, auto-generate entries from the commit log, grouped by category:
  - **Added** — new features or capabilities
  - **Changed** — changes to existing functionality
  - **Deprecated** — features that still work but will be removed
  - **Removed** — features removed in this release
  - **Fixed** — bug fixes
  - **Performance** — performance improvements
  - **Documentation** — docs changes
  - **Refactor** — code restructuring without behavior change
- If `[Unreleased]` is already populated (usual when feature PRs add their own entries), keep it. Compare it with
  the commit log and point out any merged PR without an entry.
- Present the entries to the user and ask for confirmation before proceeding.

### 2. Determine version

- Read the current version from `pyproject.toml` (`version = "..."`).
- If a version was provided as `$ARGUMENTS`, use that.
- Otherwise, infer the bump type from the changelog categories using semver conventions:
  - If there are **Added**, **Changed**, **Deprecated**, or **Removed** entries → suggest a **minor** bump
  - If there are only **Fixed**, **Documentation**, **Performance**, or **Refactor** entries → suggest a **patch** bump
- Present the suggested version to the user and ask for confirmation.

### 3. Update CHANGELOG.md

- Move the whole `[Unreleased]` body (lead-in lines, category sections, and the reference-link definitions such as
  `[#177]: https://...` at its end) under a new `[<version>] - <YYYY-MM-DD>` heading directly below `[Unreleased]`.
  Leave `[Unreleased]` empty.
- Check every link in the new section: each `[#NNN]` has a definition, and each relative link (for example the
  migration guide) points to a file that exists.

### 4. Bump version

- Update `version` in `pyproject.toml` to the new version.
- Run `uv lock` — the lockfile records the project's own version, so it must
  be regenerated with the bump (otherwise `uv lock --check` / `--frozen`
  consumers, e.g. the CI security job's `uv export --frozen`, fail).
- Update the "Current version" line near the top of `CLAUDE.md`, and retitle
  the `**Unreleased**` entry in its "Version history" section to the new
  version.

### 5. Update the roadmap

- Add a one-line row for the new version to the table in `docs/roadmap.md`, matching the existing rows, with a
  link to its migration guide if the release has one. Update any footnote there that names the previous release.

### 6. Check the package

- Run `make package-check` (build, `twine check --strict`, `check-wheel-contents`). The tag-triggered publish
  workflow runs the same gate, so a failure there would otherwise show up only after tagging.

### 7. Create release branch and PR

- Create branch `release/v<version>` from `origin/main`.
- Stage `CHANGELOG.md`, `pyproject.toml`, `uv.lock`, `CLAUDE.md`, and `docs/roadmap.md`.
- Commit with message: `Prepare release v<version>`
- Push the branch.
- Create a PR with title `Prepare release v<version>` and body summarizing the changelog entries.
- Report the PR URL to the user.
- Remind the user: after CI passes and the PR is merged, run `/release-tag` to tag and publish.
