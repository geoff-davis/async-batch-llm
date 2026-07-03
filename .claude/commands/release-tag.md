# Release Tag

Tag a merged release and publish to PyPI + GitHub Releases. Run this after a `/release-prep` PR has been merged.

## Steps

### 1. Verify merge

- Confirm we are on the `main` branch. If not, switch to it.
- Run `git pull` to get the latest.
- Read the current version from `pyproject.toml` (`version = "..."`).
- Verify the version has been bumped (i.e., no tag `v<version>` exists yet).
- If a tag already exists for this version, tell the user and stop.

### 2. Confirm with user

- Show the user the version that will be tagged and ask for confirmation before proceeding.

### 3. Tag and push

- Run `git tag v<version>` on the current HEAD.
- Run `git push origin v<version>`.
- Tell the user the tag has been pushed and that the PyPI publish workflow has been triggered.

### 4. Create GitHub Release

- Read the `[<version>]` section from `CHANGELOG.md` to use as release notes.
- Inline any reference-style links (e.g. `[#52]` → `[#52](https://github.com/...)`)
  using the definitions at the bottom of `CHANGELOG.md` — the definitions are
  not copied with the section, so un-inlined refs render as literal `[#52]`
  text on GitHub (this bit the v0.15.0 release notes).
- Run `gh release create v<version> --title "v<version>" --latest --notes "<changelog section>"`.
- Report the release URL to the user.

### 5. Clean up

- Delete the local release branch if it still exists (`release/v<version>`).
- Remind the user to verify the PyPI publish succeeded at the GitHub Actions tab.
