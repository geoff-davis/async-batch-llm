# Contributing to async-batch-llm

Thank you for your interest in contributing to async-batch-llm! This document provides guidelines and instructions for
contributing.

To report a security vulnerability, don't open a public issue; follow [SECURITY.md](SECURITY.md).

## Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/geoff-davis/async-batch-llm.git
cd async-batch-llm
```

1. **Install uv** (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. **Install dependencies**

```bash
uv sync --all-extras
```

1. **Install markdown lint tooling** (requires Node 20+; installs the version pinned in `package-lock.json`)

```bash
npm ci
```

1. **Install git hooks via prek** (recommended)

```bash
make pre-commit-install
# or manually:
uv run prek install
```

This will automatically run linting and type checking before each commit.

1. **Run tests**

```bash
uv run pytest
```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=async_batch_llm --cov-report=html

# Run specific test file
uv run pytest tests/test_basic.py

# Run with verbose output
uv run pytest -v
```

### Code Quality

We use several tools to maintain code quality:

#### Automated Git Hooks via prek (Recommended)

```bash
# Install hooks (one-time setup)
make pre-commit-install

# Run hooks manually on all files
make pre-commit-run

# Update hooks to latest versions
make pre-commit-update
```

prek will automatically run these checks before each commit:

- Ruff linting and formatting
- Mypy type checking
- Markdown linting
- General file checks (trailing whitespace, merge conflicts, etc.)

#### Manual Code Quality Checks

```bash
# Format code (same as `make format`)
uv run ruff format src/ tests/

# Lint code (same as `make lint-fix`)
uv run ruff check src/ tests/ --fix

# Type check
uv run mypy src/async_batch_llm/ --ignore-missing-imports

# Markdown lint, all tracked .md files except docs/archive/ (requires npm ci first)
make markdown-lint-fix

# Run all checks at once
make ci
```

`examples/` is excluded from ruff on purpose: example scripts check environment
variables before importing optional dependencies, which ruff reports as E402.

### Running Examples

```bash
# Run the main example (its default MockAgent example needs no API key)
uv run python examples/example.py

# The Gemini examples in it (commented out in main()) and most other examples need a key
export GOOGLE_API_KEY="your-api-key"  # GEMINI_API_KEY is also accepted
```

## Making Changes

1. **Create a branch** for your changes

```bash
git checkout -b feature/your-feature-name
```

1. **Make your changes** following these guidelines:
   - Write clear, descriptive commit messages
   - Add tests for new functionality
   - Update documentation as needed
   - Follow existing code style

2. **Test your changes**

```bash
uv run pytest
uv run ruff check src/
uv run mypy src/async_batch_llm/
```

1. **Submit a pull request**
   - Describe what your changes do
   - Reference any related issues
   - Ensure all tests pass

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for public APIs
- Keep functions focused and concise
- Use descriptive variable names

## Testing Guidelines

- Write tests for all new features
- Aim for high test coverage
- Use `MockAgent` for tests that don't require API calls
- Test both success and failure cases
- Test edge cases and error handling

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new classes and functions
- Update examples/ if adding new features
- Update CHANGELOG.md following Keep a Changelog format

## Release Process

(For maintainers.) Publishing is **tag-triggered and automated** — do not
run `uv publish` manually; the workflow uses PyPI trusted publishing (OIDC)
and a manual upload would race or fail on auth.

1. **Prepare** (the `/release-prep` skill automates this): move the body of
   the CHANGELOG `[Unreleased]` section, including its reference-link
   definitions, under a new `[X.Y.Z] - YYYY-MM-DD` heading and leave an empty
   `[Unreleased]` above it; check that every link in the section (migration
   guide, issues) resolves. Bump `version` in `pyproject.toml`; run `uv lock`
   so the lockfile picks up the new project version; update the "Current
   version" line in `CLAUDE.md`; add a one-line row for the release to
   `docs/roadmap.md`, linking its migration guide if there is one. Run
   `make package-check` locally (the publish workflow runs it too). Open a
   "Prepare release vX.Y.Z" PR from a `release/vX.Y.Z` branch and merge it
   once CI is green.
2. **Tag** (the `/release-tag` skill automates this and steps 3–4): from the
   updated `main`, `git tag vX.Y.Z && git push origin vX.Y.Z`. The tag push
   triggers `.github/workflows/publish.yml`, which re-runs the test gate,
   verifies the tag matches `pyproject.toml`, builds the package and runs
   `make package-check` (`twine check --strict` and `check-wheel-contents`),
   and publishes to PyPI. Either gate can fail the release.
3. **Confirm the publish** before announcing anything: the publish workflow
   run for the tag is green (`gh run watch`), and
   `https://pypi.org/pypi/async-batch-llm/X.Y.Z/json` resolves.
4. **GitHub release**, only after step 3: `gh release create vX.Y.Z --title "vX.Y.Z" --latest`
   with the `[X.Y.Z]` CHANGELOG section as the notes. Inline reference-style
   links such as `[#52]` (their definitions aren't part of the pasted text)
   and rewrite relative links such as `docs/migration/vX.Y.md` to absolute
   `https://github.com/geoff-davis/async-batch-llm/blob/vX.Y.Z/...` URLs;
   both render as broken text in a release body.
5. **Verify**: the GitHub release exists and the docs site redeployed
   (happens on the merge to `main`).
6. **Legacy fixtures** (next PR after the release): record what the new
   release writes, so later releases keep reading it:

    ```bash
    uv run --no-project --python 3.12 --with async-batch-llm==X.Y.Z \
        python scripts/write_legacy_fixtures.py tests/fixtures/vX_Y
    ```

    Then add `"vX_Y": "X.Y.Z"` to `RELEASES` in `tests/test_legacy_fixtures.py`
    and its entry to the expected counts in `test_every_release_has_its_fixtures`.
    Never regenerate a committed fixture directory.

## Questions?

Feel free to open an issue for:

- Bug reports
- Feature requests
- Questions about development
- Documentation improvements

Thank you for contributing!
