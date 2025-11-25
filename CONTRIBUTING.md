# Contributing to Batch LLM

Thank you for your interest in contributing to Batch LLM! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/geoff-davis/batch-llm.git
cd batch-llm
```

1. **Install uv** (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. **Install dependencies**

```bash
uv sync --all-extras
```

1. **Install markdown lint tooling** (requires Node 18+)

```bash
npm install --save-dev markdownlint-cli2
```

1. **Install pre-commit hooks** (recommended)

```bash
make pre-commit-install
# or manually:
pip install pre-commit
pre-commit install
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
uv run pytest --cov=batch_llm --cov-report=html

# Run specific test file
uv run pytest tests/test_basic.py

# Run with verbose output
uv run pytest -v
```

### Code Quality

We use several tools to maintain code quality:

#### Automated Pre-commit Hooks (Recommended)

```bash
# Install hooks (one-time setup)
make pre-commit-install

# Run hooks manually on all files
make pre-commit-run

# Update hooks to latest versions
make pre-commit-update
```

Pre-commit will automatically run these checks before each commit:

- Ruff linting and formatting
- Mypy type checking
- Markdown linting
- General file checks (trailing whitespace, merge conflicts, etc.)

#### Manual Code Quality Checks

```bash
# Format code
uv run ruff format src/ tests/ examples/

# Lint code
uv run ruff check src/ tests/ examples/ --fix

# Type check
uv run mypy src/batch_llm/ --ignore-missing-imports

# Markdown lint (requires npm install first)
npx markdownlint-cli2 "README.md" "docs/*.md" "CLAUDE.md" --fix

# Run all checks at once
make ci
```

### Running Examples

```bash
# Run the main example (requires API key)
uv run python examples/example.py

# Set API key first if needed
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
uv run mypy src/batch_llm/
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

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create a git tag: `git tag vX.Y.Z` (matching the version in `pyproject.toml`)
4. Build: `uv build`
5. Publish to PyPI: `uv publish`

## Questions?

Feel free to open an issue for:

- Bug reports
- Feature requests
- Questions about development
- Documentation improvements

Thank you for contributing!
