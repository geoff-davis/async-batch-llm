# Contributing to batch-llm

Thank you for considering contributing to batch-llm!

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/geoff-davis/batch-llm.git
cd batch-llm

# Create virtual environment and install dependencies
uv venv
uv sync --all-extras
```

### 2. Install Pre-commit Hooks

```bash
uv run pre-commit install
```

This will automatically run code quality checks before each commit.

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_basic.py -v

# Run with coverage
uv run pytest --cov=batch_llm --cov-report=html
```

### Code Quality

Always run quality checks before committing:

```bash
# Format code
uv run ruff format src/ tests/ examples/

# Lint and auto-fix issues
uv run ruff check src/ tests/ examples/ --fix

# Verify linting passes
uv run ruff check src/ tests/ examples/

# Type check
uv run mypy src/batch_llm/ --ignore-missing-imports

# Or run all checks at once
make ci
```

### Documentation

Build and preview documentation:

```bash
# Install docs dependencies
uv sync --extra docs

# Serve docs locally
uv run mkdocs serve

# Build docs
uv run mkdocs build
```

Then visit <http://localhost:8000>

### Markdown Linting

```bash
# Lint markdown files
npx markdownlint-cli2 "README.md" "docs/**/*.md" "CLAUDE.md"

# Auto-fix markdown issues
npx markdownlint-cli2 "README.md" "docs/**/*.md" "CLAUDE.md" --fix

# Or use make target
make markdown-lint-fix
```

## Pre-Commit Checklist

Before committing, ensure:

1. ✅ All tests pass: `uv run pytest`
2. ✅ Linting passes: `uv run ruff check src/ tests/`
3. ✅ Type checking passes: `uv run mypy src/batch_llm/`
4. ✅ Markdown is clean: `make markdown-lint`

Or run everything at once:

```bash
make ci
```

## Pull Request Guidelines

1. **Create a feature branch**: `git checkout -b feature/your-feature`
2. **Write tests**: Add tests for new functionality
3. **Update docs**: Update relevant documentation
4. **Run quality checks**: Ensure all checks pass
5. **Write clear commit messages**: Explain the "why" not just "what"
6. **Open PR**: Provide a clear description of changes

## Project Structure

```text
batch-llm/
├── src/batch_llm/          # Main package
│   ├── base.py             # Core data models
│   ├── parallel.py         # Main processor
│   ├── llm_strategies/     # Strategy implementations
│   ├── observers/          # Observer implementations
│   └── testing/            # Testing utilities
├── tests/                  # Test suite
├── examples/               # Example scripts
├── docs/                   # Documentation
└── CLAUDE.md              # AI assistant context
```

## Adding New Strategies

To add a new LLM provider strategy:

1. Create strategy in `src/batch_llm/llm_strategies/`
2. Implement `LLMCallStrategy` protocol
3. Add tests in `tests/`
4. Add example in `examples/`
5. Update documentation in `docs/examples/custom-strategies.md`

Example:

```python
from batch_llm import LLMCallStrategy

class MyProviderStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Your implementation
        return output, tokens
```

## Questions?

- Open an [issue](https://github.com/geoff-davis/batch-llm/issues)
- Start a [discussion](https://github.com/geoff-davis/batch-llm/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
