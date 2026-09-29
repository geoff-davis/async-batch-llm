# Contributing to async-batch-llm

Thank you for considering contributing to async-batch-llm!

To report a security vulnerability, don't open a public issue; follow the
[security policy](https://github.com/geoff-davis/async-batch-llm/blob/main/SECURITY.md).
Maintainer material, including the release process, lives in
[CONTRIBUTING.md](https://github.com/geoff-davis/async-batch-llm/blob/main/CONTRIBUTING.md).

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/geoff-davis/async-batch-llm.git
cd async-batch-llm

# Create virtual environment and install dependencies
uv venv
uv sync --all-extras
```

### 2. Install Git Hooks

Hooks are managed with [prek](https://github.com/j178/prek), a drop-in
replacement for pre-commit (same `.pre-commit-config.yaml`):

```bash
uv run prek install
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
uv run pytest --cov=async_batch_llm --cov-report=html
```

### Code Quality

Always run quality checks before committing:

```bash
# Format code (make format)
uv run ruff format src/ tests/

# Lint and auto-fix issues (make lint-fix)
uv run ruff check src/ tests/ --fix

# Verify linting passes (make lint)
uv run ruff check src/ tests/

# Type check
uv run mypy src/async_batch_llm/ --ignore-missing-imports

# Or run all checks at once
make ci
```

`examples/` is excluded from ruff on purpose: example scripts check environment
variables before importing optional dependencies, which ruff reports as E402.

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

Requires Node 20+; run `npm ci` once to install the pinned `markdownlint-cli2`.
Both targets lint every tracked `.md` file except `docs/archive/`, the same set as
the prek hook:

```bash
# Lint markdown files
make markdown-lint

# Auto-fix markdown issues
make markdown-lint-fix
```

## Pre-Commit Checklist

Before committing, ensure:

1. ✅ All tests pass: `uv run pytest`
2. ✅ Linting passes: `uv run ruff check src/ tests/`
3. ✅ Type checking passes: `uv run mypy src/async_batch_llm/`
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
async-batch-llm/
├── src/async_batch_llm/          # Main package
│   ├── base.py             # Core data models
│   ├── parallel.py         # Main processor
│   ├── streaming.py        # process_prompts / process_stream
│   ├── single.py           # call / call_result
│   ├── gateway.py          # LLMCallPool
│   ├── factory.py          # llm("provider:model")
│   ├── llm_strategies.py   # LLMCallStrategy + built-in strategies
│   ├── callable_strategy.py # CallableStrategy for existing async clients
│   ├── models.py           # Provider model classes
│   ├── artifacts.py        # JsonlArtifactStore, ArtifactIdentity, ResumePolicy
│   ├── sqlite_artifacts.py # SqliteArtifactStore
│   ├── serialization.py    # Result JSON/JSONL serialization
│   ├── classifiers/        # Provider error classifiers
│   ├── core/               # Config and protocols
│   ├── observers/          # Observer implementations
│   ├── middleware/         # Middleware protocol
│   ├── _internal/          # Private orchestration collaborators
│   └── testing/            # Testing utilities
├── tests/                  # Test suite
├── examples/               # Example scripts
├── docs/                   # Documentation
└── CLAUDE.md              # AI assistant context
```

## Adding New Strategies

To add a new LLM provider strategy:

1. Add the strategy to `src/async_batch_llm/llm_strategies.py` (or your own module)
2. Subclass the `LLMCallStrategy` abstract base class
3. Add tests in `tests/`
4. Add example in `examples/`
5. Update documentation in `docs/examples/custom-strategies.md`

Example:

```python
from async_batch_llm import LLMCallStrategy

class MyProviderStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
        # Your implementation
        return output, tokens, None
```

## Questions?

- Open an [issue](https://github.com/geoff-davis/async-batch-llm/issues)
- Start a [discussion](https://github.com/geoff-davis/async-batch-llm/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
