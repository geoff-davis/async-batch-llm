.PHONY: help test coverage lint typecheck format check-all pre-commit clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

test:  ## Run all tests
	uv run pytest tests/ -v

test-fast:  ## Run tests excluding slow tests (default)
	uv run pytest tests/ -v -m 'not slow'

test-all:  ## Run all tests including slow ones
	uv run pytest tests/ -v -m ''

coverage:  ## Run tests with coverage report
	uv run pytest --cov=batch_llm --cov-report=term-missing --cov-report=html

coverage-report:  ## Generate and open HTML coverage report
	uv run pytest --cov=batch_llm --cov-report=html
	@echo "\n==> Opening coverage report..."
	open htmlcov/index.html || xdg-open htmlcov/index.html || echo "Please open htmlcov/index.html manually"

lint:  ## Run ruff linter
	uv run ruff check src/ tests/

lint-fix:  ## Run ruff linter with auto-fix
	uv run ruff check src/ tests/ --fix

format:  ## Format code with ruff
	uv run ruff format src/ tests/

typecheck:  ## Run mypy type checker
	uv run mypy src/batch_llm/ --ignore-missing-imports

markdown-lint:  ## Check markdown files
	npx markdownlint-cli2 "README.md" "docs/*.md" "CLAUDE.md"

markdown-lint-fix:  ## Fix markdown issues
	npx markdownlint-cli2 "README.md" "docs/*.md" "CLAUDE.md" --fix

check-all:  ## Run all checks (lint + typecheck + test)
	@echo "==> Running linter..."
	@$(MAKE) lint
	@echo "\n==> Running type checker..."
	@$(MAKE) typecheck
	@echo "\n==> Running tests..."
	@$(MAKE) test-fast
	@echo "\n==> All checks passed! ✓"

ci:  ## Run CI checks (what GitHub Actions runs)
	@echo "==> Running linter..."
	@$(MAKE) lint
	@echo "\n==> Running type checker..."
	@$(MAKE) typecheck
	@echo "\n==> Running tests..."
	@$(MAKE) test
	@echo "\n==> Running markdown linter..."
	@$(MAKE) markdown-lint
	@echo "\n==> CI checks passed! ✓"

pre-commit-install:  ## Install pre-commit hooks
	pip install pre-commit
	pre-commit install
	@echo "\n==> Pre-commit hooks installed! They will run automatically on 'git commit'"

pre-commit-run:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update:  ## Update pre-commit hooks to latest versions
	pre-commit autoupdate

clean:  ## Clean up cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete

.DEFAULT_GOAL := help
