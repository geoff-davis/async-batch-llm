# Project Knowledge for Claude

Project-specific context that future Claude sessions load on startup. Keep
it tight — when in doubt, link to `docs/` rather than duplicate content
here.

---

## Project overview

**async-batch-llm** processes batches of LLM requests in parallel using a
**strategy pattern** — provider-agnostic at the framework level, with
first-class support for several providers built in.

**Current version:** v0.28.0, the last release before 1.0 and the last to
support Python 3.10 (see `CHANGELOG.md`; `pyproject.toml` is bumped
by the release-prep flow, so it may briefly lag `main` between releases).

**Key features:**

- Parallel asyncio processing with configurable concurrency
- Built-in rate limiting and exponential backoff retry logic
- Scoped, token-aware RPM/TPM admission and coordinated cooldowns
- Resumable JSONL/SQLite checkpoints, item and batch deadlines, token/cost
  budgets, and category-based fail-fast
- Concurrency-safe shared state across tasks on one event loop (`asyncio.Lock`-based)
- Provider-agnostic core: bring your own strategy/model/classifier, or wrap an
  existing async client with `CallableStrategy`
- Built-in Gemini, OpenAI, OpenRouter, DeepSeek, and PydanticAI support
- Middleware and observer patterns for extensibility
- `FakeStrategy` and `MockAgent` for testing without API calls

---

## Quick reference

High-level streaming API (`streaming.py`) for the common case — collect, or
stream as items finish. Built on the processor's first-class streaming mode
(`start()`/`add_work()`/`finish()`/`results()`), so a bounded `max_queue_size`
gives backpressure (constant memory for huge inputs). Error classifier is
auto-selected from the strategy:

```python
from async_batch_llm import llm, process_prompts, process_stream

strategy = llm("openai:gpt-6-luna")  # factory (v0.20); explicit form:
# strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-6-luna"))

result = await process_prompts(strategy, ["Summarize A", "Summarize B"])  # -> BatchResult
async for r in process_stream(strategy, prompts):  # yields WorkItemResult in completion order
    ...
```

Full-control example (drive `ParallelBatchProcessor` directly):

```python
from async_batch_llm import (
    LLMWorkItem,
    OpenAIModel,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
)

model = OpenAIModel.from_api_key("gpt-6-luna")  # reads OPENAI_API_KEY
strategy = OpenAIStrategy(model)
config = ProcessorConfig(max_workers=5, attempt_timeout=60.0)

async with ParallelBatchProcessor[None, str, None](config=config) as processor:
    for i, prompt in enumerate(prompts):
        await processor.add_work(
            LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=prompt)
        )
    result = await processor.process_all()

print(f"Succeeded: {result.succeeded}/{result.total_items}")
```

See `examples/example.py` for the full-featured walkthrough (context
passing, post-processors, middleware, observers, error handling).

---

## Architecture

### Core abstractions

- **`LLMCallStrategy[TOutput]`** (`llm_strategies.py`) — abstract base for
  LLM integrations.
  - `async prepare()` — initialize resources (caches, connections)
  - `async execute(prompt, attempt, timeout, state=None)` — make the call
  - `async on_error(exception, attempt, state=None)` — track error types
  - `async cleanup()` — release resources
- **`LLMModel` / `ManagedLLMModel`** (`core/protocols.py`) — provider-side
  protocol. `ManagedLLMModel` adds `prepare`/`cleanup` for cache or client
  lifecycle.
- **`LLMResponse`** (`base.py`) — normalized response: `text`, token counts
  (input/output/total/cached), provider metadata dict, raw response.
- **`LLMWorkItem`** (`base.py`) — work unit: `item_id`, `strategy`,
  `prompt: str`, optional `context`.
- **`ParallelBatchProcessor`** (`parallel.py`) — worker pool, rate-limit
  coordination, retry/backoff, framework-level timeout via
  `asyncio.wait_for()`. Optional `progress_callback(completed, total,
  current_item_id)` for live progress (sync or async, with configurable
  `progress_callback_timeout`).

### Built-in providers

| Provider   | Model class                         | Strategy class       | Error classifier            | Optional dep      |
|------------|-------------------------------------|----------------------|-----------------------------|-------------------|
| Gemini     | `GeminiModel`, `GeminiCachedModel`  | `GeminiStrategy`     | `GeminiErrorClassifier`     | `[gemini]`        |
| OpenAI     | `OpenAIModel`                       | `OpenAIStrategy`     | `OpenAIErrorClassifier`     | `[openai]`        |
| OpenRouter | `OpenRouterModel`                   | `OpenRouterStrategy` | `OpenRouterErrorClassifier` | `[openrouter]`    |
| DeepSeek   | `DeepSeekModel`                     | `DeepSeekStrategy`   | `OpenAIErrorClassifier`     | `[deepseek]`      |
| PydanticAI | (any model wrapped)                 | `PydanticAIStrategy` | `PydanticAIErrorClassifier` | `[pydantic-ai]`   |

`OpenAIModel` uses the Responses API by default (v0.27, `store=False`);
`api_surface="chat_completions"` opts out. `DeepSeekModel` defaults to Chat
Completions and accepts `api_surface="responses"` (needed for strict
`response_schema`); OpenRouter and plain `OpenAICompatibleModel` use Chat
Completions.

`OpenAICompatibleModel` is the base for OpenAI/OpenRouter/DeepSeek — and all
three model strategies are thin subclasses of `ModelStrategy` (shared
`execute()`/lifecycle). Subclass `OpenAICompatibleModel` for Together,
Fireworks, vLLM, etc. by overriding `_default_base_url` and optionally
`_extract_tokens` (as `DeepSeekModel` does for its native
`prompt_cache_hit_tokens` field).

For provider deep dives:

- `docs/GEMINI_INTEGRATION.md`
- `docs/OPENAI_INTEGRATION.md`
- `docs/OPENROUTER_INTEGRATION.md`

### Concurrency and locks

The locks are `asyncio.Lock`s: they serialize tasks on one event loop and give
no OS-thread safety. Most guard one piece of state and are released before
another lock is taken. Ownership:

- `_stats_lock`, `_results_lock`, `_submission_lock` — defined on the
  `BatchProcessor` base (`base.py`), used by `ParallelBatchProcessor`.
- `_rate_limit_lock` — a `ParallelBatchProcessor` property returning
  `RateLimitCoordinator._lock`.
- Collaborators own their own locks: `StrategyLifecycle`, `_internal/guardrails.py`,
  `_internal/executor_host.py` (stats for `single.py`/`call_pool.py`), the JSONL
  store, and `MetricsObserver`. Models have their own (client lifecycle, and
  `GeminiCachedModel`'s cache lock).

Known nesting: `StrategyLifecycle` holds its lock while running
`strategy.prepare()`, so a lock a model takes inside `prepare()` (for example
`GeminiCachedModel`'s cache lock) is acquired under it. Keep that order
(lifecycle, then model), and don't call back into the lifecycle from model
code. When adding a lock, check whether it can be taken while another is held.

---

## Critical design decisions

### Strategy pattern (v0.1)

Decouples framework from providers. Each strategy encapsulates how the
call is made; framework handles retry/timeout/rate limiting uniformly.
Migration from the pre-strategy API at `docs/archive/MIGRATION_V0_1.md`.

### Rate-limiting coordination

The rate-limit state machine lives on `RateLimitCoordinator`
(`_internal/rate_limit_coordinator.py`) since the v0.7.0 decomposition.
`ParallelBatchProcessor._in_cooldown` is a read-only property that
delegates to the coordinator — don't try to assign it directly.

When one worker hits a rate limit:

1. Atomic check-and-set inside the coordinator's lock: only one worker
   triggers the cooldown for a given generation. Stale callers (whose
   observed generation predates the current cooldown) silently no-op.
2. All workers pause via `asyncio.Event` (cleared on cooldown, set on
   resume).
3. Slow-start ramp after cooldown — progressive delays before workers
   resume normal throughput.
4. Consecutive rate limits trigger exponential backoff via the
   configurable `RateLimitStrategy`.

The actual implementation is in `RateLimitCoordinator._handle_rate_limit`;
read that file rather than copy a snippet here.

### Error-aware retry via `on_error()` (v0.1)

Different error types want different retry strategies:

- Validation error → escalate to smarter model (LLM quality issue)
- Network error → retry same cheap model (transient)
- Rate limit → retry same cheap model after cooldown (quota)

Strategies override `on_error()` to track error categories; `execute()`
reads counters to make per-attempt decisions. Common payoff: 60–80% cost
reduction via smart model escalation. See
`examples/example_smart_model_escalation.py`.

### Token usage tracking on failure

Tokens consumed by failed attempts are still tracked:

- `TokenExtractor` (`token_extractor.py`) reads `__cause__.result.usage()`,
  `.usage` attributes, or `__dict__["_failed_token_usage"]`.
- Strategies attach `_failed_token_usage` to exceptions when the
  underlying API has already billed but parsing fails.
- Aggregated across retry attempts; surfaces in
  `WorkItemResult.token_usage`.

### Provider-aware billing (v0.9)

`CachedTokenRates` constants (`GEMINI=0.10`, `OPENAI=0.50`,
`ANTHROPIC_READ=0.10`, `DEEPSEEK=0.02`) encode the fraction of normal
input price each provider charges for cached tokens. Pass to
`BatchResult.effective_input_tokens(rate)` / `estimated_cost(..., cached_token_rate=)`
for accurate billable counts. Omitting the rate falls back to `GEMINI` and is
deprecated since v0.27 (required in 1.0). The math conservatively rounds the billable estimate UP via `int()`
truncation of the discount.

---

## Common patterns

### PydanticAI strategy

```python
from pydantic_ai.models.google import GoogleModel

# A bare "gemini-3.5-flash" string fails on pydantic-ai 2.x; GoogleModel works on
# both 1.x (>=1.32) and 2.x. Construction needs GOOGLE_API_KEY.
agent = Agent(GoogleModel("gemini-3.5-flash"), output_type=Output)
strategy = PydanticAIStrategy(agent=agent)
work_item = LLMWorkItem(item_id="1", strategy=strategy, prompt="...")
```

### Built-in OpenAI / OpenRouter

```python
# OPENAI_API_KEY auto-resolved by SDK
model = OpenAIModel.from_api_key("gpt-6-luna")
strategy = OpenAIStrategy(model)

# OpenRouter — we read OPENROUTER_API_KEY ourselves (the SDK doesn't know
# about that env var). Raises ValueError if neither is set.
model = OpenRouterModel.from_api_key("anthropic/claude-haiku-4-5")
strategy = OpenRouterStrategy(model)
```

### Custom strategy for any provider

```python
class MyStrategy(LLMCallStrategy[str]):
    def __init__(self, client, model):
        self.client = client
        self.model = model

    async def execute(self, prompt, attempt, timeout, state=None):
        response = await self.client.generate(prompt, model=self.model)
        tokens = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return response.text, tokens
```

### Post-processing results

```python
async def save_result(result: WorkItemResult):
    if result.success and result.context:
        await db.save(result.context["id"], result.output)

processor = ParallelBatchProcessor(config=config, post_processor=save_result)
```

### Observing metrics

```python
metrics = MetricsObserver()
processor = ParallelBatchProcessor(config=config, observers=[metrics])
result = await processor.process_all()
collected = await metrics.get_metrics()
```

---

## Common pitfalls

### API pitfalls

- **Forgetting to `await` async methods.** `ParallelBatchProcessor.get_stats()`
  and `MetricsObserver.get_metrics()` are async.
- **Forgetting to wrap an Agent in a strategy.** `LLMWorkItem.strategy`
  expects a strategy instance, not a raw agent or model. Wrap:
  `PydanticAIStrategy(agent=...)`, `GeminiStrategy(model=...)`, etc.
- **Mutating results in post-processor.** Treat `result.output` as
  read-only or build new objects — concurrent post-processors share state.
- **Structured prompts.** `LLMWorkItem.prompt` is a string. For structured
  message lists (e.g. Anthropic `cache_control` markers via OpenRouter),
  build them inside a custom `execute()` and call
  `model.generate(messages_list)` directly. See
  `docs/OPENROUTER_INTEGRATION.md`.

### Workflow pitfalls

- **Fetch before you branch.** `git fetch origin` and cut new branches from
  `origin/main`, not local `main` — this repo is developed from multiple
  machines and the local clone has gone weeks stale before. On 2026-07-02 a
  full review + 22-commit fix batch was built against v0.10-era code while
  origin/main was already at v0.15.0; the resulting PR (#59) was conflicting
  and much of the work duplicated fixes main already had. See "Sync before
  working" below; the `check-branch-fresh` pre-push hook catches what the
  routine can't.
- **Use the right tool for file ops.** Read/Edit/Write/Glob/Grep — not
  bash `cat`/`sed`/`awk`. Bash is for git, npm, pytest, etc.
- **Read before editing.** The Edit tool requires a prior Read of the same
  file in this conversation.
- **Mutable defaults in Python.** Use `None` and initialize in the body:
  `def __init__(self, temps=None): self.temps = temps if temps is not None else [...]`.
- **`examples/` is excluded from ruff in pre-commit.** Examples
  intentionally check env vars before importing optional deps (E402).
  Don't try to "fix" them.

---

## Development workflow

### Sync before working

This repo is developed from multiple machines, so a locally-green checkout
can silently trail `origin/main`. Start every session with:

```bash
git fetch origin
git log --oneline main..origin/main   # anything here = local main is stale
git checkout main && git merge --ff-only origin/main
git checkout -b my-feature            # branch from the updated main
```

Then check for anything waiting on Geoff, and report it **before** starting the
session's task (an outside feature request once sat unanswered for 25 days):

```bash
python3 scripts/needs_response.py     # open issues/PRs whose latest human activity isn't a maintainer's
gh api repos/geoff-davis/async-batch-llm/dependabot/alerts \
  --jq '.[] | select(.state=="open") | [.number, .security_advisory.severity, .dependency.package.name, .security_advisory.summary] | @tsv'
```

The `Needs response` workflow (`.github/workflows/needs-response.yml`) runs the same
script weekly and keeps a `needs-response-digest` issue open, mentioning Geoff, while
anything is unanswered.

The `check-branch-fresh` pre-push hook (`scripts/check_branch_fresh.sh`)
covers what the routine can't: main moving mid-session, between when you
branched and when you push. It fetches `origin/main` (failing open when
offline) and refuses the push if the branch is missing commits from main,
printing the rebase fix. For an intentionally-behind push, skip once with
`SKIP=check-branch-fresh git push`. The hook is installed per machine/clone
by `uv run prek install` (it installs both the pre-commit and
pre-push hook types via `default_install_hook_types`).

### One-liner commands

Prefer the `make` targets — they pin the right paths (notably **`examples/`
is excluded from ruff** because example files intentionally check env vars
before importing optional deps and would fail E402).

```bash
make ci                      # full pipeline (lint + typecheck + test + markdown-lint)
make lint                    # ruff check on src/ tests/
make lint-fix                # ruff check --fix
make format                  # ruff format on src/ tests/
make typecheck               # mypy on src/async_batch_llm/
make markdown-lint-fix       # markdownlint with --fix
uv run pytest                # tests only

# Equivalents if you need to run without make (note: do NOT add examples/):
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run mypy src/async_batch_llm/ --ignore-missing-imports
```

### Git hooks (prek)

Hooks are managed with [prek](https://github.com/j178/prek), a drop-in
Rust replacement for pre-commit that reads the same
`.pre-commit-config.yaml`. `uv run prek install` once per machine/clone
(installs both the pre-commit and pre-push hook types). Hooks then run on
every commit: ruff (format + lint, with `examples/` excluded), mypy,
trailing whitespace, EOF newline, YAML/TOML validation, markdownlint,
prevention of commits to `main`/`master`. On every push,
`check-branch-fresh` blocks branches based on a stale main (see "Sync
before working"). Manual run on all files: `uv run prek run --all-files`.
Bypass with `--no-verify` only if you know what you're doing.

### Markdown config

`.markdownlint.json` relaxes line-length to 120 chars; code blocks need
language specifiers (`text` for plain output); blank lines required
around lists and code fences; HTML allowed.

### Documentation site

`uv sync --extra docs && uv run mkdocs serve` to preview locally. Pushes
to `main` auto-deploy to GitHub Pages via `.github/workflows/docs.yml`.

### Building / publishing

```bash
uv build
export UV_PUBLISH_TOKEN=...
uv publish [--index-url https://test.pypi.org/legacy/]
```

There's also a `.github/workflows/publish.yml` that handles releases via
the project's release-prep flow.

### CI workflows

- `test.yml` — on every push/PR:
  - `test` (Python 3.10–3.14; the four 3.10–3.13 legs are the required checks) and
    `test-macos`;
  - `quality` (ruff lint and format, mypy, ty, `make package-check`; markdownlint
    runs only in the prek hook and `make ci`, not in CI);
  - `sdk-compat`, a ten-leg matrix running each SDK's floor and the latest release
    of every supported major (openai, google-genai, pydantic-ai);
  - `scale-smoke`, `docs-build`, and `security` (pip-audit on the locked runtime
    deps, npm audit).
- `docs.yml` — MkDocs build & GitHub Pages deploy on push to `main`.
- `publish.yml` — tag-triggered: version check, `make package-check`, PyPI upload.
  It does not create the GitHub release; `/release-tag` does, after PyPI confirms.

GitHub Actions are pinned to exact tags. Workflow-file changes need a push
credential with the `workflow` scope (SSH works; the gh HTTPS token may not).

### Review protocol

Claude implements and Codex reviews. Follow [Direct Herdr Reviews](AGENTS.md#direct-herdr-reviews)
for authorized agent-to-agent requests and replies; send review requests directly to
`abl-reviewer` and receive verdicts as `abl-implementer`.

This repo is often worked by two sessions: one implements, one reviews.
The v0.24.0 lifecycle work, originally planned as v0.23.1, took ten review
rounds, and most repeat findings traced to missing evidence rather than
missing skill. The v0.23.1 working version was never published.

**Handing work to a review.**

- Commit first, or name the base revision. A reviewer comparing against
  the parent otherwise has to stash an uncommitted tree, which is
  destructive if the implementing session is still editing.
- Map each finding to its change and to its regression test.
- Show that test failing on the parent commit. "`make ci` passes" is not
  evidence that a fix changed behavior.
- State deliberate omissions and disagreements; don't leave them silent.

**Reviewing.**

- Verify by reproduction against the parent commit. A green suite says
  nothing about a path no test covers.
- Before calling something a regression, run the same probe on both
  trees — much of what looks new is pre-existing.
- Exercise both artifact backends whenever replay, record eligibility,
  or lookup changes; `JsonlArtifactStore` and `SqliteArtifactStore` have
  drifted apart more than once.
- Don't re-run the full suite on both interpreters by routine; the
  implementing side already runs `make ci`. Reserve the 3.10/3.13 pair
  for cancellation and asyncio-semantics changes, where they diverge.

**Changing a shared policy.** Most round-N fixes in that lifecycle work
introduced a round-N+1 defect in a sibling path. When a change touches a
category list, a swallow-or-re-raise rule, or a replay predicate,
enumerate every member and every caller and say what each one does now.
Dropping the abort-time append lost the audit trail; adding caller
attribution to capacity warnings broke warning deduplication; folding
the per-item timeout into the abort audit categories silently swallowed
store write errors.

**Background review agents.** `/code-review` finds real defects a
foreground pass misses, but it has no memory across rounds: paste the
settled decisions and out-of-scope list from `.claude/review-context.md`
into its prompt, or it re-reports accepted trade-offs. Prefer letting it
do discovery and verifying its candidates yourself over running two full
parallel passes. Keep `.claude/review-context.md` pruned at each release.

---

## Testing strategy

~3,200 tests (`slow`, `integration`, and `benchmark` deselected by default),
including ~480 parametrized doc-snippet checks (`tests/test_doc_examples.py` —
parses every fenced python block in the docs, resolves
`async_batch_llm` imports, and diffs framework-hook overrides in doc
classes against the live base-class signatures; opt a block out with
`<!-- doc-snippet: skip -->` above the fence). The doc checks parse and
signature-check snippets but don't run them. The default run takes about two
minutes, and `make ci` about two and a half (keep retries fast — see
`tests/conftest.py` for the shared `fast_retry`/`fast_rate_limit`
fixtures; use them in any test that triggers a retry, or you'll pay
1s+ per retry against the library defaults). `pytest-timeout` caps
every test at 60s so deadlock regressions fail instead of hanging CI.
Coverage spans happy paths, concurrency stress (100–200 items × 10–20
workers), edge cases, and per-provider integration with mocked SDKs.
Real API calls live behind the `integration` pytest marker and are
skipped by default.

Key test files:

- `test_basic.py` — basic processing, context, post-processors,
  metrics, timeouts.
- `test_concurrency.py` — thread safety: stats updates, rate limiting,
  metrics observer, no result loss, slow-start counter.
- `test_gemini_strategies.py` — `GeminiModel`, `GeminiCachedModel`,
  `GeminiStrategy`.
- `test_openai_compatible.py`, `test_openai_strategies.py`,
  `test_openrouter_strategies.py` — OpenAI/OpenRouter (v0.9.0).
- `test_error_classifiers.py` — every classifier branch.
- `test_token_extractor.py`, `test_token_tracking.py`,
  `test_token_tracking_on_failure.py` — token accounting.
- `test_cache_expiration_multiworker.py`, `test_cache_tag_matching.py`
  — Gemini cache lifecycle.
- `test_legacy_fixtures.py` — artifacts written by v0.18, v0.21, v0.24.1, v0.26,
  v0.27, and v0.28 still read and replay (fixtures in `tests/fixtures/`; add one per
  release via `scripts/write_legacy_fixtures.py`, see `/release-prep`).
- `test_deprecations.py`, `test_stability_page.py`, `test_categories.py` —
  deprecation warnings, every `__all__` name classified on `docs/stability.md`,
  and every produced error/timeout category in `ErrorCategory`/`TimeoutCategory`.
- `test_release_docs.py` — README/onboarding invariants and the pinned release
  version (updated by each release PR).

`MockAgent` (`testing/mocks.py`) simulates rate limits, errors, and
latency without API calls — much faster than real integration tests.

---

## Important files

### Package layout

```text
src/async_batch_llm/
├── __init__.py           # Public API exports (+ deprecated-name __getattr__)
├── base.py               # LLMWorkItem, WorkItemResult, BatchResult,
│                         # LLMResponse, RetryState, CachedTokenRates
├── py.typed              # PEP 561 marker (ships in wheel + sdist)
├── parallel.py           # ParallelBatchProcessor (orchestration)
├── streaming.py          # process_prompts / process_stream (streaming API)
├── single.py             # call / call_result (one-shot convenience API)
├── call_pool.py          # LLMCallPool (queue-less shared-cooldown service)
├── gateway.py            # old path: re-exports LLMCallPool (through 1.x) and
│                         # the deprecated LLMGateway alias (removed in 1.0)
├── factory.py            # llm("provider:model")
├── callable_strategy.py  # CallableStrategy / CallOutcome
├── categories.py         # ErrorCategory, TimeoutCategory
├── artifacts.py          # JsonlArtifactStore, ArtifactIdentity, ResumePolicy
├── sqlite_artifacts.py   # SqliteArtifactStore
├── serialization.py      # strict versioned result JSON
├── budget.py             # AttemptUsage (token/cost budgets)
├── token_estimation.py   # TokenEstimate, CharacterTokenEstimator
├── parsing.py            # JSON/code-fence response-parser helpers
├── llm_strategies.py     # LLMCallStrategy + built-in strategies
├── models.py             # GeminiModel, GeminiCachedModel,
│                         # OpenAICompatibleModel, OpenAIModel,
│                         # OpenRouterModel, DeepSeekModel
├── token_extractor.py    # TokenExtractor (failure-path token recovery)
├── provider_output.py    # Grounding/GroundingSource/ToolCall + typed
│                         # metadata views mixin (issue #52 Phase 2)
├── core/
│   ├── config.py         # ProcessorConfig, RateLimitConfig, RetryConfig
│   └── protocols.py      # LLMModel, ManagedLLMModel
├── strategies/
│   ├── errors.py         # AsyncBatchLLMError, ErrorClassifier, ErrorInfo, TokenTrackingError,
│   │                     # FrameworkTimeoutError, EmptyResponseError,
│   │                     # ProviderResponseError
│   └── rate_limit.py     # ExponentialBackoffStrategy, FixedDelayStrategy
├── classifiers/
│   ├── gemini.py         # GeminiErrorClassifier
│   ├── openai.py         # OpenAIErrorClassifier
│   ├── openrouter.py     # OpenRouterErrorClassifier (extends OpenAI)
│   └── pydantic_ai.py    # PydanticAIErrorClassifier (extends OpenAI)
├── observers/
│   ├── base.py           # ProcessorObserver protocol
│   └── metrics.py        # MetricsObserver
├── middleware/
│   └── base.py           # Middleware protocol
├── _internal/            # Shared orchestration collaborators
│   ├── admission.py      # quota scopes and FIFO admission
│   ├── artifact_codec.py # artifact records and replay predicates
│   ├── backoff.py
│   ├── budget.py         # token/cost budget accounting
│   ├── capacity.py       # provider-capacity admission
│   ├── classifier_resolver.py
│   ├── cleanup.py        # ordered teardown and detached cancellation waits
│   ├── execution_state.py # private per-item accounting
│   ├── event_dispatcher.py
│   ├── executor_host.py  # pool-less host for single.py / call_pool.py
│   ├── guardrails.py     # item/batch deadlines and fail-fast
│   ├── input_validation.py
│   ├── item_executor.py  # per-item retry/classification engine
│   ├── logical_item.py   # effective request shared by preprocessing/replay/retries
│   ├── rate_limit_coordinator.py
│   ├── responses_translation.py # Chat-style request -> OpenAI Responses
│   ├── strategy_lifecycle.py
│   └── error_logging.py
└── testing/
    ├── fake.py           # FakeStrategy, mock_strategy
    ├── mocks.py          # MockAgent
    └── strategies.py     # test-strategy helpers
```

### Documentation

- `README.md` — user-facing intro.
- `docs/getting-started.md` — installation + first batch.
- `docs/GEMINI_INTEGRATION.md` — Gemini deep dive (caching lifecycle).
- `docs/OPENAI_INTEGRATION.md` — OpenAI deep dive (v0.9.0).
- `docs/OPENROUTER_INTEGRATION.md` — OpenRouter deep dive, including the
  per-upstream caching matrix and the Anthropic `cache_control` opt-in
  pattern.
- `docs/api/` — API reference (entrypoints, core, errors, artifacts,
  single-call-pool, strategies, observers).
- `docs/stability.md` — the 1.0 compatibility promise: stable, provisional, and
  deprecated names. `tests/test_stability_page.py` requires every `__all__` name.
- `docs/migration/v0.27.md` — current migration guide (older ones alongside).
- `docs/choosing-your-limits.md`, `docs/production-checklist.md`,
  `docs/guardrails.md`, `docs/results-and-artifacts.md`, `docs/large-runs.md` —
  operational guides.
- `SECURITY.md` — vulnerability reporting.
- `docs/MIGRATION_V0_10.md` — historical migration guide
  (v0.8.x → v0.10.0; covers OpenAI/OpenRouter additions and the metadata
  3-tuple contract change).
- `docs/migration/v0.4.md` — earlier migration notes.
- `docs/archive/` — historical migration guides and design plans.
- `CHANGELOG.md` — release-by-release changes.
- `CONTRIBUTING.md` — contributor docs, including the release process
  (`/release-prep` → merge → `/release-tag`; publishing is tag-triggered).
- `CLAUDE.md` — this file.

### Examples

`examples/` directory — every pattern has a runnable script:

- `example.py` — full-featured walkthrough.
- `example_callable_application.py` — credential-free: existing async client,
  bounded streaming, and replay.
- `example_production_resume.py` — checkpoints, deadlines, and fail-fast.
- `example_single_call.py`, `example_gateway.py` — `call()` and `LLMCallPool`.
- `example_gemini_grounding.py` — Gemini search grounding and typed views.
- `example_gemini_direct.py` — built-in Gemini.
- `example_gemini_smart_retry.py` — smart retry with field-specific
  feedback.
- `example_smart_model_escalation.py` — cost-saving escalation pattern.
- `example_openai.py` — built-in OpenAI (v0.9.0).
- `example_openrouter.py` — built-in OpenRouter, including
  Anthropic `cache_control` demo (v0.9.0).
- `example_deepseek.py` — built-in DeepSeek with native cache-hit
  token tracking (v0.10.0).
- `example_anthropic.py`, `example_langchain.py` — custom-strategy
  references for providers without built-in support yet.
- `example_embeddings.py` — batch embedding generation via custom
  strategies (OpenAI `text-embedding-3-small` + Gemini
  `gemini-embedding-2`); one JSON-encoded chunk of texts per work item.
  Note the Gemini gotcha: `gemini-embedding-2` aggregates a plain string
  list into ONE embedding — wrap each text in a `Content` object for
  per-text vectors. DeepSeek offers no embeddings endpoint (as of
  2026-07).
- `example_llm_strategies.py` — custom-strategy patterns.
- `example_context_manager.py` — async context manager usage.
- `example_model_escalation.py` — earlier escalation example.
- `example_batch_benchmark.py` — flagship "why async-batch-llm" demo:
  GSM8K through DeepSeek Flash vs Gemini 3.1/2.5 Flash-Lite with
  no-think→think escalation, a per-provider 3-way wall-time race
  (sequential vs `asyncio.gather` vs the framework), a `--throughput`
  parity bench (chunked gather vs semaphore pool vs the framework),
  stdlib-`gzip` streaming I/O, token/cost reporting + a terse-vs-verbose
  sample capture, and an OpenAI LLM-as-judge fallback grader. The bake-off +
  judge use the high-level `process_prompts` API (context via 3-tuples). Writes
  `benchmark_results/summary.json` + `throughput.json`. `download_gsm8k.py`
  fetches the data (writes `examples/data/gsm8k_test.jsonl.gz`, gitignored).
  Needs `[deepseek,gemini,openai]` extras. `generate_benchmark_charts.py`
  (needs `[docs]`/matplotlib) turns the JSON into the `docs/benchmarks.md`
  figures. Docs: `docs/benchmarks.md` (results) + `docs/examples/benchmark-walkthrough.md`.

---

## Performance notes

- **Worker count.** I/O-bound LLM calls work well at 5–10 workers;
  rate-limited endpoints start at 3–5. Don't use `cpu_count()` unless
  you're CPU-bound, which you almost certainly aren't.
- **Memory.** ~10–50 MB per 1000 items, depending on output size. Each
  worker holds one item; results accumulate.
- **Throughput.** ~5–10 items/sec for current Gemini Flash with 5
  workers, mostly bounded by API latency (~200–500 ms/call).

---

## Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

```python
result = await processor.process_all()
stats = await processor.get_stats()
if stats["rate_limit_count"] > 0:
    print(f"Hit {stats['rate_limit_count']} rate limits")
    print(f"Errors: {stats['error_counts']}")

assert result.total_items == result.succeeded + result.failed
```

---

[#8]: https://github.com/geoff-davis/async-batch-llm/issues/8

## Known limitations

1. **Single-process only.** Designed for asyncio; no multi-process
   coordination. See Future Enhancements #1.
2. **No true batch API.** Parallel individual calls, not batched API
   requests. See #2.
3. **In-memory queue.** Queued work is lost on crash; results already
   checkpointed to a JSONL/SQLite artifact store replay on resume. See #4.
4. **Provider classifiers are partial.** Gemini, OpenAI, OpenRouter are
   covered; DeepSeek reuses `OpenAIErrorClassifier` (it's OpenAI-compatible);
   Anthropic native and HuggingFace pending. See #3.

---

## Future enhancements

1. **Distributed locks** — multi-process scenarios.
2. **Batch API support** — true batch APIs for ~50% cost savings.
3. **More classifiers** — Anthropic native, HuggingFace. (DeepSeek reuses
   `OpenAIErrorClassifier`.)
4. **Persistent queue** — Redis/DB-backed.
5. **Prometheus metrics** — built-in metrics export (we have
   `MetricsObserver`; this is about a Prometheus-format exporter on top).
6. **Dynamic worker scaling** — adjust workers based on load.
7. **1.0 removals.** Everything v0.27 deprecates is removed in 1.0: the 2-tuple
   `execute()` shim, omitted `cached_token_rate`, `LLMGateway`,
   `BatchProcessor`/`ProcessingStats`/`grounding_metadata_extractor` as public
   names, `gemini_safety_ratings`, `timeout_per_item`, the legacy positional
   processor parameters, callable `cache_hit_rate()`, and integer prompts. So is
   what v0.28 deprecates: positional arguments to the five config classes (they
   become keyword-only) and `ProcessorConfig.enable_detailed_logging`. 1.0
   also drops Python 3.10 (`requires-python`, mypy `python_version`, CI matrix).
   See `docs/stability.md` and `docs/internal/release-1.0-plan.md`.

---

## Where session-spanning context lives

- **`docs/internal/`** (gitignored, local to the main checkout) — plans such as
  `release-1.0-plan.md`, design docs, handoffs, and Codex review notes
  (`*-codex-review.md`). Read these to pick up an in-progress design.
- **GitHub issues** — `gh issue list -R geoff-davis/async-batch-llm`.
  Cross-referenced from "Future Enhancements" above.
- **`CHANGELOG.md`** — release-shipped changes.
- **`~/.claude/projects/-home-geoff-Projects-personal-async-batch-llm/memory/`**
  — auto-memory store; `MEMORY.md` is the index.

---

## Version history

Most recent first. See `CHANGELOG.md` for full per-release detail.

- **v0.28.0** — deprecation-only release ahead of 1.0, and the last for Python
  3.10. Positional arguments to the five config classes and
  `ProcessorConfig.enable_detailed_logging` warn (both go in 1.0); `LLMCallPool`
  moves to `call_pool.py` (`gateway.py` re-exports it); the API page becomes
  `api/single-call-pool/` with a redirect. Read `docs/migration/v0.28.md`.

- **v0.27.0** — the release that deprecates what 1.0 removes. `OpenAIModel`
  uses the Responses API by default; built-in models no longer send a default
  temperature; classification trusts exception types and status codes first
  (#177); SDK floors raised and tested by a ten-leg CI matrix (#179/#180). Adds
  token/cost budgets, `ErrorCategory`/`TimeoutCategory`, the `AsyncBatchLLMError`
  base, `llm("openai-compatible:…")`, macOS CI, and the `docs/stability.md` draft.
  Configuration failures no longer replay (#178); failed admissions keep their
  wait timing (#181); Gemini metadata uses plain enum names. Deprecates the names
  1.0 removes (see Future enhancements #7). Read `docs/migration/v0.27.md`.

- **v0.26.0** — shared strategy/model leases, artifact serialization isolation,
  provider and quota corrections, explicit processor lifecycle/events, and bounded
  diagnostic logging. Input validation and several defaults change; read
  `docs/migration/v0.26.md` before upgrading.
- **v0.25.0** — private attempt stages and outcomes isolate classification and
  failed usage from reused provider exceptions; failed admission timing remains
  consistent across retry guardrails. Public signatures and artifact schemas are
  unchanged. SQLite waiting/index retention is documented with measured evidence.
- **v0.24.1** — preserve failed-attempt usage and late error-hook reports,
  fail closed on broken quota scopes, select retry delays by resolved category,
  and verify DeepSeek Responses model/client compatibility. Includes the v0.24
  migration guide and the development-only smol-toml security fix.
- **v0.24.0** — isolated retry runtime state, ordered and retryable cleanup,
  durable stream failures, and middleware-before-replay behavior on both stores.
  Cleanup failures and callback barriers change compatibility; batch stores stay
  open until explicit close or context exit. Guardrail audit records never replay.
  See `docs/cleanup-lifecycle-contract.md` and `CHANGELOG.md` before upgrading.
- **v0.23.0** — DeepSeek Responses API strict structured output.
- **v0.22.0** — scoped, token-aware admission coordinates per-strategy
  cooldown, RPM, and TPM through one atomic FIFO gate. Public estimation APIs,
  exactly-once reservation reconciliation, refunds, underestimation debt,
  bounded quota observability, and v0.21 artifact compatibility are backed by
  the mixed-token scale scenario and reviewed 100k/1m evidence.
- **v0.21.0** — indexed SQLite artifacts for large restartable runs, bounded
  JSONL/SQLite inspection, exact queue high-water diagnostics, and a
  deterministic scale/restart harness with CI, 100k, and 1m profiles. Artifact
  identity, stored-context decoding, row-version validation, read-only
  inspection, and close-error lifecycle semantics are hardened across both
  stores.
- **v0.20.0** — `CallableStrategy`/`CallOutcome` integration for existing
  async clients, bounded completed-result handoff, item-private retry state,
  owner-scoped cooldowns, bundled progress reporting, and the `LLMCallPool`
  alias. The onboarding, application-integration, and migration path was
  refreshed for direct upgrades from v0.18.
- **v0.18.0** — stable opt-in input ordering, strict versioned result
  serialization, replayable JSONL audit/checkpoint artifacts (#81), compatible
  success or terminal-result replay, end-to-end item and batch deadlines, and
  configurable category-based fail-fast behavior. Controlled stops preserve
  accepted-work results and expose serializable termination metadata; artifact
  persistence remains privacy-safe by default and checkpointed before result
  publication.
- **v0.17.0** — provider-capacity admission outside execution timeouts
  (#74/#79), structured per-attempt timing and percentile metrics (#76),
  optional startup concurrency ramping (#77), conservative trailing-fence JSON
  recovery (#82), and the high-throughput/bounded-work documentation set
  (#75/#78/#80). Sync post-processors now run off the event loop and respect the
  configured timeout. New public surfaces include `StartupRampConfig`,
  `AttemptTiming`, `WorkItemTiming`, structured-output recovery views, and
  admission/execution/recovery metrics.
- **v0.16.0** — typed auxiliary-output views (#52 Phase 2,
  **experimental** — shapes/views may change in a minor release until
  they've seen real use). Four reserved `metadata` keys (`grounding`,
  `reasoning`, `tool_calls`, `logprobs`) carry provider-specific output as
  plain JSON-serializable dicts, and `LLMResponse`/`WorkItemResult` expose
  lazy read-only
  typed views over them (`.grounding`/`.reasoning`/`.tool_calls`/`.logprobs`,
  via the `ProviderOutputViews` mixin in `provider_output.py`; parsed on each
  access, nothing stored twice, strategy return contract untouched). Gemini
  models emit `grounding` **by default** now (`grounding_metadata_extractor`
  remains exported, redundant for built-ins); OpenAI-compatible models emit
  `reasoning` (DeepSeek `reasoning_content` → OpenRouter `reasoning`
  fallback), `tool_calls` (visibility only, raw JSON-string arguments), and
  `logprobs` — all behind `isinstance` guards so SDK drift/mocks can't leak
  non-JSON values. New exports: `Grounding`, `GroundingSource`, `ToolCall`.
  Out of scope: Gemini function-call parts, typed logprobs, aux output on
  empty/safety-blocked responses (issue Q4). Also in this release (ported
  package-review fixes): `EmptyResponseError`/`ProviderResponseError`,
  PEP 561 `py.typed`, PEP 696 TypeVar defaults, async
  `MetricsObserver.reset()` (breaking), `BatchResult` derived fields
  `init=False` (breaking), status-code-based `GeminiErrorClassifier`
  rewrite (non-429 4xx fail fast), `RateLimitConfig.max_cooldown_seconds`,
  the ~15s test suite, and the `check-branch-fresh` pre-push guard.
- **v0.15.0** — pluggable metadata extraction (#52 Phase 1). Built-in models
  take a `metadata_extractors: list[MetadataExtractor]` constructor argument
  (and the OpenAI-compatible family also accepts it via `from_api_key`;
  Gemini models have no `from_api_key`) that contribute extra keys to
  `LLMResponse.metadata` / `WorkItemResult.metadata`, merged on top of the
  built-in allowlist (user keys win; a failing extractor is logged and
  skipped). Ships `grounding_metadata_extractor` (opt-in at the time;
  built-in since the next release). New exports: `MetadataExtractor`,
  `grounding_metadata_extractor`. The per-provider `_extract_metadata`
  allowlists are still the built-in default — `_run_extractors`
  (`models.py`) merges user extractors over them. This release also brought
  first-class streaming, the retry-budget rate-limit exemption, and the
  v0.11–v0.15 fix train (see `CHANGELOG.md`).
- **v0.10.0** — response metadata reaches `WorkItemResult` ([#8]), plus
  DeepSeek support, a strategy refactor, and rate-limit/temperature fixes.
  - `LLMCallStrategy.execute()` may now return a 3-tuple
    `(output, tokens, metadata)`; legacy 2-tuple still accepted via
    `_unpack_strategy_result` compat shim (slated for removal — see Future
    Enhancements #7).
  - All built-in strategies (`GeminiStrategy`, `OpenAIStrategy`,
    `OpenRouterStrategy`, `PydanticAIStrategy`) updated to the 3-tuple
    shape; provider metadata (provider name, finish_reason, routed model,
    safety ratings) flows into `WorkItemResult.metadata`.
  - `WorkItemResult.gemini_safety_ratings` deprecated; populated from
    `metadata['safety_ratings']` for backward compat.
  - **`ModelStrategy` base** — `GeminiStrategy`/`OpenAIStrategy`/
    `OpenRouterStrategy`/`DeepSeekStrategy` are now thin subclasses sharing
    `execute()`, lifecycle delegation, and the token-on-parse-failure path
    (`_attach_token_usage`). Behavior unchanged.
  - **`DeepSeekModel` / `DeepSeekStrategy`** (`[deepseek]` extra) — direct
    DeepSeek access; `_extract_tokens` reads DeepSeek's native
    `prompt_cache_hit_tokens` into `cached_input_tokens`.
  - **`temperature=None`** is now accepted everywhere (protocol, all models,
    `ModelStrategy`) to omit the parameter — needed for OpenAI reasoning
    models (o1/o3) that reject an explicit temperature.
  - **`effective_input_tokens()`** now `warn`s when relying on the implicit
    Gemini default while cached tokens are present (pass an explicit
    `CachedTokenRates` constant to silence).
  - **`ErrorInfo.suggested_wait`** is now honored: the `RateLimitCoordinator`
    uses it as a *floor* on the cooldown. Only genuine server signals set it —
    `OpenAIErrorClassifier` parses `Retry-After`; the old hardcoded
    `DEFAULT_RATE_LIMIT_WAIT` fallbacks were removed (they're the
    `RateLimitStrategy`'s job, not the classifier's).
- **v0.9.0** — first-class OpenAI and OpenRouter.
  - `OpenAICompatibleModel` base + `OpenAIModel` / `OpenRouterModel`
    subclasses, each with `from_api_key(...)` (optional `api_key`,
    falls back to `OPENAI_API_KEY` / `OPENROUTER_API_KEY`).
  - `OpenAIStrategy`, `OpenRouterStrategy` (thin shells over the model).
  - `OpenAIErrorClassifier` / `OpenRouterErrorClassifier` (with
    `no_provider_available` handling).
  - Track-only caching: reads `usage.prompt_tokens_details.cached_tokens`.
  - `CachedTokenRates` constants for provider-aware billing;
    `effective_input_tokens()` accepts a `cached_token_rate` parameter.
  - Models implement `ManagedLLMModel`: `cleanup()` closes httpx clients
    when constructed via `from_api_key`.
  - New extras: `[openai]`, `[openrouter]`. New docs:
    `docs/OPENAI_INTEGRATION.md`, `docs/OPENROUTER_INTEGRATION.md`.
- **v0.8.0** — release prep, pip-audit scope fix.
- **v0.7.0** — internal refactor: `_internal/` collaborators
  (`EventDispatcher`, `StrategyLifecycle`, `RateLimitCoordinator`,
  `error_logging`); `TokenExtractor`;
  `ProcessorConfig.post_processor_timeout`. Public API unchanged.
- **v0.6.0** — model abstraction: `LLMModel` / `ManagedLLMModel`
  protocols; `GeminiModel` / `GeminiCachedModel` (replaces
  `GeminiCachedStrategy`).
- **v0.3.0** — `RetryState` for cross-attempt persistence.
- **v0.1.0** — strategy pattern refactor (breaking).
- **v0.0.x** — initial development; race condition fixes.

---

## Lessons from past sessions

- **Run quality checks before every commit.** `make ci` covers
  everything; pre-commit hooks catch most of it automatically. Don't
  `--no-verify`.
- **Optional dependency groups are fine.** Earlier guidance argued
  against per-provider extras; v0.9.0 added `[openai]` and `[openrouter]`
  and they work well. The right test is "does the user benefit from a
  discoverable install hint" — usually yes.
- **Keep examples runnable.** Every example file should handle missing
  API keys gracefully and use the *current* built-in API, not custom
  strategies that duplicate built-in functionality.
- **When refactoring API, search for the old patterns and update
  everywhere.** README, docs/, examples/, CLAUDE.md, and tests all need
  to move together.

---

## License

MIT — see `LICENSE`.
