# Project Knowledge for Claude

Project-specific context that future Claude sessions load on startup. Keep
it tight — when in doubt, link to `docs/` rather than duplicate content
here.

---

## Project overview

**async-batch-llm** processes batches of LLM requests in parallel using a
**strategy pattern** — provider-agnostic at the framework level, with
first-class support for several providers built in.

**Current version:** v0.10.0 work is on `main`; `pyproject.toml` still
reads `0.8.0` until the release-prep flow bumps it.

**Key features:**

- Parallel asyncio processing with configurable concurrency
- Built-in rate limiting and exponential backoff retry logic
- Thread-safe concurrent operations (`asyncio.Lock`-based, no nesting)
- Provider-agnostic core: bring your own strategy/model/classifier
- Built-in Gemini, OpenAI, and OpenRouter support
- Middleware and observer patterns for extensibility
- `MockAgent` for testing without API calls

---

## Quick reference

High-level streaming API (`streaming.py`) for the common case — collect, or
stream as items finish. Built on the processor's first-class streaming mode
(`start()`/`add_work()`/`finish()`/`results()`), so a bounded `max_queue_size`
gives backpressure (constant memory for huge inputs). Error classifier is
auto-selected from the strategy:

```python
from async_batch_llm import OpenAIModel, OpenAIStrategy, process_prompts, process_stream

strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-4o-mini"))

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

model = OpenAIModel.from_api_key("gpt-4o-mini")  # reads OPENAI_API_KEY
strategy = OpenAIStrategy(model)
config = ProcessorConfig(max_workers=5, timeout_per_item=60.0)

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
| PydanticAI | (any model wrapped)                 | `PydanticAIStrategy` | —                           | `[pydantic-ai]`   |

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

### Thread safety

`ParallelBatchProcessor` uses three independent `asyncio.Lock` instances
(`_rate_limit_lock`, `_stats_lock`, `_results_lock`). No nesting → no
deadlocks. Sub-1% overhead in benchmarks.

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
`BatchResult.effective_input_tokens(rate)` for accurate billable counts.
Default is `GEMINI` for backward compat — non-Gemini callers must opt
in. The math conservatively rounds the billable estimate UP via `int()`
truncation of the discount.

---

## Common patterns

### PydanticAI strategy

```python
agent = Agent("gemini-2.5-flash", result_type=Output)
strategy = PydanticAIStrategy(agent=agent)
work_item = LLMWorkItem(item_id="1", strategy=strategy, prompt="...")
```

### Built-in OpenAI / OpenRouter

```python
# OPENAI_API_KEY auto-resolved by SDK
model = OpenAIModel.from_api_key("gpt-4o-mini")
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

### Pre-commit hooks

`uv run pre-commit install` once. Hooks then run on every commit: ruff
(format + lint, with `examples/` excluded), mypy, trailing whitespace,
EOF newline, YAML/TOML validation, markdownlint, prevention of commits
to `main`/`master`. Manual run on all files:
`uv run pre-commit run --all-files`. Bypass with `--no-verify` only if
you know what you're doing.

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

- `test.yml` — pytest + ruff + mypy on Python 3.10–3.14; pip-audit +
  npm-audit on every push/PR.
- `docs.yml` — MkDocs build & GitHub Pages deploy on push to `main`.
- `publish.yml` — release publishing.

---

## Testing strategy

321 tests as of v0.9.0. Coverage spans happy paths, concurrency stress
(100–200 items × 10–20 workers), edge cases, and per-provider
integration with mocked SDKs. Real API calls live behind the
`integration` pytest marker and are skipped by default.

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

`MockAgent` (`testing/mocks.py`) simulates rate limits, errors, and
latency without API calls — much faster than real integration tests.

---

## Important files

### Package layout

```text
src/async_batch_llm/
├── __init__.py           # Public API exports
├── base.py               # LLMWorkItem, WorkItemResult, BatchResult,
│                         # LLMResponse, RetryState, CachedTokenRates
├── parallel.py           # ParallelBatchProcessor (orchestration)
├── streaming.py          # process_prompts / process_stream (streaming API)
├── llm_strategies.py     # LLMCallStrategy + built-in strategies
├── models.py             # GeminiModel, GeminiCachedModel,
│                         # OpenAICompatibleModel, OpenAIModel,
│                         # OpenRouterModel, DeepSeekModel
├── token_extractor.py    # TokenExtractor (failure-path token recovery)
├── core/
│   ├── config.py         # ProcessorConfig, RateLimitConfig, RetryConfig
│   └── protocols.py      # LLMModel, ManagedLLMModel, AgentLike
├── strategies/
│   ├── errors.py         # ErrorClassifier, ErrorInfo,
│   │                     # TokenTrackingError, FrameworkTimeoutError
│   └── rate_limit.py     # ExponentialBackoffStrategy, FixedDelayStrategy
├── classifiers/
│   ├── gemini.py         # GeminiErrorClassifier
│   ├── openai.py         # OpenAIErrorClassifier
│   └── openrouter.py     # OpenRouterErrorClassifier (extends OpenAI)
├── observers/
│   ├── base.py           # ProcessorObserver protocol
│   └── metrics.py        # MetricsObserver
├── middleware/
│   └── base.py           # Middleware protocol
├── _internal/            # ParallelBatchProcessor collaborators (v0.7.0)
│   ├── event_dispatcher.py
│   ├── rate_limit_coordinator.py
│   ├── strategy_lifecycle.py
│   └── error_logging.py
└── testing/
    └── mocks.py          # MockAgent
```

### Documentation

- `README.md` — user-facing intro.
- `docs/getting-started.md` — installation + first batch.
- `docs/GEMINI_INTEGRATION.md` — Gemini deep dive (caching lifecycle).
- `docs/OPENAI_INTEGRATION.md` — OpenAI deep dive (v0.9.0).
- `docs/OPENROUTER_INTEGRATION.md` — OpenRouter deep dive, including the
  per-upstream caching matrix and the Anthropic `cache_control` opt-in
  pattern.
- `docs/API.md` — API reference.
- `docs/MIGRATION_V0_10.md` — most recent migration guide
  (v0.8.x → v0.10.0; covers OpenAI/OpenRouter additions and the metadata
  3-tuple contract change).
- `docs/MIGRATION_V0_4.md` — earlier migration notes.
- `docs/archive/` — historical migration guides and design plans.
- `CHANGELOG.md` — release-by-release changes.
- `CONTRIBUTING.md`, `PUBLISHING.md` — contributor docs.
- `CLAUDE.md` — this file.

### Examples

`examples/` directory — every pattern has a runnable script:

- `example.py` — full-featured walkthrough.
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
3. **In-memory queue.** Lost on crash. See #4.
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
7. **Drop the strategy 2-tuple compat shim.** v0.10.0 added a 3-tuple
   `execute()` return shape `(output, tokens, metadata)` with a shim that
   still accepts legacy 2-tuple. Schedule the shim removal for a future
   minor or major release; once removed, also drop the
   `gemini_safety_ratings` field on `WorkItemResult` (its content lives in
   `metadata['safety_ratings']` now).

---

## Where session-spanning context lives

- **Plan-mode artifacts** — `~/.claude/plans/*.md`. Persist across
  sessions; read these to pick up an in-progress design.
- **GitHub issues** — `gh issue list -R geoff-davis/async-batch-llm`.
  Cross-referenced from "Future Enhancements" above.
- **`CHANGELOG.md`** — release-shipped changes.
- **`~/.claude/projects/-home-geoff-Projects-personal-async-batch-llm/memory/`**
  — auto-memory store; `MEMORY.md` is the index.

---

## Version history

Most recent first. See `CHANGELOG.md` for full per-release detail.

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
- **Don't promise behavior the framework can't deliver.** `WorkItemResult`
  doesn't carry `LLMResponse.metadata` (see #8). Don't write docs that
  claim it does.
- **Keep examples runnable.** Every example file should handle missing
  API keys gracefully and use the *current* built-in API, not custom
  strategies that duplicate built-in functionality.
- **When refactoring API, search for the old patterns and update
  everywhere.** README, docs/, examples/, CLAUDE.md, and tests all need
  to move together.

---

## License

MIT — see `LICENSE`.
