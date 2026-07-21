# Choosing Your Limits

One page, one pass through every limit that matters, in the order you should
decide them. Each section says what the knob bounds, how to pick a value, and
when to leave it alone. The deep dives — the
[production checklist](production-checklist.md),
[high-throughput guide](openai-high-throughput.md),
[guardrails](guardrails.md), and
[bounded work](bounded-work.md) — expand on each step; you should be able to
size a real run without leaving this page.

## The short version

```python
from async_batch_llm import JsonlArtifactStore, ResumePolicy, llm, process_prompts

strategy = llm("deepseek:deepseek-v4-flash", thinking=False)

batch = await process_prompts(
    strategy,
    prompts,                       # 10,000 items
    concurrency=64,                # step 1 — sizes everything aligned below
    progress=True,
    artifact_store=JsonlArtifactStore("runs/tagging.jsonl"),
    resume=ResumePolicy.REUSE_SUCCESSES,
)
```

`concurrency=N` (v0.20) coherently sizes the worker pool, provider-capacity
admission, and — for built-in models created via `llm()` or `from_api_key()`
without an explicit `max_connections` — the httpx connection pool. If you set
nothing else, the rest of this page is the explanation of what you just got
and when to override it.

## The decision tree

```text
1. concurrency      how many requests in flight at once?
2. connection pool  can the HTTP client actually carry that many?
3. admission        should the framework cap in-flight provider calls?
4. attempt_timeout  how long may ONE provider call take?
5. total item deadline   how long may one item take end to end?
6. batch deadline   how long may the whole run take?
7. startup ramp     should full concurrency arrive gradually?
8. cooldown         what happens when the provider says 429?
```

### 1. Concurrency — `concurrency=N`

Pick the number of simultaneous provider requests. This is a property of the
**provider**, not your CPU — LLM calls are I/O-bound, so never use
`cpu_count()`.

- Rate-limited endpoints (OpenAI/Gemini tiers): start at **5–10**.
- High-concurrency providers (DeepSeek, self-hosted vLLM): **50–200**.
- If the provider publishes an RPM limit, a serviceable estimate is
  `concurrency ≈ RPM / 60 × typical_latency_seconds`, rounded down.

Prefer the single knob over setting `max_workers`,
`max_provider_concurrency`, and `max_connections` separately — misalignment
between those three is the most common performance bug in this package.
Explicit values for any individual knob still win when you set them.

### 2. Connection pool — `max_connections`

The openai SDK's default httpx pool holds **~100 connections**. Workers above
that number don't fail — they silently queue inside httpx where no timeout is
running, which caps throughput invisibly. **This is the classic DeepSeek
footgun**: DeepSeek happily accepts hundreds of concurrent requests, so
`max_workers=150` against a default pool gives you exactly 100-wide
throughput and 50 workers waiting in the transport.

- With `concurrency=N` and a factory-built model (`llm("...")` or
  `from_api_key()` without `max_connections`), the pool is resized to `N`
  before the first request — nothing to do.
- If you build your own `AsyncOpenAI`/httpx client, size its
  `httpx.Limits(max_connections=..., max_keepalive_connections=...)` to at
  least your concurrency yourself; the framework cannot introspect or resize
  a caller-supplied client.
- An explicit `max_connections` smaller than `concurrency` is treated as a
  real contradiction and warns.

Details: [high-throughput guide](openai-high-throughput.md).

### 3. Provider admission — `max_provider_concurrency`

Admission caps how many attempts may hold a provider "slot" at once, and the
wait happens **before** `attempt_timeout` starts — so a queued attempt can't
burn its execution timeout waiting for capacity. With `concurrency=N` this is
already `N`; set it explicitly only to run the framework wider than the
provider (e.g. many cheap local workers feeding a narrow paid API), or when a
strategy advertises its own `max_concurrency` (the lower limit applies).

### 4. Per-attempt timeout — `attempt_timeout`

Bounds **one** `execute()` call (default 120s). Size it for a single slow
response, not the whole retry chain: p99 latency of one call plus margin.
30s is a sane starting point for chat-completion workloads; long-output or
reasoning models may need the default or more. Renamed from
`timeout_per_item` in v0.20 (the old name still works, with a warning).

### 5. Total item deadline — `GuardrailConfig.total_timeout_per_item`

Bounds one **logical item** end to end: admission waits, cooldowns, every
retry, and backoff. Without it, an item can legitimately take
`~max_attempts × attempt_timeout` plus waits. Set it when a downstream
consumer needs a hard per-item SLA; leave it `None` for offline jobs where
retrying through a cooldown is exactly what you want. A reasonable formula:
`max_attempts × attempt_timeout + expected_cooldown`.

### 6. Batch deadline — `GuardrailConfig.batch_timeout`

Bounds the whole run. On expiry the batch stops **accepting and executing**
work in a controlled way: accepted items get terminal results, checkpoints
flush, and `batch.termination` says why. Set it to your job scheduler's
timeout minus a shutdown margin, or leave `None`. Pair with
`abort_on_error_categories=frozenset({"authentication",
"insufficient_balance"})` so a dead credential fails the run in seconds, not
after 10,000 retries. Details: [guardrails](guardrails.md).

### 7. Startup ramp — `startup_ramp`

Opening at full concurrency against a cold endpoint can trip instant 429s.
`StartupRampConfig(initial_concurrency=4, concurrency_step=4,
ramp_interval_seconds=2.0)` walks up to full width instead. Skip it for small
runs or generous providers; reach for it when the first minute of a large run
keeps tripping cooldowns.

### 8. Cooldown — `RateLimitConfig`

What happens on 429: one worker triggers a shared cooldown, everyone pauses,
then traffic resumes through a slow-start ramp. Defaults are conservative
(300s cooldown). For providers with short quota windows, start with
`RateLimitConfig(cooldown_seconds=30.0, max_cooldown_seconds=300.0)` — the
classifier honors a server-sent `Retry-After` as a floor regardless. If you
see repeated consecutive cooldowns, your `concurrency` (step 1) is too high;
fix the cause, not the cooldown.

## Worked example: 10k items against a rate-limited provider

Target: 10,000 classification prompts against an OpenAI-tier endpoint
(~500 RPM), ~2s per call, overnight job, must survive restarts.

- **Step 1:** `500 / 60 × 2 ≈ 16` → `concurrency=16`.
- **Step 2–3:** covered by the knob (factory model resizes its pool; admission = 16).
- **Step 4:** p99 of one call ≈ 10s → `attempt_timeout=30.0`.
- **Step 5:** 3 attempts × 30s + one 60s cooldown → `total_timeout_per_item=180.0`.
- **Step 6:** cron kills the job at 2h → `batch_timeout=6600.0` (1h50m), plus
  fail-fast on `authentication`.
- **Step 7:** skip the ramp at this width.
- **Step 8:** `cooldown_seconds=60.0` — OpenAI-style minute windows.

```python
from async_batch_llm import (
    GuardrailConfig,
    JsonlArtifactStore,
    ProcessorConfig,
    RateLimitConfig,
    ResumePolicy,
    llm,
    process_prompts,
)

config = ProcessorConfig(
    concurrency=16,
    attempt_timeout=30.0,
    rate_limit=RateLimitConfig(cooldown_seconds=60.0),
    guardrails=GuardrailConfig(
        total_timeout_per_item=180.0,
        batch_timeout=6600.0,
        abort_on_error_categories=frozenset({"authentication", "insufficient_balance"}),
    ),
)

batch = await process_prompts(
    llm("openai:gpt-4o-mini"),
    prompts,
    config=config,
    progress=True,
    artifact_store=JsonlArtifactStore("runs/overnight.jsonl"),
    resume=ResumePolicy.REUSE_SUCCESSES,
)
print(batch.summary())
```

Rerunning the same command after a crash replays completed successes from the
artifact and executes only the remainder.

## If memory is the constraint

For very large or unbounded inputs, add `max_queue_size` (bounded input
buffering) and switch to `process_stream()` so results don't accumulate.
That's a memory decision, not a throughput one — see
[bounded work and backpressure](bounded-work.md).
