# Production checklist

Tuning knobs that matter when you move from a 10-item test to a real bulk run.
Each item links to the deeper reference where one exists.

## 1. Worker count (`max_workers`)

LLM calls are **I/O-bound**, so `max_workers` is "how many calls in flight at
once", not a CPU count. Do **not** use `os.cpu_count()`.

| Situation | Starting point |
| --- | --- |
| General I/O-bound (most providers) | `5`–`10` |
| Rate-limited / low-quota endpoint | `3`–`5`, lean on the coordinated cooldown |
| High-concurrency provider (e.g. DeepSeek) | `50`–`250`+ — but size the connection pool to match (below) |

Throughput from added workers flattens out well before you exhaust sockets/fds,
so raising `max_workers` past the point where you're latency-bound just adds
contention. Measure with `examples/benchmark_worker_overhead.py` (no network).

## 2. Connection pool (`max_connections`) vs `max_workers`

For the OpenAI-compatible models (`OpenAIModel` / `OpenRouterModel` /
`DeepSeekModel`), the SDK uses httpx's **default ~100-connection pool**. If
`max_workers` exceeds that, the extra workers just block waiting for a
connection — no extra throughput. **Set `max_connections >= max_workers`:**

```python
model = DeepSeekModel.from_api_key(
    "deepseek-v4-flash",
    max_connections=200,   # match your ProcessorConfig(max_workers=...)
)
```

See [OpenAI integration → connection-pool sizing](OPENAI_INTEGRATION.md#connection-pool-sizing-max_connections).

## 3. Open-file limit (`RLIMIT_NOFILE`)

Each in-flight request holds a socket (a file descriptor). A high `max_workers`
plus the connection pool plus your app's own fds can hit the OS open-file limit
— `OSError: [Errno 24] Too many open files`. This bites hardest on **macOS**
(default soft limit ~256). The processor emits a `UserWarning` at construction
when `max_workers` is close to the soft limit; it does not raise the limit for
you. Fix by raising it (`ulimit -n 8192`, or `resource.setrlimit` early in the
process) or lowering `max_workers`. Full guidance:
[Getting started → open file limits](getting-started.md#open-file-limits-and-high-concurrency).

## 4. Timeout vs. retry budget

`timeout_per_item` is **per attempt**, enforced via `asyncio.wait_for` around
each `execute()` — it is **not** a total budget across retries. With
`retry.max_attempts=3`, a single item can spend up to ~`3 × timeout_per_item`
in calls, plus backoff waits.

Rate limits are **exempt** from `max_attempts` (a 429 is "wait and retry", not a
failed attempt — see below), so a throttled item can sit through many cooldowns
without consuming its attempt budget. Bound that separately with
`retry.max_rate_limit_retries` (default `20`). Net: size `timeout_per_item` for
one slow call, and use the two retry budgets to bound total effort.

```python
ProcessorConfig(
    timeout_per_item=60.0,          # per attempt
    retry=RetryConfig(
        max_attempts=3,              # content/transport failures
        max_rate_limit_retries=20,   # throttling retries (separate budget)
    ),
)
```

## 5. Rate-limit configuration

When one worker hits a 429/quota/overload, the framework runs a **coordinated
cooldown** — all workers pause, then slow-start back up — instead of each worker
hammering a throttled endpoint. Tune via `RateLimitConfig`:

| Field | What it does |
| --- | --- |
| `cooldown_seconds` | Base pause after a rate limit (a server `Retry-After` raises it as a floor) |
| `backoff_multiplier` | Grows the cooldown on consecutive rate limits |
| `slow_start_items` / `slow_start_initial_delay` / `slow_start_final_delay` | Ramp delays as workers resume after a cooldown |

Pair with proactive limiting (`ProcessorConfig(max_requests_per_minute=...)`) to
stay under quota before you trip a 429 at all.

## 6. Constant-memory streaming for large inputs

For a very large (or unbounded) input, don't buffer all the work up front. Use
**streaming mode** with a **bounded** `max_queue_size`: workers run while you
feed, so a full queue applies **backpressure** instead of deadlocking, holding
memory roughly constant regardless of input size.

```python
from async_batch_llm import process_stream, ProcessorConfig

config = ProcessorConfig(max_workers=50, max_queue_size=200)  # ~constant memory

async for result in process_stream(strategy, huge_prompt_source, config=config):
    if result.success:
        await save(result.item_id, result.output)   # completion order
```

`huge_prompt_source` can be any sync or async iterable (e.g. a generator reading
a file lazily). The low-level equivalent is
`processor.start()` / `add_work()` / `finish()` / `results()`.

## 7. Single calls and the gateway (request paths)

For a web service's request path — where work arrives one call at a time, not as
a batch — use [`LLMGateway`](api/single-gateway.md) instead of standing up a
processor per request:

- **One long-lived gateway per app.** Create it once at startup (e.g. a FastAPI
  lifespan handler) and share it across all request handlers. A single gateway
  means one shared rate-limit cooldown — when one caller hits a 429, all callers
  briefly pause and then slow-start, instead of a thundering herd.
- **Set `max_pending` and `submit_timeout` for web paths.** `max_pending` caps
  in-flight requests (running + waiting) so an overload sheds load instantly
  (rejecting with a failed result) rather than growing an unbounded waiter list;
  `submit_timeout` bounds per-caller latency so a request stuck behind a cooldown
  returns instead of hanging the handler. Both are off by default.
- **Shutdown drains admitted requests.** `aclose()` (the `async with` exit) stops
  accepting new work, then waits for already-admitted requests to finish before
  cleaning up the shared strategy, so in-flight calls aren't cut off mid-flight.
  `submit_timeout` bounds how long that drain can take.

For a single ad-hoc call, [`call()` / `call_result()`](api/single-gateway.md)
run one prompt through the same resilience pipeline with no pool at all.

## 8. Cleanup

Use the processor as an `async with` context manager so workers, caches, and
HTTP clients are released. If you can't, call `await processor.shutdown()` when
done.
