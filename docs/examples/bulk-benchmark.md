# Bulk Benchmark: Minimizing Wall Time

This walkthrough covers `examples/example_batch_benchmark.py` — the flagship
"why async-batch-llm" demo. It runs the [GSM8K](https://github.com/openai/grade-school-math)
math benchmark through several providers and shows, in one run:

1. A **wall-time race** — the same workload three ways, per provider.
2. A **provider bake-off** — DeepSeek Flash vs Gemini 3.1 Flash-Lite vs Gemini
   2.5 Flash-Lite on accuracy, tokens, and cost.
3. **No-thinking → thinking escalation** driven by the retry path.
4. **Streaming gzip I/O** with lock-free concurrent writes (stdlib `gzip`).
5. **LLM-as-judge** as a fallback grader.

## Install and fetch data

```bash
pip install 'async-batch-llm[deepseek,gemini,openai]'
python examples/download_gsm8k.py
```

The downloader fetches the 1,319-item GSM8K test split and writes it to
`examples/data/gsm8k_test.jsonl.gz`. The benchmark reads it back with the stdlib
`gzip` module.

## Configure keys

Set the keys for whichever contestants you want — each is skipped gracefully if
its key is absent:

```bash
export DEEPSEEK_API_KEY=sk-...   # DeepSeek contestant + the wall-time race
export GOOGLE_API_KEY=...        # Gemini contestant (GEMINI_API_KEY also works)
export OPENAI_API_KEY=sk-...     # optional: ChatGPT fallback grader
```

For Gemini you can use the Vertex AI backend with Application Default
Credentials instead of an API key:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
```

```bash
python examples/example_batch_benchmark.py
```

Pass `--skip-race` to skip the wall-time race (whose sequential leg dominates
runtime) and run only the provider bake-off — handy when iterating on the
bake-off:

```bash
python examples/example_batch_benchmark.py --skip-race
```

## The architecture

```text
gzip read .jsonl.gz   (one-time, before timing)
        │
        ▼
ParallelBatchProcessor  ──►  retry · backoff · rate-limit · escalation
        │   (high concurrency)
        ▼
gzip stream-write .jsonl.gz   (concurrent post-processors → atomic blocking writes)
```

### 1. Gzip I/O (stdlib, blocking writes)

Reading is a one-time blocking gzip read, done before any timer starts:

```python
import gzip

with gzip.open(path, "rt") as f:
    for line in f:
        record = json.loads(line)
        ...
```

**Be honest about scale.** At this dataset's size (~240 KB compressed) gzip I/O
barely registers — the read happens once before the timers, and it's dwarfed by
LLM latency regardless. So the demo uses plain blocking `gzip`, *not* an async
or threaded writer. That's a deliberate choice, not a shortcut: a standalone
write-path micro-benchmark (`examples/bench_gzip_write.py`, no API) shows that
for tiny records an async-queue/thread writer runs **several times slower** than
a plain blocking `gzip.write()` — the queue hops and thread-executor overhead
dominate, and only pay off once records reach tens of KiB. At GSM8K's ~50-byte
records that overhead is pure cost. An offloaded writer earns its keep only on
multi-hundred-MB outputs, where a synchronous compress would block the event
loop and stall every concurrent worker.

**Concurrent writers are safe without a lock.** The bake-off's `post_processor`
callbacks run **concurrently**, but a synchronous `gzip.write()` with no `await`
in between is *atomic* with respect to the event loop — it can't switch tasks
mid-write — so concurrent producers share one open file with no lock and no
interleaving:

```python
class StreamingGzipWriter:
    async def __aenter__(self):
        self._fh = gzip.open(self.path, "wt")
        return self

    async def write(self, record):                   # called by each post_processor
        self._fh.write(json.dumps(record) + "\n")    # no await → atomic on the loop
```

(The blocking write does hold the loop for the duration of the compress; at this
record size that's negligible. For huge outputs you'd push it to a thread or
async writer — paying the queue/executor overhead to keep the loop free — but at
this scale that trade goes the other way, which is why we dropped the async
writer.)

Output lands in **completion order**, not input order; each record carries its
`item_id`, so the original order is recoverable downstream (sort by id) — for a
benchmark dump that's all you need.

### 2. Validation-gated thinking escalation

`EscalatingStrategy` picks the model off the attempt number — attempts 1–2 use
the cheap non-thinking mode, attempt 3 escalates to thinking. The escalation is
*validation-gated*: an answer with no parseable `#### <number>` raises, which is
what triggers the retry. The already-spent tokens are attached to the exception
so they still show up in the totals.

```python
async def execute(self, prompt, attempt, timeout, state=None):
    call = self.thinking if attempt >= self.escalate_at else self.fast
    response = await call.generate(prompt)
    answer = extract_answer(response.text)
    if answer is None and attempt < self.max_attempts:
        err = AnswerParseError("no '#### <number>' answer")
        err.__dict__["_failed_token_usage"] = response.token_usage
        raise err           # → retry → escalation
    return GSM8KAnswer(answer, response.text), response.token_usage, metadata
```

**Pitfall — your validation exception must be classified as retryable.**
The provider error classifiers (rightly) treat a generic `ValueError` as a
non-retryable *logic bug* — so raising one would fail the item on attempt 1 and
*never reach the thinking pass*. The fix is a dedicated exception plus a thin
classifier wrapper that marks it retryable and delegates everything else:

```python
class EscalationErrorClassifier(ErrorClassifier):
    def __init__(self, base): self.base = base

    def classify(self, exception):
        if isinstance(exception, AnswerParseError):
            return ErrorInfo(is_retryable=True, is_rate_limit=False,
                             is_timeout=False, error_category="answer_unparsed")
        return self.base.classify(exception)   # real API errors → provider rules
```

Providers differ only in how "thinking" is selected, hidden behind a small
`ModelCall` wrapper:

- **DeepSeek** — two `DeepSeekModel` objects, `thinking=False` / `thinking=True`.
- **Gemini 3.1** — one `GeminiModel`, per-call `thinking_level` (`minimal` vs
  `high`).
- **Gemini 2.5** — one `GeminiModel`, per-call `thinking_budget` (`0` vs a
  positive budget); 2.5 uses a numeric budget rather than a level.

**Caveat — the two Gemini fast passes aren't a matched "no thinking" setup.**
Gemini 2.5's `thinking_budget=0` turns thinking **fully off**, but Gemini 3.1's
level enum has no "off" — `minimal` is the floor and still does a little
thinking. So 3.1 carries a small thinking advantage 2.5 doesn't, and the
3.1-vs-2.5 accuracy gap shouldn't be read as pure model quality. (2.5 Flash-Lite
defaults to thinking off, so `0` matches its default; 3.1 Flash-Lite ships with
thinking on.) Also note that because escalation only fires on a parse failure, a
clean run uses the *fast* config for nearly every item — so the reported Gemini
numbers reflect `minimal` (3.1) / off (2.5), not the `high`/2048 escalation tier.

### 3. Token counting and cost

The package aggregates `total_input_tokens`, `total_cached_tokens`, and
`total_output_tokens` on the `BatchResult`. The demo turns those into dollars,
and uses `effective_input_tokens(rate)` to report cache-adjusted billable input:

```python
billable = batch_result.effective_input_tokens(pricing.cached_rate)
```

where `pricing.cached_rate` is the cache-hit price as a fraction of normal input
price (e.g. DeepSeek V4 Flash bills cache hits at ~2% of the cache-miss rate).

**Note:** the `PRICING` table at the top of the example is dated June 2026.
Always confirm against each provider's current pricing page before quoting
numbers.

### 4. LLM-as-judge fallback grader

GSM8K is exact-match scorable for free, so the judge is *not* on the critical
path. Only the handful of outputs whose answer couldn't be parsed get routed to
a second batch job, where a cheap OpenAI model (`gpt-5-nano`, the `JUDGE_MODEL`
constant) decides whether the model's response matches the gold answer:

```python
strategy = OpenAIStrategy(model, response_parser=lambda r: "YES" in r.text.upper())
```

This chains generation-batch → evaluation-batch and shows the judge pattern
honestly — used only where the cheap path fails.

## What you'll see

Two tables. The **wall-time race** has one row per provider and a column per
orchestration (sequential vs naive `asyncio.gather` vs async-batch-llm) — read
across to compare providers, down to compare orchestrations. The **provider
bake-off** reports accuracy, wall time,
input/cached/output tokens, and estimated cost, plus per-provider detail with
cache hit rates and cache-adjusted billable tokens. Per-item results are written
to `examples/data/benchmark_results/<provider>_results.jsonl.gz`, and the
aggregate numbers behind both tables (wall time, tokens, cost, accuracy) are
dumped to `examples/data/benchmark_results/summary.json` so a run can be cited
without re-running it.

### From a representative run

One full run — **1,319-item bake-off**, 30-item wall-time race, **per-provider
worker counts** (DeepSeek 250, Gemini 3.1 250, Gemini 2.5 **10**), June 2026
pricing (2026-06-09). Numbers shift run-to-run with network latency, model
sampling, and your account's rate limits — treat them as illustrative, not a
spec.

**Wall-time race** (30 items per provider, seconds):

| Provider       | Sequential | `gather` | async-batch-llm | Speedup (seq→abl) | OK |
|----------------|-----------:|---------:|----------------:|------------------:|---:|
| deepseek-flash |       57.0 |      4.3 |             3.4 |             16.9× | 30 |
| gemini-3.1     |       41.3 |      2.6 |             2.1 |             20.1× | 30 |
| gemini-2.5     |       40.1 |      5.0 |             4.4 |              9.0× | 30 |

Concurrency collapses wall time (≈9–20× here), and the framework leg is
neck-and-neck with a bare `gather` (here a touch faster) while *also* handling
the retries, backoff, and rate-limit cooldowns `gather` would silently skip.

**Provider bake-off** (1,319 items each):

| Provider (model)                   | Accuracy | Wall (s) |   Input | Cached |  Output | Cost ($) |
|------------------------------------|---------:|---------:|--------:|-------:|--------:|---------:|
| deepseek-flash (deepseek-v4-flash) |    96.9% |     28.6 | 130,171 | 16,896 | 136,127 |   0.0540 |
| gemini-3.1 (gemini-3.1-flash-lite) |    96.6% |     16.1 | 129,951 |      0 | 267,258 |   0.4334 |
| gemini-2.5 (gemini-2.5-flash-lite) |    95.3% |    679.2 | 129,748 |      0 | 443,609 |   0.1904 |

**Mind the worker counts when reading "Wall (s)" — this is not an
apples-to-apples speed race.** Gemini 2.5 Flash-Lite ran at **10 workers**
because it 503s/times-out hard above ~50, so its 679 s is mostly its rate-limit
ceiling, not raw model speed; DeepSeek and Gemini 3.1 each ran at 250.

The error/retry summary — the framework earning its keep:

- **deepseek-flash** — 1,278/1,319 correct, **0 permanent errors**; 1,328
  attempts (9 retries, 2 escalations to thinking, 9 malformed outputs). Only
  provider with cache hits (13.0% → 113,613 cache-adjusted billable input).
- **gemini-3.1** — 1,274 correct, a clean run: 1,319 attempts, **0 retries, 0
  escalations, 0 errors**.
- **gemini-2.5** — 1,257 correct but a rough session: 1,383 attempts (**64
  retries**, 32 escalations); errors by type `AnswerParseError=36,
  FrameworkTimeoutError=29, ServerError=1`, ending with just 1 unparsed and 2
  permanent failures. The framework absorbed all of it — including the 503
  (`ServerError`) that's now retryable.

Takeaways: **cost** spans ~8× for near-identical accuracy — DeepSeek cheapest
($0.054, helped by cache hits and a lean 136k output), Gemini 3.1 priciest
($0.43, high per-token rate × 267k output), 2.5 in between ($0.19) but the most
verbose (443k output). **Accuracy** is ~95–97% across all three (mind the
3.1-vs-2.5 thinking caveat above). And the LLM-as-judge fired on exactly the 1
unparsed item — used only where the free regex grader fell short.

The headline: concurrency collapses wall time, the framework matches a bare
`gather` for speed while *also* surviving transient errors and rate limits (here:
64 retries and a 503, all absorbed), and you get token/cost accounting for free.
