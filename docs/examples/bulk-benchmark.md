# Bulk Benchmark: Minimizing Wall Time

This walkthrough covers `examples/example_batch_benchmark.py` — the flagship
"why async-batch-llm" demo. It runs the [GSM8K](https://github.com/openai/grade-school-math)
math benchmark through several providers and shows, in one run:

1. A **wall-time race** — the same workload three ways, per provider.
2. A **provider bake-off** — DeepSeek Flash vs Gemini 3.1 Flash-Lite vs Gemini
   2.5 Flash-Lite on accuracy, tokens, and cost.
3. **No-thinking → thinking escalation** driven by the retry path.
4. **Async gzip I/O** on both ends via [`aiogzip`](https://github.com/geoff-davis/aiogzip).
5. **LLM-as-judge** as a fallback grader.

## Install and fetch data

```bash
pip install 'async-batch-llm[deepseek,gemini,openai]' aiogzip
python examples/download_gsm8k.py
```

The downloader fetches the 1,319-item GSM8K test split and writes it to
`examples/data/gsm8k_test.jsonl.gz`. The benchmark stream-reads it back with
`aiogzip`.

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

## The architecture

```text
aiogzip stream-read .jsonl.gz
        │   (async I/O)
        ▼
ParallelBatchProcessor  ──►  retry · backoff · rate-limit · escalation
        │   (high concurrency)
        ▼
aiogzip stream-write .jsonl.gz   (concurrent post-processors → single-consumer queue)
```

### 1. Async I/O with aiogzip

Reading is a plain async loop over the gzip lines:

```python
from aiogzip import AsyncGzipFile

async with AsyncGzipFile(str(path), "rt") as f:
    async for line in f:
        record = json.loads(line)
        ...
```

**Be honest about scale.** At this dataset's size (~240 KB compressed) gzip I/O
barely registers, so aiogzip is used as plumbing, *not* in the timing race: the
read happens once before any timer starts, and the race legs do no writing
(results are streamed to gzip only in the bake-off). A standalone write-path
benchmark (`examples/bench_gzip_write.py`, no API) makes the reason concrete —
for tiny records the async queue writer is ~2.9× *slower* than a plain blocking
`gzip.write()` (queue hops + thread-executor overhead), crossing over to faster
only once records reach tens of KiB. At ~50-byte records that overhead is ~20 µs
across the whole run — utterly dwarfed by LLM latency. aiogzip's real payoff is
on large batches, where a synchronous `gzip.read()`/`write()` of a
multi-hundred-MB file blocks the event loop and stalls every concurrent worker,
whereas aiogzip offloads zlib to a thread (above a 256 KiB chunk threshold).

**Pitfall — one aiogzip file is not safe for concurrent writers.**
The bake-off's `post_processor` callbacks run **concurrently**, and an open
`aiogzip` file isn't safe for concurrent `await`-ed writes — interleave two and
you corrupt the stream. Rather than serialize the workers behind a lock, the
demo uses a single-consumer queue: each post-processor enqueues (cheap), and one
consumer task owns the file. One writer → no corruption, no lock, loop never
blocks:

```python
class StreamingGzipWriter:
    async def write(self, record):          # called by each post_processor
        await self._queue.put(record)

    async def _consume(self):               # one task owns the file
        async with AsyncGzipFile(self.path, "wt") as out:
            while (record := await self._queue.get()) is not None:
                await out.write(json.dumps(record) + "\n")
```

(A synchronous `gzip.write()` with no `await` is also safe without a lock — it's
atomic on the loop — but it *blocks* the loop while compressing; the queue keeps
that off the critical path.)

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
to `examples/data/benchmark_results/<provider>_results.jsonl.gz`.

The headline: concurrency collapses wall time, the framework matches a bare
`gather` for speed while *also* surviving transient errors and rate limits, and
you get token/cost accounting for free.
