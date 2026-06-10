# Benchmark Walkthrough

How `examples/example_batch_benchmark.py` — the flagship "why async-batch-llm"
demo — is built. For the **results**, see [Benchmarks](../benchmarks.md); this
page is the architecture and the techniques worth stealing.

The demo runs the [GSM8K](https://github.com/openai/grade-school-math) math
benchmark through several providers and shows, in one run:

1. A **wall-time race** — the same workload three ways, per provider.
2. A **provider bake-off** — DeepSeek Flash vs Gemini 3.1 Flash-Lite vs Gemini
   2.5 Flash-Lite on accuracy, tokens, and cost.
3. **No-thinking → thinking escalation** driven by the retry path.
4. **Streaming gzip I/O** with lock-free concurrent writes (stdlib `gzip`).
5. **LLM-as-judge** as a fallback grader.

## Install and fetch data

```bash
uv sync --extra deepseek --extra gemini --extra openai
uv run python examples/download_gsm8k.py
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

## Run

```bash
uv run python examples/example_batch_benchmark.py              # race + bake-off
uv run python examples/example_batch_benchmark.py --skip-race  # bake-off only (faster)
uv run python examples/example_batch_benchmark.py --throughput # throughput parity only
```

`--skip-race` skips the wall-time race (whose sequential leg dominates runtime).
`--throughput` runs *only* the throughput benchmark. The bake-off writes
`summary.json` and `--throughput` writes `throughput.json` under
`examples/data/benchmark_results/`, plus a per-provider `<provider>_results.jsonl.gz`.

## The architecture

```text
gzip read .jsonl.gz   (one-time, before timing)
        │
        ▼
process_prompts / ParallelBatchProcessor  ──►  retry · backoff · rate-limit · escalation
        │   (high concurrency)
        ▼
gzip stream-write .jsonl.gz   (concurrent post-processors → atomic blocking writes)
```

The bake-off and judge use the high-level **`process_prompts`** API, carrying
per-item data (`gold`, `question`) through `(item_id, prompt, context)` triples
and writing each result via a forwarded `post_processor`. The throughput legs
stay on the low-level `ParallelBatchProcessor` on purpose — they're an
apples-to-apples `process_all`-vs-`gather` timing comparison.

### 1. Gzip I/O (stdlib, blocking)

The dataset is read once with stdlib `gzip` before any timer starts, and results
stream out the same way. The bake-off's `post_processor` callbacks run
concurrently, but a synchronous `gzip.write()` with no `await` in between is
*atomic* with respect to the event loop, so concurrent producers share one open
file with no lock and no interleaving:

```python
class StreamingGzipWriter:
    async def write(self, record):                   # called by each post_processor
        self._fh.write(json.dumps(record) + "\n")    # no await → atomic on the loop
```

At this dataset's size (~240 KB) gzip I/O is negligible next to LLM latency, so
the wall-time win is all concurrency. Output lands in **completion order**, not
input order; each record carries its `item_id`, so the original order is
recoverable downstream (sort by id).

### 2. Validation-gated thinking escalation

`EscalatingStrategy` picks the model off the **attempt number** — attempts 1–2
use the cheap non-thinking mode, attempt 3 escalates to thinking. The escalation
is *validation-gated*: an answer with no parseable `#### <number>` raises, which
is what triggers the retry. The already-spent tokens are attached to the
exception so they still show up in the totals.

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

Because rate-limit errors are exempt from the `max_attempts` budget, a throttled
call is retried at the *same* attempt number — so escalation tracks genuine
**output** failures, never "the endpoint was busy."

**Pitfall — your validation exception must be classified as retryable.** The
provider error classifiers (rightly) treat a generic `ValueError` as a
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
- **Gemini 3.1** — one `GeminiModel`, per-call `thinking_level` (`minimal` vs `high`).
- **Gemini 2.5** — one `GeminiModel`, per-call `thinking_budget` (`0` vs a positive budget).

**Caveat — the two Gemini fast passes aren't a matched "no thinking" setup.**
Gemini 2.5's `thinking_budget=0` turns thinking **fully off**, but Gemini 3.1's
level enum has no "off" — `minimal` is the floor and still does a little
thinking. So 3.1 carries a small thinking advantage 2.5 doesn't, and the
3.1-vs-2.5 accuracy gap shouldn't be read as pure model quality.

### 3. Token counting and cost

The package aggregates `total_input_tokens`, `total_cached_tokens`, and
`total_output_tokens` on the `BatchResult`. The demo turns those into dollars and
uses `effective_input_tokens(rate)` for cache-adjusted billable input:

```python
billable = batch_result.effective_input_tokens(pricing.cached_rate)
```

where `pricing.cached_rate` is the cache-hit price as a fraction of normal input
price (e.g. DeepSeek V4 Flash bills cache hits at ~2% of the cache-miss rate).
The `PRICING` table at the top of the example is dated — confirm against each
provider's current pricing page before quoting numbers.

### 4. LLM-as-judge fallback grader

GSM8K is exact-match scorable for free, so the judge is *not* on the critical
path. Only the handful of outputs whose answer couldn't be parsed get routed to a
second batch job, where a cheap OpenAI model (`gpt-5-nano`) decides whether the
response matches gold:

```python
strategy = OpenAIStrategy(model, response_parser=lambda r: "YES" in r.text.upper())
```

This chains generation-batch → evaluation-batch and shows the judge pattern
honestly — used only where the cheap path fails.
