# Logging

async-batch-llm logs through Python's standard `logging` module. Every logger
is named after its module and sits under the `async_batch_llm` parent (for
example `async_batch_llm.parallel` or
`async_batch_llm._internal.item_executor`), so configuring
`logging.getLogger("async_batch_llm")` controls all of them.

Logs are for people reading a run. Their wording isn't part of the API (see
[API stability](stability.md#what-it-doesnt-cover)); for anything a program
consumes, use [observers and events](api/observers.md), `get_stats()`, or the
returned `BatchResult`.

## What you see by default

The library adds no handlers of its own. If your application never configures
logging, Python's fallback handler prints **WARNING and above** to stderr, without
timestamps or logger names. That means retry warnings and terminal failures appear
even in a script that never mentions logging, while progress lines (INFO) and
detail (DEBUG) don't.

## What each level shows

| Level | What's logged |
| --- | --- |
| `ERROR` | Mostly items that failed for good, plus a few problems that don't end an item: an attempt that hit `attempt_timeout` (logged even when a retry then succeeds), a failing error classifier, a post-processor that failed or timed out, a failed cleanup step, and a cost function that raises or returns an invalid value, stopping the run. A terminal failure gets a traceback only when it's unexpected: `logic_error`, a failing classifier, an unclassified exception, or a non-retryable error raised by your own code rather than a provider SDK. Within one `process_prompts()`/`process_stream()` run, repeated terminal failures of the same kind (category, exception type and the first 200 characters of the message) are logged at ERROR once, then at DEBUG, and the run ends with one ERROR summary per repeated kind, such as "40 items failed with the same error: …". After 1,000 distinct kinds, further new kinds are logged at ERROR each time with no summary. `call()`, `call_result()` and `LLMCallPool` don't deduplicate. |
| `WARNING` | A retry ("Attempt 1/3 failed for item_7: … Retrying in 1.8s"); a rate limit that starts the shared cooldown ("Pausing all workers for 60.0s"); a progress callback that failed or timed out; a configuration value that was adjusted; and problems writing checkpoints, reading an artifact file's damaged tail, closing clients, or extracting response metadata. |
| `INFO` | A progress line every `ProcessorConfig.progress_interval` items (default 10) with throughput, tokens and error counts; the fallback progress line when `progress=True` is used without tqdm installed; a server-suggested cooldown raising the configured one; Gemini cache creation, reuse, renewal and deletion. |
| `DEBUG` | Per-attempt and lifecycle detail, and, within a batch or stream run, repeats of terminal failures already logged once at ERROR. |

## Quieter output

Hide retries, cooldowns and progress, and keep errors:

```python
import logging

logging.getLogger("async_batch_llm").setLevel(logging.ERROR)
```

ERROR isn't only terminal failures (see the table): an attempt timeout is logged
even if the retry succeeds. For an exact record of which items failed, read the
results (`result.success`, `batch.failed`) or an observer's `ITEM_FAILED` events
rather than the log.

Use `logging.CRITICAL` to silence the library's logging entirely, for example in CI
jobs that check the returned `BatchResult` instead. Retries still happen; they just
aren't logged. Python `warnings`, such as the library's `DeprecationWarning`s, go
through the `warnings` module rather than logging, so a log level doesn't hide
them.

## More detail

Configure a handler (here, the root one) and lower the library's level:

```python
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("async_batch_llm").setLevel(logging.INFO)  # or logging.DEBUG
```

INFO adds progress lines and cooldown detail; DEBUG adds per-attempt detail and is
best kept for diagnosing a specific problem on a small run.

To send the library's logs somewhere separate, attach a handler to the
`async_batch_llm` logger and set `propagate = False` on it.

## Alternatives to logs

- **A progress bar:** `process_prompts(..., progress=True)` (install the
  `progress` extra for tqdm), or your own `progress_callback`.
- **A summary at the end:** `print(batch.summary())` reports items, retries,
  tokens and timing percentiles.
- **Structured, per-event data:** an [observer](api/observers.md) receives typed
  lifecycle events (item started, completed or failed, rate limits, cooldowns) with
  documented payloads, which is the right input for metrics or dashboards.
