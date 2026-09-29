# Testing

Test code that uses async-batch-llm without spending money on API calls.

## Provider-free FakeStrategy

```python
from async_batch_llm import process_prompts
from async_batch_llm.testing import FakeStrategy

batch = await process_prompts(
    FakeStrategy(lambda prompt: prompt.upper(), token_usage={"input_tokens": 2, "output_tokens": 1}),
    ["hello", "world"],
    preserve_order=True,  # results otherwise arrive in completion order
)
assert list(batch.outputs()) == ["HELLO", "WORLD"]
```

`FakeStrategy` accepts a fixed response or synchronous `response(prompt)` function,
`latency` in seconds, and canonical or OpenAI-style token counters. It needs no extras.
`failure_schedule=[FakeRateLimitError(), None]` makes the first physical call raise
a simulated 429 and the next succeed; import `FakeRateLimitError` from
`async_batch_llm.testing`. Each entry corresponds to invocation order, including retries.
After the schedule ends, calls succeed unless `failure_rate` is set. A nonzero
`failure_rate` requires `seed=` for reproducibility. Concurrent scheduling can change
which item receives a scheduled failure. Pass `identity=ArtifactIdentity(...)` for
artifact persistence. `mock_strategy` remains available with its existing behavior.

## 1. Dry-run mode (no API calls)

`ProcessorConfig(dry_run=True)` runs the whole pipeline — workers, retries,
post-processors — but calls each strategy's `dry_run()` instead of `execute()`,
returning mock output. Good for validating configuration and wiring end to end.

```python
config = ProcessorConfig(dry_run=True)  # no API calls made

async with ParallelBatchProcessor(config=config) as processor:
    await processor.add_work(work_item)
    result = await processor.process_all()  # returns mock data
```

## 2. Mock strategies (unit tests)

`MockAgent` simulates an LLM — latency, rate limits, errors, token usage — with
no network, so it's far faster than real integration tests.

```python
from async_batch_llm.testing import MockAgent

mock_agent = MockAgent(
    response_factory=lambda p: Summary(title="Test", key_points=["A", "B"]),
    latency=0.01,           # simulate 10ms latency
    rate_limit_on_call=3,   # simulate a 429 on the 3rd call (once)
    failure_rate=0.0,       # probability of a random failure per call
)

strategy = PydanticAIStrategy(agent=mock_agent)
```

For a custom strategy, write a tiny `LLMCallStrategy` subclass whose `execute()`
returns canned `(output, tokens, metadata)`; the older `(output, tokens)` 2-tuple is
deprecated and removed in 1.0. See the framework's own
`tests/` for many examples (rate-limit exemption, token accounting, streaming,
backpressure, etc.).

## 3. Small-batch integration tests

Before a 1,000-item run, validate the real pipeline on a handful of items:

```python
test_items = full_dataset[:5]

config = ProcessorConfig(concurrency=2, attempt_timeout=30.0)
result = await process_prompts(strategy, test_items, config=config)

assert result.succeeded == len(test_items)
# ...then process the full batch
```

Real API calls in the framework's own suite live behind the `integration`
pytest marker and are skipped by default.

## 4. Preparing for 1.0

v0.27 deprecates names and behaviors that 1.0 removes. Each emits a
`DeprecationWarning`, which Python hides outside `__main__`, so turn them into
errors in your test run:

```bash
pytest -W error::DeprecationWarning
```

This fails on, among others:

- a custom `execute()` that returns the `(output, tokens)` 2-tuple;
- `effective_input_tokens()` or `estimated_cost()` called without `cached_token_rate`;
- `LLMGateway`, `BatchProcessor`, `ProcessingStats`, and `grounding_metadata_extractor`.

`from async_batch_llm import *` warns once for each deprecated name, so it also fails
under this filter; import the names you use explicitly. Other packages can emit their
own `DeprecationWarning`s (google-genai does at import on Python 3.14); add an
`ignore` filter for those modules, for example
`-W "ignore::DeprecationWarning:google.genai.types"`. The
[v0.27 migration guide](migration/v0.27.md#deprecations) lists every deprecation and
its replacement.
