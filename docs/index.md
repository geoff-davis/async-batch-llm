# async-batch-llm

Run independent LLM calls concurrently with bounded input and result handoffs,
coordinated retries, deadlines, checkpoint/replay, and terminal outcomes for
every accepted item. Use a built-in provider convenience or wrap an async
client you already have.

## First batch

Install a provider and the optional terminal progress bar:

```bash
pip install 'async-batch-llm[openai,progress]'
export OPENAI_API_KEY='...'
```

```python
import asyncio
from async_batch_llm import llm, process_prompts

async def main():
    batch = await process_prompts(
        llm("openai:gpt-4o-mini"), ["Summarize A", "Summarize B"],
        concurrency=10, progress=True,
    )
    print(batch.summary())

asyncio.run(main())
```

The same execution engine powers collection, completion-order streaming,
single calls, and the in-process `LLMCallPool`. Built-in provider wrappers are
conveniences, not requirements.

## Bring an existing client

`CallableStrategy` adapts an async SDK, gateway client, PydanticAI agent, or
internal service without introducing another executor:

```python
from async_batch_llm import ArtifactIdentity, CallOutcome, CallableStrategy

async def invoke(prompt, *, attempt, timeout, state):
    response = await client.generate(prompt, timeout=timeout)
    return CallOutcome(
        response.text,
        # token_usage must be a str -> int mapping, not an SDK usage object
        token_usage={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )

strategy = CallableStrategy(
    invoke,
    identity=ArtifactIdentity(provider="internal", model="summary-route"),
)
```

The [credential-free application example](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_callable_application.py)
demonstrates a lazy source, stateful recovery, failed-attempt token accounting,
bounded result handoff, transactional writes, and checkpoint replay.

## Where to go next

- [Getting Started](getting-started.md) — one batch through streaming and an
  existing client
- [Choosing Your Limits](choosing-your-limits.md) — concurrency, pools,
  admission, retries, and deadlines
- [Bounded Work and Backpressure](bounded-work.md) — memory behavior for large
  lazy sources
- [Results, Artifacts, and Resume](results-and-artifacts.md) — terminal results
  and durable replay
- [Compare alternatives](comparison.md) — when gather, Curator, a gateway,
  native batch, or a workflow engine is a better fit
- [Troubleshooting and FAQ](troubleshooting.md) — operational symptoms and
  fixes
- [API Reference](api/core.md) — public classes and functions
- [API Stability (draft)](stability.md) — what 1.0 keeps stable
- [v0.27 migration guide](migration/v0.27.md) — changed defaults and deprecations

## Project status

The project is beta software. APIs are typed and covered by deterministic
tests, but release notes and migration guides should be reviewed before an
upgrade. Contributions and focused production feedback are welcome.

v0.27 is the last release before 1.0, and the last to support Python 3.10; 1.0
requires Python 3.11 or newer. v0.27 deprecates several names that 1.0 removes, so
run your tests with `-W error::DeprecationWarning` before upgrading ([Testing](testing.md#4-preparing-for-10)
covers a google-genai warning to filter on Python 3.14). See
[API stability](stability.md) for what 1.0 keeps stable, and the
[v0.27 migration guide](migration/v0.27.md) for the deprecations.

## License

MIT License — see [LICENSE](https://github.com/geoff-davis/async-batch-llm/blob/main/LICENSE).
