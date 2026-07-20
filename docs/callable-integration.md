# Use Your Existing Async Client

`CallableStrategy` connects an existing async provider SDK, OpenAI-compatible
client, third-party gateway, PydanticAI agent, or internal application service
to ABL's existing execution pipeline. Built-in provider wrappers are
conveniences, not requirements.

The adapter does not execute a job or implement retries. Each callback
invocation is one physical operation. The existing `ItemExecutor` still owns
provider-capacity admission, coordinated cooldowns, logical retries, timeouts,
failed-attempt accounting, deadlines, fail-fast behavior, and terminal results.

## Minimal Adapter

```python
from async_batch_llm import (
    ArtifactIdentity,
    CallOutcome,
    CallableStrategy,
    RetryState,
)


async def invoke(
    prompt: str,
    *,
    attempt: int,
    timeout: float,
    state: RetryState | None,
) -> CallOutcome[str]:
    response = await existing_client.generate(prompt, timeout=timeout)
    return CallOutcome(
        output=response.text,
        token_usage={
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
        metadata={"route": response.route},
    )


strategy = CallableStrategy(
    invoke,
    identity=ArtifactIdentity(
        provider="application-gateway",
        model="summary-route",
        prompt_version="v3",
        parser_version="v2",
        application_version="2026.07",
    ),
)
```

The callback must be declared with `async def` and must resolve to
`CallOutcome`. ABL passes `attempt`, `timeout`, and `state` as keyword
arguments.

- `attempt` is the logical content/transport attempt. A rate limit retries the
  same logical attempt; validation or ordinary retryable failures advance it.
- `timeout` is the effective remaining per-attempt budget. It is useful for a
  client-native timeout, but ABL still enforces the outer timeout.
- `state` is private to one work item and persists across its attempts. Store
  validation feedback, partial data, or escalation choices there—never on the
  shared `CallableStrategy` instance.

Work-item prompts remain strings. Serialize application input into the prompt
and carry non-prompt application data through work-item context.

## Constructor Reference

```text
CallableStrategy(
    invoke,
    *,
    identity=None,
    error_classifier=None,
    prepare=None,
    cleanup=None,
    on_error=None,
    dry_run=None,
    max_concurrency=None,
    concurrency_scope=None,
    request_concurrency=None,
)
```

- `invoke`: required async callback shown above.
- `identity`: stable `ArtifactIdentity` for checkpoint compatibility.
- `error_classifier`: optional application/provider classifier recommended to
  execution surfaces. An explicit processor classifier takes precedence.
- `prepare` / `cleanup`: synchronous or async no-argument lifecycle callbacks.
  The existing lifecycle manager prepares once per unique strategy instance
  and cleans up every successfully prepared strategy once, including failure
  and caller-cancellation paths.
- `on_error`: synchronous or async
  `(exception, attempt, state) -> None`. It receives the same per-item state as
  `invoke`.
- `dry_run`: synchronous or async `(prompt) -> CallOutcome`. Without it, the
  standard placeholder dry-run result is used.
- `max_concurrency`: positive client/transport capacity advertised to ABL.
- `concurrency_scope`: identity shared by strategies that use the same client
  pool. The strategy instance is the default scope.
- `request_concurrency`: optional synchronous or async `(concurrency) -> bool`
  hook for clients that can safely resize their connection pool.

Callback exceptions propagate unchanged into `ItemExecutor`. External
`asyncio.CancelledError` remains cancellation; `CallableStrategy` never turns
it into a provider failure.

## Usage and Metadata

`CallOutcome.token_usage` accepts `input_tokens`, `output_tokens`,
`total_tokens`, and `cached_input_tokens`. Values must be non-negative integers
(booleans are rejected). ABL copies the mapping and derives `total_tokens` from
reported input plus output when total is omitted. An empty mapping means the
client reported no usage; ABL does not estimate it or claim complete
accounting.

Metadata may be `None` or a mapping and is copied into a plain dictionary. ABL
does not inspect or persist raw provider response objects automatically. The
reserved `grounding`, `reasoning`, `tool_calls`, and `logprobs` keys retain the
typed `WorkItemResult` views documented for provider output.

When a billed response is received but parsing or validation fails, preserve
that attempt's usage with the existing public exception:

```python
from async_batch_llm import TokenTrackingError

try:
    output = parse_response(response.text)
except ValueError as exc:
    raise TokenTrackingError(
        "billed response failed validation",
        token_usage=response.usage,
    ) from exc
```

Failed-attempt usage is then included in the eventual terminal result alongside
successful retry usage.

## Retry Recovery

```python
def on_error(
    exception: Exception,
    attempt: int,
    state: RetryState | None,
) -> None:
    if state is not None and isinstance(exception, TokenTrackingError):
        state.set("validation_feedback", "Return exactly one JSON object.")
```

Read that value from `state` on the next invocation. Concurrent work items
cannot see one another's `RetryState`.

Avoid allowing both ABL and an upstream gateway to perform the same transport
retry policy at full strength. Choose one layer as the transport-retry owner so
attempt counts, latency, and accounting remain understandable. Even when an
upstream gateway owns transport retries, ABL remains useful for application
validation recovery, total item and batch deadlines, bounded streaming,
checkpoint/replay, terminal outcomes, and application-visible accounting. ABL
cannot report retries hidden inside the upstream service unless that service
reports them.

## Artifact Identity and Replay

Arbitrary callables cannot safely reveal their provider, model, route, parser,
or application version. With `JsonlArtifactStore`, provide identity either on
the strategy or explicitly on the store. Omitting both fails before `invoke`
runs. A lambda, closure, object ID, memory address, or `repr()` is never used as
a replay identity.

An explicit store identity takes precedence. Compatible replay bypasses
`invoke`; changing callable identity invalidates replay. The artifact schema and
checkpoint-before-publication behavior are the same as for built-in strategies.

## Single and Shared Calls

All execution surfaces accept the adapter:

```python
from async_batch_llm import LLMCallPool, call, process_stream

one = await call(strategy, "one prompt")

async with LLMCallPool(strategy, config=config) as pool:
    another = await pool.submit("one service request")

async for result in process_stream(strategy, prompt_source, config=config):
    await save(result)
```

`LLMCallPool` is in-process and queue-less. It directly invokes the shared
executor under a semaphore; it has no background dispatcher. A direct SDK or a
real third-party gateway can sit beneath `CallableStrategy` or another strategy.

See the runnable, network-free
[`example_callable_application.py`](https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_callable_application.py)
for lazy database-style input, transactional output, validation recovery,
failed-attempt tokens, bounded input/output handoff, and a zero-call replay run.

## Optional OpenAI-Compatible Shape

The SDK remains an optional application dependency; it is not required by ABL:

```python
async def invoke(
    prompt: str,
    *,
    attempt: int,
    timeout: float,
    state: RetryState | None,
) -> CallOutcome[str]:
    response = await openai_client.chat.completions.create(
        model="your-route",
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )
    usage = response.usage
    return CallOutcome(
        response.choices[0].message.content or "",
        token_usage={
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
        metadata={"model": response.model},
    )
```
