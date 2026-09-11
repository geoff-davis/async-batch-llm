# Token-Aware Admission

Token-aware admission is an optional local quota smoother for provider request
and token budgets. It controls when an attempt may start; it does not change
provider concurrency, transport capacity, or retry policy.

## Four controls, four jobs

| Control | Bounds | Typical owner |
| --- | --- | --- |
| Worker concurrency | Framework tasks doing useful work | `concurrency` / `max_workers` |
| Provider concurrency | Calls holding a provider-capacity slot | `max_provider_concurrency` or the strategy/model |
| Requests per minute (RPM) | Physical attempts admitted per quota scope | `max_requests_per_minute` |
| Tokens per minute (TPM) | Estimated then reconciled token load per quota scope | `max_tokens_per_minute` plus an estimator |

Equal request counts are not equal load. Ten classification requests may fit
inside one long generation's token budget. Raising workers or connection-pool
size cannot overcome RPM or TPM; it only creates more tasks waiting at the
quota gate.

The live-attempt order is:

```text
coordinated cooldown → token estimation → atomic RPM+TPM reservation
→ provider-capacity wait → provider start
```

No provider-capacity slot is held during estimation or quota waiting.
`WorkItemResult.quota_wait_seconds` is therefore separate from
`admission_wait_seconds`, which measures provider-capacity wait.

## Configure RPM and TPM

```python
from async_batch_llm import CharacterTokenEstimator, ProcessorConfig

config = ProcessorConfig(
    concurrency=32,
    max_requests_per_minute=500,
    max_tokens_per_minute=200_000,
    token_estimator=CharacterTokenEstimator(
        characters_per_token=4.0,
        expected_output_tokens=400,
    ),
)
```

TPM is opt-in. When `max_tokens_per_minute` is configured, every live
provider attempt needs a `TokenEstimate`. Supply `token_estimator` on the
config, on `CallableStrategy`, or through a strategy's `estimate_tokens()`
hook. A missing estimator fails before provider work with
`TokenEstimatorRequired`. An individual estimate larger than the configured
bucket fails immediately with `TokenEstimateExceedsLimit`; it cannot become
admissible by waiting.

`CharacterTokenEstimator` uses a character-ratio heuristic and a fixed
expected output allowance. It is approximate and is never enabled
automatically. Prefer a provider tokenizer and a workload-specific output
estimate when accuracy matters.

## Reservation and reconciliation

Each physical attempt atomically reserves one RPM unit and its estimated input
plus output tokens from one FIFO gate. It never reserves RPM and then waits
separately for TPM.

After provider start:

- Known usage below the estimate refunds the difference.
- Known usage above the estimate records debt. Availability may go negative;
  later attempts wait until refill pays the debt.
- Explicit known zero refunds the full token reservation but still consumes
  the RPM unit because a provider attempt started.
- Unknown usage retains the estimate conservatively. This is distinct from
  known zero.
- Recoverable failed-attempt usage is reconciled before the retry. Every
  physical retry then receives a fresh estimate and reservation.

Cancellation before provider start refunds both the RPM unit and token
reservation. Cancellation after provider start follows the same known/unknown
usage rules as any other started attempt. An item or batch deadline may expire
while waiting; the waiting attempt makes no provider call and leaves no live
reservation.

Dry-run, compatible artifact replay, and middleware-filtered items bypass live
quota admission. They emit no quota-admission events and mutate no RPM/TPM
state.

## Reported usage and canonical totals

Usage knowledge is independent of the compatibility dictionary returned by
exception extraction. Missing usage may expose zero-filled counters, but those
placeholders do not authorize a token refund.

| Provider usage | Overall usage known? | Reported total |
| --- | --- | --- |
| Absent, `None`, empty, or unrelated fields only | No | `None` |
| `cached_input_tokens` only | No | `None` |
| `total_tokens: 0` | Yes | 0 |
| `input_tokens: 7`, `output_tokens: 3` | Yes | 10 |
| Input 7, output 3, optional exception total `None` | Yes | 10 |
| `input_tokens: 7`, output absent | Yes | 7 |
| Recognized counter explicitly `None` or invalid, without a valid total | No | `None` |
| Input 7, output 3, explicit total 12 | Yes | 12 |

A valid explicit total wins over the component sum. In exception usage, an
absent total or `total_tokens=None` permits derivation from input/output
counters when at least one is present and both present counters are valid.
A non-None invalid total remains unknown. An absent component counts as zero;
an invalid component does not. Successful strategy mappings still reject
`total_tokens=None` under their strict validation contract.
The derived total is written into the normalized `total_tokens` field used by
results, events, statistics, and new checkpoints. Caller-owned mappings and
historical artifacts are not modified. Cached tokens are telemetry, not an
additional charge or an automatic deduction from the total.
An optional `cached_input_tokens=None` allows fallback to `cache_read_tokens`
or `prompt_tokens_details.cached_tokens`; a valid explicit zero remains zero.

Successful strategy mappings require non-negative integers: `None`, booleans,
negative, fractional, and non-finite values fail validation. Exception extraction
is best effort and treats invalid counters as unknown rather than known zero.
It retains compatibility with integral numeric values and integer strings in
provider exception usage.

Exception usage supports mappings, attribute objects, modern/legacy provider
field names, and synchronous `.usage()` accessors. A valid framework
`_failed_token_usage` report takes precedence, followed by cause-result usage
and direct exception usage. Empty or invalid framework stamps do not hide a
valid lower-priority report. Async accessors are not started or awaited; ordinary
accessor failures are best effort, while cancellation and process-control
exceptions propagate.

The executor observes each failed physical attempt before recovery hooks or
guardrail error replacement. A synchronous processor `_extract_token_usage`
override can supply a valid positive total (including a derived one); calling
`super()` reuses the observation without calling the provider accessor again.
Zero-filled legacy overrides cannot turn unknown into known zero or erase a
known positive total. Report explicit unbilled usage on the provider exception
when a zero refund is intended.

If `strategy.on_error` adds or changes a valid `_failed_token_usage` stamp,
the executor uses that later report for the failed attempt's result accounting,
including when the hook raises or is interrupted by a deadline or batch abort.
It does not call the provider accessor again or revise the completed quota
reconciliation. A later report replaces that attempt's earlier result usage;
it is not added as another attempt. Empty/invalid stamps do not erase previously
observed usage. Explicit known zero is a valid later result report.

For example, if a failed attempt reserved 20 tokens and its usage was unknown
at reconciliation, admission retains 20. If `on_error` then reports 10, the
item records 10 tokens while its quota timing still shows unknown usage.
This preserves usage supplied during recovery without delaying reservation
finalization behind hooks that may block, fail, or be cancelled.

Quota timing and `QUOTA_RECONCILED` describe individual physical attempts.
An item that fails with 10 reported tokens and then succeeds with 15 reports
25 item tokens, while its reconciliations report 10 and 15 separately.
`ITEM_COMPLETED` reports the final attempt's tokens. Batch live counters add
newly executed usage; replay keeps the historical item usage but adds zero live
consumption. Reservation estimates for unknown attempts never become invented
provider-reported tokens.

## Quota scopes

`quota_scope` identifies the account or upstream budget shared by strategies.
Object identity defines sharing. By default it follows `concurrency_scope`, so
existing strategies preserve their established ownership. Override it when
one account quota spans multiple models or when one shared client serves
independent accounts.

Quota scope and concurrency scope answer different questions:

- `quota_scope`: which calls spend the same RPM/TPM budget and share cooldown?
- `concurrency_scope`: which calls contend for the same client/provider
  capacity?

Two models sharing one account quota:

```python
from async_batch_llm import ArtifactIdentity, CallableStrategy, TokenEstimate

shared_quota = object()


async def call_model_a(prompt, attempt, state):
    ...


async def call_model_b(prompt, attempt, state):
    ...


def estimate_a(prompt, *, strategy, attempt, state):
    return TokenEstimate(input_tokens=count_model_a(prompt), output_tokens=300)


def estimate_b(prompt, *, strategy, attempt, state):
    return TokenEstimate(input_tokens=count_model_b(prompt), output_tokens=600)


strategy_a = CallableStrategy(
    call_model_a,
    quota_scope=shared_quota,
    token_estimator=estimate_a,
    identity=ArtifactIdentity(
        provider="example",
        model="model-a",
        prompt_version="v1",
        parser_version="v1",
        application_version="v1",
    ),
)
strategy_b = CallableStrategy(
    call_model_b,
    quota_scope=shared_quota,
    token_estimator=estimate_b,
    identity=ArtifactIdentity(
        provider="example",
        model="model-b",
        prompt_version="v1",
        parser_version="v1",
        application_version="v1",
    ),
)
```

For independent account budgets, use distinct identities:

```python
account_a_quota = object()
account_b_quota = object()

strategy_a = CallableStrategy(
    call_model_a,
    quota_scope=account_a_quota,
    token_estimator=estimate_a,
    identity=identity_a,
)
strategy_b = CallableStrategy(
    call_model_b,
    quota_scope=account_b_quota,
    token_estimator=estimate_b,
    identity=identity_b,
)
```

Do not derive scope identities from credentials or place arbitrary scope
representations in logs or metrics. Events expose only run-local ordinal scope
IDs.

## Retry-aware estimators

The estimator receives the effective middleware-replaced strategy and prompt,
the logical attempt number, and the item's `RetryState`. This allows validation
recovery or model escalation to change the estimate:

```python
def estimate(prompt, *, strategy, attempt, state):
    escalated = state is not None and state.get("model") == "large"
    output_allowance = 1_200 if escalated else 300
    return TokenEstimate(
        input_tokens=provider_tokenizer(prompt),
        output_tokens=output_allowance,
    )
```

Synchronous estimators run off the event loop. Asynchronous estimators are
awaited directly. Estimator failures are redacted framework errors and are not
sent through strategy or middleware recovery hooks.

## Visibility boundaries

Token accounting covers attempts visible to ABL, including recoverable
failed-attempt usage. Retries hidden inside an upstream gateway require
gateway-reported usage to be visible. If a gateway returns only aggregate
usage, ABL cannot reconstruct its internal attempts or exact reservation
deltas.

Providers implement quotas differently: fixed windows, rolling windows,
weighted model budgets, cached-token rules, and account-level policies all
exist. ABL's gate is a local continuously refilled smoother. It reduces bursts
but does not claim to reproduce a provider's enforcement exactly.

TPM admission also does not schedule active GPU sequences, KV-cache occupancy,
or instantaneous decode-token capacity. Use a serving system or gateway that
owns those resources when that is the real constraint.

## FIFO trade-off

Admission is FIFO within each quota scope. This prevents later small requests
from repeatedly bypassing an earlier large request, but a large head item can
delay small items behind it. Split unrelated workloads into intentional quota
scopes only when they truly have independent upstream budgets; otherwise the
head-of-line behavior reflects a real shared limit.

## Observe and troubleshoot

Attempt timing reports estimate components, reserved and reported tokens,
reconciliation delta, quota wait, and run-local scope ordinal. Processor stats
and `MetricsObserver` aggregate bounded quota wait percentiles, reservations,
reported tokens, refunds, debt, known-zero/unknown attempts, estimator
failures, and scope count. `QUOTA_ADMITTED` and `QUOTA_RECONCILED` events expose
attempt-level evidence without using item or scope IDs as metric labels.

If an estimate exceeds the limit, either increase the configured bucket,
reduce expected output, or route that workload to its real independent quota
scope. Waiting cannot fix a request larger than the bucket. If timeouts or 429s
make admission look conservative, inspect `unknown_usage_attempts`: without a
reliable provider total, retaining the reservation is intentional.

See [Choosing Your Limits](choosing-your-limits.md) for sizing order and
[Troubleshooting](troubleshooting.md) for symptom-based guidance.
