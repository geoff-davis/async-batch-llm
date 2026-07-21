# Choosing Between ABL and Alternatives

**Last reviewed: 2026-07-20.** Capabilities change; follow the linked primary
documentation before making a long-lived platform decision.

async-batch-llm (ABL) is a narrow, in-process execution layer for independent
online LLM calls. It is not a dataset framework, model server, network gateway,
native batch client, scheduler, or durable workflow engine. That boundary is
often more useful than a feature checklist.

## Scenario recommendations

| Scenario | Likely best fit |
| --- | --- |
| Tiny one-off script | `asyncio.gather` plus a semaphore |
| Synthetic-data or Hugging Face dataset pipeline | Bespoke Curator |
| Central provider routing and policy | LiteLLM or another AI gateway |
| Discounted latency-tolerant provider job | Native batch API or Curator |
| Multi-step durable business workflow | Prefect, Temporal, DBOS, or another workflow engine |
| Embedded async fan-out with incremental durable outcomes | ABL |

## `asyncio.gather`, a semaphore, and a retry library

This is often the correct choice for a small script, a short finite list,
simple retries, and work that does not need durable resume or detailed
lifecycle control. Python's
[`asyncio.gather`](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
is already available, familiar, and flexible.

As requirements grow, application code must define what happens to a lazy
producer, coordinate 429 cooldowns across tasks, preserve one terminal outcome
per accepted item, enforce attempt/item/batch deadlines, account for failed
billed calls, checkpoint before publication, replay compatible work, shut down
on early consumer exit, and keep input and completed-result buffers bounded.
ABL packages those semantics together. It is a worse fit when those semantics
would be unused overhead.

The two approaches compose poorly at the same layer: either let ABL own the
item execution loop, or keep the custom gather loop small and explicit.

## Bespoke Curator

[Bespoke Curator](https://github.com/bespokelabsai/curator) is designed for
bulk inference, synthetic-data generation, curation, post-training data, and
structured extraction. Its native abstraction turns inputs and parsed outputs
into datasets, including Hugging Face `Dataset` integration. Curator provides
structured outputs, caching and interruption recovery, retries, token and cost
statistics, a data viewer, broad providers through backends such as LiteLLM,
local inference through vLLM/Ollama, and provider-native batch execution. Its
[cache fingerprinting and recovery](https://docs.bespokelabs.ai/bespoke-curator/getting-started/automatic-recovery-and-caching)
and [hosted dataset viewer](https://docs.bespokelabs.ai/bespoke-curator/getting-started/visualize-your-dataset-with-the-bespoke-curator-viewer)
are first-class data-pipeline features.

Curator is often the better choice when work begins and ends as a dataset, the
goal is synthetic training data, local/distributed model inference matters, a
viewer improves curation, or delayed native batches are desirable. ABL should
not be chosen merely because the workload has many rows.

ABL's narrower fit is embedding online fan-out inside an existing async
application: consume a database cursor without first constructing a dataset,
wrap an existing async client, yield each terminal result back into application
control flow, bound completed-result handoff, checkpoint before publication,
apply logical item and batch deadlines, and carry item-private validation
recovery state. Its core dependency surface is deliberately small.

Composition is possible. A Curator pipeline can call an application service,
and an ABL callable can sit over infrastructure also used by Curator, but using
both as the execution owner for the same rows usually duplicates caching,
retry, and accounting policy. Choose the owner that matches the surrounding
data flow.

## LiteLLM and other AI gateways

[LiteLLM](https://docs.litellm.ai/) offers a provider-normalizing Python SDK and
a central proxy. Its documented strengths include a common API across many
providers, routing and load balancing across deployments, provider fallback,
authentication and virtual keys, centralized spend/budget policy, logging,
and governance. Those are gateway concerns, and a gateway is the better tool
when many applications need shared provider access policy.

ABL can sit above LiteLLM or another gateway through `CallableStrategy`. ABL
then manages one application run: bounded production and result consumption,
application-level validation recovery, deadlines, terminal outcomes,
checkpoint/replay, and application-visible accounting. It cannot see retries
or provider attempts hidden inside the gateway.

Avoid allowing both layers to run the same transport retry policy at full
strength; nested retry loops make latency, load, and accounting difficult to
reason about. `LLMCallPool` is an in-process, queue-less shared executor, not a
network gateway or a competitor to LiteLLM.

## Provider-native batch APIs

Provider batch APIs accept asynchronous submissions and expose results later.
They are strong for very large, latency-tolerant jobs because the provider owns
scheduling and may offer discounted pricing. For example, the
[OpenAI Batch API](https://help.openai.com/en/articles/9197833-batch-api-faq)
documents a 24-hour processing window and a 50% discount for supported models;
the [Claude Message Batches API](https://platform.claude.com/docs/en/build-with-claude/message-batches)
documents delayed processing and batch pricing.

ABL currently performs incremental online calls. It is a better fit when the
application needs live completion-order results, application-specific retry
state, checkpoint-before-publication, or end-to-end item deadlines. It does not
submit provider-native batch jobs. Curator is a convenient higher-level option
when a dataset workload should use native batch mode without manually building
request files and polling.

## Workflow and durable-execution systems

Systems such as [Prefect](https://docs.prefect.io/v3/concepts/deployments),
[Temporal](https://docs.temporal.io/), and [DBOS](https://docs.dbos.dev/) are
better when work has multiple dependent steps, branching, durable waits,
schedules, multi-service coordination, distributed workers, or human approval
steps. They provide a broader execution and operations model than an LLM
fan-out library.

ABL can run inside one workflow activity or fan-out stage. The workflow engine
owns durable business orchestration; ABL owns the bounded, retry-aware set of
independent LLM calls within that stage. Do not use ABL as a DAG engine.

## When not to use ABL

Choose something else when:

- there are fewer than a handful of simple calls;
- the job is primarily synthetic-data or dataset curation;
- distributed local-model inference is required;
- provider credentials, routing, budgets, and governance must be centralized;
- delayed provider batch jobs provide the right price/latency trade-off;
- work is an arbitrary DAG with scheduling, durable waits, or approvals; or
- restart, streaming, accounting, and recovery semantics do not matter.

If the deciding factor is operational behavior rather than product category,
continue with [Choosing Your Limits](choosing-your-limits.md) and
[Bounded Work and Backpressure](bounded-work.md).
