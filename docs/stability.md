# API stability (draft for 1.0)

This page is a draft. It describes the compatibility promise planned for 1.0 and
classifies today's public API against it. Until 1.0 ships, a minor release can
still make breaking changes: each one is announced in that release's migration
guide, and removals follow the deprecation policy below.

## What semver will protect from 1.0

After 1.0, a minor or patch release will not break code that uses the following,
except where a name or field is marked provisional below:

- names exported from `async_batch_llm` (`__all__`) and their documented signatures,
  fields and defaults, wherever the name is defined (for example
  `CleanupInterruptedError`, which lives in a private module);
- the error and timeout category values in `ErrorCategory` and `TimeoutCategory`;
- `ProcessingEvent` names and their documented payload keys;
- the artifact store format (`artifact_schema_version` 1) and the serialized result
  schema: files written by one 1.x release load in every later 1.x release;
- the documented behavior of retries, deadlines, admission, replay and cleanup.

Breaking any of these requires a major release. Additions (new names, new optional
parameters, new category values, new payload keys) can arrive in minor releases.

From 1.0, the five configuration classes (`ProcessorConfig`, `RetryConfig`,
`RateLimitConfig`, `StartupRampConfig` and `GuardrailConfig`) take keyword arguments
only, so new fields can go anywhere. Other stable dataclasses, such as `LLMWorkItem`,
can still be built positionally: their field order is frozen, and new fields are
appended with defaults. Provisional dataclasses follow their provisional status.

## What it doesn't cover

- Anything in `async_batch_llm._internal`, and underscore-prefixed names anywhere.
- Submodule import paths such as `async_batch_llm.llm_strategies`,
  `async_batch_llm.observers` or `async_batch_llm.classifiers`. The names work there
  today, but import them from `async_batch_llm`; only the top-level path is covered.
- Log messages, warning text and exception message text. Match on types and
  categories, not strings.
- Exact timing, retry scheduling jitter and performance characteristics.
- **Provisional** APIs (below): they can change in a minor release, with a
  changelog entry.
- The shape of provider SDK objects in `LLMResponse.raw`.

## Deprecation policy

A name or behavior slated for removal emits a warning for at least one minor
release before it is removed; after 1.0, removals happen only in a major release.
`DeprecationWarning` is hidden outside `__main__` by default, so run your tests with
`-W error::DeprecationWarning` to catch uses early. Each release's migration guide
lists its deprecations.

Deprecated names stay in `__all__` until they are removed, so
`from async_batch_llm import *` warns once for each of them, even if your code uses
none, and raises under `-W error::DeprecationWarning`. Import the names you use
explicitly instead. On Python 3.14 an installed `google-genai` also emits its own
`DeprecationWarning` when `async_batch_llm` is imported; add
`-W "ignore::DeprecationWarning:google.genai.types"` after the error filter.

## Support policy

- **Python:** each supported version until its upstream end of life. 1.0 requires
  Python 3.11 or newer; v0.28 is the last release for 3.10.
- **Provider SDKs:** CI tests each SDK at its declared minimum and at the latest
  release of every supported major line. A new major line is supported once a CI
  leg covers it; until then it may install but is untested. A minimum can rise in a
  minor release when an older SDK can no longer be installed and used, or can't
  support a documented feature; the changelog says so.
- **Fixes:** security and bug fixes go into the latest release.

## Classification of the current API

**Stable** names are covered by the promise above. **Provisional** names can still
change in a minor release. **Deprecated** names warn now and leave the public API in
1.0.

### Entry points and configuration

| Name | Status |
| --- | --- |
| `process_prompts`, `process_stream`, `call`, `call_result`, `llm` | Stable |
| `ParallelBatchProcessor`, `LLMCallPool` | Stable |
| `ProcessorConfig`, `RetryConfig`, `RateLimitConfig`, `StartupRampConfig` | Stable |
| `GuardrailConfig`, `AbortMode` | Stable |
| `ResumePolicy` | Stable |
| `LLMWorkItem`, `WorkItemResult`, `BatchResult`, `BatchTermination` | Stable |
| `AttemptTiming`, `WorkItemTiming`, `RetryState`, `TokenUsage`, `CachedTokenRates` | Stable |
| `LLMResponse` | Stable |
| `SimpleBatchProcessor`, `SimpleWorkItem`, `SimpleResult` | Stable |
| `PostProcessorFunc`, `ProgressCallbackFunc` | Stable |
| `ErrorCategory`, `TimeoutCategory` | Stable |
| `AttemptUsage` | Provisional (new in 0.27, with the token and cost budget) |
| `LLMGateway` | Deprecated (use `LLMCallPool`) |
| `BatchProcessor` | Deprecated (use `ParallelBatchProcessor`) |
| `ProcessingStats` | Deprecated (use the dict from `get_stats()`) |

### Strategies, models and classifiers

| Name | Status |
| --- | --- |
| `LLMCallStrategy`, `ModelStrategy`, `CallableStrategy`, `CallOutcome` | Stable |
| `GeminiStrategy`, `OpenAIStrategy`, `OpenRouterStrategy`, `DeepSeekStrategy`, `PydanticAIStrategy` | Stable |
| `GeminiModel`, `GeminiCachedModel`, `OpenAIModel`, `OpenAICompatibleModel`, `OpenRouterModel`, `DeepSeekModel` | Stable |
| `LLMModel`, `ManagedLLMModel`, `MetadataExtractor` | Stable |
| `ErrorClassifier`, `ErrorInfo`, `DefaultErrorClassifier` | Stable |
| `GeminiErrorClassifier`, `OpenAIErrorClassifier`, `OpenRouterErrorClassifier`, `PydanticAIErrorClassifier` | Stable |
| `RateLimitStrategy`, `ExponentialBackoffStrategy`, `FixedDelayStrategy` | Stable |
| `TokenEstimate`, `TokenEstimator`, `CharacterTokenEstimator` | Stable |
| `pydantic_json_parser`, `strip_code_fences` | Stable |
| `Grounding`, `GroundingSource`, `ToolCall` | Provisional (typed provider-output views) |
| `grounding_metadata_extractor` | Deprecated (built-in Gemini models already emit grounding) |

Provider metadata keys and the typed provider-output views (`.grounding`,
`.reasoning`, `.tool_calls`, `.logprobs`) are provisional. They may be promoted to
stable at 1.0.0rc1 once real-provider runs confirm their shapes; `logprobs` is likely
to stay provisional after 1.0. The `metadata` dict itself is stable.

### Middleware, observers and artifacts

| Name | Status |
| --- | --- |
| `Middleware`, `BaseMiddleware` | Stable |
| `ProcessorObserver`, `BaseObserver`, `ProcessingEvent`, `MetricsObserver` | Stable |
| `ArtifactStore`, `JsonlArtifactStore`, `SqliteArtifactStore`, `SqliteDurability`, `ArtifactIdentity` | Stable |

### Exceptions

Every exception type below subclasses `AsyncBatchLLMError`.

| Name | Status |
| --- | --- |
| `AsyncBatchLLMError` | Stable |
| `ItemDeadlineExceeded`, `BatchDeadlineExceeded`, `BatchAbortedError`, `BatchBudgetExceeded` | Stable |
| `BatchAdmissionClosedError`, `BatchInterruptedError`, `StreamFinalizationError`, `CleanupInterruptedError` | Stable |
| `FrameworkTimeoutError`, `RateLimitRetriesExceeded`, `TokenTrackingError`, `LLMCallError` | Stable |
| `EmptyResponseError`, `ProviderResponseError`, `MiddlewareContractError`, `QuotaScopeError` | Stable |
| `StructuredOutputSchemaError`, `StructuredOutputValidationError` | Stable |
| `TokenEstimationError`, `TokenEstimatorRequired`, `TokenEstimateExceedsLimit` | Stable |
| `ArtifactError`, `ArtifactIdentityError`, `ArtifactFormatError`, `ArtifactIOError`, `ArtifactSerializationError` | Stable |
| `ResultSerializationError` | Stable |

`BatchBudgetExceeded`, `ErrorCategory.BATCH_BUDGET_EXCEEDED` and
`termination.kind == "budget_exceeded"` are stable outcome vocabulary: code that
catches or matches them keeps working. The budget configuration that produces them
is provisional (below).

### Testing helpers

`async_batch_llm.testing` exports `MockAgent`, `MockResult`, `FakeStrategy`,
`mock_strategy` and `FakeRateLimitError`. They are Provisional: import them from
`async_batch_llm.testing` (they aren't in the top-level `__all__`), and expect
possible changes in a minor release, announced in the changelog.

### Deprecated members

These parameters and behaviors of stable names already warn, and all of them are
removed in 1.0.

| Deprecated | Replacement |
| --- | --- |
| `ParallelBatchProcessor(max_workers=..., timeout_per_item=..., rate_limit_cooldown=...)` | `ParallelBatchProcessor(config=ProcessorConfig(...))` |
| `ProcessorConfig(timeout_per_item=...)` and `ProcessorConfig.timeout_per_item` | `attempt_timeout` |
| `WorkItemResult.gemini_safety_ratings` | `result.metadata["safety_ratings"]` |
| `BatchResult.cache_hit_rate()` (called) | `BatchResult.cache_hit_rate` (property) |
| Integer prompts in `process_prompts` / `process_stream` | String prompts |
| 2-tuple return from `LLMCallStrategy.execute()` | `(output, tokens, metadata)` |
| `effective_input_tokens()` / `estimated_cost()` with no cached-token rate | Pass a `CachedTokenRates` constant as `cached_token_rate` |
| Positional arguments to `ProcessorConfig`, `RetryConfig`, `RateLimitConfig`, `StartupRampConfig`, `GuardrailConfig` (since 0.28) | Keyword arguments |
| `ProcessorConfig(enable_detailed_logging=True)` (since 0.28; it never had an effect) | Set the `async_batch_llm` logger's level ([Logging](logging.md)) |

The legacy parameters come first in `ParallelBatchProcessor`'s signature, so pass
`config=`, `post_processor=` and the other arguments by keyword.

## Provisional through 1.0

- **Budget API.** `AttemptUsage` and the `GuardrailConfig` fields `max_total_tokens`,
  `max_total_cost` and `cost_function` are new in 0.27 and stay provisional in 1.0,
  so their shape can still change in a minor release once they have seen real use.
- **Provider-output views.** Provisional now; they may be promoted to stable at
  1.0.0rc1 (see above). `logprobs` is the most likely to stay provisional.
