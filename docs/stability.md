# API stability (draft for 1.0)

This page is a draft. It describes the compatibility promise planned for 1.0 and
classifies today's public API against it. Until 1.0 ships, a minor release can
still make breaking changes: each one is announced in that release's migration
guide, and removals follow the deprecation policy below.

## What semver will protect from 1.0

After 1.0, a minor or patch release will not break code that uses the following,
except where a name or field is marked provisional below:

- names exported from `async_batch_llm` (`__all__`) and their documented signatures,
  fields and defaults;
- the error and timeout category values in `ErrorCategory` and `TimeoutCategory`;
- `ProcessingEvent` names and their documented payload keys;
- the artifact store format (`artifact_schema_version` 1) and the serialized result
  schema: files written by one 1.x release load in every later 1.x release;
- the documented behavior of retries, deadlines, admission, replay and cleanup.

Breaking any of these requires a major release. Additions (new names, new optional
parameters, new category values, new payload keys) can arrive in minor releases.

## What it doesn't cover

- Anything in `async_batch_llm._internal`, and underscore-prefixed names anywhere.
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

## Support policy

- **Python:** each supported version until its upstream end of life. 1.0 requires
  Python 3.11 or newer; v0.27 is the last release for 3.10.
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
`.reasoning`, `.tool_calls`, `.logprobs`) are provisional until real-provider runs
confirm their shapes; `logprobs` is likely to stay provisional after 1.0.

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

## Provisional through 1.0

- **Budget API.** `AttemptUsage` and the `GuardrailConfig` fields `max_total_tokens`,
  `max_total_cost` and `cost_function` are new in 0.27 and stay provisional in 1.0,
  so their shape can still change in a minor release once they have seen real use.
- **Provider-output views.** See above; `logprobs` is the most likely to stay
  provisional.
