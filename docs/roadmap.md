# Release roadmap

| Version | Status and scope |
| --- | --- |
| v0.23.0 | Previous provider and online-execution baseline. |
| v0.24.0 | Published RetryState, middleware/replay, and lifecycle corrections. See the [migration guide](migration/v0.24.md). |
| v0.24.1 | Corrective release: usage accounting and verified quota, retry, and DeepSeek compatibility fixes. |
| v0.25.0 | Private attempt-stage and outcome hardening, isolated exception accounting, and SQLite maintenance evidence. See the [upgrade guide](migration/v0.25.md). |
| v0.26.0 | Shared ownership, artifact and provider fixes, admission/lifecycle hardening, and developer diagnostics. See the [migration guide](migration/v0.26.md). |
| v0.27.0 | Last release before 1.0 and the last to support Python 3.10: OpenAI Responses API by default, raised SDK floors, 1.0 deprecations, and the [API stability](stability.md) draft. See the [migration guide](migration/v0.27.md). |

Deferred executor, storage, and API simplification work is tracked in
[#176](https://github.com/geoff-davis/async-batch-llm/issues/176).
