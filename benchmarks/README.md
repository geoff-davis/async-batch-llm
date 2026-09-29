# async-batch-llm developer benchmarks

Deterministic, credential-free measurement tooling. Nothing here ships in the
wheel or is required at runtime.

## scale_soak

The v0.21 scale and restart soak harness (issue #127, phase one). It drives
the real `process_stream()` / `ParallelBatchProcessor` execution surfaces and
the public artifact contract with a seeded fake provider — no network calls —
and emits a versioned JSON report
(`schema_name: async-batch-llm-scale-soak`, version 2).

```bash
# Reduced deterministic regression profile (used by CI):
uv run python -m benchmarks.scale_soak --profile ci --output benchmark-results/scale-ci.json

# Reference profiles (opt in, longer):
make scale-100k
make scale-1m
```

Profiles: `ci` (~2k items/scenario), `100k`, `1m` (large-scenario subset:
healthy, slow_consumer, stop_resume, artifact_bench), and `custom`
(`--items` required). See `python -m benchmarks.scale_soak --help` for every
option.

Scenarios: `healthy`, `slow_consumer`, `rate_limit_wave`, `transport_retry`,
`validation_recovery`, `stop_resume`, `artifact_bench`, `progress_overhead`,
`token_quota_mixed`.

What a run proves (and does not): assertions cover exact order-independent
item accounting, bounded queue high-water marks, replay call elimination,
per-item smart-retry isolation, WAL plateau/truncation, and task/thread/fd
baseline recovery. Fake-provider throughput is **not** a live-provider
performance claim, and the report says so; see `docs/benchmarks.md`.

Reports land in `benchmark-results/` (gitignored). Dated reference reports are
published only after a maintainer actually runs the profile.

## Other tools

- `benchmark_progress.py` — opt-in microbenchmark for progress-dispatch overhead
  (descriptive JSON, no pass/fail threshold).
- `sqlite_wait_evidence.py`, `sqlite_wait_plugin.py`, `sqlite_index_evidence.py` —
  opt-in SQLite waiting and index experiments behind
  [docs/sqlite-maintenance-evidence.md](../docs/sqlite-maintenance-evidence.md).
  They never change production defaults.
