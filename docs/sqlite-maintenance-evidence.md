# SQLite maintenance decisions

The September 2026 evidence supports retaining the SQLite wait helper's backup
wakeup and all three existing indexes. No runtime policy or database schema
changes are proposed. These decisions use source revision `377ce4899cc0d9a59346418f4fdad5ace21c4491`.

## Waiting: retain the backup pending a reproducible failure

The owned-work bridge `_await_without_cancelling` protects preparation, executor
operations, accepted appends, detached-write settlement, reader cleanup, and
close from caller cancellation. The normal done callback wakes the caller; a
50 ms timer is a backup, not a mandatory delay on each operation. Cancelling the
caller removes its callback/timer without cancelling the owned work.

The bridge first appeared in commit `00784ce`, with a 1 ms backup. Commit
`cae2050` increased the interval to 50 ms. The historical
`reports/v0.21_session_a_handoff.md` describes a Python 3.14.2 wheel-smoke hang,
but supplies neither a standalone reproducer nor a linked CPython issue.
The repository issue search found the original SQLite feature issue
[#124](https://github.com/geoff-davis/async-batch-llm/issues/124), not a separate
upstream defect record. Targeted upstream searches did not establish a matching
issue. This is an evidence gap, not proof that the historical hang was impossible.

Python documents that registering a callback on an already-completed future
schedules that callback through the event loop. Completion during registration
alone is therefore insufficient evidence of a runtime defect. See the official
[Future callback contract](https://docs.python.org/3.14/library/asyncio-future.html#asyncio.Future.add_done_callback).
The source comment now states the uncertainty instead of asserting a confirmed
CPython bug. The implementation and its backup timer are unchanged.

### Comparison method and results

`benchmarks/sqlite_wait_evidence.py` compares the current bridge, a callback-only
bridge, `asyncio.shield`, and `asyncio.wait`. It exercises 2,000 completions per
variant/runtime: already done, queued completion, and dedicated-executor
completion. Each variant also survives three successive cancelled callers of
the same owned future, then returns its value or re-raises the original failure.
These are attempted reproductions using normal futures, not a reproduction of
the alleged registration race. They do not force an implementation to lose a
callback or simulate a broken event loop.

All completion/cancellation probes passed on Python 3.10.19, 3.11.14, 3.12.12,
3.13.7, and 3.14.2. No stranded waiter was reproduced. Python 3.14's shield
variant emitted one loop diagnostic for the failure after repeated cancellation;
the harness records it in `loop_diagnostics`. It is not treated as a hang or a
lost result. The current, callback-only, and wait variants emitted none.

The opt-in plugin `benchmarks/sqlite_wait_plugin.py` runs the existing SQLite
suite against each waiting implementation. Every variant passed **84 tests on
each of the five interpreters**: 1,680 passing cases in total. This covers
repeated reader cancellation, event-loop responsiveness, reader thread exit,
detached append success/failure, failure delivery once, close races, preparation
cancellation, WAL cleanup, executor termination, and normal replay/inspection.
Only `test_lifecycle_does_not_depend_on_asyncio_shield` is deselected in the
comparison because it asserts the current implementation choice. It remains in
normal CI; its docstring now describes that characterization accurately.

Each performance trial performs 2,000 real public SQLite lookups after warmup.
There are five trials with rotated variant order. Values below are elapsed
milliseconds, shown as minimum / median / maximum; they are not cross-version
speed comparisons.

| Python | Current timer bridge | Callback only | Shield | Wait |
| --- | --- | --- | --- | --- |
| 3.10.19 | 98.69 / 115.01 / 129.08 | 86.22 / 106.92 / 118.64 | 86.94 / 90.60 / 128.49 | 103.27 / 122.09 / 153.37 |
| 3.11.14 | 73.05 / 74.38 / 106.49 | 66.01 / 69.40 / 99.51 | 69.69 / 72.63 / 101.59 | 83.37 / 97.73 / 113.69 |
| 3.12.12 | 63.27 / 84.11 / 91.34 | 58.78 / 67.30 / 74.28 | 63.03 / 79.43 / 90.96 | 69.38 / 72.78 / 103.00 |
| 3.13.7 | 81.27 / 95.93 / 104.43 | 67.22 / 93.45 / 95.23 | 69.19 / 89.49 / 95.69 | 80.30 / 96.14 / 116.99 |
| 3.14.2 | 71.59 / 81.25 / 99.50 | 65.87 / 87.77 / 96.99 | 67.85 / 78.04 / 97.22 | 77.65 / 83.58 / 111.46 |

The current bridge created 2,000 backup timers in every 2,000-lookup trial; the
alternatives created zero. Across these short runs, the 1 ms heartbeat's largest
observed scheduling overrun was 1.24 ms. Removing timers has a measurable
allocation effect but no consistent demonstrated latency benefit on every
runtime. Timings overlap, and these local runs did not isolate CPU or filesystem
contention. Python 3.13 used SQLite 3.46.1; the other interpreters used 3.50.4.
Long-running blocked operations and heavy multi-store contention are not covered
by this throughput comparison.

### Open investigation record

Status: unresolved; retain the backup. To justify removing it, recover the
historical wheel-smoke reproducer or produce a standalone example that strands a
normal future. Record the exact interpreter build, event-loop implementation,
thread ownership, cancellation sequence, and whether it also occurs outside the
execution sandbox. Compare callback-only and shielded waits under that same
trigger and link a matching upstream issue if one exists. A reduction in timer
allocations alone does not establish that the protection is unnecessary.
This record is local to the repository; no external issue has been filed.

## Indexes: retain all three

Inspection establishes three different access paths:

| Index | Rows included | Consumer |
| --- | --- | --- |
| `idx_item_records_replay_all` | `replay_eligible = 1` | `_lookup_sync`, `REUSE_ALL` |
| `idx_item_records_replay_success` | `replay_eligible = 1 AND success = 1` | `_lookup_sync`, `REUSE_SUCCESSES` |
| `idx_item_records_success_sequence` | `success = 1` | Success-only keyset iteration and `read_results` |

The first two share the full compatibility-key columns followed by descending
sequence. Their predicates differ. The third indexes sequence alone and includes
successes regardless of replay eligibility, because inspection is not replay.
`ResumePolicy.NONE` bypasses lookup. All-results iteration uses the primary-key
sequence range. The store's existing-database validator requires all three index
names, so removal would also require an explicit compatibility/migration decision.

A narrower partial index can avoid scanning failures even when a broader index
can technically answer a success query. SQLite's
[partial-index documentation](https://www.sqlite.org/partialindex.html),
[query-plan explanation](https://www.sqlite.org/eqp.html), and
[ANALYZE documentation](https://www.sqlite.org/lang_analyze.html) describe the
mechanisms used by this experiment.

### Experiment

`benchmarks/sqlite_index_evidence.py` creates disposable databases through the
real store, then compares the complete index set with removal of each index
individually. No existing user database is opened or changed. Each run has 60
compatibility keys, alternating NULL/non-NULL context, and 300 versions per key:
18,000 rows. Three history distributions and three rotated-order trials produce
36 databases. The histories have frequent successes, sparse successes, or only
an old initial success. The newest versions include both explicitly ineligible
rows and legacy category-excluded rows marked eligible.

Lookups call production `_lookup_sync`, including row validation and legacy
exclusion. Exact selected sequences and output values are asserted; digests match
across all index designs and trials for both replay policies. Missing keys are
also queried. Success-only pages call production `_read_page_sync`. Query plans
and timings are captured before and after `ANALYZE`, for NULL and non-NULL keys.
Inserts use the real schema and 500-row transactions; fixture generation is
outside the measured insertion/commit interval. Production WAL, synchronous,
and auto-checkpoint settings remain in force. These are physical database
measurements, not public append API throughput estimates.

The table shows medians of per-trial lookup medians after `ANALYZE`, in
microseconds. Page measurements are the first success-only page (up to 1,000
rows), so compare designs within the same history, not between histories.

| History | Index set | Success replay | All replay | Missing all-replay key | Success page |
| --- | --- | ---: | ---: | ---: | ---: |
| Frequent success | Full | 8.34 | 12.69 | 2.34 | 5,983 |
| Frequent success | Without success replay | 7.50 | 12.73 | 2.36 | 6,030 |
| Frequent success | Without all replay | 9.07 | 19.09 | 2,351.15 | 5,793 |
| Frequent success | Without success sequence | 8.43 | 12.83 | 2.27 | 6,029 |
| Sparse success | Full | 7.06 | 12.71 | 2.38 | 1,454 |
| Sparse success | Without success replay | 53.06 | 12.87 | 2.32 | 1,373 |
| Sparse success | Without all replay | 7.35 | 18.86 | 2,333.91 | 1,408 |
| Sparse success | Without success sequence | 7.13 | 13.11 | 2.29 | 3,891 |
| Old success only | Full | 6.76 | 12.95 | 2.38 | 581 |
| Old success only | Without success replay | 77.91 | 12.73 | 2.40 | 578 |
| Old success only | Without all replay | 6.86 | 18.79 | 2,274.67 | 536 |
| Old success only | Without success sequence | 6.70 | 12.91 | 2.33 | 3,010 |

The success replay index avoids backward scans through failures. Removing the
all-replay index yields a table scan and makes missing-key lookup roughly three
orders of magnitude slower in these fixtures. Removing the success-sequence
index increases sparse-success inspection work. `ANALYZE` did not eliminate
these tradeoffs; the raw report includes the pre-analysis distributions too.

Indexes have write and space costs. For frequent-success history, median total
insert-plus-commit time fell from 107.85 ms to 62.37 ms without the success replay
index. Median main DB size fell from 31.48 MiB to 24.38 MiB, and the sampled WAL
size from 6.16 MiB to 5.47 MiB. For old-success history, removing that index
barely changed write time (60.07 versus 59.94 ms) or main DB size (24.22 versus
24.20 MiB), while substantially harming success lookup. Raw data records insert
and commit time separately, all index variants, page counts, and DB/WAL sizes.
Sizes are sampled before final close/checkpoint; they are not peak disk usage or
post-VACUUM estimates. Small local fixtures do not predict every workload, but
they establish a concrete reason to preserve each index. No schema migration is
warranted by this evidence.

## Reproduction and scope

From the repository root, using the pinned environment:

```bash
uv sync --frozen --extra dev --extra docs
uv run python -m benchmarks.sqlite_wait_evidence --output /tmp/sqlite-waits.json
ABL_WAIT_VARIANT=callback uv run python -m pytest \
  -p benchmarks.sqlite_wait_plugin tests/test_sqlite_artifacts.py -q \
  -k 'not lifecycle_does_not_depend_on_asyncio_shield'
uv run python -m benchmarks.sqlite_index_evidence --output /tmp/sqlite-indexes.json
```

Repeat the plugin with `current`, `callback`, `shield`, and `wait`. To reproduce
other interpreter runs, sync a separate uv environment for that interpreter and
run its `python -m ...` with `PYTHONPATH=src`. Raw distributions and query plans
are checked in under `reports/sqlite-maintenance/`. The tools are opt-in and do
not change the application's default wait function outside their own process.

No replay predicate, category set, exception-swallowing policy, stats payload,
public API, or persistence format changed. Statistics deduplication, reader pools,
batched lookup, and module rearrangement remain separate work. JSONL is unaffected;
the normal cross-store regression suite remains the compatibility check.
