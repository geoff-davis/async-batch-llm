# Cancellation CI follow-up

## Target and finding

Base: `7fb74d482377db3cc790fb64556d1a8c46a7ff93`. The first follow-up target
is `997647da6738b3db02dc036338cbeb86761b28a3`. Production code is unchanged.
Final test-file SHA-256:
`fd557eb61924a9223204c136882a48d8034df0d52a2234a83c02146eb2697086`.

GitHub Actions run `34378064467`, Python 3.11 job `102555767597`, failed
`test_c2_first_cancellation_defers_until_teardown_completes` at
`assert strategy.cleaned`. Every other check passed.

The test cancelled its context task after 50 ms without establishing that
artifact preparation and execution had completed. Cancellation before strategy
preparation owes no strategy cleanup. Delaying fake-store preparation by 100 ms
reproduces the same assertion failure; this establishes a test race, not a
production cleanup regression or proof of the runner's exact scheduling.

## Fail-first evidence

The test-only overlay on the base adds `_Store.prepare_delay` and parameterizes
the first-cancellation and cleanup-error tests with 0 and 100 ms preparation.
It does not change their synchronization, assertions, or production code.

- Overlay: `/tmp/abl-cancellation-test-only.patch`.
- SHA-256: `1678c3e7dd97e3c39864f1ae8e27abd72c4caaaeb4d9140c987f2f88acaae443`.
- Before log: `/tmp/abl-cancellation-before.log` (Python 3.11.14).
- Before: **2 failed, 2 passed, 103 deselected**. Both slow cases fail:
  missing strategy cleanup and missing cleanup-error logging, respectively.
- After log: `/tmp/abl-cancellation-after.log`: **4 passed, 103 deselected**
  on the same Python 3.11.14 environment and command.
- These are local evidence files, not published PR attachments.

Apply the overlay only to an isolated extraction of the named base. Run this
same selection before and after (with the Python 3.11 environment active):

```sh
uv run --no-sync pytest tests/test_cleanup_contract.py -q --tb=short \
  -k 'test_c2_first_cancellation or test_c2_cancellation_stays_primary'
```

## Finding-to-change mapping and scope sweep

All implementation changes are test infrastructure in
`tests/test_cleanup_contract.py`; existing cleanup assertions are retained.

| Test(s) | Established precondition |
| --- | --- |
| `test_c2_first_cancellation_defers_until_teardown_completes`, `test_c2_cancellation_stays_primary_over_ordinary_cleanup_errors` | `_context_body` signals completion of `_run_one` before cancellation; both retain slow-preparation cases |
| `test_c2_second_cancellation_force_aborts_and_skips_dependents` | First cancellation follows strategy execution entry; second follows cleanup entry |
| `test_c2_shutdown_call_cancelled_once_finishes_then_reraises` | Cancellation follows cleanup entry |
| `test_c1_shutdown_order_runtime_then_admission_then_strategies_then_store`, `test_c1_worker_threshold_is_diagnostic_only` | Shutdown follows stubborn-worker execution entry |
| `test_c1_progress_threshold_is_diagnostic_only` | Shutdown follows progress-callback entry |

The latter five tests are preventive hardening; no separate fail-first claim
is made for them. Signals replace scheduling guesses, not the intentional
delays used to check cleanup duration or diagnostic thresholds. Signal waits
have two-second bounds. This is not a claim that every timing-sensitive test
in the repository is now deterministic.

No category hierarchy, replay predicate, exception-swallowing rule, artifact
backend, or cancellation implementation changes. Existing first-cancellation
deferral, second-cancellation force-abort, error precedence, and cleanup-order
policies remain the contract under test.

## Validation

- Python 3.13.7: `make format` and `make ci`: **1,983 passed, 19 deselected,
  47 warnings**; Ruff, mypy, ty, coverage/scale-soak, and Markdown lint passed.
  Log: `/tmp/abl-cancellation-ci-final.log`.
- Python 3.11.14: `uv run --no-sync pytest tests/ -q --tb=short`:
  **1,983 passed, 19 deselected, 7 warnings**.
  Log: `/tmp/abl-cancellation-full-py311.log`.
- Python 3.10.19: `uv run --no-sync pytest tests/test_cleanup_contract.py -q --tb=short`:
  **107 passed**. Log: `/tmp/abl-cancellation-py310-final.log`.
- Python 3.11.14 cleanup suite: **107 passed** before the final one-line
  shutdown-order synchronization change, which the full 3.11 run includes.
- `npx markdownlint-cli2 reports/cancellation_ci_handoff.md` and
  `git diff --check`: passed.

## Second CI follow-up: checkpoint-test deadline placement

Base: `997647da6738b3db02dc036338cbeb86761b28a3`. Target: the next follow-up
commit containing this section. GitHub Actions run `34379315181` passed on
Python 3.10 through 3.13, but Python 3.14 job `102559938138` failed
`test_item_timeout_checkpoint_failure_still_propagates[sqlite-ArtifactFormatError-append]`:
the expected append error was never raised.

That test gave the entire item 50 ms, including store opening/preparation.
If preparation is interrupted, no artifact key exists and no append is owed.
A 100 ms preparation delay reproduces the same failure on both stores and
both exception types. It confirms the test race, not the runner's precise
scheduling or a production regression.

- Test-only overlay on the named base: `/tmp/abl-timeout-test-only.patch`.
- SHA-256: `ca128c24d6db086d66593d08800079786da09dcb98e91626086c95ef544225fc`.
- Before log `/tmp/abl-timeout-before.log`: **4 failed, 8 passed, 118 deselected**.
- After log `/tmp/abl-timeout-after.log`: **12 passed, 118 deselected**.
- Both runs: Python 3.14.2; one dependency deprecation warning. The overlay
  adds delayed-preparation cases without changing the timeout or assertions.
- Same command before and after, using the appropriate isolated target:

```sh
uv run --no-sync pytest tests/test_middleware_artifacts.py -q --tb=short \
  -k test_item_timeout_checkpoint_failure_still_propagates
```

The test now injects `ItemDeadlineExceeded` from its strategy's execution
method, after real store preparation. The append callback verifies execution
was reached and the result category is `framework_total_item_timeout` before
raising the injected persistence error. Slow-preparation cases remain in the
suite. This isolates checkpoint-error policy from wall-clock deadline placement;
it is not an end-to-end timer-expiry test for the append path.

The preparation-phase variant still expires a real timer during middleware,
then checks that audit-only preparation errors propagate. The same matrix
covers `ArtifactIOError` and `ArtifactFormatError` on JSONL and SQLite.
`test_audit_artifact_failure_preserves_controlled_stop` still covers the sibling
batch-abort/deadline best-effort policy. `tests/test_guardrails.py` retains the
real execution deadline tests. No runtime code or shared policy changes.

## Local Python 3.14 finding: cross-test SDK finalizers

The first full Python 3.14 run after the checkpoint-test change had **1 failed,
1,986 passed, 19 deselected**. Only the healthy scale scenario failed: its
task snapshot rose from one to two. The scenario passed in isolation.

- Pre-fix snapshot: base `997647d` plus `/tmp/abl-before-resource-baseline.patch`.
- SHA-256: `335fde6a1f1c348cff5e0b26763438642f247e1b9f31ee625e3336d0a3daed7e`.
- Full before log: `/tmp/abl-timeout-full-py314.log`.
- A second full run with `/tmp/abl_task_probe.py` reproduced the failure.
  Log: `/tmp/abl-timeout-full-py314-probe-final.log`.
- The extra task was `httpx2.AsyncClient.aclose()`, scheduled while the cleanup
  probe collected garbage from earlier SDK tests. It disappeared after one
  event-loop turn. No batch worker or store task remained in that diagnostic.
- An initial diagnostic run was interrupted to correct the probe's access
  to the harness's `Check` objects; it is not counted as validation.

`tests/test_scale_soak.py::isolate_harness_test` now collects old
garbage and gives scheduled finalizers one loop turn before each test. It does
not wait after a scenario or modify harness thresholds. The new
`test_cleanup_task_check_rejects_new_live_task` creates a task after its baseline
and verifies that the unchanged task-leak assertion still fails. This guard is
current-behavior evidence, not a separate historical regression.

The same uninstrumented full-suite command is the before/after check for this
test-isolation fix:

```sh
uv run --no-sync pytest tests/ -q --tb=short
```

The first uninstrumented after-run passed: **1,988 passed, 19 deselected,
8 warnings** (`/tmp/abl-final-full-py314.log`). Its Python 3.13 `make ci`
counterpart also passed (`/tmp/abl-final-ci.log`).

Running the focused Python 3.10 selection with scale tests **first** additionally
exposed pre-existing logging contamination: `run_config()` sets the library
logger to `CRITICAL`, hiding the later audit tests' expected warnings. The
terminal-result assertions passed; all 24 failures were logging assertions.
The fixture now restores the original logger level after each harness test.
This changes test isolation, not CLI behavior or audit-error handling.

- Pre-fix snapshot: `997647d` plus `/tmp/abl-before-logger-isolation.patch`.
- SHA-256: `e7935c2e2eded8565af41560dbae1d726bf29aa6f806a18f28ed6b640b50b541`.
- Before log `/tmp/abl-final-py310.log`: **24 failed, 175 passed, 1 warning**.
- After log `/tmp/abl-merge-py310.log`: **199 passed, 1 warning** on Python 3.10.19.
- Same selection for the after-run:

```sh
uv run --no-sync pytest tests/test_scale_soak.py \
  tests/test_middleware_artifacts.py tests/test_guardrails.py -q --tb=short
```

## Final validation of the combined second follow-up

- Python 3.13.7: `make format` and `make ci`: **1,988 passed, 19 deselected,
  47 warnings**. Ruff, mypy, ty, coverage/scale-soak, and Markdown lint passed.
  Log: `/tmp/abl-merge-ci.log`.
- Python 3.14.2: full suite, no diagnostic plugin: **1,988 passed,
  19 deselected, 8 warnings**. Log: `/tmp/abl-merge-py314.log`.
- Python 3.10.19: the reordered three-module selection above: **199 passed,
  1 warning**. This is focused validation, not another full Python 3.10 run.
- `npx markdownlint-cli2 reports/cancellation_ci_handoff.md` and
  `git diff --check`: passed.

Final changed-test SHA-256 values:

- `tests/test_middleware_artifacts.py`:
  `e3c648a922816df58fa2509437e191ad695a2243ff5abe63eb1bd4f657a3ce6f`.
- `tests/test_scale_soak.py`:
  `a353364d14700b774ac7896b2c0d227161bb59f61690fa9761a0b2c635324166`.
