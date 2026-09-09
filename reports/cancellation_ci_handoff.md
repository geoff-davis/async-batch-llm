# Cancellation CI follow-up

## Target and finding

Base: `7fb74d482377db3cc790fb64556d1a8c46a7ff93`. The review target is the
follow-up commit containing this report. Production code is unchanged.
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
