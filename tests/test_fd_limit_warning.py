"""Tests for the open-file-limit (RLIMIT_NOFILE) startup warning.

`ParallelBatchProcessor` warns when `max_workers` is close to the process's
soft open-file limit, since each in-flight request typically holds a socket
(an fd) and the run would otherwise fail with `OSError: [Errno 24] Too many
open files`. See `_warn_if_fd_limit_low`.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from async_batch_llm import ParallelBatchProcessor, ProcessorConfig
from async_batch_llm.parallel import _warn_if_fd_limit_low

resource = pytest.importorskip("resource", reason="RLIMIT_NOFILE is Unix-only")


class TestWarnIfFdLimitLow:
    def test_warns_when_workers_near_soft_limit(self):
        # macOS-like default soft limit; high worker count → warn.
        with patch("resource.getrlimit", return_value=(256, 1024)):
            with pytest.warns(UserWarning, match="Too many open files"):
                _warn_if_fd_limit_low(250)

    def test_warning_points_to_docs_and_remedies(self):
        with patch("resource.getrlimit", return_value=(256, 1024)):
            with pytest.warns(UserWarning) as record:
                _warn_if_fd_limit_low(250)
        msg = str(record[0].message)
        assert "ulimit -n" in msg
        assert "geoff-davis.github.io/async-batch-llm" in msg

    def test_no_warn_with_ample_headroom(self):
        # Plenty of fds above the worker count → silent.
        with patch("resource.getrlimit", return_value=(4096, 4096)):
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # any warning fails the test
                _warn_if_fd_limit_low(40)

    def test_no_warn_when_limit_unlimited(self):
        inf = resource.RLIM_INFINITY
        with patch("resource.getrlimit", return_value=(inf, inf)):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _warn_if_fd_limit_low(10_000)

    def test_processor_warns_on_construction(self):
        with patch("resource.getrlimit", return_value=(256, 1024)):
            with pytest.warns(UserWarning, match="open-file limit"):
                ParallelBatchProcessor(config=ProcessorConfig(max_workers=250))

    def test_processor_quiet_at_default_workers(self):
        # Default max_workers (5) is nowhere near even a tiny limit → no warn.
        with patch("resource.getrlimit", return_value=(256, 1024)):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ParallelBatchProcessor(config=ProcessorConfig(max_workers=5))
