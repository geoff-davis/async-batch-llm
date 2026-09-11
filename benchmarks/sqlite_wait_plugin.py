"""Opt-in pytest plugin: run existing lifecycle tests with a waiting alternative."""

import os

import pytest

import async_batch_llm.sqlite_artifacts as storage
from benchmarks.sqlite_wait_evidence import VARIANTS


@pytest.fixture(autouse=True)
def sqlite_wait_variant(monkeypatch):
    monkeypatch.setattr(
        storage, "_await_without_cancelling", VARIANTS[os.environ["ABL_WAIT_VARIANT"]]
    )
