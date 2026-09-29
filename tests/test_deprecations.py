"""v0.27 deprecations: each warns once, from the caller's code, removed in 1.0."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import warnings

import pytest

import async_batch_llm
from async_batch_llm import LLMCallPool, LLMCallStrategy, process_prompts

TOKENS = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


def _record(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [w for w in caught if "LLMGateway" in str(w.message)]


def test_llm_gateway_root_attribute_warns_from_caller():
    value, caught = _record(lambda: async_batch_llm.LLMGateway)
    assert value is LLMCallPool
    assert [w.category for w in caught] == [DeprecationWarning]
    assert "removed in 1.0" in str(caught[0].message)
    assert caught[0].filename == __file__


def test_llm_gateway_from_imports_warn_once_from_caller():
    def import_root():
        from async_batch_llm import LLMGateway

        return LLMGateway

    def import_module():
        from async_batch_llm.gateway import LLMGateway

        return LLMGateway

    for importer in (import_root, import_module):
        value, caught = _record(importer)
        assert value is LLMCallPool
        # The root import's internal importlib probe must not add a second warning.
        assert len(caught) == 1
        assert caught[0].filename == __file__


def test_plain_import_is_silent_and_star_import_keeps_the_alias():
    # A fresh interpreter, so the package import itself is observed. Wildcard
    # users keep LLMGateway through 0.27 (removed in 1.0), with one warning per
    # star import at their own line.
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import async_batch_llm\n"
        "    ours = ('LLMGateway', 'BatchProcessor', 'ProcessingStats',\n"
        "            'grounding_metadata_extractor')\n"
        "    plain = [w for w in caught if str(w.message).startswith(ours)]\n"
        "    from async_batch_llm import *\n"
        "    from async_batch_llm.gateway import *\n"
        "assert plain == [], plain\n"
        "star = [w for w in caught if 'LLMGateway' in str(w.message)]\n"
        "assert [(w.category, w.filename) for w in star] == [(DeprecationWarning, '<string>')] * 2\n"
        "assert LLMGateway is LLMCallPool\n"
        "assert 'LLMGateway' in async_batch_llm.__all__\n"
        "names = ('BatchProcessor', 'ProcessingStats', 'grounding_metadata_extractor')\n"
        "for name in names:\n"
        "    assert name in dir() and name in async_batch_llm.__all__\n"
        "    hits = [w for w in caught if str(w.message).startswith(name + ' is deprecated')]\n"
        "    assert [(w.category, w.filename) for w in hits] == [(DeprecationWarning, '<string>')]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_attributes_still_raise():
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        async_batch_llm.Nope  # noqa: B018
    gateway = importlib.import_module("async_batch_llm.gateway")
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        gateway.Nope  # noqa: B018


class Legacy(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        return prompt.upper(), dict(TOKENS)


class Current(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        return prompt.upper(), dict(TOKENS), None


@pytest.mark.asyncio
async def test_two_tuple_return_warns_at_the_strategy_execute():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = await process_prompts(Legacy(), ["a"])
    assert batch.results[0].success and batch.results[0].output == "A"
    tuple_warnings = [w for w in caught if "2-tuple" in str(w.message)]
    assert [w.category for w in tuple_warnings] == [DeprecationWarning]
    warning = tuple_warnings[0]
    assert "Legacy.execute()" in str(warning.message)
    assert "removed in 1.0" in str(warning.message)
    assert warning.filename == __file__
    assert warning.lineno == inspect.getsourcelines(Legacy.execute)[1]


@pytest.mark.asyncio
async def test_three_tuple_return_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = await process_prompts(Current(), ["a"])
    assert batch.results[0].success
    assert not [w for w in caught if "2-tuple" in str(w.message)]


@pytest.mark.parametrize(
    ("name", "module", "attribute"),
    [
        ("BatchProcessor", "async_batch_llm.base", "BatchProcessor"),
        ("ProcessingStats", "async_batch_llm.base", "ProcessingStats"),
        ("grounding_metadata_extractor", "async_batch_llm.models", "grounding_metadata_extractor"),
    ],
)
def test_names_leaving_the_public_api_warn_from_caller(name, module, attribute):
    expected = getattr(importlib.import_module(module), attribute)

    def import_root():
        namespace: dict[str, object] = {}
        exec(f"from async_batch_llm import {name}", namespace)  # noqa: S102
        return namespace[name]

    for access in (lambda: getattr(async_batch_llm, name), import_root):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = access()
        assert value is expected
        hits = [w for w in caught if str(w.message).startswith(f"{name} is deprecated")]
        assert [w.category for w in hits] == [DeprecationWarning]
        assert "public API in 1.0" in str(hits[0].message)


def test_submodule_access_stays_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from async_batch_llm.base import BatchProcessor, ProcessingStats  # noqa: F401
        from async_batch_llm.models import grounding_metadata_extractor  # noqa: F401
    assert not [w for w in caught if "deprecated" in str(w.message)]
