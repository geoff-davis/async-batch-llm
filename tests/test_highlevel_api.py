"""Tests for the high-level convenience API (process_prompts / process_stream)."""

import asyncio

import pytest

from async_batch_llm import ProcessorConfig, process_prompts, process_stream
from async_batch_llm.base import RetryState, TokenUsage, WorkItemResult
from async_batch_llm.highlevel import _normalize_prompts
from async_batch_llm.llm_strategies import LLMCallStrategy


class _UpperStrategy(LLMCallStrategy[str]):
    """Echo the prompt back upper-cased; optional per-prompt delay."""

    def __init__(self, delays: dict[str, float] | None = None) -> None:
        self.delays = delays or {}

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        if prompt in self.delays:
            await asyncio.sleep(self.delays[prompt])
        return prompt.upper(), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None


# ── _normalize_prompts ──────────────────────────────────────────────────────


def test_normalize_bare_strings_auto_ids():
    assert _normalize_prompts(["a", "b"]) == [("item_0", "a"), ("item_1", "b")]


def test_normalize_id_prompt_pairs():
    assert _normalize_prompts([("x", "a"), ("y", "b")]) == [("x", "a"), ("y", "b")]


def test_normalize_rejects_bad_shapes():
    with pytest.raises(TypeError, match="strings or"):
        _normalize_prompts([123])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        _normalize_prompts([("only-one",)])  # type: ignore[list-item]


# ── process_prompts ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_prompts_bare_strings():
    result = await process_prompts(
        _UpperStrategy(), ["hello", "world"], config=ProcessorConfig(max_workers=2)
    )
    assert result.total_items == 2
    assert result.succeeded == 2
    by_id = result.by_id()
    assert by_id["item_0"].output == "HELLO"
    assert by_id["item_1"].output == "WORLD"


@pytest.mark.asyncio
async def test_process_prompts_with_explicit_ids():
    result = await process_prompts(
        _UpperStrategy(), [("greet", "hi"), ("name", "bob")], config=ProcessorConfig(max_workers=2)
    )
    assert set(result.by_id()) == {"greet", "name"}
    assert result.by_id()["name"].output == "BOB"


@pytest.mark.asyncio
async def test_process_prompts_default_config_and_kwargs_forwarded():
    seen: list[str] = []

    async def post(result: WorkItemResult) -> None:
        seen.append(result.item_id)

    # No config (defaults), and a forwarded post_processor kwarg.
    result = await process_prompts(_UpperStrategy(), ["a"], post_processor=post)
    assert result.succeeded == 1
    assert seen == ["item_0"]


# ── process_stream ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_stream_yields_all_results():
    seen = []
    async for result in process_stream(
        _UpperStrategy(), ["a", "b", "c"], config=ProcessorConfig(max_workers=3)
    ):
        seen.append(result.item_id)
    assert set(seen) == {"item_0", "item_1", "item_2"}
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_process_stream_yields_in_completion_order():
    # item_0 is slow; item_1 fast — so item_1 must stream out first.
    strategy = _UpperStrategy(delays={"slow": 0.1})
    order = []
    async for result in process_stream(
        strategy, [("a", "slow"), ("b", "fast")], config=ProcessorConfig(max_workers=2)
    ):
        order.append(result.item_id)
    assert order[0] == "b"
    assert set(order) == {"a", "b"}


@pytest.mark.asyncio
async def test_process_stream_chains_user_post_processor():
    chained: list[str] = []

    async def post(result: WorkItemResult) -> None:
        chained.append(result.item_id)

    streamed = []
    async for result in process_stream(
        _UpperStrategy(), ["a", "b"], config=ProcessorConfig(max_workers=2), post_processor=post
    ):
        streamed.append(result.item_id)

    assert set(chained) == {"item_0", "item_1"}
    assert set(streamed) == {"item_0", "item_1"}


@pytest.mark.asyncio
async def test_process_stream_propagates_processing_errors():
    # A bounded queue smaller than the batch makes add_work() raise inside the
    # driver; the stream must surface it rather than hang.
    config = ProcessorConfig(max_workers=1, max_queue_size=1)
    with pytest.raises(ValueError, match="queue is full"):
        async for _ in process_stream(_UpperStrategy(), ["a", "b", "c"], config=config):
            pass
