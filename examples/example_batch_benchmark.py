"""Bulk LLM benchmark: minimize wall time on a big batch job, the right way.

This is the flagship "why async-batch-llm" demo. It runs the GSM8K math
benchmark through several providers and shows three things at once:

1. **Wall-time race** — the same workload run three ways, per provider:
   sequential loop vs naive ``asyncio.gather`` vs ``async-batch-llm``.
   Concurrency is what collapses wall time; the framework also adds the retry /
   backoff / rate-limit handling a bare ``gather`` lacks. Read across rows to
   compare providers, down columns to compare orchestrations.

2. **Provider bake-off** — DeepSeek Flash ``deepseek-v4-flash`` vs Gemini 3.1
   Flash-Lite ``gemini-3.1-flash-lite`` vs Gemini 2.5 Flash-Lite
   ``gemini-2.5-flash-lite`` on accuracy, tokens (input / cached / output), and
   estimated cost. Same framework, swap one strategy per provider.

3. **No-thinking → thinking escalation** — attempts 1-2 run the cheap
   non-thinking mode; attempt 3 escalates to thinking. Escalation is
   *validation-gated*: an answer with no parseable ``#### <number>`` raises,
   which is what triggers the retry (and thus the escalation).

I/O uses the stdlib ``gzip`` module on both ends: the .jsonl.gz benchmark is
read once up front, and results stream out to a .jsonl.gz file. The
post-processors run concurrently, but a synchronous ``gzip.write()`` with no
``await`` in between is atomic with respect to the event loop — the loop can't
switch tasks mid-write — so concurrent post-processors share one open file with
no lock and no corruption (``StreamingGzipWriter``). Output is in completion
order, not input order — each record carries its ``item_id`` so the original
order is recoverable downstream.

A note: at this dataset's size (~240 KB compressed) gzip I/O doesn't move wall
time — the speedup is entirely concurrency. The blocking write is deliberate:
at these record sizes, pushing writes to an async queue or a thread only adds
queue-hop / executor overhead (a standalone micro-benchmark,
``examples/bench_gzip_write.py``, shows the async path running several times
slower for tiny records). For multi-hundred-MB outputs — where a synchronous
compress would stall the loop and every concurrent worker with it — an offloaded
writer would pay off instead.

A small ChatGPT "fallback grader" batch is the LLM-as-judge showcase: GSM8K
is exact-match scorable for free, so the judge only sees the handful of
outputs whose answer we couldn't parse, and decides whether they match gold.

## Installation

```bash
pip install 'async-batch-llm[deepseek,gemini,openai]'
python examples/download_gsm8k.py
```

## Setup (set the keys for whichever contestants you want)

```bash
export DEEPSEEK_API_KEY=sk-...      # DeepSeek contestant + the wall-time race
export GOOGLE_API_KEY=...           # Gemini contestant (GEMINI_API_KEY also works)
export OPENAI_API_KEY=sk-...        # optional: ChatGPT fallback grader
```

For Gemini you can instead use the **Vertex AI** backend with Application
Default Credentials (`gcloud auth application-default login`) — no API key:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
```

Contestants and the judge are skipped automatically when their credentials
are absent, so the demo runs with whatever you have configured.

## Run

```bash
python examples/example_batch_benchmark.py
```
"""

import asyncio
import gzip
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

# ---------------------------------------------------------------------------
# Tunables — small defaults so a full run costs cents and finishes in minutes.
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "gsm8k_test.jsonl.gz"
RESULTS_DIR = Path(__file__).parent / "data" / "benchmark_results"

BAKEOFF_ITEMS = 100  # items each provider answers in the accuracy bake-off
RACE_ITEMS = 30  # items in the wall-time race (the sequential leg is the slow one)
MAX_WORKERS = 40  # high concurrency for the providers
JUDGE_WORKERS = 10  # concurrency for the fallback grader
GATHER_CHUNK_SIZE = 50  # naive baseline: gather this many at a time (barrier per chunk)

# Model ids (as of June 2026 — adjust to taste).
DEEPSEEK_MODEL = "deepseek-v4-flash"
GEMINI_31_MODEL = "gemini-3.1-flash-lite"
GEMINI_25_MODEL = "gemini-2.5-flash-lite"
JUDGE_MODEL = "gpt-5-nano"

# Gemini thinking is set per-call. Gemini 3.1 uses a thinking *level*
# (minimal/low/medium/high); Gemini 2.5 uses a numeric thinking *budget*
# (0 disables, positive enables). Each contestant's fast pass minimizes thinking
# and its escalation maximizes it.
#
# CAVEAT — the two Gemini fast passes are NOT a matched "no thinking" setup:
# 2.5's budget=0 turns thinking fully OFF, but 3.1's level enum has no "off" —
# "minimal" is the floor and still does a little thinking. So 3.1 gets a small
# thinking edge that 2.5 doesn't; don't read the 3.1-vs-2.5 gap as pure model
# quality. (2.5 Flash-Lite defaults to thinking off, so budget=0 matches its
# default; 3.1 Flash-Lite ships with thinking on by default.)
GEMINI_31_FAST_CONFIG = {"thinking_config": {"thinking_level": "minimal"}}
GEMINI_31_THINK_CONFIG = {"thinking_config": {"thinking_level": "high"}}
GEMINI_25_FAST_CONFIG = {"thinking_config": {"thinking_budget": 0}}
GEMINI_25_THINK_CONFIG = {"thinking_config": {"thinking_budget": 2048}}


@dataclass(frozen=True)
class Pricing:
    """USD per 1M tokens. ``cached_input`` is the price for a cache *hit*."""

    input: float  # cache-miss input
    cached_input: float  # cache-hit input
    output: float

    @property
    def cached_rate(self) -> float:
        """Cache-hit price as a fraction of normal input price.

        This is exactly what ``BatchResult.effective_input_tokens(rate)`` wants.
        """
        return self.cached_input / self.input if self.input else 0.0


# Verify against each provider's current pricing page before quoting numbers.
PRICING = {
    DEEPSEEK_MODEL: Pricing(input=0.14, cached_input=0.0028, output=0.28),
    GEMINI_31_MODEL: Pricing(input=0.25, cached_input=0.025, output=1.50),
    GEMINI_25_MODEL: Pricing(input=0.10, cached_input=0.010, output=0.40),
    JUDGE_MODEL: Pricing(input=0.05, cached_input=0.005, output=0.50),
}

PROMPT_TEMPLATE = (
    "Solve this grade-school math problem. Think briefly step by step, then on "
    "the FINAL line output the answer exactly as `#### <number>` and nothing "
    "after it.\n\nProblem: {question}"
)

JUDGE_TEMPLATE = (
    "A model answered a math problem. The correct final answer is: {gold}\n\n"
    "The model's full response was:\n---\n{response}\n---\n\n"
    "Did the model arrive at the correct final answer (numerically equal to "
    "{gold})? Reply with exactly YES or NO."
)


# ---------------------------------------------------------------------------
# Optional deps — import after the env check so missing keys give a clean
# message instead of an ImportError (examples are exempt from ruff E402).
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Vertex AI backend (Application Default Credentials) as an alternative to an
# API key for Gemini. google-genai also reads GOOGLE_CLOUD_PROJECT / _LOCATION.
GEMINI_USE_VERTEX = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------
@dataclass
class Item:
    """One benchmark question with its gold answer."""

    id: str
    question: str
    gold: float | None


@dataclass
class GSM8KAnswer:
    """Strategy output: the parsed numeric answer (None if unparseable) plus
    the raw model text (kept so the fallback judge can read it)."""

    value: float | None
    raw: str


# ---------------------------------------------------------------------------
# Answer parsing / scoring
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _to_number(text: str) -> float:
    cleaned = text.replace(",", "").rstrip(".")
    return float(cleaned)


def extract_answer(text: str | None) -> float | None:
    """Pull the number after the ``#### `` marker, or None if absent.

    Requiring the marker is the validation gate: a rambling answer with no
    ``#### <number>`` line returns None, which makes ``execute()`` raise and
    escalate to the thinking pass.
    """
    if not text:
        return None
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return _to_number(match.group(1))
    except ValueError:
        return None


def numbers_equal(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


# ---------------------------------------------------------------------------
# Gzip I/O (stdlib, blocking writes)
# ---------------------------------------------------------------------------
async def load_items(path: Path, limit: int) -> list[Item]:
    """Read up to ``limit`` items from a .jsonl.gz file (plain blocking gzip).

    This runs once, before any timer starts, so a brief synchronous decompress
    of the (~240 KB) file never touches the measured wall time.
    """
    items: list[Item] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            items.append(
                Item(
                    id=f"q{len(items)}",
                    question=record["question"],
                    gold=extract_answer(record["answer"]),
                )
            )
            if len(items) >= limit:
                break
    return items


class StreamingGzipWriter:
    """.jsonl.gz writer for concurrent producers, using blocking gzip writes.

    The framework's post-processors run concurrently, but a synchronous
    ``gzip.write()`` with no ``await`` in between is atomic with respect to the
    event loop — the loop can't switch tasks mid-write — so concurrent producers
    can share one open file with no lock and no corruption. We deliberately do
    *not* push writes onto an async queue or a thread: at these record sizes the
    queue-hop / executor overhead is pure cost, and the blocking write is both
    simpler and faster (see ``examples/bench_gzip_write.py``). For
    multi-hundred-MB outputs, where the compress step would stall the loop, an
    offloaded writer would pay off instead.

    Records are written in *completion order* (whatever order items finish under
    concurrency), not input order. Each record carries its ``item_id``, so the
    original order is trivially recoverable downstream (sort by id) — for a
    benchmark dump that's all you need; we don't pay to preserve order here.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: Any = None

    async def __aenter__(self) -> "StreamingGzipWriter":
        self._fh = gzip.open(self.path, "wt")
        return self

    async def write(self, record: dict[str, Any]) -> None:
        """Write one record; call this from a post_processor.

        The synchronous ``write`` is atomic on the event loop (no ``await``
        mid-write), so concurrent callers can't interleave and corrupt the
        stream — no lock needed.
        """
        self._fh.write(json.dumps(record) + "\n")

    async def __aexit__(self, *exc: Any) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Strategy: no-thinking → thinking escalation, validation-gated
# ---------------------------------------------------------------------------
from async_batch_llm import ErrorClassifier, ErrorInfo  # noqa: E402
from async_batch_llm.core.protocols import ManagedLLMModel  # noqa: E402
from async_batch_llm.llm_strategies import LLMCallStrategy  # noqa: E402


class AnswerParseError(Exception):
    """Raised when a response has no parseable ``#### <number>`` answer.

    A dedicated type (not ``ValueError``) so the classifier below can mark it
    retryable without affecting how real bugs are classified.
    """


class EscalationErrorClassifier(ErrorClassifier):
    """Make ``AnswerParseError`` retryable; delegate everything else.

    This is the key wiring for the escalation: provider classifiers
    (rightly) treat a generic ``ValueError`` as a non-retryable logic bug, so a
    parse failure would otherwise fail the item on attempt 1 — and never reach
    the thinking pass. Wrapping the provider classifier lets *our* validation
    failure trigger a retry while real API errors are still classified by the
    provider's own rules.
    """

    def __init__(self, base: ErrorClassifier) -> None:
        self.base = base

    def classify(self, exception: Exception) -> ErrorInfo:
        if isinstance(exception, AnswerParseError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="answer_unparsed",
            )
        return self.base.classify(exception)


@dataclass
class ModelCall:
    """A model plus the per-call kwargs that select its thinking mode.

    Hides provider differences from the strategy: for DeepSeek the two modes
    are two model objects (thinking baked into ``extra_body``); for Gemini it's
    one model with two different per-call ``config`` dicts.
    """

    model: Any
    label: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    async def prepare(self) -> None:
        if isinstance(self.model, ManagedLLMModel):
            await self.model.prepare()

    async def cleanup(self) -> None:
        if isinstance(self.model, ManagedLLMModel):
            await self.model.cleanup()

    async def generate(self, prompt: str) -> Any:
        return await self.model.generate(prompt, **self.kwargs)


class EscalatingStrategy(LLMCallStrategy[GSM8KAnswer]):
    """Attempts ``< escalate_at`` use the fast (non-thinking) call; attempt
    ``>= escalate_at`` uses the thinking call.

    On a non-final attempt, an unparseable answer raises (with the spent tokens
    attached so they're still accounted for) — that retry is what escalates. On
    the final attempt we return the raw text with ``value=None`` so the
    fallback judge can take a look instead of the item just failing.
    """

    def __init__(
        self,
        fast: ModelCall,
        thinking: ModelCall,
        *,
        escalate_at: int = 3,
        max_attempts: int = 3,
    ) -> None:
        self.fast = fast
        self.thinking = thinking
        self.escalate_at = escalate_at
        self.max_attempts = max_attempts

    async def prepare(self) -> None:
        await self.fast.prepare()
        await self.thinking.prepare()

    async def cleanup(self) -> None:
        await self.fast.cleanup()
        await self.thinking.cleanup()

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: Any = None
    ) -> tuple[GSM8KAnswer, dict[str, Any], dict[str, Any] | None]:
        call = self.thinking if attempt >= self.escalate_at else self.fast
        response = await call.generate(prompt)
        answer = extract_answer(response.text)

        metadata = dict(response.metadata or {})
        metadata["model_label"] = call.label
        metadata["thinking"] = call is self.thinking

        if answer is None and attempt < self.max_attempts:
            # Validation gate: trigger a retry (→ escalation) but keep the
            # already-billed tokens on the exception for failure-path accounting.
            # AnswerParseError is marked retryable by EscalationErrorClassifier.
            err = AnswerParseError(f"no '#### <number>' answer (attempt {attempt}, {call.label})")
            err.__dict__["_failed_token_usage"] = response.token_usage
            raise err

        return GSM8KAnswer(value=answer, raw=response.text), response.token_usage, metadata


# ---------------------------------------------------------------------------
# Contestant wiring
# ---------------------------------------------------------------------------
@dataclass
class Contestant:
    name: str
    model_id: str
    strategy: EscalatingStrategy
    fast_call: ModelCall
    error_classifier: Any
    pricing: Pricing


def make_deepseek() -> Contestant:
    from async_batch_llm import DeepSeekModel, OpenAIErrorClassifier

    fast = ModelCall(
        DeepSeekModel.from_api_key(DEEPSEEK_MODEL, thinking=False, max_connections=MAX_WORKERS),
        label=f"{DEEPSEEK_MODEL}:no-think",
    )
    thinking = ModelCall(
        DeepSeekModel.from_api_key(DEEPSEEK_MODEL, thinking=True, max_connections=MAX_WORKERS),
        label=f"{DEEPSEEK_MODEL}:think",
    )
    return Contestant(
        name="deepseek-flash",
        model_id=DEEPSEEK_MODEL,
        strategy=EscalatingStrategy(fast, thinking),
        fast_call=fast,
        # DeepSeek is OpenAI-compatible; wrap so parse failures escalate.
        error_classifier=EscalationErrorClassifier(OpenAIErrorClassifier()),
        pricing=PRICING[DEEPSEEK_MODEL],
    )


def _gemini_client() -> Any:
    from google import genai

    if GEMINI_USE_VERTEX:
        # Vertex AI backend: ADC (`gcloud auth application-default login`) plus
        # GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION, read from the environment.
        return genai.Client(vertexai=True)
    # Gemini Developer API (AI Studio): API key.
    return genai.Client(api_key=GOOGLE_API_KEY)


def _make_gemini(
    name: str, model_id: str, fast_config: dict[str, Any], think_config: dict[str, Any]
) -> Contestant:
    from async_batch_llm import GeminiErrorClassifier, GeminiModel

    model = GeminiModel(model_id, _gemini_client())  # one model, two thinking configs
    fast = ModelCall(model, label=f"{model_id}:fast", kwargs={"config": fast_config})
    thinking = ModelCall(model, label=f"{model_id}:think", kwargs={"config": think_config})
    return Contestant(
        name=name,
        model_id=model_id,
        strategy=EscalatingStrategy(fast, thinking),
        fast_call=fast,
        error_classifier=EscalationErrorClassifier(GeminiErrorClassifier()),
        pricing=PRICING[model_id],
    )


def make_gemini_31() -> Contestant:
    return _make_gemini("gemini-3.1", GEMINI_31_MODEL, GEMINI_31_FAST_CONFIG, GEMINI_31_THINK_CONFIG)


def make_gemini_25() -> Contestant:
    return _make_gemini("gemini-2.5", GEMINI_25_MODEL, GEMINI_25_FAST_CONFIG, GEMINI_25_THINK_CONFIG)


# ---------------------------------------------------------------------------
# Cost helpers (showcasing the package's token accounting)
# ---------------------------------------------------------------------------
def estimate_cost(batch_result: Any, pricing: Pricing) -> float:
    """Exact $ from the package's aggregated token counts."""
    cached = batch_result.total_cached_tokens
    missed = batch_result.total_input_tokens - cached
    input_cost = (missed / 1e6) * pricing.input + (cached / 1e6) * pricing.cached_input
    output_cost = (batch_result.total_output_tokens / 1e6) * pricing.output
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# The bake-off: full batch per contestant, results streamed to .jsonl.gz
# ---------------------------------------------------------------------------
async def run_contestant(
    contestant: Contestant, items: list[Item]
) -> tuple[Any, float]:
    from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
    from async_batch_llm.core import RetryConfig

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{contestant.name}_results.jsonl.gz"

    config = ProcessorConfig(
        max_workers=MAX_WORKERS,
        timeout_per_item=120.0,
        retry=RetryConfig(max_attempts=3, initial_wait=1.0),
    )

    start = perf_counter()
    # Blocking gzip writer (outer): each concurrent post-processor does a
    # synchronous, atomic write to the .jsonl.gz file — no lock, no corruption.
    async with StreamingGzipWriter(out_path) as writer:

        async def write_result(result: Any) -> None:
            # The post_processor runs for every completed item, including ones
            # that exhausted their retries — so result.output may be None.
            output = result.output if result.success else None
            await writer.write(
                {
                    "item_id": result.item_id,
                    "success": result.success,
                    "answer": output.value if output is not None else None,
                    "gold": result.context.get("gold") if result.context else None,
                    "thinking": (result.metadata or {}).get("thinking"),
                    "model": (result.metadata or {}).get("model_label"),
                    "error": result.error if not result.success else None,
                }
            )

        async with ParallelBatchProcessor(
            config=config,
            error_classifier=contestant.error_classifier,
            post_processor=write_result,
        ) as processor:
            for item in items:
                await processor.add_work(
                    LLMWorkItem(
                        item_id=item.id,
                        strategy=contestant.strategy,
                        prompt=build_prompt(item.question),
                        context={"gold": item.gold},
                    )
                )
            batch_result = await processor.process_all()
    wall = perf_counter() - start

    return batch_result, wall


@dataclass
class Scorecard:
    name: str
    model_id: str
    total: int
    exact_correct: int
    judge_correct: int
    errors: int
    ambiguous: int
    wall: float
    batch_result: Any

    @property
    def accuracy(self) -> float:
        return (self.exact_correct + self.judge_correct) / self.total if self.total else 0.0


def score_batch(name: str, model_id: str, batch_result: Any, wall: float) -> tuple[Scorecard, list[tuple]]:
    """Grade by exact match; return the ambiguous (unparseable) items for the judge."""
    exact_correct = 0
    errors = 0
    ambiguous: list[tuple[str, str, float | None]] = []

    for r in batch_result.results:
        gold = r.context.get("gold") if r.context else None
        if not r.success:
            errors += 1
            continue
        answer = r.output.value
        if answer is None:
            ambiguous.append((r.item_id, r.output.raw, gold))
        elif numbers_equal(answer, gold):
            exact_correct += 1

    card = Scorecard(
        name=name,
        model_id=model_id,
        total=len(batch_result.results),
        exact_correct=exact_correct,
        judge_correct=0,  # filled in after judging
        errors=errors,
        ambiguous=len(ambiguous),
        wall=wall,
        batch_result=batch_result,
    )
    return card, ambiguous


# ---------------------------------------------------------------------------
# Fallback grader (LLM-as-judge) — only sees the unparseable outputs
# ---------------------------------------------------------------------------
async def grade_ambiguous(ambiguous: list[tuple[str, str, float | None]]) -> tuple[dict[str, bool], Any]:
    from async_batch_llm import (
        LLMWorkItem,
        OpenAIErrorClassifier,
        OpenAIModel,
        OpenAIStrategy,
        ParallelBatchProcessor,
        ProcessorConfig,
    )

    model = OpenAIModel.from_api_key(JUDGE_MODEL, max_connections=JUDGE_WORKERS)
    strategy = OpenAIStrategy(model, response_parser=lambda resp: "YES" in resp.text.upper())
    config = ProcessorConfig(max_workers=JUDGE_WORKERS, timeout_per_item=120.0)

    async with ParallelBatchProcessor(
        config=config, error_classifier=OpenAIErrorClassifier()
    ) as processor:
        for item_id, raw, gold in ambiguous:
            await processor.add_work(
                LLMWorkItem(
                    item_id=item_id,
                    strategy=strategy,
                    prompt=JUDGE_TEMPLATE.format(gold=gold, response=raw),
                )
            )
        batch_result = await processor.process_all()

    verdicts = {r.item_id: bool(r.output) for r in batch_result.results if r.success}
    return verdicts, batch_result


# ---------------------------------------------------------------------------
# The wall-time race: same workload, three orchestrations
# ---------------------------------------------------------------------------
async def race_sequential(items: list[Item], call: ModelCall) -> tuple[float, int]:
    """Naive baseline: one call at a time. This is the slow leg."""
    start = perf_counter()
    ok = 0
    for item in items:
        try:
            await call.generate(build_prompt(item.question))
            ok += 1
        except Exception:  # noqa: BLE001 - baseline has no retry; a failure is just lost
            pass
    return perf_counter() - start, ok


async def race_naive_gather(items: list[Item], call: ModelCall) -> tuple[float, int]:
    """Realistic naive baseline: gather a fixed-size chunk at a time to bound
    concurrency. Each chunk is a barrier — the next can't start until the
    *slowest* call in the current one returns — and there's no retry/backoff, so
    a failed call is just lost. At demo sizes the batch fits in one chunk (so it
    matches firing everything at once); the per-chunk straggler cost only shows
    up once items far exceed GATHER_CHUNK_SIZE."""
    start = perf_counter()
    ok = 0
    for i in range(0, len(items), GATHER_CHUNK_SIZE):
        chunk = items[i : i + GATHER_CHUNK_SIZE]
        results = await asyncio.gather(
            *(call.generate(build_prompt(item.question)) for item in chunk),
            return_exceptions=True,
        )
        ok += sum(1 for r in results if not isinstance(r, Exception))
    return perf_counter() - start, ok


async def race_processor(items: list[Item], contestant: Contestant) -> tuple[float, int]:
    """async-batch-llm: concurrent AND robust (retry, backoff, escalation).

    This leg times orchestration only — no result writing — so the race stays a
    clean apples-to-apples comparison of how the calls are driven. (Disk I/O is
    negligible at this scale anyway; see examples/bench_gzip_write.py.)"""
    from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
    from async_batch_llm.core import RetryConfig

    config = ProcessorConfig(
        max_workers=MAX_WORKERS,
        timeout_per_item=120.0,
        retry=RetryConfig(max_attempts=3, initial_wait=1.0),
    )
    start = perf_counter()
    async with ParallelBatchProcessor(
        config=config, error_classifier=contestant.error_classifier
    ) as processor:
        for item in items:
            await processor.add_work(
                LLMWorkItem(
                    item_id=item.id,
                    strategy=contestant.strategy,
                    prompt=build_prompt(item.question),
                    context={"gold": item.gold},
                )
            )
        batch_result = await processor.process_all()
    return perf_counter() - start, batch_result.succeeded


@dataclass
class RaceResult:
    name: str
    seq: tuple[float, int]  # (wall, ok)
    gather: tuple[float, int]
    proc: tuple[float, int]  # async-batch-llm


async def run_race_for(build: Callable[[], Contestant], items: list[Item]) -> RaceResult:
    """Run all three orchestrations for one provider on the same items."""
    contestant = build()
    print(f"  racing {contestant.name}...")
    seq = await race_sequential(items, contestant.fast_call)
    gather = await race_naive_gather(items, contestant.fast_call)
    proc = await race_processor(items, contestant)  # this run's __aexit__ cleans it up
    return RaceResult(contestant.name, seq, gather, proc)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_race(rows: list[RaceResult], n: int) -> None:
    print("\n" + "=" * 76)
    print(f"WALL-TIME RACE  ({n} items per provider, three orchestrations)")
    print("=" * 76)
    print(
        f"{'Provider':<16}{'Sequential':>13}{'gather':>11}{'async-batch':>14}{'Speedup':>11}{'OK':>7}"
    )
    print(f"{'':<16}{'(s)':>13}{'(s)':>11}{'(s)':>14}{'seq->abl':>11}")
    print("-" * 76)
    for r in rows:
        seq_wall, proc_wall = r.seq[0], r.proc[0]
        speedup = f"{seq_wall / proc_wall:.1f}x" if proc_wall > 0 else "-"
        print(
            f"{r.name:<16}{seq_wall:>13.1f}{r.gather[0]:>11.1f}"
            f"{proc_wall:>14.1f}{speedup:>11}{r.proc[1]:>7}"
        )
    print(
        "\nRead across rows to compare providers; down columns to compare "
        "orchestrations.\nConcurrency collapses wall time (sequential -> gather "
        "-> async-batch-llm); the\nframework leg matches a bare gather for speed "
        "while also surviving transient\nerrors and rate limits (which gather "
        "would silently drop)."
    )


def print_bakeoff(cards: list[Scorecard], judge_card: Scorecard | None) -> None:
    print("\n" + "=" * 96)
    print(f"PROVIDER BAKE-OFF  ({cards[0].total if cards else 0} items each)")
    print("=" * 96)
    header = (
        f"{'Provider':<22}{'Accuracy':>10}{'Wall(s)':>9}"
        f"{'Input':>11}{'Cached':>10}{'Output':>11}{'Cost($)':>10}"
    )
    print(header)
    print("-" * 96)
    for card in cards:
        br = card.batch_result
        cost = estimate_cost(br, PRICING[card.model_id])
        acc = f"{card.accuracy * 100:.1f}%"
        print(
            f"{card.name + ' (' + card.model_id + ')':<22}"
            f"{acc:>10}{card.wall:>9.1f}"
            f"{br.total_input_tokens:>11,}{br.total_cached_tokens:>10,}"
            f"{br.total_output_tokens:>11,}{cost:>10.4f}"
        )
    if judge_card is not None:
        br = judge_card.batch_result
        cost = estimate_cost(br, PRICING[JUDGE_MODEL])
        print(
            f"{'judge (' + JUDGE_MODEL + ')':<22}"
            f"{'-':>10}{judge_card.wall:>9.1f}"
            f"{br.total_input_tokens:>11,}{br.total_cached_tokens:>10,}"
            f"{br.total_output_tokens:>11,}{cost:>10.4f}"
        )

    print("\nPer-provider detail:")
    for card in cards:
        br = card.batch_result
        billable = br.effective_input_tokens(PRICING[card.model_id].cached_rate)
        cache_pct = (br.total_cached_tokens / br.total_input_tokens * 100) if br.total_input_tokens else 0.0
        print(
            f"  {card.name}: exact={card.exact_correct} judge_rescued={card.judge_correct} "
            f"unparsed={card.ambiguous} errors={card.errors}"
        )
        print(
            f"      billable input tokens (cache-adjusted)={billable:,} "
            f"| cache hit rate={cache_pct:.1f}%"
        )


def write_summary(
    race_rows: list[RaceResult],
    cards: list[Scorecard],
    judge_card: Scorecard | None,
) -> Path:
    """Persist the race + bake-off aggregates to ``benchmark_results/summary.json``.

    The per-item .jsonl.gz files capture answers but not wall time, tokens, or
    cost — so this dumps the same numbers ``print_race``/``print_bakeoff`` show,
    letting a run be cited (e.g. in docs) without re-running it.
    """
    from datetime import datetime, timezone

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _bakeoff_row(card: Scorecard) -> dict[str, Any]:
        br = card.batch_result
        pricing = PRICING[card.model_id]
        cache_pct = (
            br.total_cached_tokens / br.total_input_tokens * 100
            if br.total_input_tokens
            else 0.0
        )
        return {
            "provider": card.name,
            "model_id": card.model_id,
            "accuracy_pct": round(card.accuracy * 100, 1),
            "wall_s": round(card.wall, 2),
            "input_tokens": br.total_input_tokens,
            "cached_tokens": br.total_cached_tokens,
            "output_tokens": br.total_output_tokens,
            "cost_usd": round(estimate_cost(br, pricing), 4),
            "billable_input_tokens": br.effective_input_tokens(pricing.cached_rate),
            "cache_hit_rate_pct": round(cache_pct, 1),
            "exact_correct": card.exact_correct,
            "judge_rescued": card.judge_correct,
            "unparsed": card.ambiguous,
            "errors": card.errors,
        }

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "race_items": RACE_ITEMS,
            "bakeoff_items": BAKEOFF_ITEMS,
            "max_workers": MAX_WORKERS,
            "gather_chunk_size": GATHER_CHUNK_SIZE,
        },
        "wall_time_race": [
            {
                "provider": r.name,
                "sequential_s": round(r.seq[0], 2),
                "gather_s": round(r.gather[0], 2),
                "async_batch_llm_s": round(r.proc[0], 2),
                "speedup_seq_to_abl": (round(r.seq[0] / r.proc[0], 1) if r.proc[0] else None),
                "ok": r.proc[1],
            }
            for r in race_rows
        ],
        "bakeoff": [_bakeoff_row(c) for c in cards],
    }
    if judge_card is not None:
        jbr = judge_card.batch_result
        summary["judge"] = {
            "model_id": judge_card.model_id,
            "wall_s": round(judge_card.wall, 2),
            "input_tokens": jbr.total_input_tokens,
            "cached_tokens": jbr.total_cached_tokens,
            "output_tokens": jbr.total_output_tokens,
            "cost_usd": round(estimate_cost(jbr, PRICING[JUDGE_MODEL]), 4),
            "judged": judge_card.total,
            "judge_correct": judge_card.judge_correct,
        }

    out_path = RESULTS_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def main() -> None:
    if not DATA_PATH.exists():
        print(f"Benchmark data not found at {DATA_PATH}")
        print("Run:  python examples/download_gsm8k.py")
        return

    builders: list[Callable[[], Contestant]] = []
    if DEEPSEEK_API_KEY:
        builders.append(make_deepseek)
    if GOOGLE_API_KEY or GEMINI_USE_VERTEX:
        builders.append(make_gemini_31)
        builders.append(make_gemini_25)

    if not builders:
        print("No provider credentials found. Configure at least one contestant:")
        print("  DeepSeek: export DEEPSEEK_API_KEY=sk-...")
        print("  Gemini:   export GOOGLE_API_KEY=...  (or GOOGLE_GENAI_USE_VERTEXAI=true + ADC)")
        print("Optionally OPENAI_API_KEY for the fallback grader. Then re-run.")
        return

    bakeoff_items = await load_items(DATA_PATH, BAKEOFF_ITEMS)
    race_items = bakeoff_items[:RACE_ITEMS]
    print(f"Loaded {len(bakeoff_items)} GSM8K items from {DATA_PATH.name} (gzip).")

    # Each strategy is consumed by exactly one processor, whose __aexit__ owns
    # its cleanup (it closes the provider's HTTP client). So we build a fresh
    # set of models per use rather than sharing instances across processors.

    # ---- Wall-time race (every provider × three orchestrations) ----
    print(f"\nRunning wall-time race on {RACE_ITEMS} items per provider...")
    print("  (each provider's sequential leg runs one call at a time — give it a minute)")
    race_rows: list[RaceResult] = []
    for build in builders:
        race_rows.append(await run_race_for(build, race_items))
    print_race(race_rows, RACE_ITEMS)

    # ---- Provider bake-off (fresh build per contestant) ----
    cards: list[Scorecard] = []
    all_ambiguous: list[tuple[str, str, float | None]] = []
    ambiguous_owner: dict[str, Scorecard] = {}
    for build in builders:
        contestant = build()
        print(f"\nRunning bake-off batch: {contestant.name} ({BAKEOFF_ITEMS} items)...")
        batch_result, wall = await run_contestant(contestant, bakeoff_items)
        card, ambiguous = score_batch(contestant.name, contestant.model_id, batch_result, wall)
        cards.append(card)
        for item_id, raw, gold in ambiguous:
            tagged_id = f"{contestant.name}:{item_id}"
            all_ambiguous.append((tagged_id, raw, gold))
            ambiguous_owner[tagged_id] = card

    # ---- Fallback grader (LLM-as-judge) on the unparseable outputs ----
    judge_card: Scorecard | None = None
    if all_ambiguous and OPENAI_API_KEY:
        print(f"\nFallback grader: judging {len(all_ambiguous)} unparseable outputs with {JUDGE_MODEL}...")
        start = perf_counter()
        verdicts, judge_br = await grade_ambiguous(all_ambiguous)
        judge_wall = perf_counter() - start
        for tagged_id, correct in verdicts.items():
            if correct:
                ambiguous_owner[tagged_id].judge_correct += 1
        judge_card = Scorecard(
            name="judge",
            model_id=JUDGE_MODEL,
            total=len(all_ambiguous),
            exact_correct=0,
            judge_correct=sum(verdicts.values()),
            errors=0,
            ambiguous=0,
            wall=judge_wall,
            batch_result=judge_br,
        )
    elif all_ambiguous:
        print(
            f"\n{len(all_ambiguous)} outputs were unparseable; set OPENAI_API_KEY "
            "to grade them with the ChatGPT fallback judge."
        )

    print_bakeoff(cards, judge_card)
    summary_path = write_summary(race_rows, cards, judge_card)
    print(f"\nPer-item results written to {RESULTS_DIR}/ (one .jsonl.gz per provider).")
    print(f"Aggregate summary (wall time + cost + accuracy) written to {summary_path}.")


if __name__ == "__main__":
    asyncio.run(main())
