"""Peek at raw model outputs to compare verbosity/style across providers.

The benchmark (`examples/example_batch_benchmark.py`) records only the parsed
numeric answer, not the model's full text — so when one provider's output token
count is much higher than another's, you can't see *why*. This prints the FULL
response from each model for a handful of GSM8K problems, side by side, plus its
output-token count, so you can read the terse-vs-verbose contrast directly.

Same prompt and thinking settings as the benchmark:

- DeepSeek `deepseek-v4-flash`, thinking **off**
- Gemini `gemini-3.1-flash-lite`, thinking level **minimal**
- Gemini `gemini-2.5-flash-lite`, thinking budget **0** (off)

## Setup

```bash
pip install 'async-batch-llm[deepseek,gemini]'
python examples/download_gsm8k.py
export DEEPSEEK_API_KEY=sk-...
export GOOGLE_API_KEY=...        # or GOOGLE_GENAI_USE_VERTEXAI=true + ADC
```

## Run

```bash
python examples/peek_outputs.py        # 6 problems (default)
python examples/peek_outputs.py 12     # N problems
```

Models whose credentials are absent are skipped.
"""

import asyncio
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "data" / "gsm8k_test.jsonl.gz"

PROMPT_TEMPLATE = (
    "Solve this grade-school math problem. Think briefly step by step, then on "
    "the FINAL line output the answer exactly as `#### <number>` and nothing "
    "after it.\n\nProblem: {question}"
)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
GEMINI_USE_VERTEX = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")


def load_questions(limit: int) -> list[str]:
    questions: list[str] = []
    with gzip.open(DATA_PATH, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line)["question"])
            if len(questions) >= limit:
                break
    return questions


def build_models() -> list[tuple[str, Any, dict[str, Any]]]:
    """Return ``(label, model, generate_kwargs)`` for each available provider —
    the same models/configs the benchmark uses."""
    entries: list[tuple[str, Any, dict[str, Any]]] = []

    if DEEPSEEK_API_KEY:
        from async_batch_llm import DeepSeekModel

        entries.append(
            (
                "deepseek-flash (no-think)",
                DeepSeekModel.from_api_key("deepseek-v4-flash", thinking=False),
                {},
            )
        )

    if GOOGLE_API_KEY or GEMINI_USE_VERTEX:
        from google import genai

        from async_batch_llm import GeminiModel

        client = (
            genai.Client(vertexai=True)
            if GEMINI_USE_VERTEX
            else genai.Client(api_key=GOOGLE_API_KEY)
        )
        entries.append(
            (
                "gemini-3.1 (minimal)",
                GeminiModel("gemini-3.1-flash-lite", client),
                {"config": {"thinking_config": {"thinking_level": "minimal"}}},
            )
        )
        entries.append(
            (
                "gemini-2.5 (thinking off)",
                GeminiModel("gemini-2.5-flash-lite", client),
                {"config": {"thinking_config": {"thinking_budget": 0}}},
            )
        )

    return entries


async def main() -> None:
    if not DATA_PATH.exists():
        print(f"Benchmark data not found at {DATA_PATH}")
        print("Run:  python examples/download_gsm8k.py")
        return

    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
    entries = build_models()
    if not entries:
        print("No provider credentials found. Set DEEPSEEK_API_KEY and/or GOOGLE_API_KEY")
        print("(or GOOGLE_GENAI_USE_VERTEXAI=true + ADC), then re-run.")
        return

    questions = load_questions(limit)
    token_totals = {label: 0 for label, _, _ in entries}

    try:
        for i, question in enumerate(questions):
            print("\n" + "=" * 88)
            print(f"PROBLEM {i + 1}/{len(questions)}: {question}")
            print("=" * 88)

            prompt = PROMPT_TEMPLATE.format(question=question)
            responses = await asyncio.gather(
                *(model.generate(prompt, **kwargs) for _, model, kwargs in entries),
                return_exceptions=True,
            )

            for (label, _, _), response in zip(entries, responses, strict=True):
                print(f"\n--- {label} ---")
                if isinstance(response, Exception):
                    print(f"  [error: {type(response).__name__}: {response}]")
                    continue
                token_totals[label] += response.output_tokens
                print(f"  ({response.output_tokens} output tokens)")
                print(response.text.strip())
    finally:
        from async_batch_llm.core.protocols import ManagedLLMModel

        for _, model, _ in entries:
            if isinstance(model, ManagedLLMModel):
                await model.cleanup()

    print("\n" + "=" * 88)
    print(f"AVG OUTPUT TOKENS PER ANSWER (over {len(questions)} problems)")
    print("=" * 88)
    for label, _, _ in entries:
        avg = token_totals[label] / len(questions) if questions else 0.0
        print(f"  {label:<28} {avg:6.1f}")


if __name__ == "__main__":
    asyncio.run(main())
