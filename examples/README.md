# Batch LLM Examples

This directory contains example scripts demonstrating how to use the `async-batch-llm` package.

## Running Examples

### Setup

First, install the package:

```bash
# If developing locally
uv pip install -e ..

# If installed from PyPI
uv pip install async-batch-llm
```

### What each example needs

These run with no API key:

| Example | Extra | API key |
| --- | --- | --- |
| `example_callable_application.py` | none | none |
| `example_single_call.py` | `pydantic-ai` (uses `MockAgent`) | none |
| `example_gateway.py` | `pydantic-ai` (uses `MockAgent`) | none |
| `example_context_manager.py` | `pydantic-ai` (uses `MockAgent`) | none |
| `example.py` | `pydantic-ai` | none for the `MockAgent` example; examples 1-3 need `GOOGLE_API_KEY` |

These call a provider:

| Example | Extra | API key |
| --- | --- | --- |
| `example_openai.py`, `example_production_resume.py` | `openai` | `OPENAI_API_KEY` |
| `example_openrouter.py` | `openrouter` | `OPENROUTER_API_KEY` |
| `example_deepseek.py` | `deepseek` | `DEEPSEEK_API_KEY` |
| `example_gemini_*.py`, `example_model_escalation.py`, `example_smart_model_escalation.py` | `gemini` | `GOOGLE_API_KEY` |
| `example_llm_strategies.py` | `gemini`, `pydantic-ai` | `GOOGLE_API_KEY` |
| `example_embeddings.py` | `openai` and/or `gemini` | `OPENAI_API_KEY` and/or `GOOGLE_API_KEY` |
| `example_anthropic.py` | none (`pip install anthropic`) | `ANTHROPIC_API_KEY` |
| `example_langchain.py` | none (see its docstring) | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |
| `example_batch_benchmark.py` | `deepseek`, `gemini`, `openai` | see the [benchmark walkthrough](../docs/examples/benchmark-walkthrough.md) |

`GEMINI_API_KEY` works in place of `GOOGLE_API_KEY`. Get a Gemini key from
<https://aistudio.google.com/apikey>. Most scripts that need a key print a message and exit
when it isn't set.

### Run Examples

```bash
python example_callable_application.py
```

## What's Included

### Quickstart assets

The [no-key Colab notebook](https://colab.research.google.com/github/geoff-davis/async-batch-llm/blob/main/notebooks/async_batch_llm_quickstart.ipynb)
shows coalesced progress, item-local retry feedback, a terminal failure,
`BatchResult.summary()`, and checkpoint replay. The README terminal animation
is generated from the same credential-free application scenario; see
[`docs/terminal-demo.md`](../docs/terminal-demo.md) for regeneration.

### `example_callable_application.py`

The flagship embedded-application example is fully local and needs no API key.
It wraps a fake existing async gateway client with `CallableStrategy`, streams a
paginated database-style source through bounded input and completed-result
handoffs, writes results to an async transactional sink, retains billed tokens
from validation failures, keeps retry feedback private per item, checkpoints
before publication, and performs a second run with zero live calls through
compatible replay.

### `example_production_resume.py`

A production-oriented OpenAI run with versioned JSONL checkpoints, compatible
success replay, stable collected-result ordering, item and batch deadlines, and
category-based fail-fast behavior. Run it twice with the same inputs to see
successful results replayed without another provider call.

### `example.py`

Comprehensive examples demonstrating:

1. **Simple Batch Processing** - Basic parallel processing with multiple workers
2. **Context and Post-Processing** - Using context data and post-processing hooks
3. **Error Handling** - Handling timeouts and failures gracefully
4. **Testing with MockAgent** - Testing without making real API calls

Each example is self-contained and includes detailed comments. Only example 4 runs
by default; examples 1-3 are commented out in `main()` because they need a Gemini
API key.

Output from example 1 looks like this:

```text
================================================================================
EXAMPLE 1: Simple Batch Processing (New API)
================================================================================
Processed 5 items:
  Succeeded: 5
  Failed: 0
  Total tokens: 1,234
  ✓ Pride and Prejudice: Pride and Prejudice - Romance
  ✓ 1984: Nineteen Eighty-Four - Dystopian Fiction
  ...
```

### `example_gemini_grounding.py`

Grounded Gemini batches: requests the `google_search` tool via
`generation_config` and reads web citations back through the typed views
(`result.grounding.sources` / `.queries`) — no custom strategy or extractor
needed. Requires `async-batch-llm[gemini]` and a `GOOGLE_API_KEY`.

### `example_embeddings.py`

Batch embedding generation with OpenAI (`text-embedding-3-small`) and
Gemini (`gemini-embedding-2`) via custom strategies — the framework has no
built-in embedding support, but `LLMCallStrategy` is generic over its
output type, so a strategy can return vectors and still get the worker
pool, rate-limit coordination, and retries. Each work item carries a
JSON-encoded *chunk* of texts (embedding APIs accept many inputs per
request). Runs whichever provider has an SDK + API key available.

Other provider- and pattern-specific examples (`example_openai.py`,
`example_openrouter.py`, `example_deepseek.py`, smart retry, model
escalation, benchmarks, …) live alongside this file — each script's module
docstring covers its own setup.

## Tips

- Start with `example_callable_application.py`; it needs no extras or API key.
- Size runs with `concurrency=` and `attempt_timeout`; see the
  [Choosing Your Limits guide](../docs/choosing-your-limits.md).
- Check the `BatchResult.summary()` or metrics output to monitor performance.
