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

### Set API Key

Most examples require a Google Gemini API key:

```bash
export GOOGLE_API_KEY="your-api-key-here"  # GEMINI_API_KEY also works
```

Get your API key from: <https://makersuite.google.com/app/apikey>

### Run Examples

```bash
# Run the main example
python example.py
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

Each example is self-contained and includes detailed comments.

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

## Example Output

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

## Tips

- Start with Example 4 (MockAgent) - it doesn't require an API key
- Examples 1-3 require a valid Gemini API key
- Adjust `max_workers` and `attempt_timeout` for your use case
- Check the metrics output to monitor performance
