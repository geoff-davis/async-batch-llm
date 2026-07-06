"""Example: batch embedding generation with Gemini and OpenAI.

async-batch-llm has no built-in embedding support, but ``LLMCallStrategy``
is generic over its output type — a custom strategy can return embedding
vectors instead of text and still get the worker pool, rate-limit cooldown
coordination, retries, and token accounting for free. This is the pattern
for RAG-ingestion pipelines: thousands of independent, rate-limited API
calls with bounded concurrency.

Two things make embeddings different from text generation:

1. **The efficient unit of work is a chunk, not a text.** Embedding APIs
   accept many inputs per request (OpenAI: up to 2048), so each work item
   here carries a *batch* of texts and returns a *batch* of vectors.
   ``LLMWorkItem.prompt`` is a string, so the chunk is JSON-encoded into
   the prompt and decoded inside ``execute()`` — the same pattern as the
   "structured prompts" note in the docs.
2. **Token accounting varies.** OpenAI reports usage on every embeddings
   response. The Gemini Developer API reports none (vectors are billed by
   input tokens, but the response omits usage); Vertex AI attaches
   per-embedding ``statistics.token_count``, which the strategy picks up
   when present.

Gemini gotcha (current as of mid-2026): ``gemini-embedding-2`` aggregates
a plain list of strings into ONE embedding (it's multimodal — a list reads
as parts of one document). To get one vector per text, wrap each in a
``Content`` object, as ``GeminiEmbeddingStrategy`` does below. The older
text-only ``gemini-embedding-001`` had the opposite behavior (one vector
per string) and a ``task_type`` config; ``gemini-embedding-2`` instead
takes task instructions as text prefixes, e.g.
``"task: search result | query: {text}"``.

## Installation

```bash
pip install 'async-batch-llm[gemini,openai]'
```

(Each provider's section runs only if its SDK and API key are available,
so installing just one extra is fine.)

## Setup

```bash
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...   # or GEMINI_API_KEY
```
"""

import asyncio
import json
import math
import os
from typing import Any

from async_batch_llm import (
    GeminiErrorClassifier,
    LLMWorkItem,
    OpenAIErrorClassifier,
    ParallelBatchProcessor,
    ProcessorConfig,
    TokenUsage,
)
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.strategies.errors import ErrorClassifier

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from google import genai
    from google.genai.types import Content, EmbedContentConfig, Part
except ImportError:
    genai = None

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# A tiny stand-in corpus. In a real ingestion pipeline this would be
# thousands of document chunks — the framework's bounded worker pool and
# rate-limit coordination are what keep that workload well-behaved.
CORPUS = [
    "The James Webb telescope captured infrared images of distant galaxies.",
    "Rocket boosters separate about two minutes after launch.",
    "Astronauts on the ISS experience sixteen sunrises per day.",
    "Mars rovers search for signs of ancient microbial life.",
    "Caramelizing onions slowly brings out their natural sweetness.",
    "Fresh basil should be added at the end of cooking to keep its aroma.",
    "A cast-iron skillet retains heat better than stainless steel.",
    "Proofing bread dough overnight in the fridge deepens its flavor.",
]

# How many texts to pack into each API call. Real values can be much
# higher (OpenAI accepts up to 2048 inputs per request, Gemini 100);
# kept small here so the batching is visible in the output.
CHUNK_SIZE = 4


def chunked(texts: list[str], size: int) -> list[list[str]]:
    """Split texts into chunks of at most `size`."""
    return [texts[i : i + size] for i in range(0, len(texts), size)]


class OpenAIEmbeddingStrategy(LLMCallStrategy[list[list[float]]]):
    """Embed a JSON-encoded chunk of texts in one OpenAI API call.

    The prompt is `json.dumps(list_of_texts)`; the output is one vector
    per text, in input order.
    """

    def __init__(
        self,
        client: "AsyncOpenAI",
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ):
        """
        Args:
            client: Initialized AsyncOpenAI client.
            model: Embedding model name.
            dimensions: Optional reduced output dimensionality
                (text-embedding-3-* models support shortening).
        """
        self.client = client
        self.model = model
        self.dimensions = dimensions

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state=None
    ) -> tuple[list[list[float]], TokenUsage]:
        texts: list[str] = json.loads(prompt)

        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        response = await self.client.embeddings.create(**kwargs)

        # The API documents no ordering guarantee — sort by index.
        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

        tokens: TokenUsage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": 0,  # embeddings have no completion tokens
            "total_tokens": response.usage.total_tokens,
        }
        return vectors, tokens


class GeminiEmbeddingStrategy(LLMCallStrategy[list[list[float]]]):
    """Embed a JSON-encoded chunk of texts in one Gemini API call.

    Uses ``gemini-embedding-2`` (GA since 2026, multimodal). Each text is
    wrapped in its own ``Content`` object — passing a plain list of
    strings would return a single embedding aggregating the whole chunk.
    Task hints (the old ``task_type`` config) are now text prefixes; for
    retrieval corpora Google suggests e.g. ``"task: search result |
    query: {text}"`` — omitted here to keep the vectors general-purpose.
    """

    def __init__(
        self,
        client: "genai.Client",
        model: str = "gemini-embedding-2",
        output_dimensionality: int | None = None,
    ):
        """
        Args:
            client: Initialized google-genai client.
            model: Embedding model name.
            output_dimensionality: Optional reduced dimensionality
                (128-3072; 768, 1536, and 3072 recommended). Truncated
                vectors are normalized automatically by the API.
        """
        self.client = client
        self.model = model
        self.output_dimensionality = output_dimensionality

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state=None
    ) -> tuple[list[list[float]], TokenUsage]:
        texts: list[str] = json.loads(prompt)

        config = None
        if self.output_dimensionality is not None:
            config = EmbedContentConfig(output_dimensionality=self.output_dimensionality)

        response = await self.client.aio.models.embed_content(
            model=self.model,
            # One Content per text => one embedding per text. A plain
            # list[str] would come back as ONE aggregated embedding.
            contents=[Content(parts=[Part.from_text(text=t)]) for t in texts],
            config=config,
        )

        vectors = [e.values for e in response.embeddings]

        # The Gemini Developer API returns no usage for embeddings.
        # Vertex AI attaches per-embedding statistics — pick them up
        # when present so token totals are accurate there.
        input_tokens = sum(
            int(e.statistics.token_count)
            for e in response.embeddings
            if e.statistics and e.statistics.token_count
        )
        tokens: TokenUsage = {
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "total_tokens": input_tokens,
        }
        return vectors, tokens


async def embed_corpus(
    strategy: LLMCallStrategy[list[list[float]]],
    error_classifier: ErrorClassifier,
    texts: list[str],
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, list[float]]:
    """Embed a corpus through the batch processor, one work item per chunk.

    Returns a {text: vector} mapping. Results complete in any order, so
    each chunk's texts travel alongside it in the work item's context.
    """
    config = ProcessorConfig(max_workers=4, timeout_per_item=60.0)

    async with ParallelBatchProcessor[dict[str, Any], list[list[float]], None](
        config=config,
        error_classifier=error_classifier,
    ) as processor:
        for i, chunk in enumerate(chunked(texts, chunk_size)):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"chunk_{i}",
                    strategy=strategy,
                    prompt=json.dumps(chunk),
                    context={"texts": chunk},
                )
            )
        result = await processor.process_all()

    embeddings: dict[str, list[float]] = {}
    for item_result in result.results:
        if item_result.success and item_result.context:
            for text, vector in zip(item_result.context["texts"], item_result.output):
                embeddings[text] = vector
        elif not item_result.success:
            print(f"  FAILED {item_result.item_id}: {item_result.error}")

    print(f"  Chunks: {result.succeeded}/{result.total_items} succeeded")
    print(f"  Input tokens: {result.total_input_tokens or '(not reported by this API)'}")
    return embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (stdlib only)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def show_nearest_neighbors(embeddings: dict[str, list[float]]) -> None:
    """Print each text's nearest neighbor — space texts should pair with
    space texts, cooking with cooking."""
    texts = list(embeddings)
    print("\n  Nearest neighbors:")
    for text in texts[:4]:  # first few, to keep output short
        best = max(
            (t for t in texts if t != text),
            key=lambda t: cosine_similarity(embeddings[text], embeddings[t]),
        )
        score = cosine_similarity(embeddings[text], embeddings[best])
        print(f"    {text[:52]:<52} -> {best[:52]} ({score:.3f})")


async def run_openai_embeddings() -> None:
    print("\n" + "=" * 60)
    print("OpenAI embeddings (text-embedding-3-small)")
    print("=" * 60)

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    try:
        strategy = OpenAIEmbeddingStrategy(client)
        embeddings = await embed_corpus(strategy, OpenAIErrorClassifier(), CORPUS)
        if embeddings:
            dims = len(next(iter(embeddings.values())))
            print(f"  Embedded {len(embeddings)} texts at {dims} dimensions")
            show_nearest_neighbors(embeddings)
    finally:
        await client.close()


async def run_gemini_embeddings() -> None:
    print("\n" + "=" * 60)
    print("Gemini embeddings (gemini-embedding-2)")
    print("=" * 60)

    client = genai.Client(api_key=GOOGLE_API_KEY)
    strategy = GeminiEmbeddingStrategy(client)
    embeddings = await embed_corpus(strategy, GeminiErrorClassifier(), CORPUS)
    if embeddings:
        dims = len(next(iter(embeddings.values())))
        print(f"  Embedded {len(embeddings)} texts at {dims} dimensions")
        show_nearest_neighbors(embeddings)


async def main() -> None:
    ran_any = False

    if AsyncOpenAI is not None and OPENAI_API_KEY:
        await run_openai_embeddings()
        ran_any = True
    else:
        print("Skipping OpenAI: install 'async-batch-llm[openai]' and set OPENAI_API_KEY")

    if genai is not None and GOOGLE_API_KEY:
        await run_gemini_embeddings()
        ran_any = True
    else:
        print(
            "Skipping Gemini: install 'async-batch-llm[gemini]' and set "
            "GOOGLE_API_KEY (or GEMINI_API_KEY)"
        )

    if not ran_any:
        print("\nNo provider available — nothing to run.")


if __name__ == "__main__":
    asyncio.run(main())
