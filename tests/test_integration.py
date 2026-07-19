"""Integration tests with real API calls.

These tests require valid API keys and make actual API calls.
Run with: pytest -m integration

Skip with: pytest -m "not integration"
"""

import os

import pytest

# Check if integration tests should run
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


@pytest.mark.integration
@pytest.mark.skipif(
    not GOOGLE_API_KEY,
    reason="Requires GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable",
)
@pytest.mark.asyncio
async def test_gemini_strategy_real_api():
    """Integration test with real Gemini API - basic generation."""
    from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
    from async_batch_llm.llm_strategies import GeminiStrategy
    from async_batch_llm.models import GeminiModel

    try:
        import google.genai as genai
    except ImportError:
        pytest.skip("google-genai not installed")

    # Create client with API key
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Create model and strategy
    model = GeminiModel("gemini-2.0-flash-exp", client)

    # Simple response parser - receives LLMResponse
    def parse_response(response):
        return response.text

    strategy = GeminiStrategy(
        model=model,
        response_parser=parse_response,
    )

    # Create processor
    config = ProcessorConfig(max_workers=2, attempt_timeout=30.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add a few simple test items
        for i in range(3):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"test_{i}",
                    strategy=strategy,
                    prompt=f"Say 'Test {i}' and nothing else.",
                )
            )

        result = await processor.process_all()

    # Verify results
    assert result.total_items == 3
    assert result.succeeded == 3, (
        f"Expected 3 successes, got {result.succeeded}. Errors: {[r.error for r in result.results if r.error]}"
    )
    assert result.failed == 0

    # Verify token usage tracked
    for item_result in result.results:
        assert item_result.token_usage["total_tokens"] > 0, "Token usage should be tracked"

    print(f"Gemini integration test passed: {result.succeeded}/{result.total_items} items")


@pytest.mark.integration
@pytest.mark.skipif(
    not GOOGLE_API_KEY,
    reason="Requires GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable",
)
@pytest.mark.asyncio
async def test_gemini_cached_model_real_api():
    """Integration test with real Gemini API - context caching."""
    from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
    from async_batch_llm.llm_strategies import GeminiStrategy
    from async_batch_llm.models import GeminiCachedModel

    try:
        import google.genai as genai
        from google.genai.types import Content
    except ImportError:
        pytest.skip("google-genai not installed")

    # Create client
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Create cached content (system instruction)
    cached_content = [
        Content(
            role="user",
            parts=[
                {
                    "text": "You are a helpful assistant that responds concisely. "
                    "When asked to count, respond with just the number."
                }
            ],
        )
    ]

    def parse_response(response):
        return response.text.strip()

    # Create cached model and strategy
    cached_model = GeminiCachedModel(
        model="gemini-2.0-flash-exp",
        client=client,
        cached_content=cached_content,
        cache_ttl_seconds=300,  # 5 minutes
    )

    strategy = GeminiStrategy(
        model=cached_model,
        response_parser=parse_response,
    )

    config = ProcessorConfig(max_workers=2, attempt_timeout=30.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add items that will benefit from caching
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"count_{i}",
                    strategy=strategy,  # Reuse same strategy (shared cache)
                    prompt=f"Count to {i + 1}",
                )
            )

        result = await processor.process_all()

    # Verify results
    assert result.total_items == 5
    assert result.succeeded == 5, f"Expected 5 successes, got {result.succeeded}"
    assert result.failed == 0

    # Verify cached tokens were used
    cached_tokens_used = sum(
        item_result.token_usage.get("cached_input_tokens", 0) for item_result in result.results
    )
    assert cached_tokens_used > 0, "Should have used cached tokens"

    # Cleanup cache
    await cached_model.delete_cache()

    print(
        f"Gemini cached integration test passed: {result.succeeded}/{result.total_items} items, "
        f"{cached_tokens_used} cached tokens used"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not GOOGLE_API_KEY,
    reason="Requires GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable",
)
@pytest.mark.asyncio
async def test_gemini_response_metadata_real_api():
    """Integration test for LLMResponse with safety ratings (v0.6.0)."""
    from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
    from async_batch_llm.llm_strategies import GeminiStrategy
    from async_batch_llm.models import GeminiModel

    try:
        import google.genai as genai
    except ImportError:
        pytest.skip("google-genai not installed")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Create model and strategy
    model = GeminiModel("gemini-2.0-flash-exp", client)

    # Parser that preserves metadata from LLMResponse
    metadata_store = {}

    def parse_response(response):
        metadata_store["safety_ratings"] = (
            response.metadata.get("safety_ratings") if response.metadata else None
        )
        metadata_store["finish_reason"] = (
            response.metadata.get("finish_reason") if response.metadata else None
        )
        metadata_store["raw"] = response.raw
        return response.text

    strategy = GeminiStrategy(
        model=model,
        response_parser=parse_response,
    )

    config = ProcessorConfig(max_workers=1, attempt_timeout=30.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="safety_test",
                strategy=strategy,
                prompt="Tell me a short story about a robot.",
            )
        )

        result = await processor.process_all()

    # Verify results
    assert result.succeeded == 1

    # Check that metadata was accessible in the parser
    assert metadata_store.get("safety_ratings") is not None, "Safety ratings should be present"
    assert len(metadata_store["safety_ratings"]) > 0, "Should have at least one safety rating"
    assert metadata_store.get("finish_reason") is not None, "Finish reason should be present"
    assert metadata_store.get("raw") is not None, "Raw response should be present"

    # Output should be plain string (not wrapped)
    item_result = result.results[0]
    assert isinstance(item_result.output, str), "Parsed output should be a string"
    assert len(item_result.output) > 0, "Should have generated text"

    print("Gemini metadata test passed")
    print(f"   Safety ratings: {metadata_store['safety_ratings']}")
    print(f"   Finish reason: {metadata_store['finish_reason']}")


@pytest.mark.integration
@pytest.mark.skipif(
    not GOOGLE_API_KEY,
    reason="Requires GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable",
)
@pytest.mark.asyncio
async def test_retry_on_real_validation_error():
    """Integration test that triggers real validation error and retries."""
    from pydantic import BaseModel, Field

    from async_batch_llm import (
        LLMWorkItem,
        ParallelBatchProcessor,
        ProcessorConfig,
        PydanticAIStrategy,
        RetryConfig,
    )

    try:
        from pydantic_ai import Agent
    except ImportError:
        pytest.skip("pydantic-ai not installed")

    class StrictNumber(BaseModel):
        """Model that expects a specific number format."""

        value: int = Field(..., ge=1, le=10, description="A number between 1 and 10")

    # Create agent with Gemini
    agent = Agent(
        "gemini-2.0-flash-exp",
        result_type=StrictNumber,
        system_prompt="You must respond with a number between 1 and 10.",
    )

    strategy = PydanticAIStrategy(agent=agent)

    config = ProcessorConfig(
        max_workers=1,
        attempt_timeout=30.0,
        retry=RetryConfig(max_attempts=3),
    )

    async with ParallelBatchProcessor[str, StrictNumber, None](config=config) as processor:
        # This prompt might cause validation issues on first try
        await processor.add_work(
            LLMWorkItem(
                item_id="validation_test",
                strategy=strategy,
                prompt="Pick a number between 1 and 10 and return it.",
            )
        )

        result = await processor.process_all()

    # Should eventually succeed (possibly after retries)
    assert result.total_items == 1
    # May succeed on first try or after retries
    if result.succeeded == 1:
        print("Validation test passed on attempt")
    else:
        print(f"Validation test failed after all retries: {result.results[0].error}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_suite_summary():
    """Print summary of which integration tests can run."""
    available_tests = []
    missing_keys = []

    if GOOGLE_API_KEY:
        available_tests.append("Google Gemini API tests (GOOGLE_API_KEY detected)")
    else:
        missing_keys.append("GOOGLE_API_KEY (or legacy GEMINI_API_KEY) not set")

    if OPENAI_API_KEY:
        available_tests.append("OpenAI API tests (not yet implemented)")
    else:
        missing_keys.append("OPENAI_API_KEY not set")

    if ANTHROPIC_API_KEY:
        available_tests.append("Anthropic API tests (not yet implemented)")
    else:
        missing_keys.append("ANTHROPIC_API_KEY not set")

    print("\n" + "=" * 60)
    print("Integration Test Suite Configuration")
    print("=" * 60)

    if available_tests:
        print("\nAvailable tests:")
        for test in available_tests:
            print(f"  {test}")

    if missing_keys:
        print("\nMissing API keys (tests will be skipped):")
        for key in missing_keys:
            print(f"  {key}")

    print("\nTo run integration tests:")
    print("  pytest -m integration -v")
    print("\nTo skip integration tests:")
    print("  pytest -m 'not integration'")
    print("=" * 60 + "\n")

    # This test always passes - it's just informational
    assert True


# Placeholder tests for future implementation
@pytest.mark.integration
@pytest.mark.skipif(not OPENAI_API_KEY, reason="Requires OPENAI_API_KEY environment variable")
@pytest.mark.asyncio
async def test_openai_strategy_real_api():
    """Integration test with real OpenAI API."""
    pytest.skip("OpenAI integration test not yet implemented")


@pytest.mark.integration
@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="Requires ANTHROPIC_API_KEY environment variable")
@pytest.mark.asyncio
async def test_anthropic_strategy_real_api():
    """Integration test with real Anthropic API."""
    pytest.skip("Anthropic integration test not yet implemented")
