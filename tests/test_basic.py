"""Basic tests for the batch_llm module using MockAgent.

These tests don't require any API keys and can be run in CI/CD.
"""

from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from batch_llm import (
    LLMWorkItem,
    MetricsObserver,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,
    SimpleBatchProcessor,
    SimpleResult,
    SimpleWorkItem,
)
from batch_llm.testing import MockAgent


class BookSummary(BaseModel):
    """Test output model."""

    title: Annotated[str, Field(description="Book title")]
    summary: Annotated[str, Field(description="Summary")]
    genre: Annotated[str, Field(description="Genre")]


@pytest.mark.asyncio
async def test_basic_processing():
    """Test basic batch processing with MockAgent."""

    def mock_response(prompt: str) -> BookSummary:
        return BookSummary(
            title="Mock Book",
            summary=f"Summary for: {prompt[:30]}",
            genre="Fiction",
        )

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)

    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, BookSummary, None](config=config)

    # Add work items
    for i in range(5):
        work_item = LLMWorkItem(
            item_id=f"book_{i}",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Summarize book {i}",
            context=None,
        )
        await processor.add_work(work_item)

    # Process all
    result = await processor.process_all()

    # Assertions
    assert result.total_items == 5
    assert result.succeeded == 5
    assert result.failed == 0
    assert len(result.results) == 5


@pytest.mark.asyncio
async def test_with_context():
    """Test processing with context data."""

    class BookContext(BaseModel):
        book_id: str
        source: str

    def mock_response(prompt: str) -> BookSummary:
        return BookSummary(
            title="Mock Book",
            summary="Mock summary",
            genre="Fiction",
        )

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, BookSummary, BookContext](config=config)

    # Add work with context
    context = BookContext(book_id="book_1", source="test")
    work_item = LLMWorkItem(
        item_id="book_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test prompt",
        context=context,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Check context is preserved
    assert result.results[0].context.book_id == "book_1"
    assert result.results[0].context.source == "test"


@pytest.mark.asyncio
async def test_post_processor():
    """Test post-processor is called after each success."""

    post_processed = []

    async def post_process(result):
        if result.success:
            post_processed.append(result.item_id)

    def mock_response(prompt: str) -> BookSummary:
        return BookSummary(title="Mock", summary="Mock", genre="Fiction")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, BookSummary, None](
        config=config, post_processor=post_process
    )

    # Add 3 items
    for i in range(3):
        work_item = LLMWorkItem(
            item_id=f"book_{i}",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Test {i}",
            context=None,
        )
        await processor.add_work(work_item)

    await processor.process_all()

    # Check post-processor was called for each
    assert len(post_processed) == 3
    assert "book_0" in post_processed
    assert "book_1" in post_processed
    assert "book_2" in post_processed


@pytest.mark.asyncio
async def test_metrics_observer():
    """Test metrics observer tracks stats correctly."""

    def mock_response(prompt: str) -> BookSummary:
        return BookSummary(title="Mock", summary="Mock", genre="Fiction")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)
    metrics = MetricsObserver()

    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, BookSummary, None](config=config, observers=[metrics])

    # Add 5 items
    for i in range(5):
        work_item = LLMWorkItem(
            item_id=f"book_{i}",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Test {i}",
            context=None,
        )
        await processor.add_work(work_item)

    await processor.process_all()

    # Check metrics
    collected_metrics = await metrics.get_metrics()
    assert collected_metrics["items_processed"] == 5
    assert collected_metrics["items_succeeded"] == 5
    assert collected_metrics["items_failed"] == 0
    assert collected_metrics["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_timeout_handling():
    """Test that timeouts are handled correctly."""

    def mock_response(prompt: str) -> BookSummary:
        return BookSummary(title="Mock", summary="Mock", genre="Fiction")

    # Mock agent with 1 second latency, but timeout is 0.1 seconds
    mock_agent = MockAgent(response_factory=mock_response, latency=1.0)
    config = ProcessorConfig(max_workers=1, timeout_per_item=0.1)
    processor = ParallelBatchProcessor[str, BookSummary, None](config=config)

    work_item = LLMWorkItem(
        item_id="book_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should timeout
    assert result.failed == 1
    assert result.succeeded == 0
    assert (
        "timeout" in result.results[0].error.lower()
        or "cancelled" in result.results[0].error.lower()
    )


@pytest.mark.asyncio
async def test_type_aliases():
    """Test that type aliases work correctly and are equivalent to full generic types."""

    def mock_response(prompt: str) -> BookSummary:
        return BookSummary(
            title="Alias Test",
            summary=f"Summary for: {prompt}",
            genre="Test Genre",
        )

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)
    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)

    # Test using type aliases instead of full ParallelBatchProcessor[str, BookSummary, None]
    processor = SimpleBatchProcessor[BookSummary](config=config)

    # Test SimpleWorkItem alias instead of LLMWorkItem[str, BookSummary, None]
    work_item = SimpleWorkItem[BookSummary](
        item_id="alias_test_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test prompt",
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.total_items == 1
    assert result.succeeded == 1

    # Test SimpleResult type annotation
    first_result: SimpleResult[BookSummary] = result.results[0]
    assert first_result.success
    assert first_result.output.title == "Alias Test"
    assert first_result.context is None  # SimpleResult has None context


@pytest.mark.asyncio
async def test_type_aliases_equivalent_to_full_types():
    """Verify type aliases produce equivalent results to full generic types."""

    def mock_response(prompt: str) -> BookSummary:
        return BookSummary(title="Test", summary="Test", genre="Test")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    # Process with full type
    full_processor = ParallelBatchProcessor[str, BookSummary, None](config=config)
    full_item = LLMWorkItem[str, BookSummary, None](
        item_id="full_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
    )
    await full_processor.add_work(full_item)
    full_result = await full_processor.process_all()

    # Process with alias type
    alias_processor = SimpleBatchProcessor[BookSummary](config=config)
    alias_item = SimpleWorkItem[BookSummary](
        item_id="alias_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
    )
    await alias_processor.add_work(alias_item)
    alias_result = await alias_processor.process_all()

    # Results should be equivalent
    assert full_result.succeeded == alias_result.succeeded
    assert full_result.failed == alias_result.failed
    assert isinstance(alias_result.results[0].output, type(full_result.results[0].output))
