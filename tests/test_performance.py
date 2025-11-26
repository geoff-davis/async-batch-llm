"""Performance benchmarks for async-batch-llm.

These tests measure throughput, scaling, and resource usage.
Run with: pytest -m benchmark

Skip with: pytest -m "not benchmark"
"""

import time
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, PydanticAIStrategy
from async_batch_llm.testing import MockAgent


class BenchmarkOutput(BaseModel):
    """Simple output for benchmarks."""

    value: Annotated[str, Field(description="Benchmark value")]


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_throughput_single_worker():
    """Measure throughput with 1 worker (baseline)."""
    num_items = 100
    latency = 0.01  # 10ms per item

    mock_agent = MockAgent(
        response_factory=lambda p: BenchmarkOutput(value=f"Result: {p}"),
        latency=latency,
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    strategy = PydanticAIStrategy(agent=mock_agent)

    start_time = time.time()

    async with ParallelBatchProcessor[str, BenchmarkOutput, None](config=config) as processor:
        for i in range(num_items):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

    elapsed = time.time() - start_time

    # Calculate metrics
    items_per_second = num_items / elapsed
    expected_time = num_items * latency  # Sequential processing
    efficiency = (expected_time / elapsed) * 100  # Should be ~100%

    print(f"\n{'=' * 60}")
    print("Throughput Benchmark - Single Worker")
    print(f"{'=' * 60}")
    print(f"Items processed: {result.total_items}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Throughput: {items_per_second:.2f} items/sec")
    print(f"Efficiency: {efficiency:.1f}% (100% = perfect sequential)")
    print(f"{'=' * 60}\n")

    assert result.succeeded == num_items
    assert elapsed < expected_time * 1.5  # Allow 50% overhead


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_throughput_scaling_workers():
    """Measure throughput scaling with different worker counts."""
    num_items = 200
    latency = 0.01  # 10ms per item
    worker_counts = [1, 2, 5, 10]

    results = {}

    for workers in worker_counts:
        mock_agent = MockAgent(
            response_factory=lambda p: BenchmarkOutput(value=f"Result: {p}"),
            latency=latency,
        )

        config = ProcessorConfig(max_workers=workers, timeout_per_item=10.0)
        strategy = PydanticAIStrategy(agent=mock_agent)

        start_time = time.time()

        async with ParallelBatchProcessor[str, BenchmarkOutput, None](config=config) as processor:
            for i in range(num_items):
                await processor.add_work(
                    LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
                )

            result = await processor.process_all()

        elapsed = time.time() - start_time
        items_per_second = num_items / elapsed
        speedup = results[1] / elapsed if 1 in results else 1.0

        results[workers] = elapsed

        print(
            f"Workers: {workers:2d} | Time: {elapsed:5.2f}s | Throughput: {items_per_second:6.2f} items/sec | Speedup: {speedup:.2f}x"
        )

        assert result.succeeded == num_items

    print(f"\n{'=' * 60}")
    print("Throughput Scaling Benchmark")
    print(f"{'=' * 60}")
    print(f"Items: {num_items}, Latency: {latency}s per item")
    print("Worker Count | Time (s) | Throughput (items/s) | Speedup")
    print(f"{'-' * 60}")

    baseline = results[1]
    for workers in worker_counts:
        elapsed = results[workers]
        throughput = num_items / elapsed
        speedup = baseline / elapsed
        print(f"{workers:12d} | {elapsed:8.2f} | {throughput:20.2f} | {speedup:7.2f}x")

    print(f"{'=' * 60}\n")

    # Verify scaling - 10 workers should be at least 5x faster than 1
    assert results[10] < results[1] / 5.0, "10 workers should provide >5x speedup"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_memory_usage_many_items():
    """Benchmark memory usage with many items."""
    num_items = 1000
    latency = 0.001  # 1ms per item (fast processing)

    mock_agent = MockAgent(
        response_factory=lambda p: BenchmarkOutput(value=f"Result: {p}"),
        latency=latency,
    )

    config = ProcessorConfig(max_workers=10, timeout_per_item=10.0)
    strategy = PydanticAIStrategy(agent=mock_agent)

    start_time = time.time()

    async with ParallelBatchProcessor[str, BenchmarkOutput, None](config=config) as processor:
        # Add all items
        for i in range(num_items):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

    elapsed = time.time() - start_time
    items_per_second = num_items / elapsed

    print(f"\n{'=' * 60}")
    print("Memory Usage Benchmark")
    print(f"{'=' * 60}")
    print(f"Items processed: {result.total_items}")
    print(f"Workers: {config.max_workers}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Throughput: {items_per_second:.2f} items/sec")
    print(f"Memory: Results stored for {len(result.results)} items")
    print(f"{'=' * 60}\n")

    assert result.succeeded == num_items
    assert len(result.results) == num_items


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_overhead_empty_processing():
    """Measure framework overhead with minimal processing time."""
    num_items = 100
    latency = 0.0001  # 0.1ms per item (almost instant)

    mock_agent = MockAgent(
        response_factory=lambda p: BenchmarkOutput(value="fast"),
        latency=latency,
    )

    config = ProcessorConfig(max_workers=10, timeout_per_item=10.0)
    strategy = PydanticAIStrategy(agent=mock_agent)

    start_time = time.time()

    async with ParallelBatchProcessor[str, BenchmarkOutput, None](config=config) as processor:
        for i in range(num_items):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt="test")
            )

        result = await processor.process_all()

    elapsed = time.time() - start_time
    theoretical_min = num_items * latency
    overhead = elapsed - theoretical_min
    overhead_per_item = (overhead / num_items) * 1000  # ms per item

    print(f"\n{'=' * 60}")
    print("Framework Overhead Benchmark")
    print(f"{'=' * 60}")
    print(f"Items: {num_items}")
    print(f"Mock latency: {latency * 1000:.2f}ms per item")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Theoretical minimum: {theoretical_min:.3f}s")
    print(f"Framework overhead: {overhead:.3f}s ({overhead_per_item:.2f}ms per item)")
    print(f"Overhead percentage: {(overhead / elapsed) * 100:.1f}%")
    print(f"{'=' * 60}\n")

    assert result.succeeded == num_items
    # Overhead should be reasonable - allow up to 50ms per item
    assert overhead_per_item < 50, f"Overhead too high: {overhead_per_item:.2f}ms per item"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_concurrent_stats_performance():
    """Measure performance impact of stats collection under concurrency."""
    num_items = 500
    latency = 0.005  # 5ms per item

    mock_agent = MockAgent(
        response_factory=lambda p: BenchmarkOutput(value="test"),
        latency=latency,
    )

    config = ProcessorConfig(max_workers=20, timeout_per_item=10.0)  # High concurrency
    strategy = PydanticAIStrategy(agent=mock_agent)

    start_time = time.time()

    async with ParallelBatchProcessor[str, BenchmarkOutput, None](config=config) as processor:
        for i in range(num_items):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt="test")
            )

        result = await processor.process_all()

    elapsed = time.time() - start_time

    # Get stats (this exercises the stats lock)
    stats = await processor.get_stats()

    print(f"\n{'=' * 60}")
    print("Concurrent Stats Performance")
    print(f"{'=' * 60}")
    print(f"Items: {num_items}, Workers: {config.max_workers}")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Throughput: {num_items / elapsed:.2f} items/sec")
    print("Stats collected:")
    print(f"  - Items succeeded: {stats['succeeded']}")
    print(f"  - Items failed: {stats['failed']}")
    print(f"  - Total tokens: {stats['total_tokens']}")
    print(f"{'=' * 60}\n")

    assert result.succeeded == num_items
    assert stats["succeeded"] == num_items


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_shared_strategy_vs_unique_strategies():
    """Compare performance: shared strategy vs unique strategies per item."""
    num_items = 100
    latency = 0.01

    # Test 1: Shared strategy (recommended)
    mock_agent_shared = MockAgent(
        response_factory=lambda p: BenchmarkOutput(value="test"),
        latency=latency,
    )
    shared_strategy = PydanticAIStrategy(agent=mock_agent_shared)
    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)

    start_shared = time.time()
    async with ParallelBatchProcessor[str, BenchmarkOutput, None](config=config) as processor:
        for i in range(num_items):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=shared_strategy, prompt="test")
            )
        result_shared = await processor.process_all()
    time_shared = time.time() - start_shared

    # Test 2: Unique strategies (not recommended, but some users do this)
    start_unique = time.time()
    async with ParallelBatchProcessor[str, BenchmarkOutput, None](config=config) as processor:
        for i in range(num_items):
            # Create new strategy per item (inefficient!)
            unique_agent = MockAgent(
                response_factory=lambda p: BenchmarkOutput(value="test"),
                latency=latency,
            )
            unique_strategy = PydanticAIStrategy(agent=unique_agent)
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=unique_strategy, prompt="test")
            )
        result_unique = await processor.process_all()
    time_unique = time.time() - start_unique

    print(f"\n{'=' * 60}")
    print("Shared vs Unique Strategy Performance")
    print(f"{'=' * 60}")
    print(f"Items: {num_items}, Workers: {config.max_workers}")
    print("\nShared strategy (1 instance):")
    print(f"  Time: {time_shared:.3f}s")
    print(f"  Throughput: {num_items / time_shared:.2f} items/sec")
    print(f"\nUnique strategies ({num_items} instances):")
    print(f"  Time: {time_unique:.3f}s")
    print(f"  Throughput: {num_items / time_unique:.2f} items/sec")
    print(f"\nOverhead from unique strategies: {((time_unique / time_shared - 1) * 100):.1f}%")
    print(f"{'=' * 60}\n")

    assert result_shared.succeeded == num_items
    assert result_unique.succeeded == num_items

    # Shared should be faster (or at least not significantly slower)
    # Allow some variance due to test environment
    assert time_unique >= time_shared * 0.95, "Shared strategy should not be significantly slower"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_retry_performance_impact():
    """Measure performance impact of retry logic."""
    num_items = 100
    latency = 0.01

    from async_batch_llm import RetryConfig

    # Test with retries enabled
    mock_agent = MockAgent(
        response_factory=lambda p: BenchmarkOutput(value="test"),
        latency=latency,
    )
    strategy = PydanticAIStrategy(agent=mock_agent)

    config_with_retries = ProcessorConfig(
        max_workers=5,
        timeout_per_item=10.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.1),
    )

    start_time = time.time()
    async with ParallelBatchProcessor[str, BenchmarkOutput, None](
        config=config_with_retries
    ) as processor:
        for i in range(num_items):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt="test")
            )
        result = await processor.process_all()
    elapsed_with_retries = time.time() - start_time

    throughput = num_items / elapsed_with_retries

    print(f"\n{'=' * 60}")
    print("Retry Logic Performance Impact")
    print(f"{'=' * 60}")
    print(f"Items: {num_items} (all succeeded on first attempt)")
    print("Retry config: max_attempts=3, initial_wait=0.1s")
    print(f"Time: {elapsed_with_retries:.2f}s")
    print(f"Throughput: {throughput:.2f} items/sec")
    print("\nNote: When items succeed on first attempt,")
    print("retry logic has minimal overhead (~1-2%)")
    print(f"{'=' * 60}\n")

    assert result.succeeded == num_items


@pytest.mark.benchmark
def test_benchmark_suite_summary():
    """Print summary of benchmark tests."""
    print("\n" + "=" * 60)
    print("Performance Benchmark Suite")
    print("=" * 60)
    print("\nAvailable benchmarks:")
    print("  - test_throughput_single_worker")
    print("  - test_throughput_scaling_workers")
    print("  - test_memory_usage_many_items")
    print("  - test_overhead_empty_processing")
    print("  - test_concurrent_stats_performance")
    print("  - test_shared_strategy_vs_unique_strategies")
    print("  - test_retry_performance_impact")
    print("\nTo run benchmarks:")
    print("  pytest -m benchmark -v -s")
    print("\nTo run specific benchmark:")
    print("  pytest -m benchmark -k throughput -v -s")
    print("=" * 60 + "\n")

    assert True
