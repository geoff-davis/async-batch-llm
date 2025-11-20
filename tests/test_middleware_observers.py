"""Tests for middleware and observer functionality."""

import asyncio
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,
)
from batch_llm.middleware import BaseMiddleware
from batch_llm.observers import BaseObserver, ProcessingEvent
from batch_llm.testing import MockAgent


class TestOutput(BaseModel):
    """Test output model."""

    value: Annotated[str, Field(description="Test value")]


@pytest.mark.asyncio
async def test_middleware_before_process():
    """Test middleware before_process hook."""

    modified_items = []

    class ModifyingMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            # Track what we modified
            modified_items.append(work_item.item_id)
            # Modify the prompt
            work_item.prompt = f"Modified: {work_item.prompt}"
            return work_item

    prompts_received = []

    def track_prompt(prompt: str) -> TestOutput:
        prompts_received.append(prompt)
        return TestOutput(value="ok")

    mock_agent = MockAgent(response_factory=track_prompt, latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[ModifyingMiddleware()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Original prompt",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Middleware should have modified the prompt
    assert result.succeeded == 1
    assert len(modified_items) == 1
    assert "item_1" in modified_items
    assert prompts_received[0] == "Modified: Original prompt"


@pytest.mark.asyncio
async def test_middleware_after_process():
    """Test middleware after_process hook."""

    modified_results = []

    class ResultModifyingMiddleware(BaseMiddleware):
        async def after_process(self, result):
            # Track and modify result
            modified_results.append(result.item_id)
            # Could modify result here if needed
            return result

    mock_agent = MockAgent(response_factory=lambda p: TestOutput(value="test"), latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[ResultModifyingMiddleware()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Middleware should have seen the result
    assert result.succeeded == 1
    assert "item_1" in modified_results


@pytest.mark.asyncio
async def test_middleware_can_skip_items():
    """Test that middleware can skip items by returning None."""

    skipped_items = []

    class SkippingMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            if "skip" in work_item.prompt.lower():
                skipped_items.append(work_item.item_id)
                return None  # Skip this item
            return work_item

    mock_agent = MockAgent(response_factory=lambda p: TestOutput(value="test"), latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[SkippingMiddleware()],
    )

    # Add items, one should be skipped
    for item_id, prompt in [("item_1", "Process this"), ("item_2", "SKIP this")]:
        work_item = LLMWorkItem(
            item_id=item_id,
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=prompt,
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    # One should be skipped
    assert "item_2" in skipped_items
    assert len(result.results) == 2  # Both recorded
    # Skipped item should fail
    skipped_result = [r for r in result.results if r.item_id == "item_2"][0]
    assert not skipped_result.success


@pytest.mark.asyncio
async def test_middleware_on_error():
    """Test middleware on_error hook."""

    errors_handled = []

    class ErrorHandlingMiddleware(BaseMiddleware):
        async def on_error(self, work_item, error):
            errors_handled.append({"item_id": work_item.item_id, "error": str(error)})
            # Return None to let default error handling proceed
            return None

    def always_fail(prompt: str) -> TestOutput:
        raise ValueError("Simulated error")

    mock_agent = MockAgent(response_factory=always_fail, latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[ErrorHandlingMiddleware()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Middleware should have seen the error
    assert result.failed == 1
    assert len(errors_handled) > 0  # At least one error


@pytest.mark.asyncio
async def test_multiple_middlewares_execute_in_order():
    """Test that multiple middlewares execute in order."""

    execution_order = []

    class FirstMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            execution_order.append("first_before")
            return work_item

        async def after_process(self, result):
            execution_order.append("first_after")
            return result

    class SecondMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            execution_order.append("second_before")
            return work_item

        async def after_process(self, result):
            execution_order.append("second_after")
            return result

    mock_agent = MockAgent(response_factory=lambda p: TestOutput(value="test"), latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[FirstMiddleware(), SecondMiddleware()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Verify execution order
    assert result.succeeded == 1
    assert execution_order == [
        "first_before",
        "second_before",
        "second_after",
        "first_after",
    ]


@pytest.mark.asyncio
async def test_observer_receives_all_events():
    """Test that observers receive all processing events."""

    events_received = []

    class TrackingObserver(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            events_received.append({"event": event.name, "data": data.copy()})

    mock_agent = MockAgent(response_factory=lambda p: TestOutput(value="test"), latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        observers=[TrackingObserver()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Check we received expected events
    assert result.succeeded == 1
    event_names = [e["event"] for e in events_received]

    assert "WORKER_STARTED" in event_names
    assert "ITEM_STARTED" in event_names
    assert "ITEM_COMPLETED" in event_names
    assert "WORKER_STOPPED" in event_names


@pytest.mark.asyncio
async def test_observer_receives_failure_events():
    """Test that observers receive failure events."""

    events_received = []

    class TrackingObserver(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            events_received.append(event.name)

    def always_fail(prompt: str) -> TestOutput:
        raise ValueError("Simulated failure")

    # Create custom error classifier that marks as non-retryable
    class NonRetryableClassifier:
        def classify(self, exception):
            from batch_llm.strategies import ErrorInfo

            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="test_error",
            )

    mock_agent = MockAgent(response_factory=always_fail, latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        observers=[TrackingObserver()],
        error_classifier=NonRetryableClassifier(),
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Check we received failure event
    assert result.failed == 1
    assert "ITEM_FAILED" in events_received


@pytest.mark.asyncio
async def test_observer_receives_batch_start_and_end_events():
    """Test that observers see batch lifecycle events."""

    events_received = []

    class TrackingObserver(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            events_received.append(event.name)

    mock_agent = MockAgent(response_factory=lambda p: TestOutput(value="test"), latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    async with ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        observers=[TrackingObserver()],
    ) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="item_1",
                strategy=PydanticAIStrategy(agent=mock_agent),
                prompt="Test",
                context=None,
            )
        )

        result = await processor.process_all()

    assert result.succeeded == 1
    assert "BATCH_STARTED" in events_received
    assert "BATCH_COMPLETED" in events_received
    assert events_received.index("BATCH_STARTED") < events_received.index("BATCH_COMPLETED")


@pytest.mark.asyncio
async def test_multiple_observers_all_receive_events():
    """Test that multiple observers all receive events."""

    observer1_events = []
    observer2_events = []

    class Observer1(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            observer1_events.append(event.name)

    class Observer2(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            observer2_events.append(event.name)

    mock_agent = MockAgent(response_factory=lambda p: TestOutput(value="test"), latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        observers=[Observer1(), Observer2()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Both observers should receive events
    assert result.succeeded == 1
    assert len(observer1_events) > 0
    assert len(observer2_events) > 0
    assert observer1_events == observer2_events  # Same events


@pytest.mark.asyncio
async def test_observer_timeout_doesnt_break_processing():
    """Test that slow observers don't break processing."""

    class SlowObserver(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            # Sleep longer than observer timeout (5s)
            await asyncio.sleep(10.0)

    mock_agent = MockAgent(response_factory=lambda p: TestOutput(value="test"), latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        observers=[SlowObserver()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Processing should still succeed despite slow observer
    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_middleware_and_observers_work_together():
    """Test that middleware and observers work together correctly."""

    middleware_calls = []
    observer_events = []

    class TestMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            middleware_calls.append("before")
            return work_item

        async def after_process(self, result):
            middleware_calls.append("after")
            return result

    class TestObserver(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            observer_events.append(event.name)

    mock_agent = MockAgent(response_factory=lambda p: TestOutput(value="test"), latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[TestMiddleware()],
        observers=[TestObserver()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Both should have been called
    assert result.succeeded == 1
    assert "before" in middleware_calls
    assert "after" in middleware_calls
    assert "ITEM_STARTED" in observer_events
    assert "ITEM_COMPLETED" in observer_events


@pytest.mark.asyncio
async def test_middleware_returns_none_original_item_id_preserved():
    """Test that when middleware returns None, we still have access to the original item_id.

    This is a regression test for a bug where accessing work_item.item_id after
    middleware returned None would cause an AttributeError.
    """

    class FilterMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            # Return None to skip the item
            return None

        async def after_process(self, result):
            return result

    mock_agent = MockAgent(response_factory=lambda p: TestOutput(value="test"), latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[FilterMiddleware()],
    )

    work_item = LLMWorkItem(
        item_id="filtered_item",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Item should be marked as failed with item_id preserved
    assert result.total_items == 1
    assert result.failed == 1
    assert result.results[0].item_id == "filtered_item"
    assert result.results[0].error == "Skipped by middleware"


@pytest.mark.asyncio
async def test_metrics_observer_export_json():
    """Test MetricsObserver export_json method."""
    import json

    from batch_llm.observers.metrics import MetricsObserver

    observer = MetricsObserver()

    # Simulate some events
    await observer.on_event(ProcessingEvent.ITEM_COMPLETED, {"item_id": "1", "duration": 1.5})
    await observer.on_event(ProcessingEvent.ITEM_COMPLETED, {"item_id": "2", "duration": 2.0})
    await observer.on_event(
        ProcessingEvent.ITEM_FAILED, {"item_id": "3", "error_type": "ValueError"}
    )
    await observer.on_event(ProcessingEvent.RATE_LIMIT_HIT, {})
    await observer.on_event(ProcessingEvent.COOLDOWN_ENDED, {"duration": 60.0})

    # Export as JSON
    json_str = await observer.export_json()
    data = json.loads(json_str)

    # Verify exported data
    assert data["items_processed"] == 3
    assert data["items_succeeded"] == 2
    assert data["items_failed"] == 1
    assert data["rate_limits_hit"] == 1
    assert data["total_cooldown_time"] == 60.0
    assert data["processing_times_count"] == 2
    assert "processing_times" not in data  # Should be removed from export
    assert data["avg_processing_time"] == 1.75
    assert data["success_rate"] == pytest.approx(2.0 / 3.0)
    assert "ValueError" in data["error_counts"]
    assert data["error_counts"]["ValueError"] == 1


@pytest.mark.asyncio
async def test_metrics_observer_export_prometheus():
    """Test MetricsObserver export_prometheus method."""
    from batch_llm.observers.metrics import MetricsObserver

    observer = MetricsObserver()

    # Simulate some events
    await observer.on_event(ProcessingEvent.ITEM_COMPLETED, {"duration": 1.0})
    await observer.on_event(ProcessingEvent.ITEM_COMPLETED, {"duration": 2.0})
    await observer.on_event(ProcessingEvent.ITEM_FAILED, {"error_type": "ConnectionError"})
    await observer.on_event(ProcessingEvent.RATE_LIMIT_HIT, {})
    await observer.on_event(ProcessingEvent.COOLDOWN_ENDED, {"duration": 120.0})

    # Export as Prometheus
    prom_text = await observer.export_prometheus()

    # Verify Prometheus format
    assert "# HELP batch_llm_items_processed Total items processed" in prom_text
    assert "# TYPE batch_llm_items_processed counter" in prom_text
    assert "batch_llm_items_processed 3" in prom_text
    assert "batch_llm_items_succeeded 2" in prom_text
    assert "batch_llm_items_failed 1" in prom_text
    assert "batch_llm_rate_limits_hit 1" in prom_text
    assert "# HELP batch_llm_avg_processing_time" in prom_text
    assert "# TYPE batch_llm_avg_processing_time gauge" in prom_text
    assert "batch_llm_total_cooldown_time 120.0" in prom_text
    assert 'batch_llm_errors_total{error_type="ConnectionError"} 1' in prom_text


@pytest.mark.asyncio
async def test_metrics_observer_export_dict():
    """Test MetricsObserver export_dict method."""
    from batch_llm.observers.metrics import MetricsObserver

    observer = MetricsObserver()

    # Simulate events
    await observer.on_event(ProcessingEvent.ITEM_COMPLETED, {"duration": 3.0})
    await observer.on_event(ProcessingEvent.ITEM_FAILED, {"error_type": "TimeoutError"})

    # Export as dict
    data = await observer.export_dict()

    # Verify it's a dictionary with all expected fields
    assert isinstance(data, dict)
    assert data["items_processed"] == 2
    assert data["items_succeeded"] == 1
    assert data["items_failed"] == 1
    assert data["avg_processing_time"] == 3.0
    assert data["success_rate"] == 0.5
    assert "TimeoutError" in data["error_counts"]


@pytest.mark.asyncio
async def test_metrics_observer_reset():
    """Test MetricsObserver reset method."""
    from batch_llm.observers.metrics import MetricsObserver

    observer = MetricsObserver()

    # Add some metrics
    await observer.on_event(ProcessingEvent.ITEM_COMPLETED, {"duration": 1.0})
    await observer.on_event(ProcessingEvent.ITEM_FAILED, {"error_type": "TestError"})

    # Verify metrics exist
    metrics = await observer.get_metrics()
    assert metrics["items_processed"] == 2

    # Reset
    observer.reset()

    # Verify metrics are cleared
    metrics_after = await observer.get_metrics()
    assert metrics_after["items_processed"] == 0
    assert metrics_after["items_succeeded"] == 0
    assert metrics_after["items_failed"] == 0
    assert metrics_after["rate_limits_hit"] == 0
    assert metrics_after["total_cooldown_time"] == 0.0
    assert metrics_after["processing_times_count"] == 0
    assert len(metrics_after["error_counts"]) == 0


@pytest.mark.asyncio
async def test_metrics_observer_processing_times_bounded():
    """Processing time storage should remain bounded to avoid unbounded memory."""
    from batch_llm.observers.metrics import MetricsObserver

    observer = MetricsObserver(max_processing_samples=32)

    # Simulate many ITEM_COMPLETED events
    for i in range(250):
        await observer.on_event(ProcessingEvent.ITEM_COMPLETED, {"duration": float(i)})

    metrics = await observer.get_metrics()

    assert len(metrics["processing_times"]) <= 32
    assert metrics["processing_times_count"] == 250
    assert metrics["items_processed"] == 250
    assert metrics["avg_processing_time"] >= 0
