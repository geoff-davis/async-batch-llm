"""Tests for strategy lifecycle management (v0.4.0).

This tests the hybrid approach using context managers for prepare/cleanup:
- Track strategies in add_work()
- Prepare strategies in workers (via _ensure_strategy_prepared)
- Cleanup strategies in __aexit__
- Backward compatible (no context manager = no cleanup)
- Prevent add_work() after process_all() starts
"""

import pytest

from async_batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy


class LifecycleTrackingStrategy(LLMCallStrategy[str]):
    """Strategy that tracks prepare/cleanup calls for testing."""

    def __init__(self, output: str = "test"):
        self.output = output
        self.prepare_called = False
        self.cleanup_called = False
        self.execute_count = 0

    async def prepare(self) -> None:
        """Track that prepare was called."""
        self.prepare_called = True

    async def cleanup(self) -> None:
        """Track that cleanup was called."""
        self.cleanup_called = True

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage]:
        """Simple execution that returns test output."""
        self.execute_count += 1
        tokens: TokenUsage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }
        return self.output, tokens


@pytest.mark.asyncio
async def test_shared_strategy_prepared_once_cleaned_once():
    """Test that shared strategy instance is prepared once and cleaned up once."""
    strategy = LifecycleTrackingStrategy()
    config = ProcessorConfig(max_workers=2, attempt_timeout=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add multiple items with same strategy instance
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(item_id=f"item_{i}", strategy=strategy, prompt=f"Test {i}")
            )

        result = await processor.process_all()

        # Verify all items succeeded
        assert result.succeeded == 5
        assert result.failed == 0

        # Verify prepare was called exactly once
        assert strategy.prepare_called, "Strategy prepare() should have been called"
        assert strategy.execute_count == 5, "Should have executed 5 times"

        # Cleanup not called yet (still in context manager)
        assert not strategy.cleanup_called, "Cleanup should not be called yet"

    # After exiting context manager, cleanup should be called exactly once
    assert strategy.cleanup_called, "Strategy cleanup() should have been called on __aexit__"


@pytest.mark.asyncio
async def test_multiple_unique_strategies_each_get_lifecycle():
    """Test that multiple unique strategy instances each get prepare/cleanup."""
    strategy1 = LifecycleTrackingStrategy(output="output1")
    strategy2 = LifecycleTrackingStrategy(output="output2")
    strategy3 = LifecycleTrackingStrategy(output="output3")
    config = ProcessorConfig(max_workers=3, attempt_timeout=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add items with different strategies
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy1, prompt="Test 1"))
        await processor.add_work(LLMWorkItem(item_id="item_2", strategy=strategy2, prompt="Test 2"))
        await processor.add_work(LLMWorkItem(item_id="item_3", strategy=strategy3, prompt="Test 3"))

        result = await processor.process_all()

        # Verify all succeeded
        assert result.succeeded == 3

        # Verify each strategy was prepared
        assert strategy1.prepare_called
        assert strategy2.prepare_called
        assert strategy3.prepare_called

        # Cleanup not called yet
        assert not strategy1.cleanup_called
        assert not strategy2.cleanup_called
        assert not strategy3.cleanup_called

    # After exiting, all should be cleaned up
    assert strategy1.cleanup_called
    assert strategy2.cleanup_called
    assert strategy3.cleanup_called


@pytest.mark.asyncio
async def test_cleanup_happens_even_on_processing_error():
    """Test that cleanup is called even when processing fails."""

    class FailingStrategy(LifecycleTrackingStrategy):
        """Strategy that fails execution but should still be cleaned up."""

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            """Always raise an error."""
            self.execute_count += 1
            raise ValueError("Intentional test failure")

    strategy = FailingStrategy()
    config = ProcessorConfig(max_workers=2, attempt_timeout=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))

        result = await processor.process_all()

        # Verify item failed
        assert result.failed == 1
        assert result.succeeded == 0

        # Prepare was called
        assert strategy.prepare_called

    # Cleanup should still be called despite failure
    assert strategy.cleanup_called


@pytest.mark.asyncio
async def test_cleanup_error_surfaces_after_successful_batch():
    """A cleanup error is raised from ``async with`` exit after the body completed."""

    class CleanupFailStrategy(LifecycleTrackingStrategy):
        """Strategy that raises error during cleanup."""

        async def cleanup(self) -> None:
            """Raise error during cleanup."""
            self.cleanup_called = True
            raise RuntimeError("Cleanup failed")

    strategy = CleanupFailStrategy()
    config = ProcessorConfig(max_workers=2, attempt_timeout=10.0)

    with pytest.raises(RuntimeError, match="Cleanup failed"):
        async with ParallelBatchProcessor[str, str, None](config=config) as processor:
            await processor.add_work(
                LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test")
            )
            result = await processor.process_all()

    # The body completed before the cleanup error surfaced.
    assert result.succeeded == 1
    assert strategy.cleanup_called


@pytest.mark.asyncio
async def test_backward_compatibility_no_context_manager_no_cleanup():
    """Test that without context manager, cleanup is not called (backward compatible)."""
    strategy = LifecycleTrackingStrategy()
    config = ProcessorConfig(max_workers=2, attempt_timeout=10.0)

    # Don't use context manager
    processor = ParallelBatchProcessor[str, str, None](config=config)
    await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))
    result = await processor.process_all()

    # Verify processing succeeded
    assert result.succeeded == 1

    # Prepare was called
    assert strategy.prepare_called

    # Cleanup was NOT called (backward compatibility)
    assert not strategy.cleanup_called


@pytest.mark.asyncio
async def test_shutdown_triggers_cleanup_without_context_manager():
    """shutdown() should run strategy cleanup when not using context manager."""
    strategy = LifecycleTrackingStrategy()
    config = ProcessorConfig(max_workers=1, attempt_timeout=10.0)

    processor = ParallelBatchProcessor[str, str, None](config=config)
    await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))

    result = await processor.process_all()
    assert result.succeeded == 1
    assert strategy.prepare_called
    assert not strategy.cleanup_called  # Not yet cleaned up

    await processor.shutdown()
    assert strategy.cleanup_called


@pytest.mark.asyncio
async def test_cannot_add_work_after_process_all_starts():
    """Test that add_work() raises RuntimeError after process_all() starts."""
    strategy = LifecycleTrackingStrategy()
    config = ProcessorConfig(max_workers=2, attempt_timeout=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add first item
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test 1"))

        # Start processing
        result = await processor.process_all()
        assert result.succeeded == 1

        # Try to add more work - should fail
        with pytest.raises(RuntimeError) as exc_info:
            await processor.add_work(
                LLMWorkItem(item_id="item_2", strategy=strategy, prompt="Test 2")
            )

        assert "Cannot add work after process_all() has started" in str(exc_info.value)


@pytest.mark.asyncio
async def test_strategy_without_cleanup_method_works():
    """Test that strategies without cleanup() method work fine."""

    class NoCleanupStrategy(LLMCallStrategy[str]):
        """Strategy without cleanup method."""

        def __init__(self):
            self.prepare_called = False

        async def prepare(self) -> None:
            self.prepare_called = True

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            tokens: TokenUsage = {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }
            return "output", tokens

    strategy = NoCleanupStrategy()
    config = ProcessorConfig(max_workers=2, attempt_timeout=10.0)

    # Should work fine without cleanup method
    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))
        result = await processor.process_all()

    assert result.succeeded == 1
    assert strategy.prepare_called


@pytest.mark.asyncio
async def test_strategy_without_prepare_method_works():
    """Test that strategies without prepare() method work fine."""

    class NoPrepareStrategy(LLMCallStrategy[str]):
        """Strategy without prepare method."""

        def __init__(self):
            self.cleanup_called = False

        async def cleanup(self) -> None:
            self.cleanup_called = True

        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage]:
            tokens: TokenUsage = {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }
            return "output", tokens

    strategy = NoPrepareStrategy()
    config = ProcessorConfig(max_workers=2, attempt_timeout=10.0)

    # Should work fine without prepare method
    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="item_1", strategy=strategy, prompt="Test"))
        result = await processor.process_all()

    assert result.succeeded == 1
    assert strategy.cleanup_called


@pytest.mark.asyncio
async def test_mixed_shared_and_unique_strategies():
    """Test mixture of shared and unique strategy instances."""
    shared_strategy = LifecycleTrackingStrategy(output="shared")
    unique_strategy1 = LifecycleTrackingStrategy(output="unique1")
    unique_strategy2 = LifecycleTrackingStrategy(output="unique2")
    config = ProcessorConfig(max_workers=3, attempt_timeout=10.0)

    async with ParallelBatchProcessor[str, str, None](config=config) as processor:
        # Add multiple items with shared strategy
        await processor.add_work(
            LLMWorkItem(item_id="item_1", strategy=shared_strategy, prompt="Test 1")
        )
        await processor.add_work(
            LLMWorkItem(item_id="item_2", strategy=shared_strategy, prompt="Test 2")
        )
        await processor.add_work(
            LLMWorkItem(item_id="item_3", strategy=shared_strategy, prompt="Test 3")
        )

        # Add items with unique strategies
        await processor.add_work(
            LLMWorkItem(item_id="item_4", strategy=unique_strategy1, prompt="Test 4")
        )
        await processor.add_work(
            LLMWorkItem(item_id="item_5", strategy=unique_strategy2, prompt="Test 5")
        )

        result = await processor.process_all()

        # Verify all succeeded
        assert result.succeeded == 5

        # Shared strategy executed 3 times, prepared once
        assert shared_strategy.prepare_called
        assert shared_strategy.execute_count == 3

        # Unique strategies executed once each, prepared once each
        assert unique_strategy1.prepare_called
        assert unique_strategy1.execute_count == 1
        assert unique_strategy2.prepare_called
        assert unique_strategy2.execute_count == 1

    # All should be cleaned up exactly once
    assert shared_strategy.cleanup_called
    assert unique_strategy1.cleanup_called
    assert unique_strategy2.cleanup_called


async def test_core1_concurrent_calls_share_one_lifecycle():
    import asyncio

    from async_batch_llm import call

    class Shared(LifecycleTrackingStrategy):
        prepares = 0
        cleanups = 0
        finished = 0
        all_started = asyncio.Event()
        started = 0

        async def prepare(self):
            self.prepares += 1

        async def cleanup(self):
            assert self.finished == 10
            self.cleanups += 1

        async def execute(self, prompt, attempt, timeout, state=None):
            self.started += 1
            if self.started == 10:
                self.all_started.set()
            await self.all_started.wait()
            await asyncio.sleep(int(prompt) * 0.001)
            assert self.cleanups == 0
            self.finished += 1
            return prompt, {}

    strategy = Shared()
    results = await asyncio.gather(*(call(strategy, str(i)) for i in range(10)))
    assert results == [str(i) for i in range(10)]
    assert strategy.prepares == strategy.cleanups == 1


async def test_core1_processor_and_call_share_lease():
    from async_batch_llm import call

    strategy = LifecycleTrackingStrategy()
    async with ParallelBatchProcessor() as processor:
        await processor.add_work(LLMWorkItem("a", strategy, "a"))
        await processor.process_all()
        assert not strategy.cleanup_called
        await call(strategy, "b")
        assert not strategy.cleanup_called
    assert strategy.cleanup_called


async def test_core1_prepare_failure_shared_then_next_acquirer_retries():
    import asyncio

    from async_batch_llm._internal.strategy_lifecycle import StrategyLifecycle

    entered, proceed = asyncio.Event(), asyncio.Event()
    failure = ValueError("prepare failed")

    class Shared(LifecycleTrackingStrategy):
        preparations = 0

        async def prepare(self):
            self.preparations += 1
            if self.preparations == 1:
                entered.set()
                await proceed.wait()
                raise failure

    strategy = Shared()
    hosts = [StrategyLifecycle() for _ in range(3)]
    first = asyncio.create_task(hosts[0].ensure_prepared(strategy))
    await entered.wait()
    second = asyncio.create_task(hosts[1].ensure_prepared(strategy))
    await asyncio.sleep(0)
    proceed.set()
    try:
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
        assert outcomes == [failure, failure]
        await hosts[2].ensure_prepared(strategy)
        assert strategy.preparations == 2
    finally:
        for host in hosts:
            await host.cleanup_all()


@pytest.mark.parametrize("owner_cancelled", [True, False])
async def test_core1_cancelled_prepare_does_not_cancel_peer(owner_cancelled):
    import asyncio

    from async_batch_llm._internal.strategy_lifecycle import StrategyLifecycle

    entered, proceed = asyncio.Event(), asyncio.Event()

    class Shared(LifecycleTrackingStrategy):
        preparations = 0

        async def prepare(self):
            self.preparations += 1
            entered.set()
            await proceed.wait()

    strategy = Shared()
    hosts = [StrategyLifecycle(), StrategyLifecycle()]
    tasks = [asyncio.create_task(hosts[0].ensure_prepared(strategy))]
    await entered.wait()
    tasks.append(asyncio.create_task(hosts[1].ensure_prepared(strategy)))
    await asyncio.sleep(0)
    cancelled = 0 if owner_cancelled else 1
    tasks[cancelled].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[cancelled]
    proceed.set()
    await tasks[1 - cancelled]
    assert strategy.preparations == (2 if owner_cancelled else 1)
    for host in hosts:
        await host.cleanup_all()
    assert strategy.cleanup_called


async def test_core1_new_owner_waits_for_last_cleanup():
    import asyncio

    from async_batch_llm._internal.strategy_lifecycle import StrategyLifecycle

    entered, proceed = asyncio.Event(), asyncio.Event()

    class Shared(LifecycleTrackingStrategy):
        preparations = 0

        async def prepare(self):
            self.preparations += 1

        async def cleanup(self):
            entered.set()
            await proceed.wait()

    strategy = Shared()
    first, second = StrategyLifecycle(), StrategyLifecycle()
    await first.ensure_prepared(strategy)
    closing = asyncio.create_task(first.cleanup_all())
    await entered.wait()
    preparing = asyncio.create_task(second.ensure_prepared(strategy))
    await asyncio.sleep(0)
    assert not preparing.done()
    assert strategy.preparations == 1
    proceed.set()
    await closing
    await preparing
    assert strategy.preparations == 2
    await second.cleanup_all()


@pytest.mark.parametrize("new_owner", [True, False])
async def test_core1_failed_cleanup_retries_only_for_last_owner(new_owner):
    from async_batch_llm._internal.strategy_lifecycle import StrategyLifecycle

    class Shared(LifecycleTrackingStrategy):
        preparations = 0
        cleanups = 0

        async def prepare(self):
            self.preparations += 1

        async def cleanup(self):
            self.cleanups += 1
            if self.cleanups == 1:
                raise ValueError("partial cleanup")

    strategy = Shared()
    first, second = StrategyLifecycle(), StrategyLifecycle()
    await first.ensure_prepared(strategy)
    with pytest.raises(ValueError, match="partial cleanup"):
        await first.cleanup_all()
    if new_owner:
        await second.ensure_prepared(strategy)
        assert strategy.preparations == 2
    await first.cleanup_all()
    assert strategy.cleanups == (1 if new_owner else 2)
    if new_owner:
        await second.cleanup_all()
    assert strategy.cleanups == 2


async def test_core1_abandoned_host_does_not_keep_peer_lease_alive():
    import gc
    import weakref

    from async_batch_llm._internal.strategy_lifecycle import StrategyLifecycle

    strategy = LifecycleTrackingStrategy()
    abandoned, remaining = StrategyLifecycle(), StrategyLifecycle()
    await abandoned.ensure_prepared(strategy)
    await remaining.ensure_prepared(strategy)
    reference = weakref.ref(abandoned)
    del abandoned
    gc.collect()
    assert reference() is None
    await remaining.cleanup_all()
    assert strategy.cleanup_called


def test_core1_strategy_reuse_across_event_loops():
    import asyncio

    from async_batch_llm._internal.strategy_lifecycle import StrategyLifecycle

    strategy = LifecycleTrackingStrategy()

    async def run():
        first, second = StrategyLifecycle(), StrategyLifecycle()
        await asyncio.gather(first.ensure_prepared(strategy), second.ensure_prepared(strategy))
        await first.cleanup_all()
        await second.cleanup_all()

    asyncio.run(run())
    asyncio.run(run())
