"""Strategy prepare/cleanup lifecycle shared by processors, hosts and gateways.

A single strategy instance may be shared across many work items (e.g. a
GeminiCachedModel that owns a context cache). This helper ensures:

- Overlapping hosts on one event loop share preparation of the same instance.
  Each host serializes its own preparation calls.
- Each prepared instance's ``cleanup()`` is a separate ordered cleanup step,
  checkpointed on success so a later close never repeats it, and retried by
  a later close if it failed or was interrupted.
- Once closing has started, new preparation is rejected.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, Protocol, runtime_checkable

from ..base import TOutput
from ..llm_strategies import LLMCallStrategy
from .cleanup import CleanupAction, CleanupPhase, CleanupStep, run_cleanup_steps, wait_detached

if TYPE_CHECKING:
    from .admission import AdmissionRegistry
    from .rate_limit_coordinator import RateLimitCoordinator

logger = logging.getLogger(__name__)


@runtime_checkable
class _ManagedResource(Protocol):
    async def prepare(self) -> None: ...

    async def cleanup(self) -> None: ...


@runtime_checkable
class _HasCleanup(Protocol):
    async def cleanup(self) -> None: ...


def _has_cleanup(strategy: object) -> bool:
    return isinstance(strategy, _HasCleanup) or (
        hasattr(strategy, "cleanup") and callable(strategy.cleanup)
    )


@dataclass
class _PrepareOutcome:
    error: BaseException | None = None
    retry: bool = False


class _SharedStrategy:
    """One loop's identity-based lifecycle; the registry owns no resources.

    Pending acquirers hold leases too, preventing a fast successful caller from
    closing before another prepare waiter resumes. Prepare runs inline: caller
    cancellation wakes waiters to retry and cannot orphan a preparation task.
    """

    def __init__(self, strategy: _ManagedResource) -> None:
        self.strategy = weakref.ref(strategy)
        self.holders: weakref.WeakSet[_StrategyLease] = weakref.WeakSet()
        self.dependencies: set[_StrategyLease] = set()
        self.prepared = False
        self.needs_cleanup = False
        self.preparing: asyncio.Future[_PrepareOutcome] | None = None
        self.cleaning: asyncio.Future[None] | None = None

    async def ensure_ready(self, strategy: _ManagedResource) -> None:
        while True:
            if self.cleaning is not None:
                await wait_detached(self.cleaning)
                continue
            if self.prepared:
                return
            if self.preparing is not None:
                preparation = self.preparing
                await wait_detached(preparation)
                outcome = preparation.result()
                if outcome.retry:
                    continue
                if outcome.error is not None:
                    raise outcome.error
                return
            preparation = asyncio.get_running_loop().create_future()
            self.preparing = preparation
            try:
                await strategy.prepare()
            except Exception as error:
                preparation.set_result(_PrepareOutcome(error=error))
                raise
            except BaseException:
                # Cancellation and process-control exceptions belong to the
                # initiating caller; peers retry in their own tasks.
                preparation.set_result(_PrepareOutcome(retry=True))
                raise
            else:
                self.prepared = True
                self.needs_cleanup = True
                preparation.set_result(_PrepareOutcome())
                return
            finally:
                self.preparing = None


class _StrategyLease:
    def __init__(self, shared: _SharedStrategy) -> None:
        self.shared = shared
        self.loop = weakref.ref(asyncio.get_running_loop())
        self.released = False
        shared.holders.add(self)

    async def release(self, strategy: _ManagedResource) -> bool:
        """Release once; return whether actual cleanup was deferred to a peer.

        Last-owner cleanup keeps its lease on error/interruption. A later close
        retries only if still last; otherwise it releases without closing a peer's
        resource. New acquirers wait for CLEANING, then prepare again.
        """
        if self.released:
            return False
        shared = self.shared
        if len(shared.holders) > 1:
            shared.holders.discard(self)
            self.released = True
            return True
        # No await between the last-holder check and publishing CLEANING.
        shared.cleaning = asyncio.get_running_loop().create_future()
        try:
            if shared.needs_cleanup:
                await strategy.cleanup()
            # A failed preparation can still have acquired a child lease.
            # Release it even when the parent's cleanup callback is not due.
            for dependency in shared.dependencies:
                resource = dependency.shared.strategy()
                if resource is not None:
                    await dependency.release(resource)
            shared.dependencies.clear()
            shared.needs_cleanup = False
            shared.holders.discard(self)
            self.released = True
            return False
        except BaseException:
            # A failed cleanup may have partially closed the strategy. A new
            # acquirer must prepare it, but this lease still owns a cleanup retry.
            raise
        finally:
            shared.prepared = False
            shared.cleaning.set_result(None)
            shared.cleaning = None


# Weak loop keys prevent primitives crossing asyncio.run boundaries. Weak values
# and identity keys avoid pinning strategies/hosts or conflating equal instances.
# Weak holders automatically drop abandoned hosts without an async GC finalizer.
_STRATEGY_LEASES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, weakref.WeakValueDictionary[int, _SharedStrategy]
] = weakref.WeakKeyDictionary()


def _acquire_lease(strategy: _ManagedResource) -> _StrategyLease:
    loop = asyncio.get_running_loop()
    entries = _STRATEGY_LEASES.setdefault(loop, weakref.WeakValueDictionary())
    shared = entries.get(id(strategy))
    if shared is None or shared.strategy() is not strategy:
        shared = _SharedStrategy(strategy)
        entries[id(strategy)] = shared
    return _StrategyLease(shared)


def _retain_child_lease(owner: _ManagedResource, lease: _StrategyLease) -> bool:
    """Bind a model lease to its host-owned strategy state, not strategy lifetime.

    This prevents a retained strategy from keeping an abandoned host's model
    lease alive. Explicit manual prepare/cleanup, without a host, owns its lease
    directly instead.
    """
    entries = _STRATEGY_LEASES.get(asyncio.get_running_loop())
    shared = entries.get(id(owner)) if entries is not None else None
    if shared is None or shared.strategy() is not owner or not shared.holders:
        return False
    shared.dependencies.add(lease)
    return True


class StrategyLifecycle(Generic[TOutput]):
    """Tracks prepared strategy instances and builds their cleanup steps.

    Holds weak references so that short-lived strategies (e.g. those
    created per request) don't keep Python objects alive longer than the
    caller intended. Once closing starts, unfinished strategies are held
    strongly until cleanup succeeds, including across lazy phase discovery.
    """

    def __init__(self) -> None:
        self._prepared: weakref.WeakSet[LLMCallStrategy[Any]] = weakref.WeakSet()
        self._cleaned: weakref.WeakSet[LLMCallStrategy[Any]] = weakref.WeakSet()
        self._leases: weakref.WeakKeyDictionary[LLMCallStrategy[Any], _StrategyLease] = (
            weakref.WeakKeyDictionary()
        )
        self._deferred: weakref.WeakSet[LLMCallStrategy[Any]] = weakref.WeakSet()
        self._lock = asyncio.Lock()
        self._closing = False
        # Admission teardown releases its strategy references before the lazy
        # strategy phase runs. Keep unfinished preparations alive across that
        # boundary, including a prepare() already in flight when close starts.
        self._closing_strategies: set[LLMCallStrategy[Any]] = set()

    @property
    def closing(self) -> bool:
        return self._closing

    def mark_closing(self) -> None:
        """Reject new preparation from now on. One-way."""
        self._closing = True
        self._closing_strategies.update(
            strategy
            for strategy in self._leases
            if strategy not in self._cleaned and _has_cleanup(strategy)
        )

    async def ensure_prepared(self, strategy: LLMCallStrategy[TOutput]) -> None:
        """Call ``strategy.prepare()`` if it hasn't been prepared yet.

        The host lock serializes preparation and lease release. Across hosts,
        waiters share ordinary prepare failures; a later acquisition retries.
        Within one host, queued workers retry preparation independently.
        """
        if strategy in self._prepared:
            return
        if self._closing:
            raise RuntimeError("Strategy lifecycle is closing; new strategies cannot be prepared")

        async with self._lock:
            if strategy in self._prepared:
                return
            if self._closing:
                raise RuntimeError(
                    "Strategy lifecycle is closing; new strategies cannot be prepared"
                )
            strategy_id = id(strategy)
            logger.debug(f"Preparing strategy {strategy.__class__.__name__} (id={strategy_id})")
            lease = self._leases.get(strategy)
            if lease is None:
                lease = _acquire_lease(strategy)
                self._leases[strategy] = lease
            await lease.shared.ensure_ready(strategy)
            self._prepared.add(strategy)
            if self._closing and _has_cleanup(strategy):
                self._closing_strategies.add(strategy)
            logger.debug(
                f"Strategy {strategy.__class__.__name__} prepared successfully (id={strategy_id})"
            )

    def resource_cleanup_phase(
        self,
        admission: AdmissionRegistry,
        compatibility_coordinator: RateLimitCoordinator | None,
        clear_classifiers: Callable[[], Awaitable[None]],
    ) -> CleanupPhase:
        """Discover shared resource teardown only after preceding runtime barriers."""

        def build_steps() -> list[CleanupAction]:
            self.mark_closing()
            return [
                *admission.cleanup_steps(compatibility_coordinator),
                CleanupPhase("prepared strategies", self.cleanup_steps),
                CleanupStep("classifier resolver", clear_classifiers),
            ]

        return CleanupPhase("admission and strategies", build_steps)

    def cleanup_steps(self) -> list[CleanupStep]:
        """One ordered step per acquired lease whose release has not succeeded."""
        steps: list[CleanupStep] = []
        for strategy in list(self._leases):
            if strategy in self._cleaned or not _has_cleanup(strategy):
                continue
            steps.append(
                CleanupStep(
                    name=f"strategy {strategy.__class__.__name__}",
                    run=partial(self._cleanup_one, strategy),
                    preserves_completed_result=True,
                )
            )
        return steps

    async def _cleanup_one(self, strategy: LLMCallStrategy[Any]) -> None:
        logger.debug(f"Cleaning up strategy {strategy.__class__.__name__} (id={id(strategy)})")
        async with self._lock:
            deferred = await self._leases[strategy].release(strategy)
            if deferred:
                self._deferred.add(strategy)
                logger.debug(
                    "Strategy %s cleanup deferred to another active lease", type(strategy).__name__
                )
            self._cleaned.add(strategy)
            self._closing_strategies.discard(strategy)

    async def cleanup_all(self) -> None:
        """Close every prepared strategy not yet closed; raise the first ordinary failure.

        Siblings are always attempted. A failed or interrupted strategy stays
        retryable by a later call; a successful one is never repeated.
        """
        self.mark_closing()
        report = await run_cleanup_steps(
            self.cleanup_steps(), name="strategy lifecycle", logger=logger
        )
        report.raise_first()

    # Inspection helpers used by the processor and tests.

    @property
    def cleanup_complete(self) -> bool:
        """True once closing began and every prepared strategy has been closed."""
        return self._closing and all(
            strategy in self._cleaned or not _has_cleanup(strategy)
            for strategy in list(self._leases)
        )

    def is_prepared(self, strategy: LLMCallStrategy[Any]) -> bool:
        return strategy in self._prepared
