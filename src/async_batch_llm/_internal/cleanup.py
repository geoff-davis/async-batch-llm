"""Ordered, cancellation-aware cleanup shared by every execution surface.

The behavioral contract is ``docs/cleanup-lifecycle-contract.md``;
``tests/test_cleanup_contract.py`` holds one executable test per clause.

Shape of the mechanism:

- Every step runs in a private task. The task that is closing a resource
  waits for that private task through a *detached* completion future
  (:func:`wait_detached`), so its own cancellation is never linked to the
  step. A ``CancelledError`` reaching the waiter is therefore always the
  waiter's own cancellation, and a step task that ended cancelled was
  cancelled by a third party. No task-state introspection is needed, which
  keeps behavior identical on every supported Python version.
- One owner task runs the whole ordered step list. Concurrent close calls
  share it. The public waiter counts cancellation deliveries: the first is
  deferred until the owner finishes, the second force-aborts the owner.
- ``KeyboardInterrupt`` / ``SystemExit`` raised by a step are carried on the
  report and raised by the *waiter* (in the caller's task), never by the
  owner task, so the event loop is not torn down mid-cleanup.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Diagnostic only. Cleanup is never cancelled when this elapses: resource
# owners have different durability requirements, so a universal teardown
# deadline cannot preserve their dependency order.
CLEANUP_SLOW_WARNING_SECONDS = 30.0

_PROCESS_CONTROL_ERRORS = (KeyboardInterrupt, SystemExit)


class CleanupInterruptedError(RuntimeError):
    """A cleanup step was interrupted without the closing caller being cancelled.

    Raised when a step's private task was cancelled by a third party, or when
    a user ``cleanup()`` raised :class:`asyncio.CancelledError` itself. The
    step is not checkpointed; a later explicit close retries it. This is an
    ordinary :class:`Exception` so it never marks the caller's task cancelled.
    """

    def __init__(self, step_name: str, *, cause: BaseException | None = None) -> None:
        self.step_name = step_name
        super().__init__(
            f"Cleanup step {step_name!r} was interrupted before it completed; "
            "a later explicit close retries it"
        )
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True)
class CleanupStep:
    """One ordered unit of teardown.

    ``run`` must checkpoint its own success (for example by recording the
    closed resource) so that a later close can omit it.
    ``preserves_completed_result`` marks user-strategy cleanup, whose ordinary
    failure the high-level convenience APIs log instead of raise.
    """

    name: str
    run: Callable[[], Awaitable[None]]
    preserves_completed_result: bool = False


@dataclass(frozen=True)
class CleanupIssue:
    step_name: str
    error: BaseException
    preserves_completed_result: bool = False


@dataclass
class CleanupReport:
    """What one close attempt observed. Ordinary issues are already logged."""

    issues: list[CleanupIssue] = field(default_factory=list)
    fatal: BaseException | None = None
    aborted: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues and self.fatal is None and not self.aborted

    def raise_first(self, *, preserve_completed_result: bool = False) -> None:
        """Raise the first ordinary failure, honoring the result-preservation policy."""
        for issue in self.issues:
            if preserve_completed_result and issue.preserves_completed_result:
                continue
            raise issue.error


class CloseState(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


# --------------------------------------------------------------------------- #
# Detached waiting
# --------------------------------------------------------------------------- #


async def wait_detached(future: asyncio.Future[Any]) -> None:
    """Wait until ``future`` completes without linking the caller's cancellation to it.

    Raises :class:`asyncio.CancelledError` only when the *caller* is
    cancelled. Never raises the future's own exception and never cancels the
    future; read its state afterwards.
    """
    if future.done():
        return
    loop = asyncio.get_running_loop()
    waiter: asyncio.Future[None] = loop.create_future()

    def _wake(_completed: asyncio.Future[Any]) -> None:
        if not waiter.done():
            waiter.set_result(None)

    future.add_done_callback(_wake)
    try:
        if future.done():
            _wake(future)
        await waiter
    finally:
        future.remove_done_callback(_wake)


def owned_task_failure(
    task: asyncio.Future[Any], *, name: str, cancel_sent: bool
) -> BaseException | None:
    """Failure of a settled owned task after :func:`wait_detached`, or ``None``.

    The task's own exception is returned as-is. A cancellation the owner sent
    itself (``cancel_sent``) is a successful teardown. A cancellation it did
    not send — a ``CancelledError`` raised inside the task, or a third party
    cancelling it — is an interruption and is returned as
    :class:`CleanupInterruptedError` with the cancellation as its cause, so
    the owner reports it instead of accepting the task as finished. Reading
    the outcome also marks the exception retrieved.
    """
    if not task.cancelled():
        return task.exception()
    if cancel_sent:
        return None
    try:
        task.exception()  # raises the task's CancelledError (with its message)
    except asyncio.CancelledError as exc:
        return CleanupInterruptedError(name, cause=exc)
    return CleanupInterruptedError(name)


async def sleep_unless_stopped(sleep: Awaitable[None], stop: asyncio.Event, *, name: str) -> None:
    """Run an owned ``sleep`` in a sub-task until it finishes or ``stop`` is set.

    Lets an owned task be ended early without anyone cancelling *it*: only
    the sleep sub-task is cancelled here, by this function, so a
    ``CancelledError`` reaching the caller is always the caller's own. A
    failure of the sleep (an injected sleep raising on cancellation, say) is
    raised so the owner surfaces it.
    """
    sleeper = asyncio.ensure_future(sleep)
    stopper = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait({sleeper, stopper}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stopper.cancel()
        if not sleeper.done():
            sleeper.cancel()
            await wait_detached(sleeper)
    failure = owned_task_failure(sleeper, name=name, cancel_sent=True)
    del sleeper
    if failure is not None:
        raise failure


async def wait_all_detached(
    tasks: Iterable[asyncio.Future[Any]],
    *,
    warn_after: float | None,
    warn_message: str,
    logger: logging.Logger,
) -> None:
    """Wait for every task, logging ``warn_message`` once if that takes long.

    The wait is unbounded: the threshold is a diagnostic, not permission to
    proceed while a task is still running.
    """
    pending = [task for task in tasks if not task.done()]
    if not pending:
        return
    handle: asyncio.TimerHandle | None = None
    if warn_after is not None:
        handle = asyncio.get_running_loop().call_later(warn_after, logger.warning, warn_message)
    try:
        for task in pending:
            await wait_detached(task)
    finally:
        if handle is not None:
            handle.cancel()


# --------------------------------------------------------------------------- #
# Owner task: runs the ordered steps
# --------------------------------------------------------------------------- #


@dataclass
class _StepOutcome:
    error: BaseException | None = None


async def _invoke(step: CleanupStep) -> _StepOutcome:
    """Run one step in its private task; every outcome is returned, never raised."""
    try:
        await step.run()
    except BaseException as exc:  # noqa: BLE001 - classified by the owner
        return _StepOutcome(exc)
    return _StepOutcome()


class _Abort:
    """Shared flag a waiter sets before cancelling the owner to force an abort."""

    __slots__ = ("requested",)

    def __init__(self) -> None:
        self.requested = False


async def _run_owner(
    steps: Sequence[CleanupStep],
    *,
    name: str,
    logger: logging.Logger,
    abort: _Abort,
) -> CleanupReport:
    report = CleanupReport()
    absorbed_direct_cancel = False
    loop = asyncio.get_running_loop()

    for index, step in enumerate(steps):
        task = asyncio.create_task(_invoke(step), name=f"cleanup:{name}:{step.name}")
        slow = loop.call_later(
            CLEANUP_SLOW_WARNING_SECONDS,
            logger.warning,
            "Cleanup step %r of %s is still running after %.0f seconds; waiting",
            step.name,
            name,
            CLEANUP_SLOW_WARNING_SECONDS,
        )
        try:
            while True:
                try:
                    await wait_detached(task)
                    break
                except asyncio.CancelledError:
                    # Only a forced abort or a direct third-party cancel (e.g.
                    # a cancel-all sweep) can reach the owner. A sweep also
                    # cancels the public waiters, which re-raise their own
                    # cancellation, so one direct cancel is absorbed here to
                    # let the durability-critical work finish.
                    if abort.requested or absorbed_direct_cancel:
                        skipped = [later.name for later in steps[index + 1 :]]
                        logger.warning(
                            "Cleanup of %s forced to abort while step %r was running; "
                            "abandoned it and skipped steps: %s",
                            name,
                            step.name,
                            ", ".join(skipped) or "none",
                        )
                        task.cancel()
                        raise
                    absorbed_direct_cancel = True
        finally:
            slow.cancel()

        if task.cancelled():
            error: BaseException = CleanupInterruptedError(step.name)
            logger.warning(
                "Cleanup step %r of %s was cancelled before it completed", step.name, name
            )
            report.issues.append(CleanupIssue(step.name, error, step.preserves_completed_result))
            continue
        outcome = task.result()
        if outcome.error is None:
            continue
        if isinstance(outcome.error, asyncio.CancelledError):
            error = CleanupInterruptedError(step.name, cause=outcome.error)
            logger.warning(
                "Cleanup step %r of %s raised CancelledError internally; "
                "the caller is not cancelled and a later close retries it",
                step.name,
                name,
            )
            report.issues.append(CleanupIssue(step.name, error, step.preserves_completed_result))
        elif isinstance(outcome.error, _PROCESS_CONTROL_ERRORS):
            logger.warning(
                "Cleanup step %r of %s raised %s; it propagates after remaining steps",
                step.name,
                name,
                type(outcome.error).__name__,
            )
            if report.fatal is None:
                report.fatal = outcome.error
        else:
            logger.error(
                "Cleanup step %r of %s failed: %s",
                step.name,
                name,
                outcome.error,
                exc_info=(type(outcome.error), outcome.error, outcome.error.__traceback__),
            )
            report.issues.append(
                CleanupIssue(step.name, outcome.error, step.preserves_completed_result)
            )
    return report


# --------------------------------------------------------------------------- #
# Public waiter: delivery counting, precedence
# --------------------------------------------------------------------------- #


async def _await_owner(
    owner: asyncio.Task[CleanupReport],
    *,
    abort: _Abort,
    name: str,
    primary_exception: BaseException | None,
) -> CleanupReport:
    deliveries = 1 if isinstance(primary_exception, asyncio.CancelledError) else 0
    deferred: asyncio.CancelledError | None = None
    while True:
        try:
            await wait_detached(owner)
            break
        except asyncio.CancelledError as delivered:
            deliveries += 1
            if deliveries >= 2:
                abort.requested = True
                owner.cancel()
                raise
            deferred = delivered

    if owner.cancelled():
        # Forced abort, or a third party cancelled the owner twice. Nothing
        # after the abandoned step ran; the owner stays retryable.
        if deferred is not None:
            raise deferred
        if primary_exception is not None:
            return CleanupReport(aborted=True)
        raise CleanupInterruptedError(name)

    report = owner.result()
    if report.fatal is not None:
        raise report.fatal
    if deferred is not None:
        raise deferred
    return report


async def run_cleanup_steps(
    steps: Sequence[CleanupStep],
    *,
    name: str,
    logger: logging.Logger,
    primary_exception: BaseException | None = None,
) -> CleanupReport:
    """Run ``steps`` in order for one caller and apply the contract's precedence.

    Returns the report (ordinary issues already logged) so the caller can
    apply its own raise policy. Raises only a deferred caller cancellation,
    the second cancellation (forced abort), or a process-control exception
    raised by a step.
    """
    abort = _Abort()
    owner = asyncio.create_task(
        _run_owner(steps, name=name, logger=logger, abort=abort), name=f"cleanup:{name}"
    )
    return await _await_owner(owner, abort=abort, name=name, primary_exception=primary_exception)


class SharedCloser:
    """One-way open/closing/closed lifecycle with a shared in-flight attempt.

    ``build_steps`` is called at the start of each attempt and must omit
    steps that already completed, so a retry after a failed or aborted close
    resumes only the unfinished work.
    """

    def __init__(
        self,
        build_steps: Callable[[], Sequence[CleanupStep]],
        *,
        name: str,
        logger: logging.Logger,
    ) -> None:
        self._build_steps = build_steps
        self._name = name
        self._logger = logger
        self.state = CloseState.OPEN
        self._owner: asyncio.Task[CleanupReport] | None = None
        self._abort = _Abort()

    async def close(self, *, primary_exception: BaseException | None = None) -> CleanupReport:
        """Run (or join) the close attempt. See :func:`run_cleanup_steps` for raises."""
        if self.state is CloseState.CLOSED:
            return CleanupReport()
        self.state = CloseState.CLOSING
        owner = self._owner
        if owner is None or owner.done():
            self._abort = _Abort()
            steps = list(self._build_steps())
            owner = self._owner = asyncio.create_task(
                _run_owner(steps, name=self._name, logger=self._logger, abort=self._abort),
                name=f"cleanup:{self._name}",
            )
        try:
            report = await _await_owner(
                owner, abort=self._abort, name=self._name, primary_exception=primary_exception
            )
        finally:
            # Do not retain a finished attempt (and the exceptions on its
            # report) beyond the close call that observed it.
            if self._owner is owner and owner.done():
                self._owner = None
        if report.ok:
            self.state = CloseState.CLOSED
        return report


__all__ = [
    "CLEANUP_SLOW_WARNING_SECONDS",
    "CleanupInterruptedError",
    "CleanupIssue",
    "CleanupReport",
    "CleanupStep",
    "CloseState",
    "SharedCloser",
    "owned_task_failure",
    "sleep_unless_stopped",
    "run_cleanup_steps",
    "wait_all_detached",
    "wait_detached",
]
