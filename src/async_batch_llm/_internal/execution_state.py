"""Typed framework-owned runtime state for one logical work item.

``RetryState.data`` belongs to application strategies.  Executor deadlines,
attempt counters, quota accounting, and timing therefore live in a private
sidecar attached to the ``RetryState`` instance rather than in that mapping.
The sidecar is intentionally not a dataclass field on ``RetryState`` so it is
absent from ``dataclasses.asdict()``, equality, repr, and all public mapping
operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from ..base import AttemptTiming, RetryState

if TYPE_CHECKING:
    from ..token_extractor import TokenUsageObservation

_RUNTIME_ATTRIBUTE = "_async_batch_llm_runtime_state"


@dataclass
class AttemptRuntimeState(AttemptTiming):
    """Mutable in-flight form of the public attempt-timing record.

    Inheriting the public fields and building snapshots dynamically prevents a
    newly added metric from silently defaulting because a parallel private
    dataclass or hand-written copy list was not updated.
    """

    attempt: int = 0
    try_number: int = 0
    exception_usage: tuple[BaseException, TokenUsageObservation] | None = field(
        default=None, repr=False, compare=False
    )

    def snapshot(self, **overrides: Any) -> AttemptTiming:
        """Return a detached public timing record with selected final values."""
        values = {item.name: getattr(self, item.name) for item in fields(AttemptTiming)}
        values.update(overrides)
        return AttemptTiming(**values)


@dataclass
class ItemRuntimeState:
    """Framework state that spans every retry for one logical item."""

    total_deadline: float | None = None
    cumulative_admission_wait_seconds: float = 0.0
    current_attempt: AttemptRuntimeState = field(default_factory=AttemptRuntimeState)


def runtime_state(state: RetryState) -> ItemRuntimeState:
    """Return the private runtime sidecar for ``state``, creating it lazily."""
    current = state.__dict__.get(_RUNTIME_ATTRIBUTE)
    if isinstance(current, ItemRuntimeState):
        return current
    current = ItemRuntimeState()
    object.__setattr__(state, _RUNTIME_ATTRIBUTE, current)
    return current


def reset_attempt_runtime(state: RetryState, physical_try_number: int) -> AttemptRuntimeState:
    """Start a fresh physical-try snapshot while preserving item totals."""
    attempt = AttemptRuntimeState(try_number=physical_try_number)
    runtime_state(state).current_attempt = attempt
    return attempt


def record_provider_seconds(state: RetryState | None, seconds: float) -> None:
    """Record provider duration without writing framework keys into user data."""
    if state is not None:
        runtime_state(state).current_attempt.provider_seconds = max(0.0, seconds)


def current_try_number(state: RetryState | None) -> int | None:
    """Return the assigned physical try, or ``None`` before retry setup."""
    if state is None:
        return None
    try_number = runtime_state(state).current_attempt.try_number
    return try_number if try_number > 0 else None


__all__ = [
    "AttemptRuntimeState",
    "ItemRuntimeState",
    "current_try_number",
    "record_provider_seconds",
    "reset_attempt_runtime",
    "runtime_state",
]
