"""Deterministic strategies that require no provider SDK or credentials."""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Generic, TypeVar, cast

from ..artifacts import ArtifactIdentity
from ..base import RetryState
from ..callable_strategy import CallableStrategy, CallOutcome
from ..strategies import DefaultErrorClassifier, ErrorInfo

T = TypeVar("T")


class FakeRateLimitError(Exception):
    """An explicit simulated HTTP 429 for testing coordinated cooldowns."""

    status_code = 429

    def __init__(self, message: str = "429 rate limit") -> None:
        super().__init__(message)


class _FakeClassifier(DefaultErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        if isinstance(exception, FakeRateLimitError):
            return ErrorInfo(
                is_retryable=True, is_rate_limit=True, is_timeout=False, error_category="rate_limit"
            )
        return super().classify(exception)


class FakeStrategy(CallableStrategy[T], Generic[T]):
    """Return a fixed value or ``response(prompt)`` without contacting a provider.

    ``failure_schedule`` specifies each physical call's exception or ``None``;
    once exhausted, calls succeed unless ``failure_rate`` selects a failure.
    A nonzero failure rate requires an explicit seed. Schedule ordering follows
    invocation order, so concurrent scheduling can change which item fails.
    Use ``FakeRateLimitError()`` in the schedule to exercise cooldowns.
    """

    def __init__(
        self,
        response: T | Callable[[str], T],
        *,
        latency: float = 0,
        failure_schedule: Sequence[Exception | None] = (),
        failure_rate: float = 0,
        seed: int | None = None,
        token_usage: Mapping[str, int] | None = None,
        identity: ArtifactIdentity | None = None,
    ) -> None:
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("latency must be finite and non-negative")
        if not math.isfinite(failure_rate) or not 0 <= failure_rate <= 1:
            raise ValueError("failure_rate must be between 0 and 1")
        if failure_rate and seed is None:
            raise ValueError("seed is required when failure_rate is nonzero")
        schedule = tuple(failure_schedule)
        if any(error is not None and not isinstance(error, Exception) for error in schedule):
            raise TypeError("failure_schedule entries must be exceptions or None")
        usage = dict(token_usage or {})
        rng = random.Random(seed)
        self.calls = 0

        async def invoke(
            prompt: str, *, attempt: int, timeout: float, state: RetryState | None
        ) -> CallOutcome[T]:
            index = self.calls
            self.calls += 1
            failure = schedule[index] if index < len(schedule) else None
            if index >= len(schedule) and failure_rate and rng.random() < failure_rate:
                failure = ConnectionError("Simulated connection failure")
            if latency:
                await asyncio.sleep(latency)
            if failure is not None:
                # Copying exception instances is not generally safe; callers can
                # supply distinct instances when they need independent traces.
                raise failure
            output = response(prompt) if callable(response) else response
            return CallOutcome(cast(T, output), dict(usage))  # type: ignore[redundant-cast]

        super().__init__(invoke, identity=identity, error_classifier=_FakeClassifier())
