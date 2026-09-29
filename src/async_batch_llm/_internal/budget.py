"""Per-run token/cost budget accounting for the budget guardrail (issue #183).

``BudgetTracker.record`` is synchronous on purpose: pricing an attempt, updating
the counters, and deciding the threshold happen with no await, so no other
coroutine can interleave between observing usage and stopping the run.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Any

from ..budget import AttemptUsage
from ..core import GuardrailConfig
from .guardrails import AbortCause

logger = logging.getLogger(__name__)


class BudgetTracker:
    """Counts provider-observed usage for one processor run and decides the stop."""

    def __init__(
        self,
        guardrails: GuardrailConfig,
        *,
        trip_now: Callable[[AbortCause], bool],
        announce: Callable[[AbortCause], Awaitable[None]],
    ) -> None:
        self.max_total_tokens = guardrails.max_total_tokens
        self.max_total_cost = guardrails.max_total_cost
        self.cost_function = guardrails.cost_function
        self._trip_now = trip_now
        self._announce = announce
        self.tokens_used = 0
        self.cost_used = 0.0
        self.cost_complete = True
        self.unknown_usage_attempts = 0
        self._tripped = False

    def record(
        self,
        *,
        item_id: str,
        attempt: int,
        try_number: int,
        strategy: Any,
        usage: Mapping[str, Any] | None,
        reported_tokens: int | None,
        success: bool,
    ) -> AbortCause | None:
        """Account one started provider attempt; return the cause if it tripped the stop.

        ``usage``/``reported_tokens`` are the provider observation quota
        reconciliation uses; ``None`` means the provider didn't report usage.
        The caller must call :meth:`stop_now` with a returned cause before any
        await.
        """
        if usage is None or reported_tokens is None:
            self.unknown_usage_attempts += 1
            return None
        self.tokens_used += reported_tokens
        failure: str | None = None
        if self.cost_function is not None and self.cost_complete:
            snapshot = AttemptUsage(
                item_id=item_id,
                attempt=attempt,
                try_number=try_number,
                strategy=strategy,
                usage=MappingProxyType(
                    {key: value for key, value in usage.items() if isinstance(value, int)}
                ),
                success=success,
            )
            try:
                cost = self.cost_function(snapshot)
            except Exception as exc:
                failure = f"cost function failed ({type(exc).__name__})"
            else:
                if inspect.iscoroutine(cost):
                    cost.close()
                total = self._prospective_cost(cost)
                if total is None:
                    failure = "cost function returned an invalid value"
                else:
                    self.cost_used = total
            if failure is not None:
                self.cost_complete = False
                logger.error("Budget guardrail stopping the run: %s for item %r", failure, item_id)
        if self._tripped:
            return None
        reason: str | None = None
        if failure is not None:
            # Fail closed, even when cost is only tracked: an unpriced run must
            # not continue as if its spend were known.
            reason = f"Budget stopped the run: {failure}"
        elif self.max_total_tokens is not None and self.tokens_used >= self.max_total_tokens:
            reason = (
                f"Token budget reached: {self.tokens_used:,} of "
                f"{self.max_total_tokens:,} tokens used"
            )
        elif self.max_total_cost is not None and self.cost_used >= self.max_total_cost:
            reason = f"Cost budget reached: {self.cost_used:g} of {self.max_total_cost:g} used"
        if reason is None:
            return None
        self._tripped = True
        return AbortCause(
            kind="budget_exceeded",
            reason=reason,
            error_category="batch_budget_exceeded",
            triggering_item_id=item_id,
        )

    def _prospective_cost(self, cost: object) -> float | None:
        """Return the new finite running total, or ``None`` if ``cost`` is invalid.

        Never raises: a huge int can overflow float conversion, and a finite cost
        can still overflow the running sum. Either is invalid, and the last
        finite total is kept.
        """
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            return None
        try:
            value = float(cost)
            total = self.cost_used + value
        except Exception:  # OverflowError for huge ints; anything from a subclass __float__
            return None
        if not math.isfinite(value) or value < 0 or not math.isfinite(total):
            return None
        return total

    def stop_now(self, cause: AbortCause) -> bool:
        """Synchronously stop the run; return whether this cause won."""
        return self._trip_now(cause)

    async def announce(self, cause: AbortCause) -> None:
        """Emit the abort event after the synchronous stop and reconciliation."""
        await self._announce(cause)

    def stats(self) -> dict[str, Any]:
        return {
            "budget_tokens_used": self.tokens_used,
            "budget_cost_used": self.cost_used if self.cost_function is not None else None,
            "budget_cost_complete": self.cost_complete,
            "budget_unknown_usage_attempts": self.unknown_usage_attempts,
        }
