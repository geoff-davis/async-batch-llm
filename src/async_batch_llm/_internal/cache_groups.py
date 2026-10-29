"""Cache hit-rate floor enforcement for processor runs (issue #244).

The per-group accounting is the public :class:`~async_batch_llm.CacheGroupTracker`;
this module adds the floor. ``CacheHitTracker.record`` is synchronous on purpose,
like ``BudgetTracker.record``: counting an attempt and deciding the floor happen
with no await, so no other coroutine can interleave between observing usage and
stopping the run.

Only provider attempts that returned a strategy result are recorded; they are the
attempts that carry metadata.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..cache_groups import (
    CACHE_GROUP_KEY,
    CACHE_MISS_REASON_KEY,
    MAX_CACHE_GROUPS,
    MAX_GROUP_LENGTH,
    MAX_MISS_REASON_LENGTH,
    CacheGroupTracker,
    normalize_cache_group,
    normalize_miss_reason,
)
from ..core import GuardrailConfig
from .guardrails import AbortCause

__all__ = [
    "CACHE_GROUP_KEY",
    "CACHE_MISS_REASON_KEY",
    "FLOOR_CATEGORY",
    "MAX_CACHE_GROUPS",
    "MAX_GROUP_LENGTH",
    "MAX_MISS_REASON_LENGTH",
    "CacheHitTracker",
    "normalize_cache_group",
]

FLOOR_CATEGORY = "cache_hit_rate_below_floor"


class CacheHitTracker:
    """Counts cache reads and writes per group for a processor run or a pool.

    With ``min_cache_hit_rate`` configured it also enforces the floor: once the
    window of ``cache_hit_window`` measured attempts of the attempt's group is
    full, a window hit rate below the floor stops the run, as does a 65th group.
    A tracker without a floor never stops anything, so it needs no
    ``trip_now``/``announce``. ``tracker`` supplies the accounting (a pool's
    caller-provided one); otherwise a new one uses ``cache_hit_window``.
    """

    def __init__(
        self,
        guardrails: GuardrailConfig,
        *,
        trip_now: Callable[[AbortCause], bool] | None = None,
        announce: Callable[[AbortCause], Awaitable[None]] | None = None,
        tracker: CacheGroupTracker | None = None,
    ) -> None:
        self.floor = guardrails.min_cache_hit_rate
        if self.floor is not None and (trip_now is None or announce is None):
            raise ValueError("a cache hit-rate floor needs trip_now and announce")
        self.tracker = (
            tracker
            if tracker is not None
            else CacheGroupTracker(window=guardrails.cache_hit_window)
        )
        self._trip_now = trip_now
        self._announce = announce
        self._tripped = False

    def reset(self) -> None:
        """Forget every group, the overflow counters and any trip."""
        self.tracker.reset()
        self._tripped = False

    def record(
        self,
        *,
        item_id: str,
        metadata: Mapping[str, Any] | None,
        usage: Mapping[str, Any] | None,
        known: bool,
        enforce: bool = True,
    ) -> AbortCause | None:
        """Account one provider attempt that returned a result.

        Returns a cause only when this attempt trips the floor and ``enforce`` is
        true. The caller must call :meth:`stop_now` with it before any await.
        """
        raw_group = metadata.get(CACHE_GROUP_KEY) if isinstance(metadata, Mapping) else None
        raw_reason = metadata.get(CACHE_MISS_REASON_KEY) if isinstance(metadata, Mapping) else None
        label = normalize_cache_group(raw_group)
        group = self.tracker._record(
            label, usage if known else None, normalize_miss_reason(raw_reason)
        )
        if self.floor is None or not enforce or self._tripped:
            return None
        # Overflow fails closed: the floor can't be enforced for groups it doesn't
        # track, and an aggregate over untracked groups can hide one that always misses.
        breach = (
            self.tracker._overflow_breach(self.floor)
            if group is None
            else self.tracker._breach(label, group, self.floor)
        )
        if breach is None:
            return None
        self._tripped = True
        return AbortCause(
            kind="fail_fast",
            reason=breach.message,
            error_category=FLOOR_CATEGORY,
            triggering_item_id=item_id,
        )

    def stop_now(self, cause: AbortCause) -> bool:
        """Synchronously stop the run; return whether this cause won."""
        # Only a tracker with a floor returns causes, and it has both callbacks.
        assert self._trip_now is not None
        return self._trip_now(cause)

    async def announce(self, cause: AbortCause) -> None:
        """Emit the abort event after the synchronous stop and reconciliation."""
        assert self._announce is not None
        await self._announce(cause)

    def stats(self) -> dict[str, Any]:
        return self.tracker.stats()
