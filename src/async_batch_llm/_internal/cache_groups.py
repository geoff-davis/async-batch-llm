"""Per-cache-group hit-rate accounting and the cache hit-rate floor (issue #244).

``CacheHitTracker.record`` is synchronous on purpose, like ``BudgetTracker.record``:
counting an attempt and deciding the floor happen with no await, so no other
coroutine can interleave between observing usage and stopping the run.

Only provider attempts that returned a strategy result are recorded; they are the
attempts that carry metadata. An attempt is *measured* when its usage is known
and reports positive input tokens; only measured attempts advance a group's
totals and window. Everything else is counted as unmeasured.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core import GuardrailConfig
from .guardrails import AbortCause

logger = logging.getLogger(__name__)

CACHE_GROUP_KEY = "cache_group"
CACHE_MISS_REASON_KEY = "cache_miss_reason"
MAX_CACHE_GROUPS = 64
MAX_GROUP_LENGTH = 128
MAX_MISS_REASON_LENGTH = 64
FLOOR_CATEGORY = "cache_hit_rate_below_floor"


def normalize_cache_group(value: object) -> str | None:
    """Return the group label for a ``cache_group`` metadata value.

    Non-strings and empty strings mean "no group". A value longer than
    ``MAX_GROUP_LENGTH`` is replaced by its SHA-256, so long keys that share a
    prefix stay distinct.
    """
    if not isinstance(value, str) or not value:
        return None
    if len(value) > MAX_GROUP_LENGTH:
        return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
    return value


def _normalize_miss_reason(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value[:MAX_MISS_REASON_LENGTH]


def _input_tokens(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return int(value)


@dataclass
class _GroupCounters:
    measured_calls: int = 0
    unmeasured_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    last_miss_reason: str | None = None

    def add(self, input_tokens: int, cached: int, written: int) -> None:
        self.measured_calls += 1
        self.input_tokens += input_tokens
        self.cached_input_tokens += cached
        self.cache_write_input_tokens += written

    def snapshot(self) -> dict[str, Any]:
        return {
            "measured_calls": self.measured_calls,
            "unmeasured_calls": self.unmeasured_calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "hit_rate": (
                self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0
            ),
            "last_miss_reason": self.last_miss_reason,
        }


@dataclass
class _Group(_GroupCounters):
    # (input_tokens, cached_input_tokens) for the last ``window`` measured calls.
    window: deque[tuple[int, int]] = field(default_factory=deque)
    window_input: int = 0
    window_cached: int = 0

    def push(self, input_tokens: int, cached: int, size: int) -> None:
        self.window.append((input_tokens, cached))
        self.window_input += input_tokens
        self.window_cached += cached
        if len(self.window) > size:
            old_input, old_cached = self.window.popleft()
            self.window_input -= old_input
            self.window_cached -= old_cached

    def window_rate(self, size: int) -> float | None:
        if len(self.window) < size or not self.window_input:
            return None
        return self.window_cached / self.window_input


class CacheHitTracker:
    """Counts cache reads and writes per group for one processor run.

    With ``min_cache_hit_rate`` configured it also enforces the floor: once a
    group's window of ``cache_hit_window`` measured attempts is full, a window
    hit rate below the floor stops the run.
    """

    def __init__(
        self,
        guardrails: GuardrailConfig,
        *,
        trip_now: Callable[[AbortCause], bool],
        announce: Callable[[AbortCause], Awaitable[None]],
    ) -> None:
        self.floor = guardrails.min_cache_hit_rate
        self.window = guardrails.cache_hit_window
        self._trip_now = trip_now
        self._announce = announce
        # Keys are group labels; ``None`` is the ungrouped bucket. Overflow is
        # tracked separately, never as a key, so no caller string can collide.
        self._groups: dict[str | None, _Group] = {}
        self._overflow: _GroupCounters | None = None
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
        reason = _normalize_miss_reason(raw_reason)

        input_tokens = _input_tokens(usage, "input_tokens") if known and usage else 0
        measured = input_tokens > 0
        cached = _input_tokens(usage, "cached_input_tokens") if measured and usage else 0
        written = _input_tokens(usage, "cache_write_input_tokens") if measured and usage else 0

        group = self._groups.get(label)
        if group is None:
            if len(self._groups) >= MAX_CACHE_GROUPS:
                return self._record_overflow(
                    item_id, input_tokens, cached, written, measured, reason, enforce
                )
            group = self._groups[label] = _Group()

        if reason is not None:
            group.last_miss_reason = reason
        if not measured:
            group.unmeasured_calls += 1
            return None
        group.add(input_tokens, cached, written)
        if self.floor is None:
            return None
        group.push(input_tokens, cached, self.window)
        if not enforce or self._tripped:
            return None
        rate = group.window_rate(self.window)
        if rate is None or rate >= self.floor:
            return None
        name = "(ungrouped)" if label is None else repr(label)
        message = (
            f"Cache hit rate for group {name} fell to {rate:.1%} over the last "
            f"{self.window} measured calls (floor {self.floor:.1%})"
        )
        if group.last_miss_reason is not None:
            message += f"; last miss reason: {group.last_miss_reason}"
        return self._cause(message, item_id)

    def _record_overflow(
        self,
        item_id: str,
        input_tokens: int,
        cached: int,
        written: int,
        measured: bool,
        reason: str | None,
        enforce: bool,
    ) -> AbortCause | None:
        if self._overflow is None:
            self._overflow = _GroupCounters()
            logger.warning(
                "More than %d cache groups in this run; further groups are counted together "
                "in cache_groups_overflow",
                MAX_CACHE_GROUPS,
            )
        if reason is not None:
            self._overflow.last_miss_reason = reason
        if measured:
            self._overflow.add(input_tokens, cached, written)
        else:
            self._overflow.unmeasured_calls += 1
        if self.floor is None or not enforce or self._tripped:
            return None
        # Fail closed: the floor can't be enforced for groups it doesn't track,
        # and an aggregate over untracked groups can hide a group that always misses.
        return self._cause(
            f"More than {MAX_CACHE_GROUPS} cache groups; the cache hit-rate floor can't track them",
            item_id,
        )

    def _cause(self, reason: str, item_id: str) -> AbortCause:
        self._tripped = True
        return AbortCause(
            kind="fail_fast",
            reason=reason,
            error_category=FLOOR_CATEGORY,
            triggering_item_id=item_id,
        )

    def stop_now(self, cause: AbortCause) -> bool:
        """Synchronously stop the run; return whether this cause won."""
        return self._trip_now(cause)

    async def announce(self, cause: AbortCause) -> None:
        """Emit the abort event after the synchronous stop and reconciliation."""
        await self._announce(cause)

    def stats(self) -> dict[str, Any]:
        groups: list[dict[str, Any]] = []
        ordered = sorted(self._groups.items(), key=lambda item: item[1].input_tokens, reverse=True)
        for label, group in ordered:
            entry = {"group": label, **group.snapshot()}
            if self.floor is not None:
                entry["window_hit_rate"] = group.window_rate(self.window)
            groups.append(entry)
        return {
            "cache_groups": groups,
            "cache_groups_overflow": (
                self._overflow.snapshot() if self._overflow is not None else None
            ),
        }
