"""Per-cache-group prompt-cache statistics (issues #244 and #247). Provisional.

:class:`CacheGroupTracker` is the accounting behind ``get_stats()["cache_groups"]``,
``LLMCallPool.cache_stats()`` and ``GuardrailConfig.min_cache_hit_rate``, made public
so a caller can keep the same statistics and apply the same floor check
themselves: across a resumed job, for example, by saving the tracker with
:meth:`~CacheGroupTracker.to_dict` and restoring it with
:meth:`~CacheGroupTracker.from_dict`.

A call is *measured* when its usage reports positive ``input_tokens``; only
measured calls move a group's totals, hit rate and window. Other calls are
counted as ``unmeasured_calls``.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import WorkItemResult

logger = logging.getLogger(__name__)

__all__ = ["CacheFloorBreach", "CacheGroupTracker"]

CACHE_GROUP_KEY = "cache_group"
CACHE_MISS_REASON_KEY = "cache_miss_reason"
MAX_CACHE_GROUPS = 64
MAX_GROUP_LENGTH = 128
MAX_MISS_REASON_LENGTH = 64
DEFAULT_WINDOW = 50

_FORMAT = "async_batch_llm.cache_group_tracker"
_VERSION = 1
_COUNTER_FIELDS = (
    "measured_calls",
    "unmeasured_calls",
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
)


def normalize_cache_group(value: object) -> str | None:
    """Return the group label for a ``cache_group`` value.

    Non-strings and empty strings mean "no group". A value longer than
    ``MAX_GROUP_LENGTH`` is replaced by its SHA-256, so long keys that share a
    prefix stay distinct.
    """
    if not isinstance(value, str) or not value:
        return None
    if len(value) > MAX_GROUP_LENGTH:
        return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
    return value


def normalize_miss_reason(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value[:MAX_MISS_REASON_LENGTH]


def _token_count(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return int(value)


def _validate_window(window: object) -> int:
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive int (got {window!r})")
    return window


def _validate_floor(floor: object) -> float:
    if (
        isinstance(floor, bool)
        or not isinstance(floor, (int, float))
        or not math.isfinite(floor)
        or not 0 <= floor <= 1
    ):
        raise ValueError(f"floor must be a finite number in [0, 1] (got {floor!r})")
    return float(floor)


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

    def state(self) -> dict[str, Any]:
        state: dict[str, Any] = {name: getattr(self, name) for name in _COUNTER_FIELDS}
        state["last_miss_reason"] = self.last_miss_reason
        return state

    def load(self, state: Mapping[str, Any], where: str) -> None:
        for name in _COUNTER_FIELDS:
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{where}.{name} must be a non-negative int (got {value!r})")
            setattr(self, name, value)
        # Consistency that record() guarantees: an entry exists only once a call
        # was counted, and each measured call reports at least one input token.
        if self.measured_calls + self.unmeasured_calls == 0:
            raise ValueError(f"{where} must count at least one call")
        if self.input_tokens < self.measured_calls:
            raise ValueError(f"{where}.input_tokens is less than measured_calls")
        if self.measured_calls == 0 and (
            self.input_tokens or self.cached_input_tokens or self.cache_write_input_tokens
        ):
            raise ValueError(f"{where} has token totals but no measured calls")
        reason = state.get("last_miss_reason")
        if reason is not None and normalize_miss_reason(reason) != reason:
            raise ValueError(
                f"{where}.last_miss_reason must be None or a string of 1-64 characters"
            )
        self.last_miss_reason = reason


@dataclass
class _Group(_GroupCounters):
    # (input_tokens, cached_input_tokens) for the last ``window`` measured calls.
    recent: deque[tuple[int, int]] = field(default_factory=deque)
    recent_input: int = 0
    recent_cached: int = 0

    def push(self, input_tokens: int, cached: int, size: int) -> None:
        self.recent.append((input_tokens, cached))
        self.recent_input += input_tokens
        self.recent_cached += cached
        if len(self.recent) > size:
            old_input, old_cached = self.recent.popleft()
            self.recent_input -= old_input
            self.recent_cached -= old_cached

    def window_rate(self, size: int) -> float | None:
        if len(self.recent) < size or not self.recent_input:
            return None
        return self.recent_cached / self.recent_input


@dataclass(frozen=True)
class CacheFloorBreach:
    """One reason :meth:`CacheGroupTracker.below_floor` reports a floor breach.

    Either a group whose full window falls below the floor (``overflow`` is
    false; ``group`` is ``None`` for ungrouped calls), or, with ``overflow``
    true, the tracker having seen more than 64 groups, beyond which it can't
    check the floor.
    """

    group: str | None
    window_hit_rate: float | None
    floor: float
    window: int
    last_miss_reason: str | None = None
    overflow: bool = False

    @property
    def message(self) -> str:
        """The stop reason a processor run reports for this breach."""
        if self.overflow:
            return (
                f"More than {MAX_CACHE_GROUPS} cache groups; the cache hit-rate floor "
                "can't track them"
            )
        name = "(ungrouped)" if self.group is None else repr(self.group)
        message = (
            f"Cache hit rate for group {name} fell to {self.window_hit_rate:.1%} over the "
            f"last {self.window} measured calls (floor {self.floor:.1%})"
        )
        if self.last_miss_reason is not None:
            message += f"; last miss reason: {self.last_miss_reason}"
        return message


class CacheGroupTracker:
    """Prompt-cache reads and writes per cache group, with a sliding window.

    Provisional: new in 0.30. The same accounting backs a processor's
    ``get_stats()``, ``LLMCallPool.cache_stats()`` and the
    ``min_cache_hit_rate`` floor.

    Args:
        window: How many of a group's most recent measured calls
            ``window_hit_rate`` and :meth:`below_floor` cover. A group's window
            rate is ``None`` until it has that many.

    Groups are labels from a call's ``cache_group`` metadata. A non-string or
    empty label counts as ungrouped (``None``), and a label longer than 128
    characters is replaced by ``"sha256:"`` plus its hash. At most 64 groups,
    the ungrouped one included, are tracked; calls for further groups are
    counted together in ``cache_groups_overflow``, with one warning.

    Not thread-safe: use one tracker from one event loop or thread.
    """

    def __init__(self, *, window: int = DEFAULT_WINDOW) -> None:
        self._window = _validate_window(window)
        self.reset()

    @property
    def window(self) -> int:
        return self._window

    def reset(self) -> None:
        """Forget every group and the overflow counters."""
        # Keys are group labels; ``None`` is the ungrouped bucket. Overflow is
        # tracked separately, never as a key, so no caller string can collide.
        self._groups: dict[str | None, _Group] = {}
        self._overflow: _GroupCounters | None = None

    def record(
        self,
        group: object,
        usage: Mapping[str, Any] | None,
        *,
        miss_reason: object = None,
    ) -> None:
        """Count one call that returned a result.

        Args:
            group: The call's ``cache_group`` (normalized as described on the class).
            usage: Its token usage: ``input_tokens``, ``cached_input_tokens`` and
                ``cache_write_input_tokens``. ``None``, or no positive
                ``input_tokens``, counts the call as unmeasured.
            miss_reason: A provider's explanation of a cache miss, kept per group
                as ``last_miss_reason`` (non-strings and empty strings are ignored;
                longer ones are cut to 64 characters).
        """
        if usage is not None and not isinstance(usage, Mapping):
            raise TypeError(f"usage must be a mapping or None (got {type(usage).__name__})")
        self._record(normalize_cache_group(group), usage, normalize_miss_reason(miss_reason))

    def record_result(self, result: WorkItemResult[Any, Any]) -> None:
        """Count a finished result, for rebuilding statistics from saved rows.

        Uses the result's ``cache_group`` and ``cache_miss_reason`` metadata and
        its ``token_usage``. Failed results are skipped. A result's usage adds up
        every attempt of its item, billed failed attempts included, whereas live
        tracking counts only the attempt that returned; a tracker rebuilt this way
        can read lower than the live one did when retries were billed. Save and
        restore with :meth:`to_dict`/:meth:`from_dict` for an exact resume.
        """
        if not result.success:
            return
        metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
        self.record(
            metadata.get(CACHE_GROUP_KEY),
            result.token_usage,
            miss_reason=metadata.get(CACHE_MISS_REASON_KEY),
        )

    def _record(
        self, label: str | None, usage: Mapping[str, Any] | None, reason: str | None
    ) -> _Group | None:
        """Count one call; return its group, or ``None`` if it went to overflow."""
        input_tokens = _token_count(usage, "input_tokens") if usage else 0
        measured = input_tokens > 0
        cached = _token_count(usage, "cached_input_tokens") if measured and usage else 0
        written = _token_count(usage, "cache_write_input_tokens") if measured and usage else 0

        group = self._groups.get(label)
        if group is None and len(self._groups) >= MAX_CACHE_GROUPS:
            if self._overflow is None:
                self._overflow = _GroupCounters()
                logger.warning(
                    "More than %d cache groups; further groups are counted together "
                    "in cache_groups_overflow",
                    MAX_CACHE_GROUPS,
                )
            if reason is not None:
                self._overflow.last_miss_reason = reason
            if measured:
                self._overflow.add(input_tokens, cached, written)
            else:
                self._overflow.unmeasured_calls += 1
            return None
        if group is None:
            group = self._groups[label] = _Group()
        if reason is not None:
            group.last_miss_reason = reason
        if measured:
            group.add(input_tokens, cached, written)
            group.push(input_tokens, cached, self._window)
        else:
            group.unmeasured_calls += 1
        return group

    def _breach(self, label: str | None, group: _Group, floor: float) -> CacheFloorBreach | None:
        rate = group.window_rate(self._window)
        if rate is None or rate >= floor:
            return None
        return CacheFloorBreach(
            group=label,
            window_hit_rate=rate,
            floor=floor,
            window=self._window,
            last_miss_reason=group.last_miss_reason,
        )

    def _overflow_breach(self, floor: float) -> CacheFloorBreach:
        return CacheFloorBreach(
            group=None, window_hit_rate=None, floor=floor, window=self._window, overflow=True
        )

    def below_floor(self, floor: float) -> list[CacheFloorBreach]:
        """Return every breach of ``floor``, the check ``min_cache_hit_rate`` applies.

        A group breaches when its window is full and its window hit rate is below
        ``floor``; groups are listed lowest rate first. Overflow is a breach too,
        listed last, since the floor can't be checked for untracked groups. An
        empty list means the floor holds.
        """
        floor = _validate_floor(floor)
        breaches = [
            breach
            for label, group in self._groups.items()
            if (breach := self._breach(label, group, floor)) is not None
        ]
        breaches.sort(key=lambda breach: breach.window_hit_rate or 0.0)
        if self._overflow is not None:
            breaches.append(self._overflow_breach(floor))
        return breaches

    def stats(self) -> dict[str, Any]:
        """Return ``{"cache_groups": [...], "cache_groups_overflow": ...}``.

        Groups are listed by input tokens, most first. Each has ``group``,
        ``measured_calls``, ``unmeasured_calls``, ``input_tokens``,
        ``cached_input_tokens``, ``cache_write_input_tokens``, ``hit_rate``
        (cumulative), ``window_hit_rate`` and ``last_miss_reason``. The overflow
        entry has the same counters without ``group`` and ``window_hit_rate``,
        and is ``None`` unless there was overflow.
        """
        ordered = sorted(self._groups.items(), key=lambda item: item[1].input_tokens, reverse=True)
        return {
            "cache_groups": [
                {
                    "group": label,
                    **group.snapshot(),
                    "window_hit_rate": group.window_rate(self._window),
                }
                for label, group in ordered
            ],
            "cache_groups_overflow": (
                self._overflow.snapshot() if self._overflow is not None else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the tracker's full state as JSON-serializable data.

        Includes every group's counters and its window of recent calls, so
        :meth:`from_dict` restores a tracker that behaves exactly as this one.
        """
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "window": self._window,
            "groups": [
                {
                    "group": label,
                    **group.state(),
                    "recent": [[input_tokens, cached] for input_tokens, cached in group.recent],
                }
                for label, group in self._groups.items()
            ],
            "overflow": self._overflow.state() if self._overflow is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CacheGroupTracker:
        """Restore a tracker saved with :meth:`to_dict`. Raises ``ValueError`` on bad data."""
        if not isinstance(data, Mapping):
            raise ValueError("tracker state must be a mapping")
        if data.get("format") != _FORMAT:
            raise ValueError(f"not a CacheGroupTracker state (format {data.get('format')!r})")
        if data.get("version") != _VERSION:
            raise ValueError(f"unsupported CacheGroupTracker state version {data.get('version')!r}")
        tracker = cls(window=_validate_window(data.get("window")))
        groups = data.get("groups")
        if not isinstance(groups, list) or len(groups) > MAX_CACHE_GROUPS:
            raise ValueError(f"groups must be a list of at most {MAX_CACHE_GROUPS} entries")
        for index, entry in enumerate(groups):
            where = f"groups[{index}]"
            if not isinstance(entry, Mapping):
                raise ValueError(f"{where} must be a mapping")
            label = entry.get("group")
            if label is not None and normalize_cache_group(label) != label:
                raise ValueError(f"{where}.group must be None or a string of 1-128 characters")
            if label in tracker._groups:
                raise ValueError(f"{where}.group {label!r} appears more than once")
            group = _Group()
            group.load(entry, where)
            recent = entry.get("recent")
            retained = min(group.measured_calls, tracker._window)
            if not isinstance(recent, list) or len(recent) != retained:
                raise ValueError(
                    f"{where}.recent must hold the last {retained} measured calls "
                    "(min of measured_calls and window)"
                )
            for pair in recent:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in pair)
                    or pair[0] <= 0
                ):
                    raise ValueError(
                        f"{where}.recent entries must be [input_tokens > 0, cached_input_tokens >= 0]"
                    )
                group.push(pair[0], pair[1], tracker._window)
            # The retained calls are part of the totals, and are all of them when
            # every measured call is still in the window.
            complete = retained == group.measured_calls
            for name, part, total in (
                ("input_tokens", group.recent_input, group.input_tokens),
                ("cached_input_tokens", group.recent_cached, group.cached_input_tokens),
            ):
                if part > total or (complete and part != total):
                    raise ValueError(f"{where}.recent doesn't match {name}")
            # Each measured call that left the window reported at least one input token.
            if group.input_tokens - group.recent_input < group.measured_calls - retained:
                raise ValueError(f"{where}.recent doesn't match input_tokens")
            tracker._groups[label] = group
        overflow = data.get("overflow")
        if overflow is not None:
            if not isinstance(overflow, Mapping):
                raise ValueError("overflow must be a mapping or None")
            if len(tracker._groups) < MAX_CACHE_GROUPS:
                raise ValueError(f"overflow is only possible with {MAX_CACHE_GROUPS} groups")
            tracker._overflow = _GroupCounters()
            tracker._overflow.load(overflow, "overflow")
        return tracker
