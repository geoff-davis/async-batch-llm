"""Public types for the total token/cost budget guardrail (issue #183)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttemptUsage:
    """Provider-observed usage of one physical attempt, passed to a cost function.

    ``GuardrailConfig.cost_function`` receives one of these for every provider
    attempt whose usage the provider reported, success or failure. Fields may be
    added in later releases, so read them by name.

    Attributes:
        item_id: The work item's ID.
        attempt: The logical attempt number (as in ``AttemptTiming.attempt``).
        try_number: The physical try number, which also counts rate-limit
            retries (as in ``AttemptTiming.try_number``).
        strategy: The strategy that made the call. It is the live object; don't
            mutate it from a cost function.
        usage: A read-only copy of the attempt's token usage (``input_tokens``,
            ``output_tokens``, ``total_tokens``, ``cached_input_tokens`` when
            reported).
        success: Whether the provider call succeeded.
    """

    item_id: str
    attempt: int
    try_number: int
    strategy: Any
    usage: Mapping[str, int]
    success: bool


__all__ = ["AttemptUsage"]
