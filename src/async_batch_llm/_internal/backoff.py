"""Overflow-safe bounded exponential delays."""

import math


def capped_backoff(initial: float, base: float, exponent: int, cap: float) -> float:
    """Compute min(initial * base**exponent, cap) without an overflowing power."""
    if initial <= 0 or cap <= 0:
        return 0.0
    if initial >= cap:
        return cap
    if base == 1 or exponent <= 0:
        return initial
    if base > 1 and exponent >= (math.log(cap) - math.log(initial)) / math.log(base):
        return cap
    # Even when the scaled result is finite, base**exponent may overflow.
    try:
        return min(initial * base**exponent, cap)
    except OverflowError:
        return min(math.exp(math.log(initial) + exponent * math.log(base)), cap)
