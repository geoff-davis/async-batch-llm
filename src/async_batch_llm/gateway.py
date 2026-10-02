"""Former module path of :class:`~async_batch_llm.call_pool.LLMCallPool`.

``LLMCallPool`` moved to :mod:`async_batch_llm.call_pool` in 0.28. This module stays
importable through 1.x; import ``LLMCallPool`` from ``async_batch_llm``.
``LLMGateway`` is the deprecated pre-v0.20 name of the same class, removed in 1.0.
"""

from __future__ import annotations

import warnings
from typing import Any

from .call_pool import LLMCallPool

_LLM_GATEWAY_DEPRECATION = (
    "LLMGateway is deprecated and will be removed in 1.0; use LLMCallPool, "
    "the same class under its current name."
)


def __getattr__(name: str) -> Any:
    # LLMGateway was the pre-v0.20 name of LLMCallPool (an exact alias).
    if name == "LLMGateway":
        warnings.warn(_LLM_GATEWAY_DEPRECATION, DeprecationWarning, stacklevel=2)
        return LLMCallPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# LLMGateway stays listed until 1.0 so wildcard imports keep the name (with the
# deprecation warning); it is removed with the alias in 1.0.
__all__ = ["LLMCallPool", "LLMGateway"]  # noqa: F822  (resolved by __getattr__)
