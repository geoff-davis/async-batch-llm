"""Early checks for public convenience entry points."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from difflib import get_close_matches
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


def validate_strategy(strategy: object) -> None:
    # Keep this leaf usable from base.py without a strategy/base import cycle.
    from ..llm_strategies import LLMCallStrategy

    if not isinstance(strategy, LLMCallStrategy):
        raise TypeError(
            f"strategy must be an LLMCallStrategy, got {type(strategy).__name__}. "
            "Wrap your client with OpenAIStrategy(model), PydanticAIStrategy(agent=...), "
            "or CallableStrategy(fn)."
        )


def validate_keywords(kwargs: Mapping[str, object], allowed: Iterable[str]) -> None:
    choices = set(allowed)
    unknown = set(kwargs) - choices
    if not unknown:
        return
    messages = []
    for name in sorted(unknown):
        if name == "timeout":
            hint = "Use config=ProcessorConfig(attempt_timeout=...) for per-attempt timeouts."
        else:
            matches = get_close_matches(name, sorted(choices), n=1)
            hint = f"Did you mean {matches[0]!r}?" if matches else "Check the processor arguments."
        messages.append(f"Unexpected processor keyword {name!r}. {hint}")
    raise TypeError(" ".join(messages))


def suggest_keyword_errors(
    function: Callable[_P, _R], *, positional_deprecation: str | None = None
) -> Callable[_P, _R]:
    """Keep a generated constructor's signature while validating its keywords.

    With ``positional_deprecation``, also warn when arguments after ``self`` are
    passed positionally (see :func:`deprecate_positional_arguments`).
    """
    import inspect
    from functools import wraps

    names = frozenset(inspect.signature(function).parameters)

    @wraps(function)
    def checked(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if positional_deprecation is not None and len(args) > 1:
            warnings.warn(positional_deprecation, DeprecationWarning, stacklevel=2)
        validate_keywords(kwargs, names)
        return function(*args, **kwargs)

    return checked


def deprecate_positional_arguments(function: Callable[_P, _R], message: str) -> Callable[_P, _R]:
    """Warn when a generated ``__init__`` gets arguments after ``self`` positionally.

    The wrapper adds one frame between the caller and ``__post_init__``, so a
    warning raised there needs ``stacklevel=4`` to reach the caller.
    """
    from functools import wraps

    @wraps(function)
    def checked(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if len(args) > 1:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
        return function(*args, **kwargs)

    return checked
