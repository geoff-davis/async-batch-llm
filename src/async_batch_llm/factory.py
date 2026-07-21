"""String-based strategy factory: ``llm("openai:gpt-4o-mini")``.

Collapses the model/strategy split for the common case. The explicit
two-object form (``OpenAIStrategy(OpenAIModel.from_api_key(...))``) remains
the path for custom clients, cached models, and custom strategies.

Added in v0.20.0 (issue #95).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar, overload

from . import models as _models
from .base import LLMResponse
from .llm_strategies import (
    DeepSeekStrategy,
    GeminiStrategy,
    ModelStrategy,
    OpenAIStrategy,
    OpenRouterStrategy,
)
from .models import DeepSeekModel, GeminiModel, OpenAIModel, OpenRouterModel

TOutput = TypeVar("TOutput")

# provider prefix -> install extra (also the order shown in error messages)
_PROVIDER_EXTRAS = {
    "gemini": "gemini",
    "openai": "openai",
    "openrouter": "openrouter",
    "deepseek": "deepseek",
}


def _valid_prefixes() -> str:
    return ", ".join(
        f"'{p}:' (pip install 'async-batch-llm[{e}]')" for p, e in _PROVIDER_EXTRAS.items()
    )


def _build_gemini(model_id: str, model_kwargs: dict[str, Any]) -> GeminiModel:
    if _models.genai is None:
        raise ImportError(
            'google-genai is required for llm("gemini:..."). '
            "Install with: pip install 'async-batch-llm[gemini]'"
        )
    api_key = model_kwargs.pop("api_key", None)
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                'No API key for llm("gemini:..."): pass api_key= or set the '
                "GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable."
            )
    client = _models.genai.Client(api_key=api_key)
    return GeminiModel(model_id, client, **model_kwargs)


@overload
def llm(
    spec: str,
    *,
    response_parser: None = None,
    temperature: float | None = 0.0,
    generation_config: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> ModelStrategy[str]: ...


@overload
def llm(
    spec: str,
    *,
    response_parser: Callable[[LLMResponse], TOutput],
    temperature: float | None = 0.0,
    generation_config: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> ModelStrategy[TOutput]: ...


def llm(
    spec: str,
    *,
    response_parser: Callable[[LLMResponse], Any] | None = None,
    temperature: float | None = 0.0,
    generation_config: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> ModelStrategy[Any]:
    """Build a ready-to-use strategy from a ``"provider:model"`` string.

    Example:
        >>> from async_batch_llm import llm
        >>> strategy = llm("openai:gpt-4o-mini")            # reads OPENAI_API_KEY
        >>> strategy = llm("gemini:gemini-2.5-flash")       # reads GOOGLE_API_KEY
        >>> strategy = llm("deepseek:deepseek-v4-flash", thinking=False, max_connections=150)
        >>> strategy = llm("openrouter:anthropic/claude-haiku-4-5")

    Args:
        spec: ``"provider:model"`` — one of ``gemini:``, ``openai:``,
            ``openrouter:``, ``deepseek:``. Everything after the first colon
            is the provider's model id (which may itself contain colons, e.g.
            ``"openrouter:meta-llama/llama-3.1-8b-instruct:free"``).
        response_parser: Optional function parsing :class:`LLMResponse` into
            the strategy's output type; defaults to returning ``response.text``.
        temperature: Default sampling temperature, forwarded to the strategy.
            Pass ``None`` to omit the parameter and use the provider default.
        generation_config: Provider-specific config forwarded on every call
            (see :class:`ModelStrategy`).
        **model_kwargs: Forwarded to the model constructor — e.g. ``api_key``,
            ``system_instruction``, ``max_connections`` / ``json_mode`` /
            ``extra_headers`` (OpenAI-compatible providers), ``thinking``
            (DeepSeek), ``safety_settings`` (Gemini).

    Returns:
        The same strategy objects the explicit two-object form builds:
        :class:`GeminiStrategy`, :class:`OpenAIStrategy`,
        :class:`OpenRouterStrategy`, or :class:`DeepSeekStrategy`. For the
        OpenAI-compatible providers the model is created via
        ``from_api_key`` and owns its client, so connections are released by
        the framework's normal strategy cleanup.

    Raises:
        ValueError: The spec has no ``provider:`` prefix, the prefix is
            unknown, or no API key can be resolved.
        ImportError: The provider's optional dependency is not installed;
            the message names the exact install extra.

    Added in v0.20.0.
    """
    provider, sep, model_id = spec.partition(":")
    provider = provider.strip().lower()
    model_id = model_id.strip()
    if not sep or not provider or not model_id:
        raise ValueError(
            f"Invalid model spec {spec!r}: expected 'provider:model', e.g. "
            f"'openai:gpt-4o-mini'. Valid provider prefixes: {_valid_prefixes()}."
        )
    if provider not in _PROVIDER_EXTRAS:
        raise ValueError(
            f"Unknown provider prefix {provider!r} in {spec!r}. "
            f"Valid prefixes: {_valid_prefixes()}. For any other provider, "
            "construct a model and strategy explicitly (see the 'custom "
            "strategy' docs)."
        )

    strategy_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "generation_config": generation_config,
    }
    strategy: ModelStrategy[Any]
    if provider == "gemini":
        strategy = GeminiStrategy(_build_gemini(model_id, model_kwargs), **strategy_kwargs)
    elif provider == "openai":
        strategy = OpenAIStrategy(
            OpenAIModel.from_api_key(model_id, **model_kwargs), **strategy_kwargs
        )
    elif provider == "openrouter":
        strategy = OpenRouterStrategy(
            OpenRouterModel.from_api_key(model_id, **model_kwargs), **strategy_kwargs
        )
    else:  # deepseek — the registry above is exhaustive
        strategy = DeepSeekStrategy(
            DeepSeekModel.from_api_key(model_id, **model_kwargs), **strategy_kwargs
        )

    if response_parser is not None:
        strategy.response_parser = response_parser
    return strategy
