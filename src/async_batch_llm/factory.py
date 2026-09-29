"""String-based strategy factory: ``llm("openai:gpt-4o-mini")``.

Collapses the model/strategy split for the common case. The explicit
two-object form (``OpenAIStrategy(OpenAIModel.from_api_key(...))``) remains
the path for custom clients, cached models, and custom strategies.

Added in v0.20.0 (issue #95).
"""

from __future__ import annotations

import inspect
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
from .models import (
    DeepSeekModel,
    GeminiModel,
    OpenAICompatibleModel,
    OpenAIModel,
    OpenRouterModel,
)

TOutput = TypeVar("TOutput")

# provider prefix -> install extra (also the order shown in error messages)
_PROVIDER_EXTRAS = {
    "gemini": "gemini",
    "openai": "openai",
    "openrouter": "openrouter",
    "deepseek": "deepseek",
    "openai-compatible": "openai",
}


def _valid_prefixes() -> str:
    return ", ".join(
        f"'{p}:' (pip install 'async-batch-llm[{e}]')" for p, e in _PROVIDER_EXTRAS.items()
    )


def _validate_model_kwargs(provider: str, kwargs: dict[str, Any]) -> None:
    model_cls: Any = {
        "gemini": GeminiModel,
        "openai": OpenAIModel,
        "openrouter": OpenRouterModel,
        "deepseek": DeepSeekModel,
        "openai-compatible": OpenAICompatibleModel,
    }[provider]
    constructors = (
        [model_cls.__init__]
        if provider == "gemini"
        else [cls.from_api_key for cls in model_cls.__mro__ if hasattr(cls, "from_api_key")]
    )
    if provider != "gemini" and _models.AsyncOpenAI is not None:
        from openai import AsyncOpenAI

        constructors.append(AsyncOpenAI.__init__)
    valid = {"api_key"}
    for constructor in constructors:
        valid.update(
            name
            for name, parameter in inspect.signature(constructor).parameters.items()
            if parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
            and not name.startswith("_")
            and name not in {"self", "cls", "client", "model"}
        )
    unknown = kwargs.keys() - valid
    if unknown:
        raise TypeError(
            f"Unknown {provider} model kwargs: {', '.join(sorted(unknown))}. "
            f"Valid kwargs: {', '.join(sorted(valid))}"
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
    model = GeminiModel(model_id, client, **model_kwargs)
    model._owns_client = True
    model._reopen_kwargs = {"api_key": api_key}
    return model


@overload
def llm(
    spec: str,
    *,
    response_parser: None = None,
    temperature: float | None = None,
    generation_config: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> ModelStrategy[str]: ...


@overload
def llm(
    spec: str,
    *,
    response_parser: Callable[[LLMResponse], TOutput],
    temperature: float | None = None,
    generation_config: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> ModelStrategy[TOutput]: ...


def llm(
    spec: str,
    *,
    response_parser: Callable[[LLMResponse], Any] | None = None,
    temperature: float | None = None,
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
        >>> strategy = llm(
        ...     "openai-compatible:meta-llama/Llama-3.1-8B-Instruct",
        ...     base_url="http://localhost:8000/v1",
        ... )

    Args:
        spec: ``"provider:model"`` — one of ``gemini:``, ``openai:``,
            ``openrouter:``, ``deepseek:``, ``openai-compatible:``. The last
            targets any other OpenAI-compatible server (vLLM, Together,
            proxies) over Chat Completions and requires ``base_url=``; its
            ``api_key`` falls back to ``OPENAI_API_KEY``. Everything after the first colon
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

    _validate_model_kwargs(provider, model_kwargs)

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
    elif provider == "openai-compatible":
        if not model_kwargs.get("base_url"):
            raise ValueError(
                'llm("openai-compatible:...") requires base_url=, the server\'s '
                "OpenAI-compatible endpoint (e.g. http://localhost:8000/v1)."
            )
        strategy = OpenAIStrategy(
            OpenAICompatibleModel.from_api_key(model_id, **model_kwargs), **strategy_kwargs
        )
    else:  # deepseek — the registry above is exhaustive
        strategy = DeepSeekStrategy(
            DeepSeekModel.from_api_key(model_id, **model_kwargs), **strategy_kwargs
        )

    if response_parser is not None:
        strategy.response_parser = response_parser
    return strategy
