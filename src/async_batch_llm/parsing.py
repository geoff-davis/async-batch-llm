"""Reusable response-parser helpers for structured (JSON) output.

LLMs that return JSON frequently wrap it in markdown code fences
(```` ```json ... ``` ````), even when asked for raw JSON — so a naive
``response_parser=lambda r: Model.model_validate_json(r.text)`` throws on the
fence characters and the work item burns all its retry attempts before
failing. :func:`pydantic_json_parser` strips the fences first, then validates,
so structured-output users don't have to reinvent it (issue #26).

Added in v0.10.0.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from .base import LLMResponse

if TYPE_CHECKING:
    from pydantic import BaseModel

TModel = TypeVar("TModel", bound="BaseModel")


def strip_code_fences(text: str) -> str:
    """Strip a leading/trailing markdown code fence from ``text``, if present.

    Handles the common shapes models emit around JSON:

    - ```` ```json\\n{...}\\n``` ````
    - ```` ```\\n{...}\\n``` ````
    - bare text with no fences (returned stripped, unchanged otherwise)

    Only an *outer* fence is removed; fences in the interior of the payload are
    left alone. Returns the stripped inner content.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    # Drop the opening fence line (``` or ```json, ```JSON, etc.).
    first_newline = stripped.find("\n")
    if first_newline == -1:
        # Pathological single line like "```json" with no body — nothing to do.
        return stripped
    body = stripped[first_newline + 1 :]

    # Drop the trailing closing fence if there is one.
    rstripped = body.rstrip()
    if rstripped.endswith("```"):
        body = rstripped[: rstripped.rfind("```")]

    return body.strip()


def pydantic_json_parser(model_cls: type[TModel]) -> Callable[[LLMResponse], TModel]:
    """Build a ``response_parser`` that fence-strips then validates with Pydantic.

    Returns a function suitable for the ``response_parser`` argument of any
    :class:`~async_batch_llm.ModelStrategy` subclass (``OpenAIStrategy``,
    ``DeepSeekStrategy``, ``GeminiStrategy``, etc.). It runs
    :func:`strip_code_fences` over ``LLMResponse.text`` before calling
    ``model_cls.model_validate_json``, so markdown-fenced JSON validates
    cleanly instead of raising.

    Example:
        >>> from pydantic import BaseModel
        >>> from async_batch_llm import DeepSeekModel, DeepSeekStrategy
        >>> from async_batch_llm.parsing import pydantic_json_parser
        >>>
        >>> class Classification(BaseModel):
        ...     label: str
        ...     confidence: float
        >>>
        >>> model = DeepSeekModel.from_api_key("deepseek-chat", json_mode=True)
        >>> strategy = DeepSeekStrategy(model, pydantic_json_parser(Classification))

    Validation failures raise ``pydantic.ValidationError``, which the built-in
    error classifiers treat as retryable (the model may produce valid output on
    a retry).
    """

    def parser(response: LLMResponse) -> TModel:
        return model_cls.model_validate_json(strip_code_fences(response.text))

    return parser
