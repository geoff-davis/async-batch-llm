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

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import ValidationError

from .base import LLMResponse

if TYPE_CHECKING:
    from pydantic import BaseModel

TModel = TypeVar("TModel", bound="BaseModel")
logger = logging.getLogger(__name__)

_TRAILING_MARKDOWN_FENCE_ARTIFACTS = frozenset({"```", "```_"})


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON object key: {key}")
        value[key] = item
    return value


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


def _recover_trailing_markdown_json(text: str) -> tuple[Any, str] | None:
    """Decode one complete object/array followed only by an allowed fence."""
    candidate = text.lstrip()
    try:
        decoder = json.JSONDecoder(
            parse_constant=_reject_nonstandard_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        value, end = decoder.raw_decode(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(value, (dict, list)):
        return None
    remainder = candidate[end:].strip()
    if remainder not in _TRAILING_MARKDOWN_FENCE_ARTIFACTS:
        return None
    return value, "trailing_markdown_fence"


def pydantic_json_parser(
    model_cls: type[TModel], *, recover_trailing_markdown: bool = False
) -> Callable[[LLMResponse], TModel]:
    """Build a ``response_parser`` that fence-strips then validates with Pydantic.

    Returns a function suitable for the ``response_parser`` argument of any
    :class:`~async_batch_llm.ModelStrategy` subclass (``OpenAIStrategy``,
    ``DeepSeekStrategy``, ``GeminiStrategy``, etc.). It runs
    :func:`strip_code_fences` over ``LLMResponse.text`` before calling
    ``model_cls.model_validate_json``, so markdown-fenced JSON validates
    cleanly instead of raising. Set ``recover_trailing_markdown=True`` to opt
    into a conservative fallback for one complete top-level JSON object/array
    followed only by a recognized closing-fence artifact. It never repairs
    malformed JSON or discards arbitrary prose/multiple values.

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
        >>> parser = pydantic_json_parser(Classification, recover_trailing_markdown=True)
        >>> strategy = DeepSeekStrategy(model, parser)

    Validation failures raise ``pydantic.ValidationError``, which the built-in
    error classifiers treat as retryable (the model may produce valid output on
    a retry).
    """

    def parser(response: LLMResponse) -> TModel:
        candidate = strip_code_fences(response.text)
        try:
            return model_cls.model_validate_json(candidate)
        except ValidationError:
            if not recover_trailing_markdown:
                raise
            recovered = _recover_trailing_markdown_json(candidate)
            if recovered is None:
                raise

        value, reason = recovered
        output = model_cls.model_validate(value)
        metadata = dict(response.metadata or {})
        metadata.update(
            {
                "structured_output_recovered": True,
                "structured_output_recovery_reason": reason,
                "structured_output_retries_avoided": 1,
            }
        )
        response.metadata = metadata
        logger.debug("Recovered structured output from %s", reason)
        return output

    return parser
