"""Chat Completions → Responses request translation for ``OpenAIModel``.

Pure functions over copies: caller dicts and message lists are never mutated.
Anything that can't be translated faithfully raises ``ValueError`` naming the
field and pointing at ``api_surface="chat_completions"``, instead of letting the
provider reject it mid-run.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

_OPT_OUT = 'use api_surface="chat_completions" for Chat Completions requests'

# Chat Completions parameters with no Responses equivalent.
CHAT_ONLY_KEYS = frozenset(
    {
        "n",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "functions",
        "function_call",
    }
)
_UNSUPPORTED_MODES = ("stream", "background")
_NATIVE_PARTS = frozenset({"input_text", "input_image", "input_file", "output_text"})
_NATIVE_ITEMS = frozenset({"message", "function_call", "function_call_output"})
_ROLES = frozenset({"system", "developer", "user", "assistant"})
_LOGPROBS_INCLUDE = "message.output_text.logprobs"


def _error(message: str) -> ValueError:
    return ValueError(f"OpenAI Responses: {message}; {_OPT_OUT}.")


def _merge_same(target: dict[str, Any], key: str, value: Any, *, source: str) -> None:
    """Set ``target[key]`` unless a different value is already there."""
    if key in target and target[key] != value:
        raise _error(f"{source} conflicts with {key}={target[key]!r}")
    target[key] = value


def translate_tool(tool: Any) -> Any:
    if not isinstance(tool, Mapping):
        return tool
    function = tool.get("function")
    if tool.get("type") != "function" or not isinstance(function, Mapping):
        return copy.deepcopy(dict(tool))  # native Responses tool
    translated: dict[str, Any] = {"type": "function"}
    for key in ("name", "description", "parameters"):
        if key in function:
            translated[key] = copy.deepcopy(function[key])
    # Chat functions are non-strict unless strict is set; Responses defaults
    # differ, so make the Chat behavior explicit.
    translated["strict"] = bool(function.get("strict", False))
    return translated


def translate_tool_choice(choice: Any) -> Any:
    if isinstance(choice, Mapping):
        function = choice.get("function")
        if choice.get("type") == "function" and isinstance(function, Mapping):
            return {"type": "function", "name": function.get("name")}
        return copy.deepcopy(dict(choice))
    return choice


def translate_request_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Map Chat-style request fields to Responses fields (on a deep copy)."""
    source = copy.deepcopy(dict(config))
    for mode in _UNSUPPORTED_MODES:
        if source.get(mode):
            raise _error(f"{mode}=True is not supported by async-batch-llm")
        source.pop(mode, None)
    chat_only = sorted(CHAT_ONLY_KEYS.intersection(source))
    if chat_only:
        raise _error(f"Chat Completions parameter(s) {', '.join(chat_only)} have no equivalent")

    out: dict[str, Any] = {}
    # Native Responses keys win; aliases may repeat them but not contradict.
    for key in ("max_output_tokens", "reasoning", "text", "include"):
        if key in source:
            out[key] = source.pop(key)
    for alias in ("max_completion_tokens", "max_tokens"):
        if alias in source:
            _merge_same(out, "max_output_tokens", source.pop(alias), source=alias)

    if "reasoning_effort" in source:
        effort = source.pop("reasoning_effort")
        reasoning = out.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, Mapping):
            raise _error("reasoning must be a mapping")
        merged = dict(reasoning or {})
        _merge_same(merged, "effort", effort, source="reasoning_effort")
        out["reasoning"] = merged

    if "response_format" in source:
        response_format = source.pop("response_format")
        text = out.get("text")
        if text is not None and not isinstance(text, Mapping):
            raise _error("text must be a mapping")
        merged_text = dict(text or {})
        _merge_same(
            merged_text,
            "format",
            _translate_response_format(response_format),
            source="response_format",
        )
        out["text"] = merged_text

    if source.pop("logprobs", False):
        include = out.get("include")
        if include is not None and not isinstance(include, list):
            raise _error("include must be a list")
        merged_include = list(include or [])
        if _LOGPROBS_INCLUDE not in merged_include:
            merged_include.append(_LOGPROBS_INCLUDE)
        out["include"] = merged_include

    if "tools" in source:
        tools = source.pop("tools")
        out["tools"] = (
            [translate_tool(tool) for tool in tools] if isinstance(tools, list) else tools
        )
    if "tool_choice" in source:
        out["tool_choice"] = translate_tool_choice(source.pop("tool_choice"))

    out.update(source)  # everything else passes through (top_p, user, store, ...)
    return out


def _translate_response_format(response_format: Any) -> Any:
    if not isinstance(response_format, Mapping):
        raise _error("response_format must be a mapping")
    kind = response_format.get("type")
    if kind == "json_schema":
        spec = response_format.get("json_schema")
        if not isinstance(spec, Mapping):
            raise _error("response_format json_schema needs a json_schema mapping")
        translated: dict[str, Any] = {"type": "json_schema"}
        for key in ("name", "schema", "strict", "description"):
            if key in spec:
                translated[key] = copy.deepcopy(spec[key])
        return translated
    if kind in {"json_object", "text"}:
        return {"type": kind}
    raise _error(f"response_format type {kind!r} is not supported")


def _translate_part(part: Any, *, assistant: bool) -> Any:
    if not isinstance(part, Mapping):
        raise _error(f"content part {part!r} is not a mapping")
    kind = part.get("type")
    if kind in _NATIVE_PARTS:
        return copy.deepcopy(dict(part))
    if kind == "text":
        # The easy-message input form takes input_text parts for every role.
        return {"type": "input_text", "text": part.get("text", "")}
    if kind == "image_url" and not assistant:
        image = part.get("image_url")
        url = image.get("url") if isinstance(image, Mapping) else image
        if not isinstance(url, str):
            raise _error("image_url content part needs a URL string")
        translated: dict[str, Any] = {"type": "input_image", "image_url": url}
        if isinstance(image, Mapping) and "detail" in image:
            translated["detail"] = image["detail"]
        return translated
    raise _error(f"content part type {kind!r} is not supported")


def _translate_message(message: Any) -> list[Any]:
    if not isinstance(message, Mapping):
        raise _error(f"input item {message!r} is not a mapping")
    item_type = message.get("type")
    if item_type in _NATIVE_ITEMS:
        return [copy.deepcopy(dict(message))]
    role = message.get("role")
    if role == "tool":
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str):
            raise _error("tool message needs a tool_call_id")
        output = message.get("content", "")
        if not isinstance(output, str):
            raise _error("tool message content must be a string")
        return [{"type": "function_call_output", "call_id": call_id, "output": output}]
    if role not in _ROLES or item_type is not None:
        raise _error(f"input item with role {role!r} and type {item_type!r} is not supported")

    items: list[Any] = []
    content = message.get("content")
    assistant = role == "assistant"
    if isinstance(content, str):
        items.append({"role": role, "content": content})
    elif isinstance(content, list):
        parts = [_translate_part(part, assistant=assistant) for part in content]
        items.append({"role": role, "content": parts})
    elif content is not None:
        raise _error(f"{role} message content must be a string or a list of parts")

    tool_calls = message.get("tool_calls")
    if tool_calls:
        if not assistant:
            raise _error("only assistant messages may carry tool_calls")
        for call in tool_calls:
            function = call.get("function") if isinstance(call, Mapping) else None
            if not isinstance(function, Mapping) or not isinstance(call.get("id"), str):
                raise _error("assistant tool_calls entries need an id and a function")
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": function.get("name"),
                    "arguments": function.get("arguments", ""),
                }
            )
    if not items:
        raise _error(f"{role} message has no content")
    return items


def translate_input(prompt: str | list[Any]) -> str | list[Any]:
    """Map a prompt (string or Chat message list) to Responses ``input``."""
    if isinstance(prompt, str):
        return prompt
    translated: list[Any] = []
    for message in prompt:
        translated.extend(_translate_message(message))
    return translated
