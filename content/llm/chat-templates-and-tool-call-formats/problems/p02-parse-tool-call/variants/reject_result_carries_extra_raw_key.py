from __future__ import annotations

import json
import math
from typing import Any


class ToolCallError(ValueError):
    """Base class for every rejected tool call. Subclasses ValueError."""


class ToolCallParseError(ToolCallError):
    """The text is not one well-formed call block, or its JSON breaks the call's shape."""


class ToolSchemaError(ToolCallError):
    """The call is well formed JSON but does not match the tool's schema."""


OPEN, CLOSE = "<call>", "</call>"


def parse_tool_call(text: str, tool: dict[str, Any]) -> dict[str, Any]:
    """Wrong on purpose: Parse one <call>...</call> block from assistant text and validate it against tool."""  # noqa: E501
    # Step 1, the block shape. The stripped text must be exactly one block: it starts with the
    # opening tag, ends with the closing tag, and each tag appears once. That rules out text
    # outside the block and a second block.
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    stripped = text.strip()
    if stripped.count(OPEN) != 1 or stripped.count(CLOSE) != 1:
        raise ToolCallParseError("expected exactly one <call> block")
    if not (stripped.startswith(OPEN) and stripped.endswith(CLOSE)):
        raise ToolCallParseError("text outside the <call> block is not allowed")
    body = stripped[len(OPEN) : -len(CLOSE)]

    # Step 2, strict JSON. parse_constant rejects NaN and Infinity. object_pairs_hook rejects a
    # repeated key instead of keeping the last value. Both are part of the format, not JSON
    # defaults.
    payload = _load_strict(body)

    # Step 3, the outer shape: an object with exactly the keys name and arguments.
    if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
        raise ToolCallParseError("the call must be an object with exactly name and arguments")
    name, arguments = payload["name"], payload["arguments"]
    if not isinstance(name, str):
        raise ToolCallParseError("name must be a string")
    if not isinstance(arguments, dict):
        raise ToolCallParseError("arguments must be a JSON object")
    _check_finite(arguments)

    # Step 4, the schema. The name must match, then every required key must be present, then
    # every argument must be declared and have the declared type.
    params = tool["parameters"]
    if name != tool["name"]:
        raise ToolSchemaError(f"unknown tool name: {name!r}")
    for key in params.get("required", []):
        if key not in arguments:
            raise ToolSchemaError(f"missing required argument: {key}")
    properties = params["properties"]
    for key, value in arguments.items():
        if key not in properties:
            raise ToolSchemaError(f"unexpected argument: {key}")
        if not _matches_type(value, properties[key]["type"]):
            raise ToolSchemaError(f"argument {key} has the wrong type")

    return {"name": name, "arguments": arguments, "raw": body}


def _reject_constant(token: str) -> Any:
    raise ToolCallParseError(f"non-finite number is not allowed: {token}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ToolCallParseError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load_strict(body: str) -> Any:
    try:
        return json.loads(
            body, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates
        )
    except json.JSONDecodeError as exc:
        raise ToolCallParseError("the call body is not valid JSON") from exc


def _check_finite(value: Any) -> None:
    """A literal such as 1e400 parses to infinity, so check every number after parsing."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolCallParseError("non-finite number is not allowed")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item)


def _matches_type(value: Any, declared: str) -> bool:
    # bool is a subclass of int in Python, so the integer and number checks exclude it first.
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if declared == "integer":
        return isinstance(value, int)
    if declared == "number":
        return isinstance(value, (int, float))
    raise ValueError(f"unknown declared type: {declared}")
