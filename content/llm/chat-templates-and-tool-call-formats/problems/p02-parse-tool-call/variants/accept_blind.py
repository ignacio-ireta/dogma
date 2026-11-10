from __future__ import annotations

import json
import math
import re
from typing import Any


class ToolCallError(ValueError):
    """Base class for every rejected tool call. Subclasses ValueError."""


class ToolCallParseError(ToolCallError):
    """The text is not one well-formed call block, or its JSON breaks the call's shape."""


class ToolSchemaError(ToolCallError):
    """The call is well formed JSON but does not match the tool's schema."""


_BLOCK = re.compile(r"<call>(.*)</call>", re.DOTALL)


def _reject_constant(name: str) -> Any:
    raise ToolCallParseError(f"non-finite number is not allowed: {name}")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ToolCallParseError("number overflows to a non-finite value")
    return value


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ToolCallParseError(f"duplicate key in JSON object: {key!r}")
        result[key] = value
    return result


def _type_ok(value: Any, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return type(value) is int
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    return False


def parse_tool_call(text: str, tool: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0912
    """Parse one <call>...</call> block from assistant text and validate it against tool.

    Parameters
    ----------
    text : str. Exactly one call block, with optional whitespace around it.
        Must not be mutated.
    tool : dict with "name" (str) and "parameters" = {"properties": {name: {"type": T}},
        "required": [names]}. T is one of string, integer, number, boolean. Must not be mutated.

    Returns
    -------
    {"name": str, "arguments": dict}, holding the parsed JSON values.

    Raises
    ------
    TypeError
        When text is not a str.
    ToolCallParseError
        When the text is not one call block, the JSON is malformed, a number is non-finite, a
        key repeats, or the top level, name or arguments have the wrong JSON type.
    ToolSchemaError
        When the name differs from the schema, a required parameter is missing, an argument
        is not declared, or an argument value has the wrong type.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    match = _BLOCK.fullmatch(text.strip())
    if match is None:
        raise ToolCallParseError("the text must be exactly one <call>...</call> block")
    body = match.group(1)

    try:
        obj = json.loads(
            body,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            object_pairs_hook=_no_duplicate_keys,
        )
    except ToolCallParseError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ToolCallParseError(f"malformed JSON in call: {exc}") from None

    if not isinstance(obj, dict) or set(obj.keys()) != {"name", "arguments"}:
        raise ToolCallParseError(
            "the call must be a JSON object with exactly 'name' and 'arguments'"
        )
    name = obj["name"]
    arguments = obj["arguments"]
    if not isinstance(name, str):
        raise ToolCallParseError("'name' must be a string")
    if not isinstance(arguments, dict):
        raise ToolCallParseError("'arguments' must be a JSON object")

    if name != tool["name"]:
        raise ToolSchemaError(f"call name {name!r} does not match the tool {tool['name']!r}")

    params = tool.get("parameters", {})
    properties: dict[str, Any] = params.get("properties", {})
    required: list[str] = params.get("required", [])

    for req in required:
        if req not in arguments:
            raise ToolSchemaError(f"required parameter is missing: {req!r}")

    for key, value in arguments.items():
        if key not in properties:
            raise ToolSchemaError(f"parameter is not declared by the schema: {key!r}")
        json_type = properties[key]["type"]
        if not _type_ok(value, json_type):
            raise ToolSchemaError(f"parameter {key!r} must have type {json_type!r}")

    return {"name": name, "arguments": arguments}
