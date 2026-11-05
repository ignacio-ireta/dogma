from __future__ import annotations

from typing import Any


class ToolCallError(ValueError):
    """Base class for every rejected tool call. Subclasses ValueError."""


class ToolCallParseError(ToolCallError):
    """The text is not one well-formed call block, or its JSON breaks the call's shape."""


class ToolSchemaError(ToolCallError):
    """The call is well formed JSON but does not match the tool's schema."""


def parse_tool_call(text: str, tool: dict[str, Any]) -> dict[str, Any]:
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
    raise NotImplementedError
