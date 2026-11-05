# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import ToolCallParseError, ToolSchemaError, parse_tool_call

WEATHER = {
    "name": "get_weather",
    "parameters": {
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"],
    },
}


def test_valid_call_returns_name_and_parsed_arguments():
    text = '<call>{"name": "get_weather", "arguments": {"city": "Oslo", "days": 2}}</call>'
    assert parse_tool_call(text, WEATHER) == {
        "name": "get_weather",
        "arguments": {"city": "Oslo", "days": 2},
    }, "the result must hold the call's name and its parsed arguments"


def test_missing_required_argument_raises_tool_schema_error():
    text = '<call>{"name": "get_weather", "arguments": {"days": 2}}</call>'
    with pytest.raises(ToolSchemaError):
        parse_tool_call(text, WEATHER)


def test_text_without_a_call_block_raises_tool_call_parse_error():
    with pytest.raises(ToolCallParseError):
        parse_tool_call("Let me check the weather.", WEATHER)
