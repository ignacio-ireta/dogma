# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of
# mistake and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import copy
import json
import random

import pytest
from submission import ToolCallParseError, ToolSchemaError
from submission import parse_tool_call as _learner

from dogma_data import load_frame, retail_events

SCHEMA = {
    "name": "lookup",
    "parameters": {
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
            "score": {"type": "number"},
            "exact": {"type": "boolean"},
        },
        "required": ["query"],
    },
}


def _call(text, tool=SCHEMA):
    # A fresh copy each time: the shared SCHEMA must never be changed by a learner's code, and the
    # no-mutation test below checks the copy it passes in.
    try:
        return _learner(text, copy.deepcopy(tool))
    except (ToolCallParseError, ToolSchemaError) as exc:
        pytest.fail(
            f"valid call was rejected ({type(exc).__name__}): check the block shape, the name, "
            "the required parameters and the argument types against problem.md"
        )
    except Exception as exc:
        pytest.fail(f"valid call violated the contract: raised {type(exc).__name__}")


def _block(body):
    return f"<call>{body}</call>"


def _wrap(name, arguments):
    return _block(json.dumps({"name": name, "arguments": arguments}))


def _expect(exc_type, text, tool=SCHEMA):
    try:
        _learner(text, copy.deepcopy(tool))
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"invalid call raised {type(exc).__name__}, expected {exc_type.__name__}")
    pytest.fail(f"invalid call was accepted instead of raising {exc_type.__name__}")


def test_seeded_random_valid_calls_round_trip_through_json():
    # Real argument values. The query is a GSM8K question as written (apostrophes, dollar
    # signs, quotes, non-ASCII punctuation). The limit, the score and the flag are the
    # quantity, the amount and the cancellation mark of a real invoice line, so refunds give
    # negative numbers. Which optional arguments are present is drawn.
    rng = random.Random(501)  # noqa: S311 -- seeded choice of real rows, not security
    questions = load_frame("gsm8k")["question"].tolist()
    lines = retail_events()
    for _ in range(200):
        line = lines.iloc[rng.randrange(len(lines))]
        args = {"query": rng.choice(questions)}
        if rng.random() < 0.5:
            args["limit"] = int(line["quantity"])
        if rng.random() < 0.5:
            args["score"] = float(line["amount"])
        if rng.random() < 0.5:
            args["exact"] = bool(line["cancelled"])
        pad = rng.choice(["", " ", "\n\t "])
        text = pad + _wrap("lookup", args) + pad
        want = json.loads(json.dumps({"name": "lookup", "arguments": args}))
        assert _call(text) == want, (
            "parsed call disagrees with the JSON reference: check the arguments and the name"
        )


def test_integer_above_two_to_the_53_is_kept_exact():
    got = _call(_wrap("lookup", {"query": "x", "limit": 2**53 + 1}))
    assert got["arguments"]["limit"] == 2**53 + 1, (
        "integers must stay exact: do not convert them to float"
    )
    assert type(got["arguments"]["limit"]) is int, "an integer argument must come back as int"


def test_boolean_in_an_integer_slot_raises_tool_schema_error():
    _expect(ToolSchemaError, _wrap("lookup", {"query": "x", "limit": True}))


def test_float_in_an_integer_slot_raises_tool_schema_error():
    _expect(
        ToolSchemaError, _block('{"name": "lookup", "arguments": {"query": "x", "limit": 2.0}}')
    )


def test_integer_in_a_number_slot_is_accepted():
    got = _call(_wrap("lookup", {"query": "x", "score": 3}))
    assert got["arguments"]["score"] == 3, "an integer is a valid number"


def test_boolean_in_a_number_slot_raises_tool_schema_error():
    _expect(ToolSchemaError, _wrap("lookup", {"query": "x", "score": False}))


def test_integer_in_a_boolean_slot_raises_tool_schema_error():
    _expect(ToolSchemaError, _wrap("lookup", {"query": "x", "exact": 1}))


@pytest.mark.parametrize(
    "token", ["NaN", "Infinity", "-Infinity"], ids=["nan", "infinity", "negative-infinity"]
)
def test_non_finite_json_constant_raises_tool_call_parse_error(token):
    _expect(
        ToolCallParseError,
        _block('{"name": "lookup", "arguments": {"query": "x", "score": ' + token + "}}"),
    )


def test_literal_that_overflows_to_infinity_raises_tool_call_parse_error():
    _expect(
        ToolCallParseError,
        _block('{"name": "lookup", "arguments": {"query": "x", "score": 1e400}}'),
    )


def test_repeated_key_raises_tool_call_parse_error():
    _expect(
        ToolCallParseError,
        _block('{"name": "lookup", "name": "lookup", "arguments": {"query": "x"}}'),
    )


def test_two_call_blocks_raise_tool_call_parse_error():
    text = _wrap("lookup", {"query": "a"}) + _wrap("lookup", {"query": "b"})
    _expect(ToolCallParseError, text)


def test_text_outside_the_block_raises_tool_call_parse_error():
    _expect(ToolCallParseError, "Sure. " + _wrap("lookup", {"query": "a"}))


def test_unclosed_block_raises_tool_call_parse_error():
    _expect(ToolCallParseError, '<call>{"name": "lookup", "arguments": {"query": "a"}}')


def test_malformed_json_body_raises_tool_call_parse_error():
    _expect(ToolCallParseError, _block('{"name": "lookup", "arguments": {"query": "a"'))


def test_wrong_tool_name_raises_tool_schema_error():
    _expect(ToolSchemaError, _wrap("search", {"query": "a"}))


def test_missing_required_argument_raises_tool_schema_error():
    _expect(ToolSchemaError, _wrap("lookup", {"limit": 2}))


def test_undeclared_argument_raises_tool_schema_error():
    _expect(ToolSchemaError, _wrap("lookup", {"query": "a", "page": 2}))


def test_top_level_list_raises_tool_call_parse_error():
    _expect(ToolCallParseError, _block('[{"name": "lookup", "arguments": {}}]'))


def test_missing_arguments_key_raises_tool_call_parse_error():
    _expect(ToolCallParseError, _block('{"name": "lookup"}'))


def test_extra_top_level_key_raises_tool_call_parse_error():
    _expect(
        ToolCallParseError,
        _block('{"name": "lookup", "arguments": {"query": "a"}, "id": 1}'),
    )


def test_non_string_name_raises_tool_call_parse_error():
    _expect(ToolCallParseError, _block('{"name": 7, "arguments": {"query": "a"}}'))


def test_arguments_that_are_not_an_object_raise_tool_call_parse_error():
    _expect(ToolCallParseError, _block('{"name": "lookup", "arguments": ["a"]}'))


def test_non_string_text_raises_type_error():
    try:
        _learner(None, SCHEMA)
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"non-string text raised {type(exc).__name__}, expected TypeError")
    pytest.fail("non-string text was accepted instead of raising TypeError")


def test_inputs_are_not_mutated():
    # Call the learner directly on a private copy, and compare it with a snapshot taken from the
    # shared SCHEMA, which no test changes. A learner that writes into the schema fails here.
    tool = copy.deepcopy(SCHEMA)
    text = _wrap("lookup", {"query": "a", "limit": 3})
    before_text, before_tool = text, copy.deepcopy(SCHEMA)
    try:
        _learner(text, tool)
    except Exception as exc:
        pytest.fail(f"valid call violated the contract: raised {type(exc).__name__}")
    assert text == before_text, "the text must not be modified"
    assert tool == before_tool, "the tool schema must not be modified: do not write into it"


def _must_reject_as_parse_error(text, mistake):
    try:
        _learner(text, copy.deepcopy(SCHEMA))
    except ToolCallParseError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}", pytrace=False)
    pytest.fail(mistake, pytrace=False)


def test_non_finite_number_inside_a_list_raises_tool_call_parse_error():
    _must_reject_as_parse_error(
        _block('{"name": "lookup", "arguments": {"query": "x", "limit": [1e400]}}'),
        "every JSON number must be finite, including one nested in a list: "
        "raise ToolCallParseError",
    )


def test_text_after_the_closing_tag_raises_tool_call_parse_error():
    text = "<call>" + '{"name": "lookup", "arguments": {"query": "</call>"}}' + "garbage"
    _expect(ToolCallParseError, text)


def test_text_before_the_opening_tag_raises_tool_call_parse_error():
    text = "Hello!" + '{"name": "lookup", "arguments": {"query": "<call>"}}' + "</call>"
    _expect(ToolCallParseError, text)
