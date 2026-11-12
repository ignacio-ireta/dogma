# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of
# mistake and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import functools
import random

import pytest
from submission import render_chat as _learner

from dogma_data import load_frame

ROLES_AFTER = {
    "user": ["assistant"],
    "assistant": ["user", "tool"],
    "tool": ["tool", "assistant"],
}


@functools.cache
def _gsm8k():
    """GSM8K questions and worked answers: real message contents, with newlines and symbols."""
    table = load_frame("gsm8k")
    return tuple(table["question"]), tuple(table["answer"])


def _call(messages, gen=False):
    try:
        return _learner(messages, gen)
    except (TypeError, ValueError) as exc:
        pytest.fail(
            f"valid conversation was rejected ({type(exc).__name__}): check the role-order, "
            "key, type and reserved-string rules against problem.md"
        )
    except Exception as exc:
        pytest.fail(f"valid conversation violated the contract: raised {type(exc).__name__}")


def _expect(exc_type, messages, gen=False, why=""):
    try:
        _learner(messages, gen)
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(
            f"invalid input raised {type(exc).__name__}, expected {exc_type.__name__}. {why}"
        )
    pytest.fail(f"invalid input was accepted instead of raising {exc_type.__name__}. {why}")


def _msg(role, content):
    return {"role": role, "content": content}


def _reference(messages, gen):
    """Independent renderer: the template written as a join over the messages."""
    text = "<bos>" + "".join(f"<msg:{m['role']}>{m['content']}<eom>" for m in messages)
    return text + ("<msg:assistant>" if gen else "")


def _random_conversation(rng):
    """A conversation with real contents: GSM8K questions from the user, worked answers after.

    The answers hold newlines, operators and a final line that starts with ####, so the content
    must be kept as it is. Three turns in ten also get a tab and trailing spaces appended.
    """
    questions, answers = _gsm8k()
    msgs = []
    if rng.random() < 0.5:
        instruction = "Solve the problem. Give the final number after ####."
        msgs.append(_msg("system", f"{instruction} (task {rng.randrange(100)})"))
    role = "user"
    msgs.append(_msg("user", questions[rng.randrange(len(questions))]))
    for _ in range(rng.randrange(0, 4)):
        role = rng.choice(ROLES_AFTER[role])
        source = questions if role == "user" else answers
        whitespace = "\n\t " if rng.random() < 0.3 else ""
        msgs.append(_msg(role, f"{source[rng.randrange(len(source))]}{whitespace}"))
    gen = msgs[-1]["role"] in ("user", "tool") and rng.random() < 0.5
    return msgs, gen


def test_random_valid_conversations_render_like_the_reference():
    rng = random.Random(401)  # noqa: S311 -- seeded synthetic conversations, not security
    for _ in range(300):
        messages, gen = _random_conversation(rng)
        assert _call(messages, gen) == _reference(messages, gen), (
            "rendered string disagrees with the template: check the order of the parts"
        )


def test_content_is_kept_byte_for_byte_including_whitespace_and_unicode():
    content = "  spaced\tand\nnewlines  é ß 中  "
    got = _call([_msg("user", content)])
    assert got == f"<bos><msg:user>{content}<eom>", (
        "content must be copied exactly: no stripping, no escaping, no added whitespace"
    )


def test_empty_content_renders_an_empty_message():
    assert _call([_msg("user", "")]) == "<bos><msg:user><eom>", (
        "an empty message is the open and close markers with nothing between them"
    )


@pytest.mark.parametrize(
    "content",
    ["a <eom> b", "<bos>", "<msg:user>x"],
    ids=["eom", "bos", "msg-open"],
)
def test_reserved_control_string_inside_content_raises_value_error(content):
    _expect(
        ValueError,
        [_msg("user", content)],
        why="content that contains a reserved control string must be rejected",
    )


def test_first_turn_must_be_user_raises_value_error():
    # Rule 3: the first non-system message is user. Nothing else in the order rules
    # catches these, so each case needs this check on its own.
    _expect(ValueError, [_msg("assistant", "a")], why="the first turn must be a user message")
    _expect(ValueError, [_msg("tool", "a")], why="the first turn must be a user message")
    _expect(
        ValueError,
        [_msg("system", "s"), _msg("assistant", "a")],
        why="after an optional system message, the first turn must be a user message",
    )


def test_system_message_after_user_raises_value_error():
    _expect(
        ValueError,
        [_msg("user", "a"), _msg("system", "b")],
        why="system may appear only as the first message",
    )


def test_second_system_message_raises_value_error():
    _expect(
        ValueError,
        [_msg("system", "a"), _msg("system", "b"), _msg("user", "c")],
        why="system may appear at most once",
    )


def test_user_followed_by_user_raises_value_error():
    _expect(
        ValueError,
        [_msg("user", "a"), _msg("user", "b")],
        why="a user message must be followed by an assistant message or the end",
    )


def test_tool_result_directly_after_user_raises_value_error():
    _expect(
        ValueError,
        [_msg("user", "a"), _msg("tool", "b")],
        why="a tool result must follow an assistant message",
    )


def test_assistant_followed_by_assistant_raises_value_error():
    _expect(
        ValueError,
        [_msg("user", "a"), _msg("assistant", "b"), _msg("assistant", "c")],
        why="an assistant message must be followed by a user or tool message, or the end",
    )


def test_user_turn_directly_after_tool_result_raises_value_error():
    messages = [_msg("user", "a"), _msg("assistant", "call"), _msg("tool", "r"), _msg("user", "b")]
    _expect(
        ValueError,
        messages,
        why="a tool message must be followed by a tool or assistant message, or the end",
    )


def test_generation_prompt_after_assistant_raises_value_error():
    _expect(
        ValueError,
        [_msg("user", "a"), _msg("assistant", "b")],
        True,
        why="add_generation_prompt needs the last message to be user or tool",
    )


def test_generation_prompt_after_tool_result_is_allowed():
    messages = [_msg("user", "a"), _msg("assistant", "call"), _msg("tool", "r")]
    assert _call(messages, True).endswith("<msg:assistant>"), (
        "after a tool result the generation prompt must open the next assistant turn"
    )


def test_generation_flag_must_be_a_bool_not_an_int():
    _expect(TypeError, [_msg("user", "a")], 1, why="add_generation_prompt must be a bool")


@pytest.mark.parametrize(
    "item",
    [{"role": "user"}, {"role": "user", "content": "a", "name": "x"}],
    ids=["missing-content", "extra-key"],
)
def test_message_with_missing_or_extra_key_raises_value_error(item):
    _expect(ValueError, [item], why="each message has exactly the keys role and content")


@pytest.mark.parametrize("role", ["developer", "User", ""], ids=["unknown", "wrong-case", "empty"])
def test_unknown_role_raises_value_error(role):
    _expect(
        ValueError, [_msg(role, "a")], why="role must be exactly system, user, assistant or tool"
    )


@pytest.mark.parametrize("content", [None, 3], ids=["none", "int"])
def test_non_string_content_raises_type_error(content):
    _expect(TypeError, [_msg("user", content)], why="content must be a str")


def test_empty_conversation_raises_value_error():
    _expect(ValueError, [], why="a conversation needs at least one non-system message")


def test_system_only_conversation_raises_value_error():
    _expect(
        ValueError,
        [_msg("system", "rules")],
        why="a conversation needs at least one non-system message",
    )


def test_tuple_of_messages_raises_type_error():
    try:
        _learner((_msg("user", "a"),))
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"non-list messages raised {type(exc).__name__}, expected TypeError")
    pytest.fail("non-list messages were accepted instead of raising TypeError")


def test_inputs_are_not_mutated():
    messages = [_msg("system", "s"), _msg("user", "u"), _msg("assistant", "a")]
    before = [dict(m) for m in messages]
    _call(messages)
    assert messages == before, "the messages list and its dicts must not be modified"


def test_item_that_is_a_string_instead_of_a_dict_raises_type_error():
    _expect(TypeError, ["not a message"], why="each item of messages must be a dict")


def test_list_content_raises_type_error():
    _expect(TypeError, [_msg("user", ["hi"])], why="content must be a str, not a list")
