# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import render_chat


def test_single_user_message_with_generation_prompt():
    got = render_chat([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    assert got == "<bos><msg:user>Hi<eom><msg:assistant>", (
        "the template is <bos>, then <msg:ROLE>content<eom> per message, then the open "
        "assistant marker when add_generation_prompt is true"
    )


def test_system_then_user_then_assistant_renders_in_order_without_extra_characters():
    got = render_chat(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    )
    assert got == "<bos><msg:system>Be brief.<eom><msg:user>2+2?<eom><msg:assistant>4<eom>", (
        "messages must follow each other with no spaces or newlines between them"
    )


def test_assistant_first_conversation_raises_value_error():
    with pytest.raises(ValueError):
        render_chat([{"role": "assistant", "content": "Hi"}])
