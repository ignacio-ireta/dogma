from __future__ import annotations

import itertools
from typing import Any

ROLES = ("system", "user", "assistant", "tool")
RESERVED = ("<bos>", "<eom>", "<msg:")

# The roles that may legally follow each non-system role (the turn-order rules in problem.md).
# A user message must be answered by an assistant message. An assistant message may be followed
# by a user message or a tool result. A tool result may be followed by another tool result or
# by the assistant's answer.
ALLOWED_NEXT: dict[str, tuple[str, ...]] = {
    "user": ("assistant",),
    "assistant": ("user", "tool"),
    "tool": ("tool", "assistant"),
}


def render_chat(messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
    """Wrong on purpose: Render {role, content} messages into <bos>, <msg:ROLE>content<eom>, in order.

    The template is defined in problem.md. Validation runs first, so a bad conversation
    fails with a named error before any text is built.
    """  # noqa: E501
    # Type check first. bool is a subclass of int, so the generation flag is checked for bool.
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    if not isinstance(add_generation_prompt, bool):
        raise TypeError("add_generation_prompt must be a bool")

    roles = [_check_message(item) for item in messages]
    _check_turn_order(roles, add_generation_prompt)

    # Render. Concatenate the fixed strings with the content, byte for byte. Nothing is added
    # between the pieces, so the output has no extra whitespace.
    parts = ["<bos>"]
    for item in messages:
        parts.append(f"<msg:{item['role']}>{item['content']}<eom>")
    if add_generation_prompt:
        parts.append("<msg:assistant>")
    return "".join(parts)


def _check_message(item: Any) -> str:
    """Check one message's shape, role, content and reserved strings. Returns its role."""
    if not isinstance(item, dict):
        raise TypeError("each message must be a dict")
    if "role" not in item or "content" not in item:
        raise ValueError("each message must have exactly the keys role and content")
    role, content = item["role"], item["content"]
    if not isinstance(role, str) or role not in ROLES:
        raise ValueError(f"unknown role: {role!r}")
    if not isinstance(content, str):
        raise TypeError("content must be a str")
    if any(token in content for token in RESERVED):
        raise ValueError("content contains a reserved control string")
    return role


def _check_turn_order(roles: list[str], add_generation_prompt: bool) -> None:
    """Enforce the turn-order rules. A system message is legal only first and only once."""
    body = roles[1:] if roles and roles[0] == "system" else roles
    if not body:
        raise ValueError("a conversation needs at least one message that is not system")
    if "system" in body:
        raise ValueError("system may appear only as the first message, at most once")
    if body[0] != "user":
        raise ValueError("the first message after system must be user")
    for current, following in itertools.pairwise(body):
        if following not in ALLOWED_NEXT[current]:
            raise ValueError(f"{following} may not follow {current}")
    # The generation prompt asks for the next assistant turn. That is only legal when the
    # conversation is waiting for an answer: after a user or tool message, not an assistant one.
    if add_generation_prompt and body[-1] == "assistant":
        raise ValueError("the generation prompt cannot follow an assistant message")
