from __future__ import annotations

from typing import Any

_ROLES = ("system", "user", "assistant", "tool")
_RESERVED = ("<bos>", "<eom>", "<msg:")


def render_chat(messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:  # noqa: PLR0912
    """Render {role, content} messages into the template <bos>, <msg:ROLE>content<eom>.

    Parameters
    ----------
    messages : list of dict with exactly the keys "role" and "content". Roles are system,
        user, assistant and tool. Content is a str with no reserved strings (<bos>, <eom>,
        <msg:). Must not be mutated.
    add_generation_prompt : bool. When True, append <msg:assistant>, which requires the last
        message to be user or tool.

    Returns
    -------
    str : "<bos>" followed by the messages in order, with no extra characters.

    Raises
    ------
    TypeError
        When messages is not a list, an item is not a dict, content is not a str, or
        add_generation_prompt is not a bool.
    ValueError
        For a missing or extra key, an unknown role, a reserved string in content, an
        invalid turn order, or a generation prompt after an assistant message.
    """
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    if not isinstance(add_generation_prompt, bool):
        raise TypeError("add_generation_prompt must be a bool")

    roles: list[str] = []
    parts: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            raise TypeError("each message must be a dict")
        if set(item.keys()) != {"role", "content"}:
            raise ValueError("each message must have exactly the keys 'role' and 'content'")
        role = item["role"]
        content = item["content"]
        if not isinstance(role, str) or role not in _ROLES:
            raise ValueError(f"unknown role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("content must be a str")
        if any(tok in content for tok in _RESERVED):
            raise ValueError("content contains a reserved template string")
        roles.append(role)
        parts.append(f"<msg:{role}>{content}<eom>")

    if not roles:
        raise ValueError("the conversation must not be empty")
    if all(r == "system" for r in roles):
        raise ValueError("the conversation must contain a non-system message")

    # system: only at position 0, at most once.
    for i, r in enumerate(roles):
        if r == "system" and i != 0:
            raise ValueError("system may appear only as the first message")

    # first non-system message must be user.
    first = next(r for r in roles if r != "system")
    if first != "user":
        raise ValueError("the first non-system message must be from the user")

    for i, r in enumerate(roles):
        nxt = roles[i + 1] if i + 1 < len(roles) else None
        if r == "user":
            if nxt is not None and nxt != "assistant":
                raise ValueError("a user message must be followed by an assistant message")
        elif r == "assistant":
            if nxt is not None and nxt not in ("user", "tool"):
                raise ValueError("an assistant message must be followed by user or tool")
        elif r == "tool":  # noqa: SIM102
            if nxt is not None and nxt not in ("tool", "assistant"):
                raise ValueError("a tool message must be followed by tool or assistant")

    if add_generation_prompt and roles[-1] not in ("user", "tool"):
        raise ValueError("add_generation_prompt requires the last message to be user or tool")

    out = "<bos>" + "".join(parts)
    if add_generation_prompt:
        out += "<msg:assistant>"
    return out
