from __future__ import annotations

from typing import Any


def render_chat(messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
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
    raise NotImplementedError
