"""Shape checks for ids that arrive in paths and queries (ENGINEERING_STANDARDS §4.7).

A check only decides the shape. Whether an id exists is decided by the content loader or the
database, and anything that fails either check is answered as not found.
"""

from __future__ import annotations

import re
from typing import Final

from dogma.content.schemas import PROBLEM_DIR_PATTERN, TOPIC_ID_PATTERN

_TOPIC_BODY: Final = TOPIC_ID_PATTERN.removeprefix("^").removesuffix("$")
_PROBLEM_DIR_BODY: Final = PROBLEM_DIR_PATTERN.removeprefix("^").removesuffix("$")

TOPIC_ID: Final = re.compile(_TOPIC_BODY)
PROBLEM_ID: Final = re.compile(rf"(?P<topic>{_TOPIC_BODY})/(?P<dir>{_PROBLEM_DIR_BODY})")
CARD_ID: Final = re.compile(rf"{_TOPIC_BODY}#\d{{1,6}}")
ATTEMPT_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
SESSION_ID: Final = re.compile(r"[0-9a-f]{32}")


def is_topic_id(value: str) -> bool:
    return TOPIC_ID.fullmatch(value) is not None


def problem_parts(value: str) -> tuple[str, str] | None:
    """``(topic id, problem directory)`` for a well-formed problem id, else None."""
    match = PROBLEM_ID.fullmatch(value)
    if match is None:
        return None
    return match.group("topic"), match.group("dir")


def is_card_id(value: str) -> bool:
    return CARD_ID.fullmatch(value) is not None


def is_attempt_id(value: str) -> bool:
    return ATTEMPT_ID.fullmatch(value) is not None


def is_session_id(value: str) -> bool:
    return SESSION_ID.fullmatch(value) is not None
