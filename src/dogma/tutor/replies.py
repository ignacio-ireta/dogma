"""Strict reading of a model reply that must be one JSON object (score and interviewer modes).

A reply wrapped in one Markdown code fence is accepted, because models add it often. Anything
else around the object, such as prose before or after it, makes the whole reply invalid.
"""

from __future__ import annotations

import json
import re

from dogma.core.errors import DogmaError, ErrorCode

_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n(.*)\n```\s*\Z", re.DOTALL)


def json_object(reply: str, code: ErrorCode, message: str) -> dict[str, object]:
    """The reply as a JSON object, or a DogmaError with ``code`` and ``message``."""
    body = reply.strip()
    fenced = _FENCE.match(body)
    if fenced is not None:
        body = fenced.group(1)
    try:
        data: object = json.loads(body)
    except ValueError as exc:
        raise DogmaError(code, message) from exc
    if not isinstance(data, dict):
        raise DogmaError(code, message)
    return {str(key): value for key, value in data.items()}
