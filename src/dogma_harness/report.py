"""Harness output: the JSON document the runner reads back.

Stdlib only: this module never imports ``dogma``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

MAX_STDOUT_BYTES = 64 * 1024
MAX_MESSAGE_CHARS = 2000
_TRUNCATION_MARKER = "\n[... stdout truncated at 64 KB ...]\n"


@dataclass(frozen=True)
class TestRecord:
    """One test function. ``file`` is ``hidden`` only for ``test_hidden.py``."""

    name: str
    file: Literal["public", "hidden"]
    outcome: Literal["pass", "fail", "error", "skip"]
    message: str
    duration_ms: int


@dataclass(frozen=True)
class HarnessResult:
    """What the harness writes to ``report.json``."""

    status: Literal["passed", "failed", "error"]
    tests: tuple[TestRecord, ...] = ()
    stdout: str = ""
    collection_error: str | None = None

    @classmethod
    def errored(cls, message: str) -> HarnessResult:
        """A run that never reached the tests: the harness itself could not proceed."""
        return cls(status="error", collection_error=trim_message(message))


def trim_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Keep the head of ``text`` within ``limit`` characters."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def trim_tail(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Keep the end of ``text``: where pytest puts the exception line of a collection error."""
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3) :]


def truncate_stdout(text: str) -> str:
    """Cap captured stdout at 64 KB of UTF-8, marker included, never splitting a character."""
    data = text.encode("utf-8", errors="replace")
    if len(data) <= MAX_STDOUT_BYTES:
        return text
    marker = _TRUNCATION_MARKER.encode("utf-8")
    keep = MAX_STDOUT_BYTES - len(marker)
    return data[:keep].decode("utf-8", errors="ignore") + _TRUNCATION_MARKER


def write_report(path: Path, result: HarnessResult) -> None:
    """Write the report atomically: temp file in the same directory, then rename."""
    payload = json.dumps(asdict(result), ensure_ascii=False)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
