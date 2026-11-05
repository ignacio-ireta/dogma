"""Deterministic heading/page-aware Markdown chunking."""

from __future__ import annotations

import re

import yaml

from dogma.corpus.models import Chunk

CHUNKER_VERSION = "1"
# Four characters per token approximates 500 tokens; the cap also bounds long words.
MAX_CHARS = 2000
_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_PAGE = re.compile(r"^Page ([1-9][0-9]*)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _body(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is None:
            raise ValueError("Unclosed Markdown front matter")
        yaml.safe_load("\n".join(lines[1:end]))
        return "\n".join(lines[end + 1 :])
    return text


def _split(text: str) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > MAX_CHARS:
        boundary = remaining.rfind("\n\n", 0, MAX_CHARS + 1)
        if boundary < MAX_CHARS // 2:
            boundary = max(
                remaining.rfind(" ", 0, MAX_CHARS + 1), remaining.rfind("\n", 0, MAX_CHARS + 1)
            )
        if boundary <= 0:
            boundary = MAX_CHARS
        parts.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def chunk_markdown(
    text: str,
    *,
    source_id: str,
    path: str,
    topic_ids: tuple[str, ...] = (),
) -> tuple[Chunk, ...]:
    """Split at structural changes, then at at most 2000 characters (~500 tokens).

    YAML front matter is omitted. Page markers are docmd's exact H3 ``Page N``;
    fenced examples never affect provenance. No chunk combines different pages.
    """
    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []
    page: int | None = None
    buffer: list[str] = []
    fence: str | None = None

    def flush() -> None:
        for part in _split("\n".join(buffer)):
            chunks.append(
                Chunk(
                    source_id,
                    path,
                    page,
                    tuple(h[1] for h in headings),
                    len(chunks),
                    part,
                    topic_ids,
                )
            )
        buffer.clear()

    for line in _body(text).splitlines():
        marker = _FENCE.match(line)
        if fence is not None:
            buffer.append(line)
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not marker[2].strip()
            ):
                fence = None
        elif marker:
            fence = marker[1]
            buffer.append(line)
        elif heading := _HEADING.match(line):
            flush()
            level, title = len(heading[1]), heading[2]
            headings[:] = [(n, h) for n, h in headings if n < level]
            page_match = _PAGE.fullmatch(title) if level == 3 else None
            if page_match:
                page = int(page_match[1])
            else:
                headings.append((level, title))
        else:
            buffer.append(line)
    flush()
    return tuple(chunks)
