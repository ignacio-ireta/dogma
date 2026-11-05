"""Markdown structure for content files: YAML front matter, the first H1 and the H2 sections.

Only what the content gate needs. Lines inside fenced code blocks are never headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_H1 = re.compile(r"^# (.+)$")
_H2 = re.compile(r"^## (.+)$")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")
_DOCTEST = re.compile(r"^\s*>>> ")


@dataclass(frozen=True)
class Section:
    """One H2 section: its heading, its body text, and how many examples it shows."""

    heading: str
    fences: int
    doctests: int
    text: str

    @property
    def examples(self) -> int:
        """Examples shown: a fenced block or a ``>>>`` line each count once; take the larger."""
        return max(self.fences, self.doctests)


@dataclass(frozen=True)
class Outline:
    """The first H1 title (None if absent) and every H2 section in document order."""

    title: str | None
    sections: tuple[Section, ...]

    @property
    def headings(self) -> tuple[str, ...]:
        return tuple(section.heading for section in self.sections)


def split_front_matter(text: str) -> tuple[str | None, str]:
    """Return ``(front_matter_yaml, body)``.

    Front matter exists only when the text opens with a ``---`` line. Without one the YAML part
    is None and the body is the whole text. An opened block that never closes raises ValueError.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1 :])
    raise ValueError("front matter opened with '---' but never closed")


def parse_outline(body: str) -> Outline:
    """Find the first H1 and the H2 sections (fenced code is skipped for headings)."""
    title: str | None = None
    sections: list[Section] = []
    current: str | None = None
    fences = 0
    doctests = 0
    lines: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if _DOCTEST.match(line):
            doctests += 1
        if _FENCE.match(line):
            if not in_fence:
                fences += 1
            in_fence = not in_fence
            lines.append(line)
            continue
        if in_fence:
            lines.append(line)
            continue
        h1 = _H1.match(line)
        if h1 and title is None:
            title = h1.group(1).strip()
        h2 = _H2.match(line)
        if h2:
            if current is not None:
                sections.append(Section(current, fences, doctests, "\n".join(lines)))
            current, fences, doctests, lines = h2.group(1).strip(), 0, 0, []
        else:
            lines.append(line)
    if current is not None:
        sections.append(Section(current, fences, doctests, "\n".join(lines)))
    return Outline(title, tuple(sections))
