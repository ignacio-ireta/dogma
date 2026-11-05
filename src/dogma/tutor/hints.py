"""The reviewed hints of a problem: its ``hints.md``, read for the tutor (CONTENT_SPEC §6.5, I-22).

The file has three H2 sections, ``Hint 1`` to ``Hint 3``, each with a label after a separator.
Both separators in use, a colon and an em dash, are accepted. A file that is missing, unreadable or
not in that shape gives no hints and a one-line note, and the tutor then works from the model alone,
as in I-12. Reading never raises.
"""

from __future__ import annotations

import re
from pathlib import Path

from dogma.content.markdown import parse_outline
from dogma.tutor.models import ReviewedHint, ReviewedHints

HINTS_FILE = "hints.md"
REVIEWED_LEVELS = (1, 2, 3)
_DEFAULT_LABELS = {1: "concept", 2: "formula / approach", 3: "code shape"}
_HEADING = re.compile(r"^Hint\s+([1-3])\b(.*)$")
_SEPARATORS = " :\N{EM DASH}\N{EN DASH}-"  # space, colon, em dash, en dash, hyphen
_NO_HINTS = "so the hint comes from the model alone"


def load_reviewed_hints(problem_dir: Path) -> ReviewedHints:
    """Read ``hints.md`` of a problem directory. A missing or broken file is a note, not a raise."""
    path = problem_dir / HINTS_FILE
    if not path.is_file():
        return ReviewedHints(note=f"hints.md is missing, {_NO_HINTS}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ReviewedHints(note=f"hints.md cannot be read, {_NO_HINTS}")
    return parse_reviewed_hints(text)


def parse_reviewed_hints(text: str) -> ReviewedHints:
    """The three reviewed hints in order, or a note when the text does not have that shape."""
    sections = parse_outline(text).sections
    if len(sections) != len(REVIEWED_LEVELS):
        return _malformed()
    hints: list[ReviewedHint] = []
    for expected, section in zip(REVIEWED_LEVELS, sections, strict=True):
        match = _HEADING.match(section.heading.strip())
        body = section.text.strip()
        if match is None or int(match.group(1)) != expected or not body:
            return _malformed()
        label = match.group(2).strip(_SEPARATORS) or _DEFAULT_LABELS[expected]
        hints.append(ReviewedHint(level=expected, label=label, text=body))
    return ReviewedHints(hints=tuple(hints))


def _malformed() -> ReviewedHints:
    return ReviewedHints(note=f"hints.md is malformed (expected Hint 1 to Hint 3), {_NO_HINTS}")
