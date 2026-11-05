"""Relocate a conversion when its source moved (same content, new path).

Moving the existing ``.md`` + sidecar and rewriting their ``source_file``
references is dramatically cheaper than re-extraction (OCR PDFs especially),
and the content is identical by definition — the move was detected by hash.
The new location is written atomically first and the old pair deleted last,
so a crash mid-relocation leaves both copies and a re-run converges.
Frontmatter is rewritten line-targeted with the same JSON-scalar encoding the
writer uses, so everything but that one line stays byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from docmd.fsio import atomic_write_text
from docmd.markdown.metadata import SidecarMeta, sidecar_path_for, write_sidecar

_DELIM = "---"
_SOURCE_KEY = "source_file: "


def rewrite_frontmatter_source(markdown: str, new_source: str) -> tuple[str, bool]:
    """Replace the frontmatter ``source_file`` value; ``(text, False)`` if absent.

    Only lines inside a terminated leading frontmatter block are considered —
    a hand-edited file without one is returned untouched.
    """
    if not markdown.startswith(_DELIM + "\n"):
        return markdown, False
    lines = markdown.split("\n")
    try:
        closing = lines.index(_DELIM, 1)
    except ValueError:
        return markdown, False
    for i in range(1, closing):
        if lines[i].startswith(_SOURCE_KEY):
            lines[i] = _SOURCE_KEY + json.dumps(new_source, ensure_ascii=False)
            return "\n".join(lines), True
    return markdown, False


def relocate_output(old_md: Path, new_md: Path, new_source: str) -> bool:
    """Move ``.md`` + sidecar to a new path, rewriting ``source_file`` in both.

    Returns ``False`` when the pair cannot be relocated cleanly (old output or
    sidecar missing/corrupt, no rewritable frontmatter) — the caller archives
    what exists and re-extracts instead. ``OSError`` propagates.
    """
    old_sidecar = sidecar_path_for(old_md)
    if not (old_md.is_file() and old_sidecar.is_file()):
        return False
    try:
        meta = SidecarMeta.model_validate_json(old_sidecar.read_text(encoding="utf-8"))
    except ValueError:
        return False
    rewritten, ok = rewrite_frontmatter_source(old_md.read_text(encoding="utf-8"), new_source)
    if not ok:
        return False
    # Case-only rename on a case-insensitive filesystem: old and new resolve
    # to one file, so deleting "the old pair" last would destroy the result.
    # Rename first to fix the stored case, then rewrite in place.
    same_pair = new_md.exists() and old_md.samefile(new_md)
    if same_pair:
        old_md.rename(new_md)
        old_sidecar.rename(sidecar_path_for(new_md))
    atomic_write_text(new_md, rewritten)
    write_sidecar(new_md, meta.model_copy(update={"source_file": new_source}))
    if not same_pair:
        old_md.unlink(missing_ok=True)
        old_sidecar.unlink(missing_ok=True)
    return True
