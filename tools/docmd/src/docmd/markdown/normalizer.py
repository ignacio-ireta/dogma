"""Pure IR-to-IR text hygiene applied between extraction and rendering.

Scope: unicode NFC, control-character stripping, whitespace collapsing, line
ending normalization, and dropping blocks that end up empty. Markdown escaping
is deliberately NOT done here — that is format-specific and belongs to the
writer.
"""

from __future__ import annotations

import re
import unicodedata

from docmd.types import (
    Block,
    CodeBlock,
    DocumentIR,
    Heading,
    ImageRef,
    ListItem,
    Paragraph,
    Section,
    Table,
)

# Control characters except \t and \n (carriage returns are folded first).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Horizontal whitespace runs, including NBSP and tabs.
_WS_RUN = re.compile(r"[ \t\u00a0]+")


def clean_text(text: str, *, fold_newlines: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    if fold_newlines:
        text = text.replace("\n", " ")
        return _WS_RUN.sub(" ", text).strip()
    # Multi-line text (code blocks): keep internal whitespace, trim line ends.
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def _normalize_block(block: Block) -> Block | None:
    if isinstance(block, Heading):
        text = clean_text(block.text)
        return Heading(level=block.level, text=text) if text else None
    if isinstance(block, Paragraph):
        text = clean_text(block.text)
        return Paragraph(text=text) if text else None
    if isinstance(block, ListItem):
        text = clean_text(block.text)
        if not text:
            return None
        return ListItem(text=text, level=block.level, ordered=block.ordered)
    if isinstance(block, Table):
        rows = [[clean_text(cell) for cell in row] for row in block.rows]
        if not any(cell for row in rows for cell in row):
            return None
        caption = clean_text(block.caption) if block.caption else None
        return Table(rows=rows, has_header=block.has_header, caption=caption or None)
    if isinstance(block, CodeBlock):
        text = clean_text(block.text, fold_newlines=False)
        return CodeBlock(text=text, language=block.language) if text else None
    return ImageRef(
        name=clean_text(block.name) or None if block.name else None,
        content_type=block.content_type,
        alt=clean_text(block.alt) or None if block.alt else None,
    )


def normalize(doc: DocumentIR) -> DocumentIR:
    sections: list[Section] = []
    for section in doc.sections:
        blocks = [b for b in (_normalize_block(block) for block in section.blocks) if b]
        sections.append(
            Section(
                kind=section.kind,
                index=section.index,
                title=clean_text(section.title) or None if section.title else None,
                blocks=blocks,
            )
        )
    return DocumentIR(
        source_path=doc.source_path,
        source_type=doc.source_type,
        title=clean_text(doc.title) or None if doc.title else None,
        sections=sections,
        properties={clean_text(k): clean_text(v) for k, v in doc.properties.items()},
    )
