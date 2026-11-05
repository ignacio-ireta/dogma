"""Render a normalized DocumentIR to Markdown following the output contract.

Document shape (see docs/extraction_contract.md):

    ---
    frontmatter
    ---

    # <source filename>

    ## Document metadata        (only when properties exist)

    ## Content

    ### <section label>         (slide/sheet/page/notes; body blocks render bare)

    ## Extraction warnings      (only when warnings exist)

Content headings are shifted +2 so they nest under ``## Content``. Escaping of
Markdown-significant characters happens here (the normalizer only does text
hygiene): pipes/newlines in table cells, and structural markers at the start of
paragraph or list text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import PurePosixPath

from docmd.markdown.metadata import FrontMatter, render_frontmatter
from docmd.types import (
    Block,
    CodeBlock,
    DocumentIR,
    ExtractionWarning,
    Heading,
    ImageRef,
    ListItem,
    Paragraph,
    Section,
    SectionKind,
    Table,
)

HEADING_OFFSET = 2

_ORDERED_MARKER = re.compile(r"^(\d+)([.)])(\s|$)")
_STRUCTURAL_MARKER = re.compile(r"^[#>*+-](\s|$)")


def escape_leading_marker(text: str) -> str:
    """Backslash-escape text that would otherwise parse as Markdown structure."""
    m = _ORDERED_MARKER.match(text)
    if m:
        return f"{m.group(1)}\\{m.group(2)}{text[m.end(2) :]}"
    if _STRUCTURAL_MARKER.match(text):
        return f"\\{text}"
    return text


def _escape_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


class MarkdownWriter:
    def render(
        self,
        doc: DocumentIR,
        front: FrontMatter,
        warnings: Sequence[ExtractionWarning] = (),
    ) -> str:
        parts = [render_frontmatter(front).rstrip("\n")]
        parts.append(f"# {PurePosixPath(doc.source_path).name}")

        meta_lines = []
        if doc.title:
            meta_lines.append(f"- Title: {doc.title}")
        meta_lines.extend(f"- {key}: {value}" for key, value in doc.properties.items())
        if meta_lines:
            parts.append("## Document metadata")
            parts.append("\n".join(meta_lines))

        parts.append("## Content")
        for section in doc.sections:
            rendered = self.render_section(section)
            if rendered:
                parts.append(rendered)

        if warnings:
            parts.append("## Extraction warnings")
            parts.append("\n".join(self._render_warning(w) for w in warnings))

        return "\n\n".join(parts) + "\n"

    def render_section(self, section: Section) -> str:
        parts = []
        label = self._section_label(section)
        if label:
            parts.append(f"### {label}")
        parts.extend(self._render_blocks(section.blocks))
        return "\n\n".join(parts)

    def _section_label(self, section: Section) -> str | None:
        if section.kind is SectionKind.BODY:
            return section.title or None
        if section.kind is SectionKind.SLIDE:
            base = f"Slide {section.index}" if section.index is not None else "Slide"
            return f"{base}: {section.title}" if section.title else base
        if section.kind is SectionKind.SHEET:
            return f"Sheet: {section.title}" if section.title else "Sheet"
        if section.kind is SectionKind.PAGE:
            return f"Page {section.index}" if section.index is not None else "Page"
        if section.index is not None:
            return f"Speaker notes (slide {section.index})"
        return "Speaker notes"

    def _render_blocks(self, blocks: Sequence[Block]) -> list[str]:
        """Render blocks, gluing consecutive list items into one Markdown list."""
        parts: list[str] = []
        pending_list: list[str] = []
        for block in blocks:
            if isinstance(block, ListItem):
                pending_list.append(self._render_list_item(block))
                continue
            if pending_list:
                parts.append("\n".join(pending_list))
                pending_list = []
            rendered = self.render_block(block)
            if rendered:
                parts.append(rendered)
        if pending_list:
            parts.append("\n".join(pending_list))
        return parts

    def render_block(self, block: Block) -> str:
        if isinstance(block, Heading):
            level = min(block.level + HEADING_OFFSET, 6)
            return f"{'#' * level} {block.text}"
        if isinstance(block, Paragraph):
            return escape_leading_marker(block.text)
        if isinstance(block, ListItem):
            return self._render_list_item(block)
        if isinstance(block, Table):
            return self.render_table(block)
        if isinstance(block, ImageRef):
            return self._render_image(block)
        return self._render_code(block)

    def _render_list_item(self, item: ListItem) -> str:
        marker = "1." if item.ordered else "-"
        indent = "  " * item.level
        return f"{indent}{marker} {escape_leading_marker(item.text)}"

    def render_table(self, table: Table) -> str:
        if not table.rows:
            return ""
        width = max(len(row) for row in table.rows)
        if width == 0:
            return ""
        cells = [[_escape_cell(c) for c in row] + [""] * (width - len(row)) for row in table.rows]
        if table.has_header:
            header, body = cells[0], cells[1:]
        else:
            header, body = [f"Column {i + 1}" for i in range(width)], cells
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        rendered = "\n".join(lines)
        if table.caption:
            return f"*{table.caption}*\n\n{rendered}"
        return rendered

    def _render_image(self, image: ImageRef) -> str:
        desc = image.name or "unnamed"
        if image.content_type:
            desc += f" ({image.content_type})"
        if image.alt:
            desc += f"; alt: {image.alt}"
        return f"`[image: {desc}]`"

    def _render_code(self, code: CodeBlock) -> str:
        fence = "```"
        while fence in code.text:
            fence += "`"
        return f"{fence}{code.language or ''}\n{code.text}\n{fence}"

    def _render_warning(self, warning: ExtractionWarning) -> str:
        location = f" ({warning.location})" if warning.location else ""
        return f"- **{warning.code}**{location}: {warning.message}"
