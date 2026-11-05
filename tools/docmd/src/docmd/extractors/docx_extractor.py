"""DOCX extraction via python-docx (>= 1.1 required for iter_inner_content).

Block order matters: ``Document.iter_inner_content()`` yields paragraphs and
tables interleaved in document order — reading ``.paragraphs`` and ``.tables``
separately would lose that ordering. Footnotes, comments, and tracked changes
are not extracted in v0.1 (declared in docs/extraction_contract.md); their
presence is reported as a warning.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import docx
from docx.document import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.hyperlink import Hyperlink
from docx.text.paragraph import Paragraph as DocxParagraph
from docx.text.run import Run

from docmd.errors import wrap_library_error
from docmd.extractors.base import ExtractionResult, Extractor
from docmd.types import (
    Block,
    DocumentIR,
    ExtractionWarning,
    Heading,
    ImageRef,
    ListItem,
    Paragraph,
    Section,
    SectionKind,
    SourceType,
    Table,
)

_HEADING_STYLE = re.compile(r"^Heading (\d)$")
_LIST_STYLE = re.compile(r"^List (Bullet|Number)(?: (\d))?$")


class DocxExtractor(Extractor):
    name = "docx"
    version = "0.1.0"
    file_type = SourceType.DOCX

    def extract(self, path: Path, *, relative_path: str | None = None) -> ExtractionResult:
        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise wrap_library_error(exc, path) from exc

        warnings: list[ExtractionWarning] = []
        blocks: list[Block] = []
        for item in document.iter_inner_content():
            if isinstance(item, DocxParagraph):
                blocks.extend(self._convert_paragraph(item))
            elif isinstance(item, DocxTable):
                blocks.append(self._convert_table(item))

        self._warn_unsupported_elements(document, warnings)

        return ExtractionResult(
            document=DocumentIR(
                source_path=relative_path or path.name,
                source_type=SourceType.DOCX,
                title=document.core_properties.title or None,
                sections=[Section(kind=SectionKind.BODY, blocks=blocks)],
                properties=self._properties(document),
            ),
            warnings=warnings,
        )

    def _convert_paragraph(self, paragraph: DocxParagraph) -> list[Block]:
        # Reconstruct text in run order so hyperlinks become inline MD links.
        pieces: list[str] = []
        for item in paragraph.iter_inner_content():
            if isinstance(item, Run):
                pieces.append(item.text)
            elif isinstance(item, Hyperlink):
                if item.address:
                    pieces.append(f"[{item.text}]({item.address})")
                else:
                    pieces.append(item.text)
        text = "".join(pieces)

        blocks: list[Block] = []
        style_name = self._style_name(paragraph)
        heading = _HEADING_STYLE.match(style_name)
        if style_name == "Title":
            blocks.append(Heading(level=1, text=text))
        elif heading:
            blocks.append(Heading(level=min(int(heading.group(1)), 6), text=text))
        elif list_match := _LIST_STYLE.match(style_name):
            ordered = list_match.group(1) == "Number"
            # "List Bullet 2" means second indent level -> level 1.
            level = int(list_match.group(2)) - 1 if list_match.group(2) else self._ilvl(paragraph)
            blocks.append(ListItem(text=text, level=level, ordered=ordered))
        elif text.strip():
            blocks.append(Paragraph(text=text))

        blocks.extend(self._images_in(paragraph))
        return blocks

    @staticmethod
    def _style_name(paragraph: DocxParagraph) -> str:
        try:
            style = paragraph.style
            return str(style.name) if style is not None and style.name else ""
        except (AttributeError, KeyError):
            return ""

    @staticmethod
    def _ilvl(paragraph: DocxParagraph) -> int:
        """Nesting level from numbering XML; the one raw-XML escape hatch."""
        try:
            return int(paragraph._p.pPr.numPr.ilvl.val)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return 0

    @staticmethod
    def _images_in(paragraph: DocxParagraph) -> list[ImageRef]:
        images: list[ImageRef] = []
        try:
            element = paragraph._p
        except AttributeError:
            return images
        try:
            rids = element.xpath(".//a:blip/@r:embed")
        except Exception:
            rids = []
        if not rids:
            try:
                if element.xpath(".//w:drawing"):
                    images.append(ImageRef())
            except Exception:
                pass
            return images
        for rid in rids:
            image_part = paragraph.part.related_parts.get(rid)
            if image_part is None:
                images.append(ImageRef())
                continue
            images.append(
                ImageRef(
                    name=PurePosixPath(str(image_part.partname)).name,
                    content_type=image_part.content_type,
                )
            )
        return images

    @staticmethod
    def _convert_table(table: DocxTable) -> Table:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        return Table(rows=rows, has_header=True)

    @staticmethod
    def _warn_unsupported_elements(
        document: DocxDocument, warnings: list[ExtractionWarning]
    ) -> None:
        body = document.element.body
        if body.xpath(".//w:footnoteReference"):
            warnings.append(
                ExtractionWarning(
                    code="FOOTNOTES_NOT_EXTRACTED",
                    message="document contains footnotes; not extracted in v0.1",
                )
            )
        if body.xpath(".//w:commentReference"):
            warnings.append(
                ExtractionWarning(
                    code="COMMENTS_NOT_EXTRACTED",
                    message="document contains comments; not extracted in v0.1",
                )
            )
        if body.xpath(".//w:ins | .//w:del"):
            warnings.append(
                ExtractionWarning(
                    code="TRACK_CHANGES_NOT_EXTRACTED",
                    message="document contains tracked changes; markup ignored in v0.1",
                )
            )

    @staticmethod
    def _properties(document: DocxDocument) -> dict[str, str]:
        core = document.core_properties
        props: dict[str, str] = {}
        if core.author:
            props["Author"] = str(core.author)
        if core.created:
            props["Created"] = core.created.isoformat()
        if core.modified:
            props["Modified"] = core.modified.isoformat()
        return props
