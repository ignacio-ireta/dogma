"""PPTX extraction via python-pptx.

Shapes are read in XML (z-)order, which usually matches authoring order but
not necessarily visual order — documented in the extraction contract. Group
shapes are recursed into (their text is silently lost otherwise). Speaker
notes become a separate NOTES section right after their slide.

Text mapping rule (v0.1): paragraphs inside body/content placeholders become
list items (carrying their indent level); text in other shapes becomes plain
paragraphs.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER

from docmd.errors import wrap_library_error
from docmd.extractors.base import ExtractionResult, Extractor
from docmd.types import (
    Block,
    DocumentIR,
    ExtractionWarning,
    ImageRef,
    ListItem,
    Paragraph,
    Section,
    SectionKind,
    SourceType,
    Table,
)

_TITLE_PLACEHOLDERS = {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE}
_LIST_PLACEHOLDERS = {PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT}


class PptxExtractor(Extractor):
    name = "pptx"
    version = "0.1.0"
    file_type = SourceType.PPTX

    def extract(self, path: Path, *, relative_path: str | None = None) -> ExtractionResult:
        try:
            presentation = Presentation(str(path))
        except Exception as exc:
            raise wrap_library_error(exc, path) from exc

        warnings: list[ExtractionWarning] = []
        sections: list[Section] = []
        for index, slide in enumerate(presentation.slides, start=1):
            title = None
            try:
                if slide.shapes.title is not None and slide.shapes.title.text.strip():
                    title = slide.shapes.title.text
            except (AttributeError, KeyError):
                pass

            blocks: list[Block] = []
            self._walk_shapes(slide.shapes, blocks, warnings, f"slide {index}")
            sections.append(
                Section(kind=SectionKind.SLIDE, index=index, title=title, blocks=blocks)
            )

            notes = self._notes_text(slide)
            if notes:
                sections.append(
                    Section(
                        kind=SectionKind.NOTES,
                        index=index,
                        blocks=[Paragraph(text=notes)],
                    )
                )

        return ExtractionResult(
            document=DocumentIR(
                source_path=relative_path or path.name,
                source_type=SourceType.PPTX,
                title=presentation.core_properties.title or None,
                sections=sections,
                properties=self._properties(presentation),
            ),
            warnings=warnings,
        )

    def _walk_shapes(
        self,
        shapes: object,
        blocks: list[Block],
        warnings: list[ExtractionWarning],
        location: str,
    ) -> None:
        for shape in shapes:  # type: ignore[attr-defined]
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                self._walk_shapes(shape.shapes, blocks, warnings, location)
                continue
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                blocks.append(self._picture_ref(shape))
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                blocks.append(Table(rows=rows, has_header=True))
                continue
            if not getattr(shape, "has_text_frame", False) or not shape.has_text_frame:
                continue
            if self._is_title_placeholder(shape):
                continue  # already used as the section title
            as_list = self._is_list_placeholder(shape)
            for paragraph in shape.text_frame.paragraphs:
                text = "".join(run.text for run in paragraph.runs)
                if not text.strip():
                    continue
                if as_list:
                    blocks.append(ListItem(text=text, level=paragraph.level or 0))
                else:
                    blocks.append(Paragraph(text=text))

    @staticmethod
    def _is_title_placeholder(shape: object) -> bool:
        try:
            return (
                shape.is_placeholder  # type: ignore[attr-defined]
                and shape.placeholder_format.type in _TITLE_PLACEHOLDERS  # type: ignore[attr-defined]
            )
        except (AttributeError, ValueError):
            return False

    @staticmethod
    def _is_list_placeholder(shape: object) -> bool:
        try:
            return (
                shape.is_placeholder  # type: ignore[attr-defined]
                and shape.placeholder_format.type in _LIST_PLACEHOLDERS  # type: ignore[attr-defined]
            )
        except (AttributeError, ValueError):
            return False

    @staticmethod
    def _picture_ref(shape: object) -> ImageRef:
        name = None
        content_type = None
        alt = None
        try:
            name = shape.image.filename or None  # type: ignore[attr-defined]
            content_type = shape.image.content_type  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
        with contextlib.suppress(AttributeError):
            alt = shape._element._nvXxPr.cNvPr.get("descr") or None  # type: ignore[attr-defined]
        return ImageRef(name=name, content_type=content_type, alt=alt)

    @staticmethod
    def _notes_text(slide: object) -> str | None:
        try:
            if not slide.has_notes_slide:  # type: ignore[attr-defined]
                return None
            text = slide.notes_slide.notes_text_frame.text  # type: ignore[attr-defined]
            return text if text.strip() else None
        except (AttributeError, ValueError):
            return None

    @staticmethod
    def _properties(presentation: object) -> dict[str, str]:
        core = presentation.core_properties  # type: ignore[attr-defined]
        props: dict[str, str] = {}
        if core.author:
            props["Author"] = str(core.author)
        if core.created:
            props["Created"] = core.created.isoformat()
        if core.modified:
            props["Modified"] = core.modified.isoformat()
        return props
