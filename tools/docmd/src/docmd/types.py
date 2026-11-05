"""Intermediate representation (IR) shared by all extractors and writers.

Extractors produce a ``DocumentIR``; the Markdown writer consumes one. No
extractor writes Markdown directly — this is the contract that keeps the four
formats normalized to a single structure.

v0.1 carries plain text only (no inline bold/italic spans).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    PDF = "pdf"


class SectionKind(StrEnum):
    BODY = "body"
    SLIDE = "slide"
    SHEET = "sheet"
    PAGE = "page"
    NOTES = "notes"


class Heading(BaseModel):
    kind: Literal["heading"] = "heading"
    level: int = Field(ge=1, le=6)
    text: str


class Paragraph(BaseModel):
    kind: Literal["paragraph"] = "paragraph"
    text: str


class ListItem(BaseModel):
    kind: Literal["list_item"] = "list_item"
    text: str
    level: int = Field(default=0, ge=0)
    ordered: bool = False


class Table(BaseModel):
    kind: Literal["table"] = "table"
    rows: list[list[str]]
    has_header: bool = True
    caption: str | None = None


class ImageRef(BaseModel):
    """Image placeholder — v0.1 extracts metadata only, never pixel data."""

    kind: Literal["image"] = "image"
    name: str | None = None
    content_type: str | None = None
    alt: str | None = None


class CodeBlock(BaseModel):
    kind: Literal["code"] = "code"
    text: str
    language: str | None = None


Block = Annotated[
    Heading | Paragraph | ListItem | Table | ImageRef | CodeBlock,
    Field(discriminator="kind"),
]


class ExtractionWarning(BaseModel):
    """A recoverable problem found while extracting (rendered + persisted)."""

    code: str
    message: str
    location: str | None = None


class Section(BaseModel):
    kind: SectionKind
    index: int | None = None
    title: str | None = None
    blocks: list[Block] = Field(default_factory=list)


class DocumentIR(BaseModel):
    source_path: str
    source_type: SourceType
    title: str | None = None
    sections: list[Section] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)
