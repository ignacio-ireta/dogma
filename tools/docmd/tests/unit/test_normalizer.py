"""Text hygiene: unicode, whitespace, control chars, empty-block dropping."""

from docmd.markdown.normalizer import clean_text, normalize
from docmd.types import (
    CodeBlock,
    DocumentIR,
    Paragraph,
    Section,
    SectionKind,
    SourceType,
    Table,
)


def make_doc(blocks: list) -> DocumentIR:  # type: ignore[type-arg]
    return DocumentIR(
        source_path="x.docx",
        source_type=SourceType.DOCX,
        sections=[Section(kind=SectionKind.BODY, blocks=blocks)],
    )


def test_nfc_normalization() -> None:
    # "e" + combining acute accent (U+0301) becomes the single NFC codepoint
    assert clean_text("Cafe\u0301") == "Caf\u00e9"


def test_control_chars_stripped_nbsp_collapsed() -> None:
    assert clean_text("a\x00b\x1fc  d\u00a0e") == "abc d e"


def test_crlf_and_newline_folding() -> None:
    assert clean_text("line1\r\nline2\rline3") == "line1 line2 line3"


def test_code_blocks_keep_newlines() -> None:
    doc = make_doc([CodeBlock(text="def f():\r\n    return 1 ")])
    block = normalize(doc).sections[0].blocks[0]
    assert isinstance(block, CodeBlock)
    assert block.text == "def f():\n    return 1"


def test_empty_paragraphs_dropped() -> None:
    doc = make_doc([Paragraph(text="  \u00a0 "), Paragraph(text="kept")])
    blocks = normalize(doc).sections[0].blocks
    assert len(blocks) == 1
    assert isinstance(blocks[0], Paragraph) and blocks[0].text == "kept"


def test_fully_empty_table_dropped() -> None:
    doc = make_doc([Table(rows=[["", " "], ["", ""]])])
    assert normalize(doc).sections[0].blocks == []


def test_table_cells_cleaned() -> None:
    doc = make_doc([Table(rows=[["a\u00a0b", "c\nd"]])])
    block = normalize(doc).sections[0].blocks[0]
    assert isinstance(block, Table)
    assert block.rows == [["a b", "c d"]]


def test_unicode_text_preserved() -> None:
    assert clean_text("Sørlandet: ¿dónde? 🎉") == "Sørlandet: ¿dónde? 🎉"
