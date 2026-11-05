"""Extractor integration tests over generated fixtures: structure + errors."""

from pathlib import Path

import pytest

from docmd.errors import (
    FileCorruptionError,
    PasswordProtectedError,
    ScannedPdfError,
)
from docmd.extractors import get_extractor
from docmd.types import (
    Heading,
    ImageRef,
    ListItem,
    Paragraph,
    SectionKind,
    SourceType,
    Table,
)

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------------ docx


def test_docx_basic_structure(fixtures_dir: Path) -> None:
    result = get_extractor(SourceType.DOCX).extract(fixtures_dir / "docx" / "basic.docx")
    assert result.status == "success"
    doc = result.document
    assert doc.title == "Fixture Document"
    assert doc.properties["Author"] == "Fixture Author"
    blocks = doc.sections[0].blocks

    headings = [b for b in blocks if isinstance(b, Heading)]
    assert [h.text for h in headings][:2] == ["Fixture Document", "Lists"]

    lists = [b for b in blocks if isinstance(b, ListItem)]
    assert [(li.text, li.level, li.ordered) for li in lists] == [
        ("First bullet", 0, False),
        ("Nested bullet", 1, False),
        ("Second bullet", 0, False),
        ("Numbered one", 0, True),
        ("Numbered two", 0, True),
    ]

    # Interleaving: paragraph -> table -> paragraph order is preserved.
    kinds = [type(b).__name__ for b in blocks]
    table_idx = kinds.index("Table")
    assert isinstance(blocks[table_idx - 1], Paragraph)
    assert blocks[table_idx - 1].text == "Before the table."
    assert isinstance(blocks[table_idx + 1], Paragraph)
    assert blocks[table_idx + 1].text == "After the table."

    table = blocks[table_idx]
    assert isinstance(table, Table)
    assert table.rows[0] == ["Name", "Qty", "Price"]

    links = [b for b in blocks if isinstance(b, Paragraph) and "](" in b.text]
    assert links and "[the example site](https://example.com/)" in links[0].text

    images = [b for b in blocks if isinstance(b, ImageRef)]
    assert images and images[0].content_type == "image/png"


def test_docx_empty(fixtures_dir: Path) -> None:
    result = get_extractor(SourceType.DOCX).extract(fixtures_dir / "docx" / "empty.docx")
    assert result.status == "success"
    assert result.document.sections[0].blocks == []


# ------------------------------------------------------------------------ pptx


def test_pptx_basic_structure(fixtures_dir: Path) -> None:
    result = get_extractor(SourceType.PPTX).extract(fixtures_dir / "pptx" / "basic.pptx")
    doc = result.document
    slide_sections = [s for s in doc.sections if s.kind is SectionKind.SLIDE]
    notes_sections = [s for s in doc.sections if s.kind is SectionKind.NOTES]
    assert len(slide_sections) == 4
    assert slide_sections[0].title == "Fixture Deck"
    assert slide_sections[1].title == "Agenda"

    agenda = slide_sections[1].blocks
    assert [(b.text, b.level) for b in agenda if isinstance(b, ListItem)] == [
        ("Point one", 0),
        ("Sub point", 1),
        ("Point two", 0),
    ]

    assert len(notes_sections) == 1
    assert notes_sections[0].index == 2
    notes_block = notes_sections[0].blocks[0]
    assert isinstance(notes_block, Paragraph)
    assert "Speaker notes" in notes_block.text

    tables = [b for s in doc.sections for b in s.blocks if isinstance(b, Table)]
    assert tables and tables[0].rows[0] == ["Metric", "Value"]

    # Group recursion: slide 4 text must not be lost.
    slide4_texts = [b.text for b in slide_sections[3].blocks if isinstance(b, Paragraph)]
    assert "Text inside a group" in slide4_texts
    assert "More grouped text" in slide4_texts


def test_pptx_empty(fixtures_dir: Path) -> None:
    result = get_extractor(SourceType.PPTX).extract(fixtures_dir / "pptx" / "empty.pptx")
    assert result.status == "success"
    assert len(result.document.sections) == 1
    assert result.document.sections[0].blocks == []


# ------------------------------------------------------------------------ xlsx


def test_xlsx_basic_structure(fixtures_dir: Path) -> None:
    result = get_extractor(SourceType.XLSX).extract(fixtures_dir / "xlsx" / "basic.xlsx")
    doc = result.document
    assert [s.title for s in doc.sections] == ["Data", "Notes"]

    data = doc.sections[0].blocks[0]
    assert isinstance(data, Table)
    assert data.rows[0] == ["Item", "Qty", "Price", "InStock", "Added", "Notes"]
    assert data.rows[1][3] == "TRUE"
    assert data.rows[1][4] == "2020-05-17"
    # Formula without cached value renders as formula text + warning.
    assert any("=SUM(B2:B4)" in cell for row in data.rows for cell in row)
    assert any(w.code == "FORMULA_NO_CACHED_VALUE" for w in result.warnings)
    # Merged cell: anchor carries the value.
    assert any(row[0] == "Merged note" for row in data.rows)


def test_xlsx_cached_formula(fixtures_dir: Path) -> None:
    result = get_extractor(SourceType.XLSX).extract(fixtures_dir / "xlsx" / "cached_formula.xlsx")
    assert result.status == "success"
    table = result.document.sections[0].blocks[0]
    assert isinstance(table, Table)
    assert table.rows[-1] == ["Total", "42 (=SUM(B2:B3))"]


def test_xlsx_empty(fixtures_dir: Path) -> None:
    result = get_extractor(SourceType.XLSX).extract(fixtures_dir / "xlsx" / "empty.xlsx")
    assert result.status == "success"
    assert result.document.sections[0].blocks == []


def test_xlsx_truncation_warning(fixtures_dir: Path) -> None:
    from docmd.extractors.xlsx_extractor import XlsxExtractor

    extractor = XlsxExtractor(max_rows_per_sheet=2, max_cols_per_sheet=3)
    result = extractor.extract(fixtures_dir / "xlsx" / "basic.xlsx")
    truncations = [w for w in result.warnings if w.code == "SHEET_TRUNCATED"]
    assert truncations
    table = result.document.sections[0].blocks[0]
    assert isinstance(table, Table)
    assert len(table.rows) <= 2
    assert all(len(row) <= 3 for row in table.rows)


# ------------------------------------------------------------------------- pdf


def test_pdf_basic_structure(fixtures_dir: Path) -> None:
    result = get_extractor(SourceType.PDF).extract(fixtures_dir / "pdf" / "basic.pdf")
    doc = result.document
    assert doc.title == "Fixture PDF"
    assert doc.properties["Pages"] == "2"
    assert [s.index for s in doc.sections] == [1, 2]

    page1 = doc.sections[0].blocks
    headings = [b for b in page1 if isinstance(b, Heading)]
    assert headings[0].text.strip() == "Fixture PDF Heading"
    assert headings[0].level == 1
    assert any(isinstance(b, Heading) and b.level >= 2 for b in headings[1:])

    page2 = doc.sections[1].blocks
    tables = [b for b in page2 if isinstance(b, Table)]
    assert tables, "ruled table should be detected on page 2"
    flat = [cell for row in tables[0].rows for cell in row]
    assert "Region" in " ".join(flat)
    # Table text must not be duplicated as paragraphs.
    paragraph_text = " ".join(b.text for b in page2 if isinstance(b, Paragraph))
    assert "North" not in paragraph_text


def test_pdf_empty_page(fixtures_dir: Path) -> None:
    result = get_extractor(SourceType.PDF).extract(fixtures_dir / "pdf" / "empty.pdf")
    assert result.status == "success"
    assert result.document.sections[0].blocks == []


def test_pdf_scanned_fails_structured(fixtures_dir: Path) -> None:
    with pytest.raises(ScannedPdfError) as exc_info:
        get_extractor(SourceType.PDF).extract(fixtures_dir / "pdf" / "scanned.pdf")
    assert exc_info.value.error_code == "SCANNED_PDF_OCR_DISABLED"


def test_pdf_encrypted_fails_structured(fixtures_dir: Path) -> None:
    with pytest.raises(PasswordProtectedError):
        get_extractor(SourceType.PDF).extract(fixtures_dir / "pdf" / "encrypted.pdf")


# ----------------------------------------------------------------- error paths


@pytest.mark.parametrize(
    ("rel", "source_type", "expected"),
    [
        ("corrupt/truncated.docx", SourceType.DOCX, FileCorruptionError),
        ("corrupt/not_a_zip.docx", SourceType.DOCX, FileCorruptionError),
        ("corrupt/zero_bytes.docx", SourceType.DOCX, FileCorruptionError),
        ("corrupt/zip_not_office.docx", SourceType.DOCX, FileCorruptionError),
        ("corrupt/ole_encrypted.docx", SourceType.DOCX, PasswordProtectedError),
        ("corrupt/truncated.pdf", SourceType.PDF, FileCorruptionError),
        ("corrupt/docx_named.pdf", SourceType.PDF, FileCorruptionError),
    ],
)
def test_corrupt_files_raise_domain_errors(
    fixtures_dir: Path, rel: str, source_type: SourceType, expected: type[Exception]
) -> None:
    with pytest.raises(expected):
        get_extractor(source_type).extract(fixtures_dir / rel)
