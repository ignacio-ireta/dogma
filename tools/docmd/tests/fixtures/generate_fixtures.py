"""Reproducible binary fixture generator.

Fixtures are built programmatically (never committed) into ``_build/`` so they
have a readable, reviewable source of truth instead of being mystery blobs.
Document core properties are pinned to a fixed epoch so extracted *content* is
deterministic even though the container bytes are not byte-stable.

Run standalone:
    uv run python tests/fixtures/generate_fixtures.py [--out DIR] [--include-huge]

Known limitation: these files are library-authored, not Word/Excel-authored.
Real-world samples can be added later under tests/fixtures/external/ with a
provenance README.
"""

from __future__ import annotations

import argparse
import base64
import io
import shutil
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

FIXED_DT = datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC)
FIXTURE_AUTHOR = "Fixture Author"
OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")

# A valid 1x1 red PNG.
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


# --------------------------------------------------------------------------- docx


def build_docx_basic(path: Path) -> None:
    import docx
    from docx.opc.constants import RELATIONSHIP_TYPE
    from docx.oxml.ns import qn
    from docx.oxml.parser import OxmlElement

    doc = docx.Document()
    doc.core_properties.author = FIXTURE_AUTHOR
    doc.core_properties.created = FIXED_DT
    doc.core_properties.modified = FIXED_DT
    doc.core_properties.title = "Fixture Document"

    doc.add_heading("Fixture Document", level=1)
    p = doc.add_paragraph("Intro with ")
    p.add_run("bold").bold = True
    p.add_run(" and ")
    p.add_run("italic").italic = True
    p.add_run(" runs.")

    doc.add_heading("Lists", level=2)
    doc.add_paragraph("First bullet", style="List Bullet")
    doc.add_paragraph("Nested bullet", style="List Bullet 2")
    doc.add_paragraph("Second bullet", style="List Bullet")
    doc.add_paragraph("Numbered one", style="List Number")
    doc.add_paragraph("Numbered two", style="List Number")

    doc.add_heading("Table", level=2)
    doc.add_paragraph("Before the table.")
    table = doc.add_table(rows=3, cols=3)
    for col, text in enumerate(["Name", "Qty", "Price"]):
        table.cell(0, col).text = text
    for row, values in enumerate([["Apples", "3", "1.50"], ["Pears", "5", "2.00"]], start=1):
        for col, text in enumerate(values):
            table.cell(row, col).text = text
    doc.add_paragraph("After the table.")

    link_para = doc.add_paragraph("Visit ")
    part = link_para.part
    r_id = part.relate_to("https://example.com/", RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    text_el = OxmlElement("w:t")
    text_el.text = "the example site"
    run.append(text_el)
    hyperlink.append(run)
    link_para._p.append(hyperlink)

    doc.add_heading("Image", level=2)
    doc.add_picture(io.BytesIO(PNG_1PX))

    doc.save(str(path))


def build_docx_empty(path: Path) -> None:
    import docx

    doc = docx.Document()
    doc.core_properties.created = FIXED_DT
    doc.core_properties.modified = FIXED_DT
    doc.save(str(path))


def build_docx_special_chars(path: Path) -> None:
    import docx

    doc = docx.Document()
    doc.core_properties.created = FIXED_DT
    doc.core_properties.modified = FIXED_DT
    doc.add_paragraph("Pipes | stars * underscores _ hashes # backticks `")
    doc.add_paragraph("# starts like a heading but is not")
    doc.add_paragraph("- starts like a list but is not")
    doc.add_paragraph("Café naïve Sørlandet ¿dónde?")
    doc.add_paragraph("日本語テキスト and emoji \U0001f389")
    doc.add_paragraph("non\u00a0breaking\u00a0spaces")
    doc.save(str(path))


# --------------------------------------------------------------------------- pptx


def build_pptx_basic(path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Emu, Inches

    prs = Presentation()
    prs.core_properties.author = FIXTURE_AUTHOR
    prs.core_properties.created = FIXED_DT
    prs.core_properties.modified = FIXED_DT
    prs.core_properties.title = "Fixture Deck"

    slide1 = prs.slides.add_slide(prs.slide_layouts[0])
    slide1.shapes.title.text = "Fixture Deck"
    slide1.placeholders[1].text = "A deck for extractor tests"

    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "Agenda"
    body = slide2.placeholders[1].text_frame
    body.text = "Point one"
    p = body.add_paragraph()
    p.text = "Sub point"
    p.level = 1
    p = body.add_paragraph()
    p.text = "Point two"
    slide2.notes_slide.notes_text_frame.text = "Speaker notes for slide two."

    slide3 = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    table_shape = slide3.shapes.add_table(3, 2, Inches(1), Inches(1), Inches(6), Inches(2))
    cells = [["Metric", "Value"], ["Users", "1200"], ["Churn", "3%"]]
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            table_shape.table.cell(r, c).text = text
    textbox = slide3.shapes.add_textbox(Inches(1), Inches(4), Inches(4), Inches(1))
    textbox.text_frame.text = "Stray textbox"

    slide4 = prs.slides.add_slide(prs.slide_layouts[6])
    group = slide4.shapes.add_group_shape()
    inner = group.shapes.add_textbox(Emu(914400), Emu(914400), Emu(2743200), Emu(457200))
    inner.text_frame.text = "Text inside a group"
    inner2 = group.shapes.add_textbox(Emu(914400), Emu(1828800), Emu(2743200), Emu(457200))
    inner2.text_frame.text = "More grouped text"

    prs.save(str(path))


def build_pptx_empty(path: Path) -> None:
    from pptx import Presentation

    prs = Presentation()
    prs.core_properties.created = FIXED_DT
    prs.core_properties.modified = FIXED_DT
    prs.slides.add_slide(prs.slide_layouts[6])
    prs.save(str(path))


# --------------------------------------------------------------------------- xlsx


def build_xlsx_basic(path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    wb.properties.creator = FIXTURE_AUTHOR
    wb.properties.created = FIXED_DT.replace(tzinfo=None)
    wb.properties.modified = FIXED_DT.replace(tzinfo=None)
    ws = wb.active
    assert ws is not None
    ws.title = "Data"
    ws.append(["Item", "Qty", "Price", "InStock", "Added", "Notes"])
    ws.append(["Laptop", 2, 1999.99, True, date(2020, 5, 17), "with | pipe"])
    ws.append(["Mouse", 10, 25.5, False, date(2020, 6, 1), None])
    ws.append(["Desk", 1, 450.0, True, date(2020, 7, 4), "unicode æøå"])
    # Formula written by openpyxl has NO cached value -> warning path.
    ws["B6"] = "=SUM(B2:B4)"
    ws["A6"] = "Total"
    ws.merge_cells("A8:B8")
    ws["A8"] = "Merged note"

    notes = wb.create_sheet("Notes")
    notes["C3"] = "Sparse sheet content"

    wb.save(str(path))


def build_xlsx_cached_formula(path: Path) -> None:
    """xlsxwriter can store a formula together with its cached value."""
    import xlsxwriter

    wb = xlsxwriter.Workbook(str(path))
    wb.set_properties({"author": FIXTURE_AUTHOR, "created": FIXED_DT.replace(tzinfo=None)})
    ws = wb.add_worksheet("Calc")
    ws.write_string(0, 0, "Label")
    ws.write_string(0, 1, "Amount")
    ws.write_string(1, 0, "Alpha")
    ws.write_number(1, 1, 40)
    ws.write_string(2, 0, "Beta")
    ws.write_number(2, 1, 2)
    ws.write_string(3, 0, "Total")
    ws.write_formula(3, 1, "=SUM(B2:B3)", None, 42)
    wb.close()


def build_xlsx_empty(path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    wb.properties.created = FIXED_DT.replace(tzinfo=None)
    wb.properties.modified = FIXED_DT.replace(tzinfo=None)
    wb.save(str(path))


def build_xlsx_huge(path: Path, rows: int = 50_000, cols: int = 10) -> None:
    from openpyxl import Workbook

    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Big")
    ws.append([f"col{i}" for i in range(cols)])
    for r in range(rows):
        ws.append([f"r{r}c{c}" for c in range(cols)])
    wb.save(str(path))


# --------------------------------------------------------------------------- pdf


def _pdf_metadata() -> dict[str, str]:
    return {
        "title": "Fixture PDF",
        "author": FIXTURE_AUTHOR,
        "creationDate": "D:20200101120000Z",
        "modDate": "D:20200101120000Z",
    }


def build_pdf_basic(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "Fixture PDF Heading", fontsize=24)
    page.insert_text((72, 130), "First body paragraph with enough text to matter.", fontsize=11)
    page.insert_text((72, 150), "Second body line continues the paragraph.", fontsize=11)
    page.insert_text((72, 190), "Subheading on page one", fontsize=16)
    page.insert_text((72, 220), "Trailing paragraph after the subheading.", fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((72, 80), "Page two opens with a ruled table:", fontsize=11)
    # Draw a ruled 3x2 table so pymupdf's find_tables() can detect it.
    x0, y0, col_w, row_h, n_rows, n_cols = 72, 100, 150, 24, 3, 2
    cells = [["Region", "Revenue"], ["North", "1200"], ["South", "900"]]
    for r in range(n_rows + 1):
        y = y0 + r * row_h
        page2.draw_line((x0, y), (x0 + n_cols * col_w, y))
    for c in range(n_cols + 1):
        x = x0 + c * col_w
        page2.draw_line((x, y0), (x, y0 + n_rows * row_h))
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            page2.insert_text((x0 + c * col_w + 6, y0 + r * row_h + 16), text, fontsize=10)
    page2.insert_text((72, y0 + n_rows * row_h + 40), "Paragraph after the table.", fontsize=11)

    doc.set_metadata(_pdf_metadata())
    doc.save(str(path))
    doc.close()


def build_pdf_scanned(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=PNG_1PX)
    doc.set_metadata(_pdf_metadata())
    doc.save(str(path))
    doc.close()


def build_pdf_scanned_text(path: Path) -> None:
    """Image-only page whose image contains real rendered text, for OCR tests."""
    import pymupdf

    source = pymupdf.open()
    page = source.new_page()
    page.insert_text((72, 100), "Scanned Fixture Title", fontsize=24)
    page.insert_text((72, 160), "Quarterly revenue grew nine percent year over year.", fontsize=12)
    page.insert_text((72, 180), "Operating costs stayed flat across all regions.", fontsize=12)
    pix = source[0].get_pixmap(dpi=150)
    source.close()

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=pix)
    doc.set_metadata(_pdf_metadata())
    doc.save(str(path))
    doc.close()


def build_pdf_mixed_scan(path: Path) -> None:
    """Page 1 has a native text layer; page 2 is a scan (auto OCRs only page 2)."""
    import pymupdf

    source = pymupdf.open()
    spage = source.new_page()
    spage.insert_text((72, 120), "This sentence exists only as pixels on page two.", fontsize=14)
    pix = source[0].get_pixmap(dpi=150)
    source.close()

    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 90), "Native Text Heading", fontsize=24)
    page1.insert_text((72, 130), "Page one keeps its embedded text layer.", fontsize=11)
    page2 = doc.new_page()
    page2.insert_image(page2.rect, pixmap=pix)
    doc.set_metadata(_pdf_metadata())
    doc.save(str(path))
    doc.close()


def build_pdf_encrypted(path: Path, password: str = "secret") -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "You should not be able to read this without a password.")
    doc.set_metadata(_pdf_metadata())
    doc.save(
        str(path),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw=password,
    )
    doc.close()


def build_pdf_empty(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    doc.set_metadata(_pdf_metadata())
    doc.save(str(path))
    doc.close()


# ------------------------------------------------------------------ corrupt & misc


def build_corrupt_set(directory: Path, docx_source: Path, pdf_source: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    docx_bytes = docx_source.read_bytes()

    (directory / "truncated.docx").write_bytes(docx_bytes[: int(len(docx_bytes) * 0.6)])
    (directory / "not_a_zip.docx").write_bytes(b"this is not a zip archive at all")
    (directory / "zero_bytes.docx").write_bytes(b"")
    (directory / "ole_encrypted.docx").write_bytes(OLE_MAGIC + b"\x00" * 512)
    (directory / "truncated.pdf").write_bytes(b"%PDF-1.7\n% nothing useful follows\n")
    # Real DOCX bytes under a .pdf name: extension/magic mismatch.
    (directory / "docx_named.pdf").write_bytes(docx_bytes)

    with zipfile.ZipFile(directory / "zip_not_office.docx", "w") as zf:
        zf.writestr("readme.txt", "a zip, but not an OOXML package")

    _ = pdf_source  # reserved for future corrupt-pdf variants


def build_misc(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "readme.txt").write_text("unsupported file type\n")
    (directory / "~$lock.docx").write_bytes(b"\x00" * 16)


def build_tree(tree_dir: Path, build_dir: Path) -> None:
    """Nested folder combining good, broken, and ignorable files for e2e runs."""
    if tree_dir.exists():
        shutil.rmtree(tree_dir)
    (tree_dir / "slides").mkdir(parents=True)
    (tree_dir / "finance" / "2026").mkdir(parents=True)
    (tree_dir / "broken").mkdir(parents=True)

    shutil.copy(build_dir / "docx" / "basic.docx", tree_dir / "report.docx")
    shutil.copy(build_dir / "pdf" / "basic.pdf", tree_dir / "report.pdf")
    shutil.copy(build_dir / "pptx" / "basic.pptx", tree_dir / "slides" / "deck.pptx")
    shutil.copy(build_dir / "xlsx" / "basic.xlsx", tree_dir / "finance" / "2026" / "budget.xlsx")
    shutil.copy(build_dir / "corrupt" / "truncated.docx", tree_dir / "broken" / "truncated.docx")
    shutil.copy(build_dir / "pdf" / "scanned.pdf", tree_dir / "broken" / "scanned.pdf")
    (tree_dir / "notes.txt").write_text("plain text, unsupported\n")
    (tree_dir / "~$report.docx").write_bytes(b"\x00" * 16)


# ----------------------------------------------------------------------- driver

GENERATORS = {
    "docx/basic.docx": build_docx_basic,
    "docx/empty.docx": build_docx_empty,
    "docx/special_chars.docx": build_docx_special_chars,
    "pptx/basic.pptx": build_pptx_basic,
    "pptx/empty.pptx": build_pptx_empty,
    "xlsx/basic.xlsx": build_xlsx_basic,
    "xlsx/cached_formula.xlsx": build_xlsx_cached_formula,
    "xlsx/empty.xlsx": build_xlsx_empty,
    "pdf/basic.pdf": build_pdf_basic,
    "pdf/scanned.pdf": build_pdf_scanned,
    "pdf/scanned_text.pdf": build_pdf_scanned_text,
    "pdf/mixed_scan.pdf": build_pdf_mixed_scan,
    "pdf/encrypted.pdf": build_pdf_encrypted,
    "pdf/empty.pdf": build_pdf_empty,
}


def generate_all(out_dir: Path, *, include_huge: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel, builder in GENERATORS.items():
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        builder(target)
    build_corrupt_set(
        out_dir / "corrupt", out_dir / "docx" / "basic.docx", out_dir / "pdf" / "basic.pdf"
    )
    build_misc(out_dir / "misc")
    build_tree(out_dir / "tree", out_dir)
    if include_huge:
        build_xlsx_huge(out_dir / "xlsx" / "huge.xlsx")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "_build",
        help="output directory (default: tests/fixtures/_build)",
    )
    parser.add_argument("--include-huge", action="store_true", help="also build the 50k-row xlsx")
    args = parser.parse_args()
    out = generate_all(args.out, include_huge=args.include_huge)
    count = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"generated {count} fixture files under {out}")


if __name__ == "__main__":
    main()
