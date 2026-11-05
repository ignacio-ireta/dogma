"""The fixture generator produces the full, valid fixture set."""

import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

EXPECTED_FILES = [
    "docx/basic.docx",
    "docx/empty.docx",
    "docx/special_chars.docx",
    "pptx/basic.pptx",
    "pptx/empty.pptx",
    "xlsx/basic.xlsx",
    "xlsx/cached_formula.xlsx",
    "xlsx/empty.xlsx",
    "pdf/basic.pdf",
    "pdf/scanned.pdf",
    "pdf/scanned_text.pdf",
    "pdf/mixed_scan.pdf",
    "pdf/encrypted.pdf",
    "pdf/empty.pdf",
    "corrupt/truncated.docx",
    "corrupt/not_a_zip.docx",
    "corrupt/zero_bytes.docx",
    "corrupt/zip_not_office.docx",
    "corrupt/ole_encrypted.docx",
    "corrupt/truncated.pdf",
    "corrupt/docx_named.pdf",
    "misc/readme.txt",
    "misc/~$lock.docx",
    "tree/report.docx",
    "tree/report.pdf",
    "tree/slides/deck.pptx",
    "tree/finance/2026/budget.xlsx",
    "tree/broken/truncated.docx",
    "tree/broken/scanned.pdf",
    "tree/notes.txt",
    "tree/~$report.docx",
]


def test_all_fixtures_exist(fixtures_dir: Path) -> None:
    missing = [rel for rel in EXPECTED_FILES if not (fixtures_dir / rel).exists()]
    assert not missing, f"missing fixtures: {missing}"


def test_good_office_fixtures_are_valid_zips(fixtures_dir: Path) -> None:
    for rel in ["docx/basic.docx", "pptx/basic.pptx", "xlsx/basic.xlsx"]:
        assert zipfile.is_zipfile(fixtures_dir / rel), f"{rel} is not a valid zip"


def test_corrupt_fixtures_are_actually_broken(fixtures_dir: Path) -> None:
    assert not zipfile.is_zipfile(fixtures_dir / "corrupt" / "not_a_zip.docx")
    assert (fixtures_dir / "corrupt" / "zero_bytes.docx").stat().st_size == 0
    head = (fixtures_dir / "corrupt" / "ole_encrypted.docx").read_bytes()[:8]
    assert head == bytes.fromhex("D0CF11E0A1B11AE1")


def test_encrypted_pdf_needs_password(fixtures_dir: Path) -> None:
    import pymupdf

    doc = pymupdf.open(fixtures_dir / "pdf" / "encrypted.pdf")
    assert doc.needs_pass
    doc.close()


def test_scanned_pdf_has_no_text(fixtures_dir: Path) -> None:
    import pymupdf

    doc = pymupdf.open(fixtures_dir / "pdf" / "scanned.pdf")
    assert not doc[0].get_text().strip()
    assert doc[0].get_images()
    doc.close()


def test_ocr_fixtures_are_image_only_where_expected(fixtures_dir: Path) -> None:
    import pymupdf

    doc = pymupdf.open(fixtures_dir / "pdf" / "scanned_text.pdf")
    assert not doc[0].get_text().strip()
    assert doc[0].get_images()
    doc.close()

    doc = pymupdf.open(fixtures_dir / "pdf" / "mixed_scan.pdf")
    assert "Native Text Heading" in doc[0].get_text()
    assert not doc[1].get_text().strip()
    assert doc[1].get_images()
    doc.close()
