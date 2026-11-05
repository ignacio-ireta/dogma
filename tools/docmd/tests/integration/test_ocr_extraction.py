"""OCR extraction paths.

TestFakeBackend exercises the extractor's OCR wiring with a stub engine and
runs on every platform; TestAppleVision drives the real Vision engine and is
skipped wherever it cannot run (non-macOS or missing ocr extra).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from docmd.config import OcrMode
from docmd.errors import OcrError, OcrNoTextError, ScannedPdfError
from docmd.extractors.pdf_extractor import PdfExtractor
from docmd.ocr import OcrLine, is_ocr_available
from docmd.types import Heading, Paragraph, Table

pytestmark = pytest.mark.integration


class FakeBackend:
    name = "fake"

    def __init__(self, lines: list[OcrLine] | None = None, error: Exception | None = None) -> None:
        self.lines = lines or []
        self.error = error
        self.calls = 0

    def recognize(self, png: bytes) -> list[OcrLine]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.lines


def fake_line(text: str, *, y: float = 0.5, confidence: float = 1.0) -> OcrLine:
    return OcrLine(text=text, confidence=confidence, bbox=(0.1, y, 0.5, 0.02))


class TestFakeBackend:
    def test_auto_replaces_scanned_page_with_ocr_text(self, fixtures_dir: Path) -> None:
        backend = FakeBackend(lines=[fake_line("Recovered sentence.")])
        extractor = PdfExtractor(ocr_mode=OcrMode.AUTO, ocr_backend=backend)
        result = extractor.extract(fixtures_dir / "pdf" / "scanned.pdf")

        assert backend.calls == 1
        assert result.status == "warning"
        codes = [w.code for w in result.warnings]
        assert "OCR_APPLIED" in codes
        assert "SCANNED_PAGE" not in codes
        paragraphs = [b for b in result.document.sections[0].blocks if isinstance(b, Paragraph)]
        assert paragraphs and paragraphs[0].text == "Recovered sentence."

    def test_auto_skips_pages_with_native_text(self, fixtures_dir: Path) -> None:
        backend = FakeBackend(lines=[fake_line("should never be used")])
        extractor = PdfExtractor(ocr_mode=OcrMode.AUTO, ocr_backend=backend)
        extractor.extract(fixtures_dir / "pdf" / "basic.pdf")
        assert backend.calls == 0

    def test_low_confidence_warning(self, fixtures_dir: Path) -> None:
        backend = FakeBackend(lines=[fake_line("blurry text", confidence=0.2)])
        extractor = PdfExtractor(ocr_mode=OcrMode.AUTO, ocr_backend=backend)
        result = extractor.extract(fixtures_dir / "pdf" / "scanned.pdf")
        assert "OCR_LOW_CONFIDENCE" in [w.code for w in result.warnings]

    def test_engine_failure_wrapped_as_ocr_error(self, fixtures_dir: Path) -> None:
        backend = FakeBackend(error=RuntimeError("vision exploded"))
        extractor = PdfExtractor(ocr_mode=OcrMode.AUTO, ocr_backend=backend)
        with pytest.raises(OcrError) as exc_info:
            extractor.extract(fixtures_dir / "pdf" / "scanned.pdf")
        assert exc_info.value.error_code == "OCR_FAILED"
        assert "vision exploded" in (exc_info.value.detail or "")

    def test_no_text_found_fails_structured(self, fixtures_dir: Path) -> None:
        extractor = PdfExtractor(ocr_mode=OcrMode.AUTO, ocr_backend=FakeBackend())
        with pytest.raises(OcrNoTextError) as exc_info:
            extractor.extract(fixtures_dir / "pdf" / "scanned.pdf")
        assert exc_info.value.error_code == "OCR_NO_TEXT"

    def test_force_ocrs_every_page(self, fixtures_dir: Path) -> None:
        backend = FakeBackend(lines=[fake_line("forced output")])
        extractor = PdfExtractor(ocr_mode=OcrMode.FORCE, ocr_backend=backend)
        result = extractor.extract(fixtures_dir / "pdf" / "basic.pdf")

        assert backend.calls == 2  # basic.pdf has two pages
        for section in result.document.sections:
            assert all(isinstance(b, Paragraph) for b in section.blocks)
            assert not any(isinstance(b, Heading | Table) for b in section.blocks)

    def test_off_mode_error_hints_at_ocr_flag(self, fixtures_dir: Path) -> None:
        with pytest.raises(ScannedPdfError) as exc_info:
            PdfExtractor().extract(fixtures_dir / "pdf" / "scanned_text.pdf")
        assert "--ocr auto" in exc_info.value.message


@pytest.mark.slow
def test_concurrent_mixed_pdfs_never_lose_tables(fixtures_dir: Path) -> None:
    """Regression: pymupdf's table module uses module-level globals, so
    unsynchronized concurrent find_tables()/extract() across threads silently
    dropped tables (~5% of extractions before the lock in pdf_extractor)."""
    from concurrent.futures import ThreadPoolExecutor

    base = fixtures_dir / "pdf"

    def job(i: int) -> bool | None:
        if i % 2 == 0:
            result = PdfExtractor().extract(base / "basic.pdf")
            return any(isinstance(b, Table) for s in result.document.sections for b in s.blocks)
        other = ["empty.pdf", "scanned_text.pdf", "mixed_scan.pdf"][i % 3]
        backend = FakeBackend(lines=[fake_line("noise")])
        extractor = PdfExtractor(ocr_mode=OcrMode.AUTO, ocr_backend=backend)
        with contextlib.suppress(OcrNoTextError):
            extractor.extract(base / other)
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [r for r in pool.map(job, range(80)) if r is not None]
    assert all(results), f"tables lost in {results.count(False)}/{len(results)} extractions"


@pytest.mark.ocr
@pytest.mark.skipif(not is_ocr_available(), reason="needs macOS with the ocr extra")
class TestAppleVision:
    def test_scanned_text_pdf_is_recognized(self, fixtures_dir: Path) -> None:
        extractor = PdfExtractor(ocr_mode=OcrMode.AUTO)
        result = extractor.extract(fixtures_dir / "pdf" / "scanned_text.pdf")

        assert result.status == "warning"
        applied = [w for w in result.warnings if w.code == "OCR_APPLIED"]
        assert applied and applied[0].location == "page 1"
        text = " ".join(
            b.text for b in result.document.sections[0].blocks if isinstance(b, Paragraph)
        )
        assert "Scanned Fixture Title" in text
        assert "Quarterly revenue" in text

    def test_mixed_scan_only_ocrs_the_scanned_page(self, fixtures_dir: Path) -> None:
        extractor = PdfExtractor(ocr_mode=OcrMode.AUTO)
        result = extractor.extract(fixtures_dir / "pdf" / "mixed_scan.pdf")

        applied = [w for w in result.warnings if w.code == "OCR_APPLIED"]
        assert [w.location for w in applied] == ["page 2"]
        page1 = result.document.sections[0].blocks
        assert any(isinstance(b, Heading) and "Native Text Heading" in b.text for b in page1), (
            "page 1 native heading must come from the text layer, not OCR"
        )
        page2_text = " ".join(
            b.text for b in result.document.sections[1].blocks if isinstance(b, Paragraph)
        )
        assert "pixels on page two" in page2_text

    def test_blank_scan_yields_no_text_error(self, fixtures_dir: Path) -> None:
        extractor = PdfExtractor(ocr_mode=OcrMode.AUTO)
        with pytest.raises(OcrNoTextError):
            extractor.extract(fixtures_dir / "pdf" / "scanned.pdf")

    def test_force_mode_reads_native_pages_as_images(self, fixtures_dir: Path) -> None:
        extractor = PdfExtractor(ocr_mode=OcrMode.FORCE)
        result = extractor.extract(fixtures_dir / "pdf" / "basic.pdf")

        all_text = " ".join(
            b.text
            for section in result.document.sections
            for b in section.blocks
            if isinstance(b, Paragraph)
        )
        assert "Fixture PDF Heading" in all_text
        assert len([w for w in result.warnings if w.code == "OCR_APPLIED"]) == 2
