"""PDF extraction via PyMuPDF (AGPL-3.0 — see README licensing note).

Per page: tables are detected first (``page.find_tables()``); text blocks whose
center falls inside a detected table's bbox are excluded from the text pass to
avoid duplication, then blocks and tables are interleaved by vertical position.
Heading levels are inferred from font size relative to the page's dominant
body size — a heuristic, flagged as such in the extraction contract.

Scanned pages (no text layer, image present): with OCR off they get a
SCANNED_PAGE warning, and a fully image-only document fails with
ScannedPdfError. With ``--ocr auto`` such pages are rendered and recognized via
the configured OCR backend (Apple Vision in v0.2), producing plain paragraphs —
OCR gives no reliable font sizes or table structure, so no heading inference or
table detection happens on OCR'd pages. ``--ocr force`` OCRs every page and
ignores any existing text layer (for garbled embedded text). Encrypted files
are tried with an empty user password before failing with
PasswordProtectedError.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

# Stop pymupdf advertising its layout package on stdout, where it would
# corrupt CLI output.
with __import__("contextlib").suppress(AttributeError):
    pymupdf.no_recommend_layout()

from docmd.config import OcrMode
from docmd.errors import (
    FileCorruptionError,
    OcrError,
    OcrNoTextError,
    PasswordProtectedError,
    ScannedPdfError,
    wrap_library_error,
)
from docmd.extractors.base import ExtractionResult, Extractor
from docmd.ocr import OcrBackend, get_ocr_backend, group_into_paragraphs
from docmd.types import (
    Block,
    DocumentIR,
    ExtractionWarning,
    Heading,
    Paragraph,
    Section,
    SectionKind,
    SourceType,
    Table,
)

# Font-size ratio (vs body text) -> heading level.
_HEADING_THRESHOLDS = ((1.6, 1), (1.3, 2), (1.15, 3))

# Mean per-page Vision confidence below this gets an OCR_LOW_CONFIDENCE warning.
_LOW_CONFIDENCE = 0.5

# pymupdf's table module keeps its working state in module-level globals
# (pymupdf.table.EDGES/CHARS/TEXTPAGE), so concurrent find_tables()/extract()
# calls from worker threads clobber each other and silently lose tables.
# Serialize the table-detection phase only; text and OCR stay parallel.
_TABLE_DETECTION_LOCK = threading.Lock()


class PdfExtractor(Extractor):
    name = "pdf"
    version = "0.2.0"
    file_type = SourceType.PDF

    def __init__(
        self,
        *,
        ocr_mode: OcrMode = OcrMode.OFF,
        ocr_languages: list[str] | None = None,
        ocr_dpi: int = 300,
        ocr_backend: OcrBackend | None = None,
    ) -> None:
        self.ocr_mode = ocr_mode
        self.ocr_dpi = ocr_dpi
        if ocr_backend is None and ocr_mode is not OcrMode.OFF:
            ocr_backend = get_ocr_backend(languages=ocr_languages)
        self._ocr_backend = ocr_backend

    def extract(self, path: Path, *, relative_path: str | None = None) -> ExtractionResult:
        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:
            raise wrap_library_error(exc, path) from exc

        try:
            # MuPDF opens many non-PDF formats (XPS, EPUB, even Office files);
            # refuse them so misnamed files fail predictably instead of being
            # half-heartedly rendered by the wrong extractor.
            if not doc.is_pdf:
                raise FileCorruptionError(
                    "file opened but is not a PDF (misnamed file?)", path=path
                )
            if doc.needs_pass and not doc.authenticate(""):
                raise PasswordProtectedError("PDF is encrypted and requires a password", path=path)

            warnings: list[ExtractionWarning] = []
            sections: list[Section] = []
            pages_with_text = 0
            image_only_pages = 0

            for page_no in range(doc.page_count):
                page = doc[page_no]
                if self.ocr_mode is OcrMode.FORCE:
                    blocks, page_warnings, has_text, image_only = self._convert_page_forced_ocr(
                        page, page_no + 1, path
                    )
                else:
                    blocks, page_warnings, has_text, image_only = self._convert_page(
                        page, page_no + 1
                    )
                    if image_only and self.ocr_mode is OcrMode.AUTO:
                        ocr_blocks, ocr_warnings = self._ocr_page(page, page_no + 1, path)
                        if ocr_blocks:
                            blocks = ocr_blocks
                            page_warnings = [
                                w for w in page_warnings if w.code != "SCANNED_PAGE"
                            ] + ocr_warnings
                            has_text, image_only = True, False
                warnings.extend(page_warnings)
                pages_with_text += int(has_text)
                image_only_pages += int(image_only)
                sections.append(Section(kind=SectionKind.PAGE, index=page_no + 1, blocks=blocks))

            if pages_with_text == 0 and image_only_pages > 0:
                if self.ocr_mode is OcrMode.OFF:
                    raise ScannedPdfError(
                        "PDF contains no extractable text (scanned document?); "
                        "re-run with --ocr auto to recognize it",
                        path=path,
                    )
                raise OcrNoTextError(
                    "OCR found no recognizable text in this image-only PDF", path=path
                )

            properties = self._properties(doc)
            title = doc.metadata.get("title") or None if doc.metadata else None
        finally:
            doc.close()

        return ExtractionResult(
            document=DocumentIR(
                source_path=relative_path or path.name,
                source_type=SourceType.PDF,
                title=title,
                sections=sections,
                properties=properties,
            ),
            warnings=warnings,
        )

    def _convert_page(
        self, page: Any, page_no: int
    ) -> tuple[list[Block], list[ExtractionWarning], bool, bool]:
        warnings: list[ExtractionWarning] = []
        positioned: list[tuple[float, Block]] = []

        table_bboxes: list[Any] = []
        try:
            with _TABLE_DETECTION_LOCK:
                finder = page.find_tables()
                for table in finder.tables:
                    rows = [
                        ["" if cell is None else str(cell) for cell in row]
                        for row in table.extract()
                    ]
                    if any(any(cell for cell in row) for row in rows):
                        bbox = pymupdf.Rect(table.bbox)
                        table_bboxes.append(bbox)
                        positioned.append((bbox.y0, Table(rows=rows, has_header=True)))
        except Exception as exc:
            warnings.append(
                ExtractionWarning(
                    code="TABLE_DETECTION_FAILED",
                    message=f"table detection failed: {exc}",
                    location=f"page {page_no}",
                )
            )

        body_size = self._body_font_size(page)
        text_found = False
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # not a text block
                continue
            rect = pymupdf.Rect(block["bbox"])
            if any(bbox.contains(rect.tl + (rect.br - rect.tl) * 0.5) for bbox in table_bboxes):
                continue  # already captured as table content
            text, max_size = self._block_text(block)
            if not text.strip():
                continue
            text_found = True
            positioned.append((rect.y0, self._classify(text, max_size, body_size)))

        has_images = bool(page.get_images())
        image_only = not text_found and not table_bboxes and has_images
        if image_only:
            warnings.append(
                ExtractionWarning(
                    code="SCANNED_PAGE",
                    message="page has image content but no extractable text",
                    location=f"page {page_no}",
                )
            )

        positioned.sort(key=lambda item: item[0])
        return [b for _, b in positioned], warnings, text_found or bool(table_bboxes), image_only

    def _convert_page_forced_ocr(
        self, page: Any, page_no: int, path: Path
    ) -> tuple[list[Block], list[ExtractionWarning], bool, bool]:
        """Force mode: OCR the rendered page, ignoring any embedded text layer."""
        blocks, warnings = self._ocr_page(page, page_no, path)
        if blocks:
            return blocks, warnings, True, False
        has_images = bool(page.get_images())
        if has_images:
            warnings.append(
                ExtractionWarning(
                    code="SCANNED_PAGE",
                    message="page has image content but no OCR-recognizable text",
                    location=f"page {page_no}",
                )
            )
        return [], warnings, False, has_images

    def _ocr_page(
        self, page: Any, page_no: int, path: Path
    ) -> tuple[list[Block], list[ExtractionWarning]]:
        """Render the page and recognize text; returns ([], []) when nothing is found."""
        if self._ocr_backend is None:  # defensive: callers gate on ocr_mode
            return [], []
        png: bytes = page.get_pixmap(dpi=self.ocr_dpi).tobytes("png")
        try:
            lines = self._ocr_backend.recognize(png)
        except Exception as exc:
            raise OcrError(
                "OCR engine failed",
                path=path,
                detail=(f"page {page_no}: {type(exc).__module__}.{type(exc).__qualname__}: {exc}"),
            ) from exc

        paragraphs = group_into_paragraphs(lines)
        if not paragraphs:
            return [], []

        warnings = [
            ExtractionWarning(
                code="OCR_APPLIED",
                message=(f"{len(lines)} line(s) recognized via {self._ocr_backend.name} OCR"),
                location=f"page {page_no}",
            )
        ]
        confidences = [line.confidence for line in lines if line.text.strip()]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        if mean_confidence < _LOW_CONFIDENCE:
            warnings.append(
                ExtractionWarning(
                    code="OCR_LOW_CONFIDENCE",
                    message=f"mean OCR confidence is low ({mean_confidence:.2f})",
                    location=f"page {page_no}",
                )
            )
        return [Paragraph(text=text) for text in paragraphs], warnings

    @staticmethod
    def _body_font_size(page: Any) -> float:
        """Dominant font size weighted by text length; 0 when page has no text."""
        sizes: Counter[float] = Counter()
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes[round(span["size"], 1)] += len(span.get("text", ""))
        if not sizes:
            return 0.0
        return sizes.most_common(1)[0][0]

    @staticmethod
    def _block_text(block: dict[str, Any]) -> tuple[str, float]:
        lines: list[str] = []
        max_size = 0.0
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            lines.append("".join(span.get("text", "") for span in spans))
            for span in spans:
                max_size = max(max_size, float(span.get("size", 0)))
        return "\n".join(lines), max_size

    @staticmethod
    def _classify(text: str, size: float, body_size: float) -> Block:
        if body_size > 0 and "\n" not in text.strip():
            ratio = size / body_size
            for threshold, level in _HEADING_THRESHOLDS:
                if ratio >= threshold:
                    return Heading(level=level, text=text)
        return Paragraph(text=text)

    @staticmethod
    def _properties(doc: Any) -> dict[str, str]:
        props: dict[str, str] = {}
        metadata = doc.metadata or {}
        if metadata.get("author"):
            props["Author"] = str(metadata["author"])
        if metadata.get("creationDate"):
            props["Created"] = str(metadata["creationDate"])
        props["Pages"] = str(doc.page_count)
        return props
