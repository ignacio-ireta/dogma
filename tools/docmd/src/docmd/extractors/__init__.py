"""Extractor registry. Imports are lazy so `docmd version` stays fast."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docmd.errors import UnsupportedFileTypeError
from docmd.extractors.base import ExtractionResult, Extractor
from docmd.types import SourceType

if TYPE_CHECKING:
    from docmd.config import AppConfig

__all__ = ["ExtractionResult", "Extractor", "get_extractor"]


def get_extractor(file_type: SourceType, config: AppConfig | None = None) -> Extractor:
    if file_type is SourceType.DOCX:
        from docmd.extractors.docx_extractor import DocxExtractor

        return DocxExtractor()
    if file_type is SourceType.PPTX:
        from docmd.extractors.pptx_extractor import PptxExtractor

        return PptxExtractor()
    if file_type is SourceType.XLSX:
        from docmd.extractors.xlsx_extractor import XlsxExtractor

        if config is not None:
            return XlsxExtractor(
                max_rows_per_sheet=config.max_rows_per_sheet,
                max_cols_per_sheet=config.max_cols_per_sheet,
            )
        return XlsxExtractor()
    if file_type is SourceType.PDF:
        from docmd.extractors.pdf_extractor import PdfExtractor

        if config is not None:
            return PdfExtractor(
                ocr_mode=config.ocr,
                ocr_languages=config.ocr_languages,
                ocr_dpi=config.ocr_dpi,
            )
        return PdfExtractor()
    raise UnsupportedFileTypeError(f"no extractor for file type {file_type!r}")
