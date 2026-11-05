"""XLSX extraction via openpyxl, values + formula text.

Each workbook is opened twice in read_only mode: once with ``data_only=True``
(cached computed values) and once with ``data_only=False`` (formula text).
Formula cells render as ``value (=FORMULA)``; a formula with no cached value
(file never saved by a calculating application) renders as just the formula
plus a FORMULA_NO_CACHED_VALUE warning. ``reset_dimensions()`` guards against
stale/absent dimension records; row/column caps guard memory on huge sheets.
Merged cells: only the anchor cell carries the value (others are empty).
"""

from __future__ import annotations

import contextlib
import itertools
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from docmd.errors import wrap_library_error
from docmd.extractors.base import ExtractionResult, Extractor
from docmd.types import (
    Block,
    DocumentIR,
    ExtractionWarning,
    Section,
    SectionKind,
    SourceType,
    Table,
)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, datetime):
        # Excel datetimes at midnight are usually plain dates.
        if value.time() == time.min:
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return f"{value:g}" if value == int(value) else repr(value)
    return str(value)


class XlsxExtractor(Extractor):
    name = "xlsx"
    version = "0.1.0"
    file_type = SourceType.XLSX

    def __init__(self, max_rows_per_sheet: int = 10_000, max_cols_per_sheet: int = 256) -> None:
        self.max_rows = max_rows_per_sheet
        self.max_cols = max_cols_per_sheet

    def extract(self, path: Path, *, relative_path: str | None = None) -> ExtractionResult:
        try:
            values_wb = load_workbook(str(path), read_only=True, data_only=True)
        except Exception as exc:
            raise wrap_library_error(exc, path) from exc
        try:
            formulas_wb = load_workbook(str(path), read_only=True, data_only=False)
        except Exception as exc:
            values_wb.close()
            raise wrap_library_error(exc, path) from exc

        warnings: list[ExtractionWarning] = []
        sections: list[Section] = []
        try:
            properties = self._properties(values_wb)
            for index, (ws_values, ws_formulas) in enumerate(
                zip(values_wb.worksheets, formulas_wb.worksheets, strict=True), start=1
            ):
                section = self._convert_sheet(ws_values, ws_formulas, index, warnings)
                sections.append(section)
        except Exception as exc:
            raise wrap_library_error(exc, path) from exc
        finally:
            values_wb.close()
            formulas_wb.close()

        return ExtractionResult(
            document=DocumentIR(
                source_path=relative_path or path.name,
                source_type=SourceType.XLSX,
                sections=sections,
                properties=properties,
            ),
            warnings=warnings,
        )

    def _convert_sheet(
        self,
        ws_values: Any,
        ws_formulas: Any,
        index: int,
        warnings: list[ExtractionWarning],
    ) -> Section:
        title = str(ws_values.title)
        for ws in (ws_values, ws_formulas):
            with contextlib.suppress(AttributeError):
                ws.reset_dimensions()  # absent on non read-only worksheets

        rows: list[list[str]] = []
        truncated_rows = False
        truncated_cols = False
        missing_cache_cells = 0

        value_iter = ws_values.iter_rows(values_only=True)
        formula_iter = ws_formulas.iter_rows(values_only=True)
        for row_no, (value_row, formula_row) in enumerate(
            itertools.zip_longest(value_iter, formula_iter, fillvalue=()), start=1
        ):
            if row_no > self.max_rows:
                truncated_rows = True
                break
            if len(value_row) > self.max_cols or len(formula_row) > self.max_cols:
                truncated_cols = True
            cells: list[str] = []
            for value, formula in itertools.zip_longest(
                value_row[: self.max_cols], formula_row[: self.max_cols], fillvalue=None
            ):
                is_formula = isinstance(formula, str) and formula.startswith("=")
                if is_formula:
                    if value is None or value == formula:
                        missing_cache_cells += 1
                        cells.append(str(formula))
                    else:
                        cells.append(f"{_format_value(value)} ({formula})")
                else:
                    cells.append(_format_value(value))
            rows.append(cells)

        rows = self._trim(rows)

        if truncated_rows:
            warnings.append(
                ExtractionWarning(
                    code="SHEET_TRUNCATED",
                    message=f"sheet exceeds {self.max_rows} rows; output truncated",
                    location=f"sheet {title}",
                )
            )
        if truncated_cols:
            warnings.append(
                ExtractionWarning(
                    code="SHEET_TRUNCATED",
                    message=f"sheet exceeds {self.max_cols} columns; output truncated",
                    location=f"sheet {title}",
                )
            )
        if missing_cache_cells:
            warnings.append(
                ExtractionWarning(
                    code="FORMULA_NO_CACHED_VALUE",
                    message=(
                        f"{missing_cache_cells} formula cell(s) have no cached value; "
                        "showing formula text only"
                    ),
                    location=f"sheet {title}",
                )
            )

        blocks: list[Block] = [Table(rows=rows, has_header=True)] if rows else []
        return Section(kind=SectionKind.SHEET, index=index, title=title, blocks=blocks)

    @staticmethod
    def _trim(rows: list[list[str]]) -> list[list[str]]:
        """Drop trailing empty rows and columns (keeps interior gaps)."""
        while rows and not any(cell for cell in rows[-1]):
            rows.pop()
        if not rows:
            return rows
        last_used_col = max(
            (i for row in rows for i, cell in enumerate(row) if cell),
            default=-1,
        )
        width = last_used_col + 1
        return [row[:width] + [""] * (width - len(row[:width])) for row in rows]

    @staticmethod
    def _properties(workbook: Any) -> dict[str, str]:
        props: dict[str, str] = {}
        info = workbook.properties
        if info is None:
            return props
        if info.creator:
            props["Author"] = str(info.creator)
        if info.created:
            props["Created"] = info.created.isoformat()
        # "Modified" is omitted: writer libraries reset it at save time, which
        # would make otherwise-identical extractions differ.
        return props
