"""Memory/format stability on a 50k-row workbook. Marked slow."""

from pathlib import Path

import pytest

from docmd.extractors.xlsx_extractor import XlsxExtractor
from docmd.types import Table

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_huge_sheet_truncates_with_warning(fixtures_dir: Path) -> None:
    from tests.fixtures.generate_fixtures import build_xlsx_huge

    huge = fixtures_dir / "xlsx" / "huge.xlsx"
    if not huge.exists():
        build_xlsx_huge(huge)

    extractor = XlsxExtractor(max_rows_per_sheet=10_000)
    result = extractor.extract(huge)
    assert any(w.code == "SHEET_TRUNCATED" for w in result.warnings)
    table = result.document.sections[0].blocks[0]
    assert isinstance(table, Table)
    assert len(table.rows) == 10_000
