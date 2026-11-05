"""Golden rendering tests: hand-built IR documents through normalize + write.

These lock the full output contract (frontmatter, section labels, heading
shift, warnings section) against regressions. Update via:
uv run pytest -m golden --update-golden  (then review git diff tests/expected)
"""

import pytest

from docmd.markdown.metadata import FrontMatter
from docmd.markdown.normalizer import normalize
from docmd.markdown.writer import MarkdownWriter
from docmd.types import (
    DocumentIR,
    ExtractionWarning,
    Heading,
    ImageRef,
    ListItem,
    Paragraph,
    Section,
    SectionKind,
    SourceType,
    Table,
)
from tests.conftest import EXPECTED_DIR, GoldenCheck

pytestmark = pytest.mark.golden

writer = MarkdownWriter()


def render(doc: DocumentIR, warnings: list[ExtractionWarning] | None = None) -> str:
    front = FrontMatter(
        source_file=doc.source_path,
        source_type=doc.source_type,
        extraction_timestamp=None,  # deterministic
        extractor_version=f"{doc.source_type.value}/0.1.0",
        status="warning" if warnings else "success",
    )
    return writer.render(normalize(doc), front, warnings or [])


def test_golden_body_document(assert_matches_golden: GoldenCheck) -> None:
    doc = DocumentIR(
        source_path="docs/quarterly_report.docx",
        source_type=SourceType.DOCX,
        title="Quarterly Report",
        properties={"Author": "Example Author", "Created": "2026-01-15"},
        sections=[
            Section(
                kind=SectionKind.BODY,
                blocks=[
                    Heading(level=1, text="Introduction"),
                    Paragraph(text="This  quarter   exceeded expectations."),
                    Heading(level=2, text="Key results"),
                    ListItem(text="Revenue up 12%"),
                    ListItem(text="Churn down 3%", level=1),
                    Table(
                        rows=[["Region", "Revenue"], ["North", "1,200"], ["South", "900"]],
                        caption="Revenue by region",
                    ),
                    Paragraph(text="# This line must not become a heading"),
                    ImageRef(name="chart.png", content_type="image/png"),
                ],
            )
        ],
    )
    assert_matches_golden(render(doc), EXPECTED_DIR / "ir" / "body_document.md")


def test_golden_slides_document(assert_matches_golden: GoldenCheck) -> None:
    doc = DocumentIR(
        source_path="decks/townhall.pptx",
        source_type=SourceType.PPTX,
        sections=[
            Section(
                kind=SectionKind.SLIDE,
                index=1,
                title="Welcome",
                blocks=[Paragraph(text="Agenda for today")],
            ),
            Section(
                kind=SectionKind.SLIDE,
                index=2,
                blocks=[
                    ListItem(text="Point one"),
                    ListItem(text="Point two"),
                ],
            ),
            Section(
                kind=SectionKind.NOTES,
                index=2,
                blocks=[Paragraph(text="Remember to pause here.")],
            ),
        ],
    )
    warnings = [
        ExtractionWarning(
            code="SHAPE_ORDER_HEURISTIC",
            message="shape order inferred from position",
            location="slide 2",
        )
    ]
    assert_matches_golden(render(doc, warnings), EXPECTED_DIR / "ir" / "slides_document.md")


def test_golden_sheet_document(assert_matches_golden: GoldenCheck) -> None:
    doc = DocumentIR(
        source_path="finance/budget.xlsx",
        source_type=SourceType.XLSX,
        sections=[
            Section(
                kind=SectionKind.SHEET,
                index=1,
                title="Data",
                blocks=[
                    Table(
                        rows=[
                            ["Item", "Cost | Notes"],
                            ["Laptop", "2,000"],
                            ["Total", "2,000 (=SUM(B2:B2))"],
                        ]
                    )
                ],
            ),
            Section(kind=SectionKind.SHEET, index=2, title="Empty sheet", blocks=[]),
        ],
    )
    assert_matches_golden(render(doc), EXPECTED_DIR / "ir" / "sheet_document.md")
