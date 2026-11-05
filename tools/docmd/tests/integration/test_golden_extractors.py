"""Golden tests: fixture file -> extract -> normalize -> full Markdown.

These lock each format's end-to-end output. Deterministic mode is forced so
timestamps never appear. Update workflow:
uv run pytest -m golden --update-golden && git diff tests/expected
"""

from pathlib import Path

import pytest

from docmd.extractors import get_extractor
from docmd.markdown.metadata import FrontMatter
from docmd.markdown.normalizer import normalize
from docmd.markdown.writer import MarkdownWriter
from docmd.types import SourceType
from tests.conftest import EXPECTED_DIR, GoldenCheck

pytestmark = [pytest.mark.integration, pytest.mark.golden]

CASES = [
    ("docx/basic.docx", SourceType.DOCX),
    ("docx/special_chars.docx", SourceType.DOCX),
    ("pptx/basic.pptx", SourceType.PPTX),
    ("xlsx/basic.xlsx", SourceType.XLSX),
    ("xlsx/cached_formula.xlsx", SourceType.XLSX),
    ("pdf/basic.pdf", SourceType.PDF),
]


def render_fixture(fixtures_dir: Path, rel: str, source_type: SourceType) -> str:
    extractor = get_extractor(source_type)
    result = extractor.extract(fixtures_dir / rel, relative_path=rel)
    doc = normalize(result.document)
    front = FrontMatter(
        source_file=rel,
        source_type=source_type,
        extraction_timestamp=None,  # deterministic
        extractor_version=extractor.versioned_name,
        status=result.status,
    )
    return MarkdownWriter().render(doc, front, result.warnings)


@pytest.mark.parametrize(("rel", "source_type"), CASES, ids=[c[0] for c in CASES])
def test_golden_extraction(
    fixtures_dir: Path,
    assert_matches_golden: GoldenCheck,
    rel: str,
    source_type: SourceType,
) -> None:
    actual = render_fixture(fixtures_dir, rel, source_type)
    expected = EXPECTED_DIR / Path(rel).parent / (Path(rel).name + ".md")
    assert_matches_golden(actual, expected)
