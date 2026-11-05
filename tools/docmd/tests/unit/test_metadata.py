"""Frontmatter rendering and sidecar persistence."""

import json
from datetime import UTC, datetime
from pathlib import Path

from docmd.markdown.metadata import (
    FrontMatter,
    SidecarMeta,
    render_frontmatter,
    sidecar_path_for,
    write_sidecar,
)
from docmd.types import ExtractionWarning, SourceType


def make_front(**overrides: object) -> FrontMatter:
    defaults: dict[str, object] = {
        "source_file": "docs/report.docx",
        "source_type": SourceType.DOCX,
        "extraction_timestamp": datetime(2026, 6, 10, 15, 30, 12, tzinfo=UTC),
        "extractor_version": "docx/0.1.0",
        "status": "success",
    }
    defaults.update(overrides)
    return FrontMatter.model_validate(defaults)


def test_frontmatter_key_order_and_quoting() -> None:
    out = render_frontmatter(make_front())
    assert out == (
        "---\n"
        'source_file: "docs/report.docx"\n'
        'source_type: "docx"\n'
        'extraction_timestamp: "2026-06-10T15:30:12+00:00"\n'
        'extractor_version: "docx/0.1.0"\n'
        'status: "success"\n'
        "---\n"
    )


def test_frontmatter_deterministic_omits_timestamp() -> None:
    out = render_frontmatter(make_front(extraction_timestamp=None))
    assert "extraction_timestamp" not in out


def test_frontmatter_escapes_quotes_and_unicode() -> None:
    out = render_frontmatter(make_front(source_file='we"ird: påth.docx'))
    assert 'source_file: "we\\"ird: påth.docx"' in out


def test_sidecar_path_naming() -> None:
    assert sidecar_path_for(Path("out/report.docx.md")) == Path("out/report.docx.md.meta.json")


def test_write_sidecar_round_trip(tmp_path: Path) -> None:
    md_path = tmp_path / "report.docx.md"
    meta = SidecarMeta(
        source_file="docs/report.docx",
        source_sha256="ab" * 32,
        source_size=1234,
        extractor="docx",
        extractor_version="0.1.0",
        tool_version="0.1.0",
        pipeline_steps=["extract", "normalize", "write"],
        warnings=[ExtractionWarning(code="X", message="m", location="table 3")],
        created_at=datetime(2026, 6, 10, tzinfo=UTC),
    )
    sidecar = write_sidecar(md_path, meta)
    data = json.loads(sidecar.read_text())
    assert data["schema_version"] == 1
    assert data["warnings"][0]["location"] == "table 3"
    assert data["created_at"].startswith("2026-06-10")


def test_write_sidecar_deterministic_omits_created_at(tmp_path: Path) -> None:
    md_path = tmp_path / "r.pdf.md"
    meta = SidecarMeta(
        source_file="r.pdf",
        source_sha256="00" * 32,
        source_size=1,
        extractor="pdf",
        extractor_version="0.1.0",
        tool_version="0.1.0",
        pipeline_steps=["extract"],
        created_at=None,
    )
    data = json.loads(write_sidecar(md_path, meta).read_text())
    assert "created_at" not in data
