"""Casefold output-collision pre-flight (synthetic entries; no fs needed)."""

from datetime import UTC, datetime
from pathlib import Path

from docmd.config import AppConfig
from docmd.discovery.scanner import DiscoveredFile
from docmd.pipeline.runner import PipelineRunner
from docmd.pipeline.state import FileStatus, Manifest, ManifestStore
from docmd.pipeline.steps import steps_from_names
from docmd.types import SourceType


def make_file(rel: str) -> DiscoveredFile:
    return DiscoveredFile(
        path=Path("/src") / rel, relative_path=rel, file_type=SourceType.DOCX, size=10
    )


def test_casefold_collision_fails_second_file(tmp_path: Path) -> None:
    store = ManifestStore(
        Manifest(
            run_id="t",
            tool_version="0",
            command=[],
            input_root="i",
            output_root="o",
            started_at=datetime.now(UTC),
        ),
        tmp_path / "manifest.json",
    )
    runner = PipelineRunner(
        config=AppConfig(),
        steps=steps_from_names(),
        store=store,
        out_root=tmp_path / "out",
    )
    files = [make_file("a/Report.docx"), make_file("a/REPORT.DOCX"), make_file("a/other.docx")]
    ok, bad = runner._preflight_collisions(files)

    assert [f.relative_path for f in ok] == ["a/Report.docx", "a/other.docx"]
    assert len(bad) == 1
    assert bad[0].status is FileStatus.FAILED
    assert bad[0].error is not None and bad[0].error.error_code == "OUTPUT_COLLISION"
    assert "a/Report.docx" in bad[0].error.message
