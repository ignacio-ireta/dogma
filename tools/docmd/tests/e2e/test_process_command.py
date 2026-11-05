"""End-to-end: `docmd process` on real fixtures via the CLI runner."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docmd.cli import app
from docmd.ocr import is_ocr_available

pytestmark = pytest.mark.e2e

runner = CliRunner()


def test_process_docx(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "docx" / "basic.docx"
    result = runner.invoke(app, ["process", str(src), "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output

    md = tmp_path / "basic.docx.md"
    sidecar = tmp_path / "basic.docx.md.meta.json"
    assert md.exists() and sidecar.exists()
    content = md.read_text()
    assert content.startswith("---\n")
    assert "extraction_timestamp" in content  # non-deterministic by default

    meta = json.loads(sidecar.read_text())
    assert meta["extractor"] == "docx"
    assert meta["pipeline_steps"] == ["extract", "normalize", "write"]
    assert len(meta["source_sha256"]) == 64


def test_process_deterministic_is_repeatable(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "pdf" / "basic.pdf"
    out1, out2 = tmp_path / "one", tmp_path / "two"
    for out in (out1, out2):
        result = runner.invoke(app, ["process", str(src), "--out", str(out), "--deterministic"])
        assert result.exit_code == 0, result.output
    assert (out1 / "basic.pdf.md").read_bytes() == (out2 / "basic.pdf.md").read_bytes()


def test_process_corrupt_file_fails_with_code(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "corrupt" / "truncated.docx"
    result = runner.invoke(app, ["process", str(src), "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "CORRUPT_FILE" in result.output
    assert not list(tmp_path.iterdir())  # no half-written outputs


def test_process_password_protected(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "pdf" / "encrypted.pdf"
    result = runner.invoke(app, ["process", str(src), "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "PASSWORD_PROTECTED" in result.output


def test_process_unsupported_file(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "misc" / "readme.txt"
    result = runner.invoke(app, ["process", str(src), "--out", str(tmp_path)])
    assert result.exit_code == 4


def test_process_partial_steps_writes_nothing(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "docx" / "basic.docx"
    result = runner.invoke(
        app,
        ["process", str(src), "--out", str(tmp_path), "--steps", "extract", "--steps", "normalize"],
    )
    assert result.exit_code == 0, result.output
    assert not list(tmp_path.iterdir())
    assert "extract, normalize" in result.output


def test_process_unknown_step_is_usage_error(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "docx" / "basic.docx"
    result = runner.invoke(
        app, ["process", str(src), "--out", str(tmp_path), "--steps", "frobnicate"]
    )
    assert result.exit_code == 2


def test_process_ocr_unavailable_is_usage_error(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("docmd.ocr.is_ocr_available", lambda: False)
    src = fixtures_dir / "pdf" / "scanned_text.pdf"
    result = runner.invoke(app, ["process", str(src), "--out", str(tmp_path), "--ocr", "auto"])
    assert result.exit_code == 2
    assert "uv sync --extra ocr" in result.output
    assert not list(tmp_path.iterdir())


def test_process_scanned_pdf_without_ocr_hints_at_flag(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "pdf" / "scanned_text.pdf"
    result = runner.invoke(app, ["process", str(src), "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "SCANNED_PDF_OCR_DISABLED" in result.output
    assert "--ocr auto" in result.output


@pytest.mark.ocr
@pytest.mark.skipif(not is_ocr_available(), reason="needs macOS with the ocr extra")
def test_process_ocr_auto_converts_scanned_pdf(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "pdf" / "scanned_text.pdf"
    result = runner.invoke(
        app, ["process", str(src), "--out", str(tmp_path), "--ocr", "auto", "--deterministic"]
    )
    assert result.exit_code == 0, result.output

    content = (tmp_path / "scanned_text.pdf.md").read_text()
    assert "Quarterly revenue" in content
    assert "OCR_APPLIED" in content  # surfaced in the warnings section

    meta = json.loads((tmp_path / "scanned_text.pdf.md.meta.json").read_text())
    assert meta["extractor_version"] == "0.2.0"
    assert any(w["code"] == "OCR_APPLIED" for w in meta["warnings"])
