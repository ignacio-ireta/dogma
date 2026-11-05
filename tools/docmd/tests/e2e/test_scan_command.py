"""End-to-end `docmd scan` runs: exit codes, reports, resume."""

import json
import signal
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docmd.cli import app

pytestmark = pytest.mark.e2e

runner = CliRunner()


@pytest.fixture(autouse=True)
def restore_sigint() -> Iterator[None]:
    """scan() installs its own SIGINT handler; keep pytest's intact."""
    previous = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, previous)


def latest_run_dir(out: Path) -> Path:
    return max((out / "runs").iterdir(), key=lambda p: p.name)


def test_scan_tree_partial_failure(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "md"
    result = runner.invoke(
        app,
        ["scan", str(fixtures_dir / "tree"), "--out", str(out), "--recursive", "--no-progress"],
    )
    assert result.exit_code == 1, result.output  # corrupt + scanned fixtures fail
    assert "Failed files:" in result.output
    assert "broken/truncated.docx" in result.output

    assert (out / "report.docx.md").is_file()
    assert (out / "report.docx.md.meta.json").is_file()
    assert (out / "finance" / "2026" / "budget.xlsx.md").is_file()

    run_dir = latest_run_dir(out)
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "errors.json").is_file()
    assert (run_dir / "run.log").is_file()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["summary"]["failed"] == 2
    statuses = {rec["source"]: rec["status"] for rec in manifest["files"].values()}
    assert statuses["notes.txt"] == "skipped"
    assert statuses["broken/scanned.pdf"] == "failed"

    errors = json.loads((run_dir / "errors.json").read_text())
    codes = {e["error"]["error_code"] for e in errors}
    assert "SCANNED_PDF_OCR_DISABLED" in codes
    assert "CORRUPT_FILE" in codes


def test_scan_clean_folder_exit_zero_and_rerun_skips(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "md"
    src = fixtures_dir / "docx"
    first = runner.invoke(app, ["scan", str(src), "--out", str(out), "--no-progress"])
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, ["scan", str(src), "--out", str(out), "--no-progress"])
    assert second.exit_code == 0, second.output
    manifest = json.loads((latest_run_dir(out) / "manifest.json").read_text())
    assert manifest["summary"]["skipped"] == manifest["summary"]["total"]


def test_scan_resume_skips_completed(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "md"
    src = fixtures_dir / "tree"
    first = runner.invoke(
        app, ["scan", str(src), "--out", str(out), "--recursive", "--no-progress"]
    )
    assert first.exit_code == 1

    resumed = runner.invoke(
        app,
        ["scan", str(src), "--out", str(out), "--recursive", "--resume", "--no-progress"],
    )
    assert resumed.exit_code == 1  # failures fail again
    manifest = json.loads((latest_run_dir(out) / "manifest.json").read_text())
    reasons = [
        rec.get("skip_reason", "") for rec in manifest["files"].values() if rec.get("skip_reason")
    ]
    assert any(r.startswith("resume: completed in run") for r in reasons)
    # The corrupt files were re-attempted, not resume-skipped.
    statuses = {rec["source"]: rec["status"] for rec in manifest["files"].values()}
    assert statuses["broken/truncated.docx"] == "failed"


def test_scan_resume_ignores_other_source_manifest(fixtures_dir: Path, tmp_path: Path) -> None:
    import shutil

    out = tmp_path / "md"
    first = runner.invoke(
        app, ["scan", str(fixtures_dir / "docx"), "--out", str(out), "--no-progress"]
    )
    assert first.exit_code == 0

    other = tmp_path / "other"
    other.mkdir()
    shutil.copy(fixtures_dir / "xlsx" / "basic.xlsx", other / "data.xlsx")
    resumed = runner.invoke(
        app, ["scan", str(other), "--out", str(out), "--resume", "--no-progress"]
    )
    assert resumed.exit_code == 0
    manifest = json.loads((latest_run_dir(out) / "manifest.json").read_text())
    record = manifest["files"]["data.xlsx"]
    # Converted for real — not resume-skipped off the other source's manifest.
    assert record["status"] in ("success", "warning")
    assert not (record.get("skip_reason") or "").startswith("resume:")


def test_scan_no_supported_files_exit_4(fixtures_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["scan", str(fixtures_dir / "misc"), "--out", str(tmp_path / "md"), "--no-progress"],
    )
    assert result.exit_code == 4


def test_scan_missing_source_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scan", str(tmp_path / "nope"), "--out", str(tmp_path / "md")])
    assert result.exit_code == 2  # typer argument validation


def test_scan_invalid_log_level_exit_2(fixtures_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "scan",
            str(fixtures_dir / "docx"),
            "--out",
            str(tmp_path / "md"),
            "--log-level",
            "verbose",
        ],
    )
    assert result.exit_code == 2


def test_scan_log_file_contains_no_document_content(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "md"
    result = runner.invoke(
        app, ["scan", str(fixtures_dir / "docx"), "--out", str(out), "--no-progress"]
    )
    assert result.exit_code == 0
    log_text = (latest_run_dir(out) / "run.log").read_text()
    assert "Pipes | stars" not in log_text  # special_chars.docx body text
    assert "Intro with" not in log_text  # basic.docx body text
    assert "processing file" in log_text
