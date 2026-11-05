"""Runner over the fixture tree: failure isolation, skip-unchanged, fail-fast."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from docmd.config import AppConfig
from docmd.discovery.scanner import scan_with_skips
from docmd.pipeline.runner import PipelineRunner
from docmd.pipeline.state import FileStatus, Manifest, ManifestStore, RunSummary
from docmd.pipeline.steps import steps_from_names

pytestmark = pytest.mark.integration


def make_runner(out: Path, **config_overrides: object) -> PipelineRunner:
    config = AppConfig.from_cli(**{k: v for k, v in config_overrides.items() if v is not None})
    store = ManifestStore(
        Manifest(
            run_id="test",
            tool_version="0.1.0",
            command=["test"],
            input_root="in",
            output_root=str(out),
            started_at=datetime.now(UTC),
        ),
        out / "runs" / "test" / "manifest.json",
    )
    return PipelineRunner(config=config, steps=steps_from_names(), store=store, out_root=out)


def run_tree(fixtures_dir: Path, out: Path, **config_overrides: object) -> RunSummary:
    files, skipped = scan_with_skips(fixtures_dir / "tree", recursive=True, out_root=out)
    runner = make_runner(out, **config_overrides)
    return runner.run(files, skipped)


def test_tree_run_isolates_failures(fixtures_dir: Path, tmp_path: Path) -> None:
    summary = run_tree(fixtures_dir, tmp_path / "md", deterministic=True)
    # 4 good (report.docx, report.pdf, deck.pptx, budget.xlsx),
    # 2 bad (broken/truncated.docx, broken/scanned.pdf),
    # 2 scanner skips (notes.txt, ~$report.docx)
    assert summary.success + summary.warning == 4
    assert summary.failed == 2
    assert summary.skipped == 2
    assert summary.interrupted == 0

    out = tmp_path / "md"
    assert (out / "report.docx.md").is_file()
    assert (out / "report.pdf.md").is_file()
    assert (out / "slides" / "deck.pptx.md").is_file()
    assert (out / "finance" / "2026" / "budget.xlsx.md").is_file()
    assert not (out / "broken" / "truncated.docx.md").exists()


def test_rerun_skips_unchanged(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "md"
    run_tree(fixtures_dir, out, deterministic=True)
    summary2 = run_tree(fixtures_dir, out, deterministic=True)
    # The 4 previously-converted files are now skipped as unchanged.
    assert summary2.success + summary2.warning == 0
    assert summary2.skipped == 2 + 4
    assert summary2.failed == 2  # corrupt files fail again


def test_changed_source_is_reprocessed(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "md"
    src_tree = tmp_path / "tree"
    import shutil

    shutil.copytree(fixtures_dir / "tree", src_tree)
    files, skipped = scan_with_skips(src_tree, recursive=True, out_root=out)
    make_runner(out, deterministic=True).run(files, skipped)

    # Modify one source: sha changes, must be reprocessed; others skip.
    target = src_tree / "report.pdf"
    target.write_bytes((fixtures_dir / "pdf" / "empty.pdf").read_bytes())
    files, skipped = scan_with_skips(src_tree, recursive=True, out_root=out)
    store_records = make_runner(out, deterministic=True)
    summary = store_records.run(files, skipped)
    records = {r.source: r for r in store_records.store.records}
    assert records["report.pdf"].status in (FileStatus.SUCCESS, FileStatus.WARNING)
    assert records["report.docx"].status is FileStatus.SKIPPED
    assert summary.success + summary.warning == 1


def test_force_reprocesses_everything(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "md"
    run_tree(fixtures_dir, out, deterministic=True)
    summary = run_tree(fixtures_dir, out, deterministic=True, force=True)
    assert summary.success + summary.warning == 4
    assert summary.skipped == 2  # only scanner skips remain


def test_fail_fast_interrupts_remaining(fixtures_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "md"
    files, skipped = scan_with_skips(fixtures_dir / "tree", recursive=True, out_root=out)
    runner = make_runner(out, deterministic=True, fail_fast=True, workers=1)
    summary = runner.run(files, skipped)
    assert summary.failed >= 1
    # With one worker, everything queued after the first failure is interrupted.
    assert (
        summary.failed + summary.interrupted + summary.success + summary.warning + (summary.skipped)
        == summary.total
    )


def test_parallel_output_matches_serial(fixtures_dir: Path, tmp_path: Path) -> None:
    out1, out8 = tmp_path / "w1", tmp_path / "w8"
    run_tree(fixtures_dir, out1, deterministic=True, workers=1)
    run_tree(fixtures_dir, out8, deterministic=True, workers=8)
    for rel in [
        "report.docx.md",
        "report.pdf.md",
        "slides/deck.pptx.md",
        "finance/2026/budget.xlsx.md",
    ]:
        assert (out1 / rel).read_bytes() == (out8 / rel).read_bytes(), rel
