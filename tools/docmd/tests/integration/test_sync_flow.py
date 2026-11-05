"""run_sync over real fixtures: hash-failure safety and multi-repo registries.

These cover behavior below the CLI that the e2e sync tests do not reach:
a tracked-but-unreadable source must classify as nothing (never as a
deletion), and syncs of different repos into one output root must not lose
each other's registry entries.
"""

import json
import os
import shutil
import signal
from collections.abc import Iterator
from pathlib import Path

import pytest

from docmd.config import AppConfig
from docmd.tracking.sync import SyncOutcome, run_sync

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def restore_sigint() -> Iterator[None]:
    """run_sync installs its own SIGINT handler; keep pytest's intact."""
    previous = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, previous)


def sync(src: Path, out: Path) -> SyncOutcome:
    config = AppConfig.from_cli(recursive=True)
    return run_sync(
        source=src,
        out_root=out,
        config=config,
        command=["sync", str(src), "--out", str(out)],
        show_progress=False,
    )


def make_repo(fixtures_dir: Path, root: Path, name: str) -> Path:
    src = root / name
    (src / "sub").mkdir(parents=True)
    shutil.copy(fixtures_dir / "docx" / "basic.docx", src / "a.docx")
    shutil.copy(fixtures_dir / "docx" / "special_chars.docx", src / "sub" / "b.docx")
    return src


def read_snapshot(out: Path, repo: str) -> dict:
    path = out / ".docmd" / "repos" / repo / "snapshot.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.skipif(os.geteuid() == 0, reason="chmod cannot revoke access from root")
def test_unreadable_tracked_file_never_classifies_as_deleted(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    src = make_repo(fixtures_dir, tmp_path, "handbook")
    out = tmp_path / "kb"
    assert sync(src, out).exit_code == 0

    target = src / "a.docx"
    target.chmod(0)
    try:
        outcome = sync(src, out)
    finally:
        target.chmod(0o644)

    # The run reports the failure, but the prior conversion must survive it:
    # still live, still tracked, and nothing archived for this run.
    assert outcome.exit_code == 1
    assert (out / "handbook" / "a.docx.md").is_file()
    snapshot = read_snapshot(out, "handbook")
    assert "a.docx" in snapshot["files"]
    report = outcome.report
    assert report is not None
    assert all(change.source != "a.docx" for change in report.changes)
    assert any("could not be hashed" in note for note in report.notes)
    failed = [r for r in outcome.records if r.source == "a.docx"]
    assert failed and failed[0].error is not None
    assert failed[0].error.error_code == "HASH_FAILED"
    run_archive = out / ".docmd" / "archive" / "handbook" / outcome.run_dir.name
    assert not run_archive.exists()

    # Once readable again the file settles back to unchanged: no re-conversion.
    recovered = sync(src, out)
    assert recovered.exit_code == 0, recovered
    assert recovered.report is not None
    assert recovered.report.counts.unchanged == 2
    assert recovered.report.counts.failed == 0


def test_case_only_rename_relocates_without_archive_or_reextraction(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    src = make_repo(fixtures_dir, tmp_path, "handbook")
    out = tmp_path / "kb"
    assert sync(src, out).exit_code == 0
    original_md = (out / "handbook" / "a.docx.md").read_text(encoding="utf-8")

    (src / "a.docx").rename(src / "A.docx")
    outcome = sync(src, out)

    assert outcome.exit_code == 0
    report = outcome.report
    assert report is not None
    assert report.counts.moved == 1
    assert report.counts.added == report.counts.updated == report.counts.deleted == 0
    # Relocated, not re-extracted: no conversions ran and only the
    # frontmatter source_file line changed.
    assert outcome.summary is not None and outcome.summary.total == 0
    new_md = (out / "handbook" / "A.docx.md").read_text(encoding="utf-8")
    assert new_md == original_md.replace('source_file: "a.docx"', 'source_file: "A.docx"')
    # Nothing was displaced into this run's archive.
    assert not (out / ".docmd" / "archive" / "handbook" / outcome.run_dir.name).exists()
    assert "A.docx" in read_snapshot(out, "handbook")["files"]
    assert "a.docx" not in read_snapshot(out, "handbook")["files"]


def test_deleted_path_whose_output_belongs_to_another_source_is_not_archived(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    src = make_repo(fixtures_dir, tmp_path, "handbook")
    out = tmp_path / "kb"
    assert sync(src, out).exit_code == 0

    # Simulate a foreign conversion sitting at the tracked path (as happens
    # when a casefold-equal move target was relocated there): rewrite the
    # sidecar with a different source hash, then delete the source.
    sidecar = out / "handbook" / "a.docx.md.meta.json"
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    meta["source_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    (src / "a.docx").unlink()

    outcome = sync(src, out)
    assert outcome.exit_code == 0
    # The deletion is untracked, but the foreign output stays alive in place.
    assert (out / "handbook" / "a.docx.md").is_file()
    assert not (out / ".docmd" / "archive" / "handbook" / outcome.run_dir.name).exists()
    assert "a.docx" not in read_snapshot(out, "handbook")["files"]
    report = outcome.report
    assert report is not None
    deleted = [c for c in report.changes if c.action.value == "deleted"]
    assert len(deleted) == 1
    assert deleted[0].archived_to is None


def test_fail_fast_failure_exits_1_not_130(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "handbook"
    src.mkdir()
    shutil.copy(fixtures_dir / "corrupt" / "truncated.docx", src / "broken.docx")
    config = AppConfig.from_cli(recursive=True, fail_fast=True)
    outcome = run_sync(
        source=src,
        out_root=tmp_path / "kb",
        config=config,
        command=["sync", str(src), "--out", str(tmp_path / "kb")],
        show_progress=False,
    )
    # fail-fast cancels the token, but with nothing left undone this is a
    # plain failure to report, not an interruption to resume.
    assert outcome.exit_code == 1


def test_two_repos_share_an_output_root_without_losing_entries(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    src_alpha = make_repo(fixtures_dir, tmp_path, "alpha")
    src_beta = make_repo(fixtures_dir, tmp_path, "beta")
    out = tmp_path / "kb"

    assert sync(src_alpha, out).exit_code == 0
    assert sync(src_beta, out).exit_code == 0
    assert sync(src_alpha, out).exit_code == 0  # rerun must not drop beta

    registry = json.loads((out / "registry.json").read_text(encoding="utf-8"))
    assert set(registry["repos"]) == {"alpha", "beta"}
    assert registry["repos"]["alpha"]["run_count"] == 2
    assert registry["repos"]["beta"]["run_count"] == 1
    assert (out / "alpha" / "a.docx.md").is_file()
    assert (out / "beta" / "a.docx.md").is_file()
    assert (out / "alpha" / "sub" / "b.docx.md").is_file()
    # Snapshots stay per-repo.
    assert set(read_snapshot(out, "alpha")["files"]) == {"a.docx", "sub/b.docx"}
    assert set(read_snapshot(out, "beta")["files"]) == {"a.docx", "sub/b.docx"}
