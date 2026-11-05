"""Snapshot persistence and diff-baseline folding: roundtrip, corruption, apply_run_results."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from docmd.errors import SnapshotCorruptError
from docmd.pipeline.state import FileRecord, FileStatus
from docmd.tracking.snapshot import (
    RepoSnapshot,
    SnapshotEntry,
    apply_run_results,
    load_snapshot,
    output_relpath_for,
    save_snapshot,
    snapshot_path,
)

RUN_1 = "20260701T090000Z"
RUN_2 = "20260702T090000Z"
SHA_A = "a" * 64
SHA_B = "b" * 64


def make_entry(
    sha256: str = SHA_A,
    size: int = 10,
    output: str = "docs/a.docx.md",
    run_id: str = RUN_1,
) -> SnapshotEntry:
    return SnapshotEntry(sha256=sha256, size=size, output=output, last_run_id=run_id)


def make_snapshot(
    files: dict[str, SnapshotEntry] | None = None, run_id: str = RUN_1
) -> RepoSnapshot:
    return RepoSnapshot(
        repo="proj",
        source_root="/src/proj",
        run_id=run_id,
        updated_at=datetime(2026, 7, 1, tzinfo=UTC),
        files=files or {},
    )


def apply(previous: RepoSnapshot | None, **kwargs: object) -> RepoSnapshot:
    kwargs.setdefault("recursive", True)
    return apply_run_results(previous, repo="proj", source_root="/src/proj", run_id=RUN_2, **kwargs)


def test_apply_records_scan_depth_of_the_writing_run() -> None:
    assert apply(None, recursive=True).recursive is True
    assert apply(make_snapshot(), recursive=False).recursive is False


def test_output_relpath_agrees_with_pipeline_naming(tmp_path: Path) -> None:
    """snapshot.output_relpath_for and steps.output_path_for share one contract."""
    from pathlib import PurePosixPath

    from docmd.pipeline.steps import output_path_for

    for rel in ("a.docx", "guides/setup.docx", "a b/c.d/report.v2.pdf"):
        via_relpath = tmp_path.joinpath(*PurePosixPath(output_relpath_for(rel)).parts)
        assert output_path_for(rel, tmp_path) == via_relpath


def test_snapshot_path_layout(tmp_path: Path) -> None:
    expected = tmp_path / ".docmd" / "repos" / "proj" / "snapshot.json"
    assert snapshot_path(tmp_path, "proj") == expected


def test_load_snapshot_missing_returns_none(tmp_path: Path) -> None:
    assert load_snapshot(tmp_path, "proj") is None


def test_save_load_roundtrip_preserves_entries(tmp_path: Path) -> None:
    snap = make_snapshot({"docs/a.docx": make_entry()})
    path = save_snapshot(tmp_path, snap)
    assert path == snapshot_path(tmp_path, "proj")
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 1
    assert on_disk["files"]["docs/a.docx"]["sha256"] == SHA_A
    assert on_disk["files"]["docs/a.docx"]["output"] == "docs/a.docx.md"
    reloaded = load_snapshot(tmp_path, "proj")
    assert reloaded == snap


def test_load_snapshot_garbage_raises_snapshot_corrupt(tmp_path: Path) -> None:
    path = snapshot_path(tmp_path, "proj")
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SnapshotCorruptError) as excinfo:
        load_snapshot(tmp_path, "proj")
    assert excinfo.value.error_code == "SNAPSHOT_CORRUPT"


def test_load_snapshot_wrong_schema_raises_snapshot_corrupt(tmp_path: Path) -> None:
    path = snapshot_path(tmp_path, "proj")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "files": []}), encoding="utf-8")
    with pytest.raises(SnapshotCorruptError):
        load_snapshot(tmp_path, "proj")


def test_output_relpath_for_appends_md() -> None:
    assert output_relpath_for("docs/a.docx") == "docs/a.docx.md"
    assert output_relpath_for("b.pdf") == "b.pdf.md"


def test_apply_success_record_upserts_entry() -> None:
    record = FileRecord(
        source="docs/a.docx", sha256=SHA_B, size=5, status=FileStatus.SUCCESS, output="out"
    )
    snap = apply(make_snapshot(), records=[record])
    entry = snap.files["docs/a.docx"]
    assert entry.sha256 == SHA_B
    assert entry.size == 5
    assert entry.output == "docs/a.docx.md"
    assert entry.last_run_id == RUN_2


def test_apply_warning_record_overwrites_prior_entry() -> None:
    prev = make_snapshot({"docs/a.docx": make_entry(sha256=SHA_A, size=10)})
    record = FileRecord(
        source="docs/a.docx", sha256=SHA_B, size=7, status=FileStatus.WARNING, output="out"
    )
    snap = apply(prev, records=[record])
    entry = snap.files["docs/a.docx"]
    assert entry.sha256 == SHA_B
    assert entry.size == 7
    assert entry.last_run_id == RUN_2


def test_apply_skipped_with_output_and_sha_upserts() -> None:
    # skip-unchanged adoption: the file was not reconverted but is tracked as current
    record = FileRecord(
        source="docs/a.docx",
        sha256=SHA_A,
        size=10,
        status=FileStatus.SKIPPED,
        output="out",
        skip_reason="unchanged",
    )
    snap = apply(None, records=[record])
    assert snap.files["docs/a.docx"].sha256 == SHA_A
    assert snap.files["docs/a.docx"].last_run_id == RUN_2


def test_apply_scanner_skip_without_output_or_sha_ignored() -> None:
    prev = make_snapshot({"docs/a.docx": make_entry()})
    records = [
        FileRecord(source="docs/a.docx", status=FileStatus.SKIPPED, skip_reason="unsupported"),
        FileRecord(source="new.bin", sha256=SHA_B, status=FileStatus.SKIPPED, skip_reason="x"),
    ]
    snap = apply(prev, records=records)
    assert "new.bin" not in snap.files
    assert snap.files["docs/a.docx"] == prev.files["docs/a.docx"]
    assert snap.files["docs/a.docx"].last_run_id == RUN_1


def test_apply_failed_record_keeps_prior_entry_untouched() -> None:
    prev = make_snapshot({"docs/a.docx": make_entry(sha256=SHA_A)})
    # even a failed record carrying sha256+output must not move the baseline
    record = FileRecord(
        source="docs/a.docx", sha256=SHA_B, size=9, status=FileStatus.FAILED, output="out"
    )
    snap = apply(prev, records=[record])
    entry = snap.files["docs/a.docx"]
    assert entry.sha256 == SHA_A
    assert entry.last_run_id == RUN_1


def test_apply_failed_new_file_adds_no_entry() -> None:
    record = FileRecord(source="new.docx", sha256=SHA_B, size=9, status=FileStatus.FAILED)
    snap = apply(None, records=[record])
    assert snap.files == {}


def test_apply_interrupted_record_keeps_prior_entry_untouched() -> None:
    prev = make_snapshot({"docs/a.docx": make_entry()})
    record = FileRecord(
        source="docs/a.docx", sha256=SHA_B, size=9, status=FileStatus.INTERRUPTED, output="out"
    )
    snap = apply(prev, records=[record])
    assert snap.files["docs/a.docx"] == prev.files["docs/a.docx"]


def test_apply_completed_moves_rekey_entry_and_rewrite_output() -> None:
    prev = make_snapshot({"old/a.docx": make_entry(sha256=SHA_A, size=10, output="old/a.docx.md")})
    snap = apply(prev, completed_moves=[("old/a.docx", "new/a.docx")])
    assert "old/a.docx" not in snap.files
    entry = snap.files["new/a.docx"]
    assert entry.sha256 == SHA_A
    assert entry.size == 10
    assert entry.output == "new/a.docx.md"
    assert entry.last_run_id == RUN_2


def test_apply_completed_deletes_remove_entries() -> None:
    prev = make_snapshot({"gone.docx": make_entry(), "kept.docx": make_entry()})
    snap = apply(prev, completed_deletes=["gone.docx", "never-tracked.docx"])
    assert "gone.docx" not in snap.files
    assert "kept.docx" in snap.files


def test_apply_previous_none_starts_empty() -> None:
    snap = apply(None)
    assert snap.files == {}
    assert snap.repo == "proj"
    assert snap.source_root == "/src/proj"
    assert snap.run_id == RUN_2


def test_apply_unmentioned_entries_carry_over_with_old_run_id() -> None:
    prev = make_snapshot({"stale.docx": make_entry(run_id=RUN_1)})
    record = FileRecord(
        source="fresh.docx", sha256=SHA_B, size=3, status=FileStatus.SUCCESS, output="out"
    )
    snap = apply(prev, records=[record])
    assert snap.files["stale.docx"] == prev.files["stale.docx"]
    assert snap.files["stale.docx"].last_run_id == RUN_1
    assert snap.files["fresh.docx"].last_run_id == RUN_2


def test_apply_then_save_persists_folded_state(tmp_path: Path) -> None:
    prev = make_snapshot(
        {
            "old/a.docx": make_entry(output="old/a.docx.md"),
            "gone.docx": make_entry(),
            "kept.docx": make_entry(),
        }
    )
    record = FileRecord(
        source="new.docx", sha256=SHA_B, size=3, status=FileStatus.SUCCESS, output="out"
    )
    snap = apply(
        prev,
        completed_moves=[("old/a.docx", "new/a.docx")],
        completed_deletes=["gone.docx"],
        records=[record],
    )
    save_snapshot(tmp_path, snap)
    on_disk = json.loads(snapshot_path(tmp_path, "proj").read_text(encoding="utf-8"))
    assert sorted(on_disk["files"]) == ["kept.docx", "new.docx", "new/a.docx"]
    assert on_disk["files"]["new/a.docx"]["output"] == "new/a.docx.md"
    assert on_disk["files"]["kept.docx"]["last_run_id"] == RUN_1
    assert on_disk["run_id"] == RUN_2
