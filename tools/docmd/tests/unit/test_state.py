"""Manifest store: atomic persistence, thread safety, latest-run loading."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from docmd.errors import ErrorInfo
from docmd.pipeline.state import (
    FileRecord,
    FileStatus,
    Manifest,
    ManifestStore,
    RunSummary,
    allocate_run_dir,
    new_run_id,
    write_errors_json,
)


def make_manifest(run_id: str = "20260610T120000Z") -> Manifest:
    return Manifest(
        run_id=run_id,
        tool_version="0.1.0",
        command=["scan", "in", "--out", "out"],
        input_root="in",
        output_root="out",
        started_at=datetime(2026, 6, 10, tzinfo=UTC),
    )


def test_new_run_id_format() -> None:
    rid = new_run_id(datetime(2026, 6, 10, 15, 30, 12, tzinfo=UTC))
    assert rid == "20260610T153012Z"


def test_allocate_run_dir_uniqueness(tmp_path: Path) -> None:
    id1, dir1 = allocate_run_dir(tmp_path)
    id2, dir2 = allocate_run_dir(tmp_path)
    assert dir1 != dir2 and id1 != id2
    assert dir1.is_dir() and dir2.is_dir()


def test_store_write_through(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    store = ManifestStore(make_manifest(), path)
    assert path.exists()  # saved on construction
    store.update(FileRecord(source="a.docx", status=FileStatus.SUCCESS))
    on_disk = json.loads(path.read_text())
    assert on_disk["files"]["a.docx"]["status"] == "success"


def test_store_concurrent_updates(tmp_path: Path) -> None:
    store = ManifestStore(make_manifest(), tmp_path / "manifest.json")

    def work(i: int) -> None:
        store.update(FileRecord(source=f"f{i}.docx", status=FileStatus.SUCCESS))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    reloaded = Manifest.model_validate_json((tmp_path / "manifest.json").read_text())
    assert len(reloaded.files) == 50


def test_finalize_sets_summary(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    store = ManifestStore(make_manifest(), path)
    store.finalize(RunSummary(total=1, success=1, duration_s=0.5))
    reloaded = Manifest.model_validate_json(path.read_text())
    assert reloaded.summary is not None and reloaded.summary.success == 1
    assert reloaded.finished_at is not None


def test_load_latest_picks_newest_valid(tmp_path: Path) -> None:
    for rid in ["20260610T100000Z", "20260610T110000Z"]:
        d = tmp_path / rid
        d.mkdir()
        ManifestStore(make_manifest(rid), d / "manifest.json")
    # newest dir holds garbage -> falls back to the previous valid one
    bad = tmp_path / "20260610T120000Z"
    bad.mkdir()
    (bad / "manifest.json").write_text("{not json")
    latest = ManifestStore.load_latest(tmp_path)
    assert latest is not None and latest.run_id == "20260610T110000Z"


def test_load_latest_empty(tmp_path: Path) -> None:
    assert ManifestStore.load_latest(tmp_path / "missing") is None


def test_run_summary_counts_pending_as_interrupted() -> None:
    records = [
        FileRecord(source="a", status=FileStatus.SUCCESS),
        FileRecord(source="b", status=FileStatus.PENDING),
        FileRecord(source="c", status=FileStatus.INTERRUPTED),
        FileRecord(source="d", status=FileStatus.WARNING),
    ]
    summary = RunSummary.from_records(records, 1.0)
    assert summary.total == 4
    assert summary.interrupted == 2
    assert summary.warning == 1


def test_write_errors_json_only_bad_records(tmp_path: Path) -> None:
    records = [
        FileRecord(source="ok.docx", status=FileStatus.SUCCESS),
        FileRecord(
            source="bad.pdf",
            status=FileStatus.FAILED,
            error=ErrorInfo(error_code="CORRUPT_FILE", error_type="X", message="m"),
        ),
        FileRecord(source="late.docx", status=FileStatus.INTERRUPTED),
    ]
    path = write_errors_json(tmp_path, records)
    data = json.loads(path.read_text())
    assert [d["source"] for d in data] == ["bad.pdf", "late.docx"]
    assert data[0]["error"]["error_code"] == "CORRUPT_FILE"
