"""Archiving displaced outputs under <out>/.docmd/archive/<repo>/<run_id>/."""

from pathlib import Path

from docmd.markdown.metadata import sidecar_path_for
from docmd.tracking.archive import archive_output, archive_path_for, archive_root, prune_archive

MD_BODY = "# Report\n\nSome converted content.\n"
SIDECAR_BODY = '{"schema_version": 1, "source_file": "docs/report.docx"}\n'


def make_live_pair(md_path: Path) -> Path:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(MD_BODY, encoding="utf-8")
    sidecar = sidecar_path_for(md_path)
    sidecar.write_text(SIDECAR_BODY, encoding="utf-8")
    return sidecar


def test_archive_path_for_nested_relpath_maps_under_hidden_archive() -> None:
    out = Path("/out")
    run_id = "20260702T090000Z"
    path = archive_path_for(out, "notes", run_id, "docs/sub/report.docx.md")
    assert path == out / ".docmd" / "archive" / "notes" / run_id / "docs/sub/report.docx.md"
    assert path.is_relative_to(archive_root(out, "notes", run_id))


def test_archive_output_move_relocates_pair_and_preserves_content(tmp_path: Path) -> None:
    md = tmp_path / "notes" / "docs" / "report.docx.md"
    sidecar = make_live_pair(md)
    dest_md = archive_path_for(tmp_path, "notes", "run-1", "docs/report.docx.md")

    archived = archive_output(md, dest_md, mode="move")

    assert archived == [dest_md, sidecar_path_for(dest_md)]
    assert not md.exists()
    assert not sidecar.exists()
    assert dest_md.read_text(encoding="utf-8") == MD_BODY
    assert sidecar_path_for(dest_md).read_text(encoding="utf-8") == SIDECAR_BODY


def test_archive_output_copy_leaves_originals_in_place(tmp_path: Path) -> None:
    md = tmp_path / "notes" / "docs" / "report.docx.md"
    sidecar = make_live_pair(md)
    dest_md = archive_path_for(tmp_path, "notes", "run-1", "docs/report.docx.md")

    archived = archive_output(md, dest_md, mode="copy")

    assert archived == [dest_md, sidecar_path_for(dest_md)]
    assert md.read_text(encoding="utf-8") == MD_BODY
    assert sidecar.read_text(encoding="utf-8") == SIDECAR_BODY
    assert dest_md.read_text(encoding="utf-8") == MD_BODY
    assert sidecar_path_for(dest_md).read_text(encoding="utf-8") == SIDECAR_BODY


def test_archive_output_missing_sidecar_archives_only_md(tmp_path: Path) -> None:
    md = tmp_path / "notes" / "report.docx.md"
    md.parent.mkdir(parents=True)
    md.write_text(MD_BODY, encoding="utf-8")
    dest_md = archive_path_for(tmp_path, "notes", "run-1", "report.docx.md")

    archived = archive_output(md, dest_md, mode="move")

    assert archived == [dest_md]
    assert not md.exists()
    assert dest_md.read_text(encoding="utf-8") == MD_BODY
    assert not sidecar_path_for(dest_md).exists()


def test_archive_output_missing_md_archives_only_sidecar(tmp_path: Path) -> None:
    md = tmp_path / "notes" / "report.docx.md"
    md.parent.mkdir(parents=True)
    sidecar = sidecar_path_for(md)
    sidecar.write_text(SIDECAR_BODY, encoding="utf-8")
    dest_md = archive_path_for(tmp_path, "notes", "run-1", "report.docx.md")

    archived = archive_output(md, dest_md, mode="copy")

    assert archived == [sidecar_path_for(dest_md)]
    assert not dest_md.exists()
    assert sidecar_path_for(dest_md).read_text(encoding="utf-8") == SIDECAR_BODY


def test_archive_output_both_missing_returns_empty_and_creates_nothing(tmp_path: Path) -> None:
    md = tmp_path / "notes" / "gone.docx.md"  # never written
    dest_md = archive_path_for(tmp_path, "notes", "run-1", "gone.docx.md")

    assert archive_output(md, dest_md, mode="copy") == []
    assert archive_output(md, dest_md, mode="move") == []
    assert not (tmp_path / ".docmd").exists()  # no archive dirs materialized


def seed_archived_run(out: Path, repo: str, run_id: str) -> Path:
    run_dir = archive_root(out, repo, run_id)
    run_dir.mkdir(parents=True)
    (run_dir / "old.docx.md").write_text(MD_BODY, encoding="utf-8")
    return run_dir


def test_prune_archive_zero_keeps_everything(tmp_path: Path) -> None:
    seed_archived_run(tmp_path, "notes", "20260701T090000Z")
    assert prune_archive(tmp_path, "notes", keep_runs=0) == []
    assert archive_root(tmp_path, "notes", "20260701T090000Z").is_dir()


def test_prune_archive_missing_repo_dir_is_noop(tmp_path: Path) -> None:
    assert prune_archive(tmp_path, "notes", keep_runs=3) == []


def test_prune_archive_fewer_runs_than_keep_prunes_nothing(tmp_path: Path) -> None:
    seed_archived_run(tmp_path, "notes", "20260701T090000Z")
    seed_archived_run(tmp_path, "notes", "20260702T090000Z")
    assert prune_archive(tmp_path, "notes", keep_runs=2) == []
    assert prune_archive(tmp_path, "notes", keep_runs=3) == []


def test_prune_archive_drops_oldest_beyond_keep(tmp_path: Path) -> None:
    runs = ["20260630T090000Z", "20260701T090000Z", "20260702T090000Z"]
    for run_id in runs:
        seed_archived_run(tmp_path, "notes", run_id)
    seed_archived_run(tmp_path, "other", "20260629T090000Z")  # other repos untouched

    assert prune_archive(tmp_path, "notes", keep_runs=1) == runs[:2]
    assert not archive_root(tmp_path, "notes", runs[0]).exists()
    assert not archive_root(tmp_path, "notes", runs[1]).exists()
    assert archive_root(tmp_path, "notes", runs[2]).is_dir()
    assert archive_root(tmp_path, "other", "20260629T090000Z").is_dir()
