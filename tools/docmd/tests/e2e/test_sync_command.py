"""End-to-end `docmd sync` runs: tracking, diffing, archiving, guards, repos views."""

import hashlib
import json
import os
import shutil
import signal
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from docmd.cli import app

pytestmark = pytest.mark.e2e

runner = CliRunner()


@pytest.fixture(autouse=True)
def restore_sigint() -> Iterator[None]:
    """sync() installs its own SIGINT handler; keep pytest's intact."""
    previous = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, previous)


def sync(src: Path, out: Path, *extra: str) -> Any:
    return runner.invoke(app, ["sync", str(src), "--out", str(out), "--no-progress", *extra])


def combined_output(result: Any) -> str:
    """stdout plus stderr regardless of the installed click's capture mode."""
    try:
        return str(result.output) + str(result.stderr)
    except (ValueError, AttributeError):
        return str(result.output)


def latest_run_dir(out: Path) -> Path:
    return max((out / "runs").iterdir(), key=lambda p: p.name)


def read_changes(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((latest_run_dir(out) / "changes.json").read_text())
    return data


def read_registry(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "registry.json").read_text())
    return data


def read_snapshot(out: Path, repo: str) -> dict[str, Any]:
    path = out / ".docmd" / "repos" / repo / "snapshot.json"
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def by_action(changes: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for change in changes["changes"]:
        grouped.setdefault(change["action"], []).append(change)
    return grouped


def tree_state(root: Path) -> dict[str, tuple[Any, ...]]:
    """Capture contents and metadata so dry-run tests detect transient state writes."""
    state: dict[str, tuple[Any, ...]] = {}
    if not root.exists():
        return state
    for path in [root, *sorted(root.rglob("*"))]:
        stat = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        if path.is_dir():
            state[relative] = ("dir", stat.st_mode, stat.st_mtime_ns)
        elif path.is_symlink():
            state[relative] = ("symlink", path.readlink().as_posix(), stat.st_mode)
        else:
            state[relative] = (
                "file",
                stat.st_mode,
                stat.st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return state


@pytest.fixture
def repo(fixtures_dir: Path, tmp_path: Path) -> Path:
    src = tmp_path / "handbook"
    (src / "guides").mkdir(parents=True)
    shutil.copy(fixtures_dir / "docx" / "basic.docx", src / "guides" / "setup.docx")
    shutil.copy(fixtures_dir / "xlsx" / "basic.xlsx", src / "budget.xlsx")
    shutil.copy(fixtures_dir / "docx" / "special_chars.docx", src / "notes.docx")
    return src


def test_sync_first_run_registers_and_converts(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    result = sync(repo, out)
    assert result.exit_code == 0, combined_output(result)

    assert (out / "handbook" / "guides" / "setup.docx.md").is_file()
    assert (out / "handbook" / "guides" / "setup.docx.md.meta.json").is_file()
    assert (out / "handbook" / "budget.xlsx.md").is_file()

    registry = read_registry(out)
    entry = registry["repos"]["handbook"]
    assert entry["run_count"] == 1
    assert entry["file_count"] == 3
    assert entry["source_root"] == str(repo)

    snapshot = read_snapshot(out, "handbook")
    assert set(snapshot["files"]) == {"guides/setup.docx", "budget.xlsx", "notes.docx"}
    assert snapshot["recursive"] is True

    changes = read_changes(out)
    assert changes["counts"]["added"] == 3
    assert all(c["action"] == "added" and c["status"] == "completed" for c in changes["changes"])
    assert all(c["output"] for c in changes["changes"])


def test_sync_rerun_unchanged_does_no_work(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    result = sync(repo, out)
    assert result.exit_code == 0, combined_output(result)

    changes = read_changes(out)
    assert changes["counts"]["unchanged"] == 3
    assert changes["changes"] == []
    manifest = json.loads((latest_run_dir(out) / "manifest.json").read_text())
    assert manifest["summary"]["success"] == 0  # nothing was converted


def test_sync_updated_file_archives_previous_version(
    repo: Path, tmp_path: Path, fixtures_dir: Path
) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    old_sha = read_snapshot(out, "handbook")["files"]["budget.xlsx"]["sha256"]
    old_md = (out / "handbook" / "budget.xlsx.md").read_text()

    shutil.copy(fixtures_dir / "xlsx" / "cached_formula.xlsx", repo / "budget.xlsx")
    result = sync(repo, out)
    assert result.exit_code == 0, combined_output(result)

    run_id = latest_run_dir(out).name
    archived = out / ".docmd" / "archive" / "handbook" / run_id / "budget.xlsx.md"
    assert archived.is_file()
    assert archived.read_text() == old_md
    assert archived.with_name(archived.name + ".meta.json").is_file()

    new_sha = read_snapshot(out, "handbook")["files"]["budget.xlsx"]["sha256"]
    assert new_sha != old_sha

    (updated,) = by_action(read_changes(out))["updated"]
    assert updated["status"] == "completed"
    assert updated["archived_to"] == f".docmd/archive/handbook/{run_id}/budget.xlsx.md"
    assert updated["previous_sha256"] == old_sha
    assert updated["sha256"] == new_sha


def test_sync_deleted_file_archived_and_untracked(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    (repo / "notes.docx").unlink()
    result = sync(repo, out)
    assert result.exit_code == 0, combined_output(result)

    assert not (out / "handbook" / "notes.docx.md").exists()
    assert not (out / "handbook" / "notes.docx.md.meta.json").exists()
    run_id = latest_run_dir(out).name
    archive = out / ".docmd" / "archive" / "handbook" / run_id
    assert (archive / "notes.docx.md").is_file()
    assert (archive / "notes.docx.md.meta.json").is_file()

    assert "notes.docx" not in read_snapshot(out, "handbook")["files"]
    assert read_registry(out)["repos"]["handbook"]["file_count"] == 2
    (deleted,) = by_action(read_changes(out))["deleted"]
    assert deleted["status"] == "completed"
    assert deleted["archived_to"] == f".docmd/archive/handbook/{run_id}/notes.docx.md"


def test_sync_moved_file_relocates_without_reextraction(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    old_md = (out / "handbook" / "guides" / "setup.docx.md").read_text()

    (repo / "manual").mkdir()
    (repo / "guides" / "setup.docx").rename(repo / "manual" / "install.docx")
    (repo / "guides").rmdir()
    result = sync(repo, out)
    assert result.exit_code == 0, combined_output(result)

    new_path = out / "handbook" / "manual" / "install.docx.md"
    assert new_path.is_file()
    assert not (out / "handbook" / "guides").exists()  # old output pruned with its dir

    new_md = new_path.read_text()
    assert 'source_file: "manual/install.docx"' in new_md
    strip = "source_file: "
    old_rest = [line for line in old_md.splitlines() if not line.startswith(strip)]
    new_rest = [line for line in new_md.splitlines() if not line.startswith(strip)]
    assert old_rest == new_rest  # body byte-identical: nothing was re-extracted

    sidecar = json.loads((out / "handbook" / "manual" / "install.docx.md.meta.json").read_text())
    assert sidecar["source_file"] == "manual/install.docx"

    manifest = json.loads((latest_run_dir(out) / "manifest.json").read_text())
    assert manifest["summary"]["success"] == 0  # no conversion happened

    (moved,) = by_action(read_changes(out))["moved"]
    assert moved["previous_source"] == "guides/setup.docx"
    assert moved["status"] == "completed"
    snapshot = read_snapshot(out, "handbook")["files"]
    assert "guides/setup.docx" not in snapshot
    assert snapshot["manual/install.docx"]["output"] == "manual/install.docx.md"


def test_sync_case_only_rename_relocates_in_place(repo: Path, tmp_path: Path) -> None:
    """`notes.docx` -> `NOTES.docx`: on case-insensitive filesystems the old and
    new outputs are one file; relocation must fix the stored case, never
    archive the pair away as a stray or destroy its own result."""
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    body_before = (out / "handbook" / "notes.docx.md").read_text()

    (repo / "notes.docx").rename(repo / "NOTES.docx")
    result = sync(repo, out)
    assert result.exit_code == 0, combined_output(result)

    names = {p.name for p in (out / "handbook").iterdir()}
    assert "NOTES.docx.md" in names
    assert "notes.docx.md" not in names
    md = (out / "handbook" / "NOTES.docx.md").read_text()
    assert 'source_file: "NOTES.docx"' in md
    strip = "source_file: "
    assert [line for line in md.splitlines() if not line.startswith(strip)] == [
        line for line in body_before.splitlines() if not line.startswith(strip)
    ]

    manifest = json.loads((latest_run_dir(out) / "manifest.json").read_text())
    assert manifest["summary"]["success"] == 0  # relocated, not re-extracted
    (moved,) = by_action(read_changes(out))["moved"]
    assert moved["previous_source"] == "notes.docx"
    assert moved["status"] == "completed"
    snapshot = read_snapshot(out, "handbook")["files"]
    assert "NOTES.docx" in snapshot
    assert "notes.docx" not in snapshot


def test_sync_duplicate_content_moves_fall_back_to_delete_add(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    src = tmp_path / "twins"
    src.mkdir()
    shutil.copy(fixtures_dir / "docx" / "basic.docx", src / "a.docx")
    shutil.copy(fixtures_dir / "docx" / "basic.docx", src / "b.docx")
    out = tmp_path / "kb"
    assert sync(src, out).exit_code == 0

    (src / "a.docx").unlink()
    (src / "b.docx").rename(src / "c.docx")
    result = sync(src, out)
    assert result.exit_code == 0, combined_output(result)

    changes = read_changes(out)
    assert changes["counts"]["moved"] == 0
    assert changes["counts"]["deleted"] == 2
    assert changes["counts"]["added"] == 1
    grouped = by_action(changes)
    assert "moved" not in grouped
    assert {c["source"] for c in grouped["deleted"]} == {"a.docx", "b.docx"}
    run_id = latest_run_dir(out).name
    archive = out / ".docmd" / "archive" / "twins" / run_id
    assert (archive / "a.docx.md").is_file()
    assert (archive / "b.docx.md").is_file()


def test_sync_casefold_repo_name_clash_exit_2(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    # Simulate a repo tracked under a different casing of the same name — on
    # case-insensitive filesystems both would share one output directory.
    registry_file = out / "registry.json"
    registry_file.write_text(registry_file.read_text().replace('"handbook"', '"HANDBOOK"'))

    result = sync(repo, out)
    assert result.exit_code == 2
    assert "REPO_NAME_RESERVED" in combined_output(result)
    assert "HANDBOOK" in combined_output(result)


def test_sync_reserved_repo_name_exit_2(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "runs"
    src.mkdir()
    shutil.copy(fixtures_dir / "docx" / "basic.docx", src / "a.docx")
    result = sync(src, tmp_path / "kb")
    assert result.exit_code == 2
    assert "REPO_NAME_RESERVED" in combined_output(result)


def test_sync_dry_run_new_repo_writes_nothing(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    result = sync(repo, out, "--dry-run")
    assert result.exit_code == 0, combined_output(result)

    assert not out.exists()
    output = combined_output(result)
    assert "Sync planned (dry run)" in output
    assert "added: 3" in output
    assert "Run report:" not in output


def test_sync_dry_run_changes_nothing_on_known_repo(
    repo: Path, tmp_path: Path, fixtures_dir: Path
) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    first_snapshot = read_snapshot(out, "handbook")

    shutil.copy(fixtures_dir / "xlsx" / "cached_formula.xlsx", repo / "budget.xlsx")
    (repo / "notes.docx").unlink()
    before = tree_state(out)
    result = sync(repo, out, "--dry-run")
    assert result.exit_code == 0, combined_output(result)

    assert tree_state(out) == before
    output = combined_output(result)
    assert "updated: 1" in output
    assert "deleted: 1" in output
    assert "Run report:" not in output
    assert (out / "handbook" / "notes.docx.md").is_file()  # nothing archived
    assert read_snapshot(out, "handbook") == first_snapshot
    assert read_registry(out)["repos"]["handbook"]["run_count"] == 1


def test_sync_corrupt_registry_fails_loudly(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    (out / "registry.json").write_text("{ not json")
    result = sync(repo, out)
    assert result.exit_code == 3
    assert "REGISTRY_CORRUPT" in combined_output(result)
    assert (out / "handbook" / "budget.xlsx.md").is_file()  # outputs untouched


def test_sync_corrupt_snapshot_treated_as_new(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    (out / ".docmd" / "repos" / "handbook" / "snapshot.json").write_text("garbage")

    result = sync(repo, out)
    assert result.exit_code == 0, combined_output(result)
    changes = read_changes(out)
    assert any("snapshot was corrupt" in note for note in changes["notes"])
    snapshot = read_snapshot(out, "handbook")
    assert len(snapshot["files"]) == 3  # rebuilt in one run
    # Existing outputs matched by hash were adopted, not re-archived.
    run_id = latest_run_dir(out).name
    assert not (out / ".docmd" / "archive" / "handbook" / run_id).exists()


def test_sync_failed_update_keeps_previous_output_live(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    old_sha = read_snapshot(out, "handbook")["files"]["budget.xlsx"]["sha256"]
    old_md = (out / "handbook" / "budget.xlsx.md").read_text()

    (repo / "budget.xlsx").write_bytes(b"this is not a spreadsheet")
    result = sync(repo, out)
    assert result.exit_code == 1, combined_output(result)

    assert (out / "handbook" / "budget.xlsx.md").read_text() == old_md
    run_id = latest_run_dir(out).name
    assert (out / ".docmd" / "archive" / "handbook" / run_id / "budget.xlsx.md").is_file()
    assert read_snapshot(out, "handbook")["files"]["budget.xlsx"]["sha256"] == old_sha
    (updated,) = by_action(read_changes(out))["updated"]
    assert updated["status"] == "failed"
    assert updated["error"]["error_code"] == "CORRUPT_FILE"

    retry = sync(repo, out)
    assert retry.exit_code == 1  # still corrupt: re-detected as updated, retried
    (updated,) = by_action(read_changes(out))["updated"]
    assert updated["source"] == "budget.xlsx"


def test_sync_empty_source_known_repo_exit_4(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    for path in list(repo.rglob("*")):
        if path.is_file():
            path.unlink()

    result = sync(repo, out)
    assert result.exit_code == 4
    assert "deletions were NOT processed" in combined_output(result)
    assert len(read_snapshot(out, "handbook")["files"]) == 3
    assert (out / "handbook" / "budget.xlsx.md").is_file()


def test_sync_self_heals_missing_output(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    (out / "handbook" / "notes.docx.md").unlink()  # .md gone, sidecar intact
    (out / "handbook" / "budget.xlsx.md.meta.json").unlink()  # sidecar gone, .md intact

    result = sync(repo, out)
    assert result.exit_code == 0, combined_output(result)
    assert (out / "handbook" / "notes.docx.md").is_file()
    assert (out / "handbook" / "budget.xlsx.md.meta.json").is_file()
    changes = read_changes(out)
    assert any("missing on disk" in note for note in changes["notes"])
    healed = by_action(changes)["updated"]
    assert {c["source"] for c in healed} == {"notes.docx", "budget.xlsx"}
    assert all(c["status"] == "completed" for c in healed)


def test_sync_mass_deletion_guard_requires_confirm(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "bulk"
    src.mkdir()
    for i in range(12):
        shutil.copy(fixtures_dir / "docx" / "basic.docx", src / f"doc_{i:02d}.docx")
    out = tmp_path / "kb"
    assert sync(src, out).exit_code == 0

    for i in range(11):  # 11 deletions > max(10, 12 // 4)
        (src / f"doc_{i:02d}.docx").unlink()
    result = sync(src, out)
    assert result.exit_code == 0, combined_output(result)
    changes = read_changes(out)
    assert any("safety threshold" in note for note in changes["notes"])
    assert changes["counts"]["deleted"] == 0
    assert len(read_snapshot(out, "bulk")["files"]) == 12
    assert (out / "bulk" / "doc_00.docx.md").is_file()  # nothing archived

    confirmed = sync(src, out, "--confirm-deletes")
    assert confirmed.exit_code == 0, combined_output(confirmed)
    assert read_changes(out)["counts"]["deleted"] == 11
    assert len(read_snapshot(out, "bulk")["files"]) == 1
    run_id = latest_run_dir(out).name
    assert (out / ".docmd" / "archive" / "bulk" / run_id / "doc_00.docx.md").is_file()


def test_sync_depth_change_skips_deletions(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "layered"
    (src / "nested").mkdir(parents=True)
    shutil.copy(fixtures_dir / "docx" / "basic.docx", src / "top.docx")
    shutil.copy(fixtures_dir / "docx" / "special_chars.docx", src / "nested" / "deep.docx")
    out = tmp_path / "kb"
    assert sync(src, out).exit_code == 0
    assert read_snapshot(out, "layered")["recursive"] is True

    result = sync(src, out, "--no-recursive")
    assert result.exit_code == 0, combined_output(result)
    changes = read_changes(out)
    assert any("scan depth changed" in note for note in changes["notes"])
    assert changes["counts"]["deleted"] == 0
    assert (out / "layered" / "nested" / "deep.docx.md").is_file()  # not archived away
    snapshot = read_snapshot(out, "layered")
    assert "nested/deep.docx" in snapshot["files"]
    assert snapshot["recursive"] is False


def test_sync_locked_repo_exit_2(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    lock = out / ".docmd" / "locks" / "handbook.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "hostname": socket.gethostname()}))

    result = sync(repo, out)
    assert result.exit_code == 2
    assert "REPO_LOCKED" in combined_output(result)
    lock.unlink()


def test_sync_stale_lock_is_replaced(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    lock = out / ".docmd" / "locks" / "handbook.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": 99999999, "hostname": socket.gethostname()}))

    result = sync(repo, out)
    assert result.exit_code == 0, combined_output(result)
    assert not lock.exists()  # released after the run


def test_sync_archive_keep_runs_prunes_oldest(
    repo: Path, tmp_path: Path, fixtures_dir: Path
) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    shutil.copy(fixtures_dir / "xlsx" / "cached_formula.xlsx", repo / "budget.xlsx")
    assert sync(repo, out).exit_code == 0
    shutil.copy(fixtures_dir / "xlsx" / "basic.xlsx", repo / "budget.xlsx")
    result = sync(repo, out, "--archive-keep-runs", "1")
    assert result.exit_code == 0, combined_output(result)

    archive = out / ".docmd" / "archive" / "handbook"
    remaining = sorted(p.name for p in archive.iterdir())
    assert remaining == [latest_run_dir(out).name]
    assert any("pruned" in note for note in read_changes(out)["notes"])


def test_repos_list_show_history_after_syncs(
    repo: Path, tmp_path: Path, fixtures_dir: Path
) -> None:
    out = tmp_path / "kb"
    assert sync(repo, out).exit_code == 0
    shutil.copy(fixtures_dir / "xlsx" / "cached_formula.xlsx", repo / "budget.xlsx")
    assert sync(repo, out).exit_code == 0

    listed = runner.invoke(app, ["repos", "list", "--out", str(out)])
    assert listed.exit_code == 0, combined_output(listed)
    assert "handbook" in listed.output

    shown = runner.invoke(app, ["repos", "show", "handbook", "--out", str(out), "--files"])
    assert shown.exit_code == 0, combined_output(shown)
    assert "budget.xlsx" in shown.output
    assert "runs:         2" in shown.output

    history = runner.invoke(app, ["repos", "history", "handbook", "--out", str(out)])
    assert history.exit_code == 0, combined_output(history)
    assert "updated" in history.output
    assert "archived_to=.docmd/archive/handbook/" in history.output

    unknown = runner.invoke(app, ["repos", "show", "nope", "--out", str(out)])
    assert unknown.exit_code == 1
