"""Per-repo snapshot: the last synced state of every tracked source file.

The snapshot is the diff baseline: source relpath -> (sha256, size, output).
It is rewritten atomically after each sync and only ever reflects completed
work, so re-running sync after a crash or interrupt converges instead of
losing state. A corrupt snapshot is recoverable — the repo is treated as new
and the snapshot rebuilds in one run — unlike the registry, its blast radius
is a single repo.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from docmd.errors import SnapshotCorruptError
from docmd.fsio import atomic_write_text
from docmd.pipeline.state import FileRecord, FileStatus

SNAPSHOT_NAME = "snapshot.json"


class SnapshotEntry(BaseModel):
    sha256: str
    size: int
    output: str  # posix relpath under <out>/<repo>/
    last_run_id: str  # run that produced (or last rewrote) this output


class RepoSnapshot(BaseModel):
    schema_version: int = 1
    repo: str
    source_root: str
    run_id: str  # run that last wrote this snapshot
    updated_at: datetime
    recursive: bool | None = None  # scan depth of the run that wrote this; None = pre-field
    files: dict[str, SnapshotEntry] = Field(default_factory=dict)  # key: source posix relpath


def snapshot_path(out_root: Path, repo: str) -> Path:
    return out_root / ".docmd" / "repos" / repo / SNAPSHOT_NAME


def load_snapshot(out_root: Path, repo: str) -> RepoSnapshot | None:
    path = snapshot_path(out_root, repo)
    if not path.exists():
        return None
    try:
        return RepoSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SnapshotCorruptError(
            f"snapshot for repo {repo!r} is unreadable: {path}",
            path=path,
            detail=str(exc),
        ) from exc


def save_snapshot(out_root: Path, snapshot: RepoSnapshot) -> Path:
    path = snapshot_path(out_root, snapshot.repo)
    atomic_write_text(path, snapshot.model_dump_json(indent=2) + "\n")
    return path


def output_relpath_for(source_relpath: str) -> str:
    """Mirror of ``pipeline.steps.output_path_for`` in relpath space."""
    return source_relpath + ".md"


def apply_run_results(
    previous: RepoSnapshot | None,
    *,
    repo: str,
    source_root: str,
    run_id: str,
    recursive: bool,
    completed_moves: Iterable[tuple[str, str]] = (),
    completed_deletes: Iterable[str] = (),
    records: Iterable[FileRecord] = (),
) -> RepoSnapshot:
    """Fold one run's completed work into the snapshot.

    Only completed work moves the baseline: failed or interrupted conversions
    of previously-tracked files keep their prior entry (the old output is
    still live — the pipeline overwrites only on success), and failed new
    files get no entry so the next sync retries them.
    """
    files = dict(previous.files) if previous is not None else {}

    for old, new in completed_moves:
        entry = files.pop(old, None)
        if entry is not None:
            files[new] = entry.model_copy(
                update={"output": output_relpath_for(new), "last_run_id": run_id}
            )

    for path in completed_deletes:
        files.pop(path, None)

    for record in records:
        if record.sha256 is None or record.output is None:
            continue
        if record.status not in (FileStatus.SUCCESS, FileStatus.WARNING, FileStatus.SKIPPED):
            continue
        files[record.source] = SnapshotEntry(
            sha256=record.sha256,
            size=record.size,
            output=output_relpath_for(record.source),
            last_run_id=run_id,
        )

    return RepoSnapshot(
        repo=repo,
        source_root=source_root,
        run_id=run_id,
        updated_at=datetime.now(UTC),
        recursive=recursive,
        files=files,
    )
