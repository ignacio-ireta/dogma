"""Run state: per-file records, the run manifest, and its atomic store.

The manifest is written through on every file completion (atomic replace), so
a crash or Ctrl+C always leaves a loadable manifest behind for ``--resume``.
File keys are posix relative paths — stable across platforms.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from docmd.errors import ErrorInfo
from docmd.fsio import atomic_write_text

MANIFEST_NAME = "manifest.json"
ERRORS_NAME = "errors.json"


class FileStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


class FileRecord(BaseModel):
    source: str  # posix relative path; also the manifest key
    sha256: str | None = None
    size: int = 0
    status: FileStatus
    output: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    extractor_version: str | None = None
    warning_count: int = 0
    error: ErrorInfo | None = None
    skip_reason: str | None = None


class RunSummary(BaseModel):
    total: int = 0
    success: int = 0
    warning: int = 0
    failed: int = 0
    skipped: int = 0
    interrupted: int = 0
    duration_s: float = 0.0

    @classmethod
    def from_records(cls, records: list[FileRecord], duration_s: float) -> RunSummary:
        counts = dict.fromkeys(FileStatus, 0)
        for record in records:
            counts[record.status] += 1
        return cls(
            total=len(records),
            success=counts[FileStatus.SUCCESS],
            warning=counts[FileStatus.WARNING],
            failed=counts[FileStatus.FAILED],
            skipped=counts[FileStatus.SKIPPED],
            interrupted=counts[FileStatus.INTERRUPTED]
            + counts[FileStatus.PENDING]
            + counts[FileStatus.RUNNING],
            duration_s=round(duration_s, 3),
        )


class Manifest(BaseModel):
    schema_version: int = 1
    run_id: str
    tool_version: str
    command: list[str]
    input_root: str
    output_root: str
    started_at: datetime
    finished_at: datetime | None = None
    files: dict[str, FileRecord] = Field(default_factory=dict)
    summary: RunSummary | None = None


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return now.strftime("%Y%m%dT%H%M%SZ")


def allocate_run_dir(runs_root: Path) -> tuple[str, Path]:
    """Create a unique run directory; suffix the id on same-second collisions.

    ``mkdir`` itself is the existence check — a concurrent process racing for
    the same second loses the create and retries with the next suffix instead
    of crashing on ``FileExistsError``.
    """
    base = new_run_id()
    run_id = base
    n = 1
    while True:
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(parents=True)
        except FileExistsError:
            run_id = f"{base}-{n}"
            n += 1
            continue
        return run_id, run_dir


class ManifestStore:
    """Thread-safe write-through persistence for a run manifest."""

    def __init__(self, manifest: Manifest, path: Path) -> None:
        self.manifest = manifest
        self.path = path
        self._lock = threading.Lock()
        self.save()

    def update(self, record: FileRecord) -> None:
        with self._lock:
            self.manifest.files[record.source] = record
            self._save_locked()

    def update_many(self, records: list[FileRecord]) -> None:
        with self._lock:
            for record in records:
                self.manifest.files[record.source] = record
            self._save_locked()

    def finalize(self, summary: RunSummary) -> None:
        with self._lock:
            self.manifest.summary = summary
            self.manifest.finished_at = datetime.now(UTC)
            self._save_locked()

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        atomic_write_text(self.path, self.manifest.model_dump_json(indent=2) + "\n")

    @property
    def records(self) -> list[FileRecord]:
        with self._lock:
            return list(self.manifest.files.values())

    @staticmethod
    def load_latest(runs_dir: Path) -> Manifest | None:
        if not runs_dir.is_dir():
            return None
        candidates = sorted(
            (p for p in runs_dir.iterdir() if (p / MANIFEST_NAME).is_file()),
            key=lambda p: p.name,
        )
        for run_dir in reversed(candidates):
            try:
                raw = (run_dir / MANIFEST_NAME).read_text(encoding="utf-8")
                return Manifest.model_validate_json(raw)
            except (OSError, ValueError):
                continue
        return None


def write_errors_json(run_dir: Path, records: list[FileRecord]) -> Path:
    """Persist failed/interrupted records as a machine-readable report."""
    path = run_dir / ERRORS_NAME
    bad = [
        r.model_dump(mode="json", exclude_none=True)
        for r in records
        if r.status in (FileStatus.FAILED, FileStatus.INTERRUPTED)
    ]
    atomic_write_text(path, json.dumps(bad, indent=2, ensure_ascii=False) + "\n")
    return path
