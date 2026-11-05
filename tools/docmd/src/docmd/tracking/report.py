"""Per-run sync change report (``changes.json``).

The report is the traceability artifact of a sync run: every action —
added/updated/moved/deleted — with where the previous conversion went
(``archived_to``) and how the action ended. Unchanged files are counted, never
listed, so the report stays small on large repos.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from docmd.errors import ErrorInfo
from docmd.fsio import atomic_write_text

CHANGES_NAME = "changes.json"


class ChangeKind(StrEnum):
    ADDED = "added"
    UPDATED = "updated"
    MOVED = "moved"
    DELETED = "deleted"


class ChangeStatus(StrEnum):
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ChangeRecord(BaseModel):
    """One planned or executed action on a tracked document."""

    action: ChangeKind
    source: str  # current posix relpath; for deleted: the vanished relpath
    previous_source: str | None = None  # set when content came from another path (moves)
    sha256: str | None = None  # current content hash; None for deleted
    previous_sha256: str | None = None  # hash from the prior snapshot, when one existed
    output: str | None = None  # posix relpath under <out>/<repo>/; None for deleted
    archived_to: str | None = None  # posix relpath under <out>, when a version was archived
    status: ChangeStatus
    error: ErrorInfo | None = None


class SyncCounts(BaseModel):
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    moved: int = 0
    deleted: int = 0
    failed: int = 0
    interrupted: int = 0


class SyncReport(BaseModel):
    schema_version: int = 1
    run_id: str
    repo: str
    source_root: str
    output_root: str
    generated_at: datetime
    dry_run: bool = False
    notes: list[str] = Field(default_factory=list)
    counts: SyncCounts
    changes: list[ChangeRecord] = Field(default_factory=list)


def write_changes_json(run_dir: Path, report: SyncReport) -> Path:
    path = run_dir / CHANGES_NAME
    atomic_write_text(path, report.model_dump_json(indent=2) + "\n")
    return path
