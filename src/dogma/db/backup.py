"""``dogma backup``: every table, as one JSON file under var/backups (ENGINEERING_STANDARDS §9).

The file is written atomically, so an interrupted backup never leaves a truncated export behind.
The schema revision is recorded so a restore knows which migrations the data needs.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from sqlalchemy import Table, select
from sqlalchemy.orm import Session

from dogma.core.config import Settings
from dogma.core.fsio import atomic_write

# Every table is exported: importing the model modules registers them on Base.metadata.
from dogma.db import (  # noqa: F401
    answer_models,
    calendar_models,
    mock_models,
    models,
    resolve_models,
    review_models,
    session_models,
)
from dogma.db.base import Base
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.migrate import current_revision
from dogma.db.session import database_path, open_session

BACKUP_FORMAT: Final = "dogma-backup"
BACKUP_FORMAT_VERSION: Final = 1


@dataclass(frozen=True)
class BackupResult:
    path: Path
    rows: Mapping[str, int]
    snapshot_path: Path | None = None


def write_backup(settings: Settings, *, now: datetime) -> BackupResult:
    """Export every table to ``var/backups/dogma-<UTC time>.json`` and return where it went."""
    stamp = now.astimezone(UTC)
    suffix = f"{stamp:%Y%m%dT%H%M%S}" + (f"{stamp:%f}" if stamp.microsecond else "") + "Z"
    path = settings.var_dir / "backups" / f"dogma-{suffix}.json"
    snapshot = path.with_suffix(".sqlite3") if database_path(settings).is_file() else None
    if snapshot is not None:
        _snapshot_database(database_path(settings), snapshot)
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        revision = current_revision(session.connection())
        tables = {table.name: _table_rows(session, table) for table in Base.metadata.sorted_tables}
    payload = {
        "format": BACKUP_FORMAT,
        "format_version": BACKUP_FORMAT_VERSION,
        "exported_at": stamp.isoformat(timespec="seconds"),
        "schema_revision": revision,
        "tables": tables,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        atomic_write(path, text)
    except OSError as exc:
        raise StorageError(
            StorageErrorCode.BACKUP_WRITE_FAILED, f"cannot write {path}: {exc.strerror or exc}"
        ) from exc
    return BackupResult(
        path=path, rows={name: len(rows) for name, rows in tables.items()}, snapshot_path=snapshot
    )


def _snapshot_database(source: Path, destination: Path) -> None:
    """Capture the old schema and committed WAL rows before open_session can migrate them."""
    try:
        with (
            closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as original,
            closing(sqlite3.connect(":memory:")) as snapshot,
        ):
            original.backup(snapshot)
            atomic_write(destination, snapshot.serialize())
    except (OSError, sqlite3.Error) as exc:
        raise StorageError(
            StorageErrorCode.BACKUP_WRITE_FAILED,
            f"cannot save the pre-migration database: {exc}",
        ) from exc


def _table_rows(session: Session, table: Table) -> list[dict[str, object]]:
    result = session.execute(select(table).order_by(*table.primary_key.columns)).mappings()
    return [{key: _jsonable(value) for key, value in row.items()} for row in result]


def _jsonable(value: object) -> object:
    return value.isoformat(timespec="microseconds") if isinstance(value, datetime) else value
