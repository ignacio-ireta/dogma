"""Read-only snapshot of the run artifacts the learner database still references.

Pruning must never delete a recorded attempt (ARCHITECTURE §4.9). This module reads the ids and
stored ``run_dir`` strings with a read-only SQLite connection. It never creates, migrates or writes
the database (so it does not use ``open_session``), and any doubt about the file aborts the caller.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from dogma.core.config import Settings
from dogma.db.engine import DB_FILENAME
from dogma.db.errors import StorageError, StorageErrorCode

_REQUIRED_COLUMNS = frozenset({"id", "run_dir"})


def protected_runs(settings: Settings) -> tuple[frozenset[str], frozenset[str]]:
    """Attempt ids and non-null stored ``run_dir`` strings, exactly as the database holds them.

    A missing database has nothing to protect and yields empty sets. An existing database that is
    unreadable, malformed, or lacks the ``attempt`` columns raises ``StorageError``, so the caller
    never prunes against a protection set it could not read.
    """
    path = settings.var_dir / DB_FILENAME
    if not path.exists() and not path.is_symlink():
        return frozenset(), frozenset()
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise StorageError(
            StorageErrorCode.DB_UNAVAILABLE, f"cannot open {path} read-only: {_reason(exc)}"
        ) from exc
    try:
        return _read(conn, path)
    finally:
        conn.close()


def _read(conn: sqlite3.Connection, path: Path) -> tuple[frozenset[str], frozenset[str]]:
    try:
        verdict = conn.execute("PRAGMA quick_check").fetchone()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(attempt)")}
        if verdict is None or verdict[0] != "ok":
            raise StorageError(
                StorageErrorCode.DB_READ_FAILED, f"{path} failed its integrity check"
            )
        if not columns >= _REQUIRED_COLUMNS:
            raise StorageError(
                StorageErrorCode.DB_READ_FAILED,
                f"{path} has no attempt table with id and run_dir; not pruning",
            )
        rows = conn.execute("SELECT id, run_dir FROM attempt").fetchall()
    except sqlite3.Error as exc:
        raise StorageError(
            StorageErrorCode.DB_READ_FAILED, f"cannot read {path}: {_reason(exc)}"
        ) from exc
    ids = frozenset(str(row[0]) for row in rows if row[0] is not None)
    run_dirs = frozenset(str(row[1]) for row in rows if row[1] is not None)
    return ids, run_dirs


def _reason(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__
