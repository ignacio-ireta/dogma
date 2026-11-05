"""Backups: every table in one JSON file, written atomically, with the schema revision recorded."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from tests.unit.db.support import make_report

from dogma.core.config import Settings
from dogma.db.backup import BACKUP_FORMAT, write_backup
from dogma.db.engine import make_engine
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.migrate import alembic_config
from dogma.db.models import AttemptMode
from dogma.db.session import database_path, upgrade_database
from dogma.db.store import record_attempt

HEAD = "a9b0c1d2e3f4"
NOW = datetime(2026, 10, 24, 9, 30, 15, tzinfo=UTC)
EMPTY_ROWS = {
    "attempt": 0,
    "topic_state": 0,
    "card": 0,
    "card_review": 0,
    "study_session": 0,
    "session_block": 0,
    "resolve_window": 0,
    "answer": 0,
    "mock_session": 0,
    "calendar_warp": 0,
}


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def test_a_backup_of_a_new_database_holds_empty_tables_and_the_revision(
    settings: Settings,
) -> None:
    result = write_backup(settings, now=NOW)

    assert result.path == settings.var_dir / "backups" / "dogma-20261024T093015Z.json"
    assert dict(result.rows) == EMPTY_ROWS
    payload = _load(result.path)
    assert payload["format"] == BACKUP_FORMAT
    assert payload["format_version"] == 1
    assert payload["schema_revision"] == HEAD
    assert payload["exported_at"] == "2026-10-24T09:30:15+00:00"
    assert payload["tables"] == {name: [] for name in EMPTY_ROWS}


def test_a_backup_holds_every_row_with_utc_timestamps(settings: Settings) -> None:
    upgrade_database(settings)
    started = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)
    record_attempt(
        settings,
        make_report(public=(4, 4), hidden=(11, 11)),
        mode=AttemptMode.SUBMIT,
        core=True,
        started_at=started,
        submitted_at=started + timedelta(milliseconds=500),
    )

    result = write_backup(settings, now=NOW)

    payload = _load(result.path)
    [attempt] = payload["tables"]["attempt"]
    assert attempt["problem_id"] == "cml/vector-operations/p01-vector-projection"
    assert (attempt["public_passed"], attempt["hidden_passed"]) == (4, 11)
    assert attempt["started_at"] == "2026-10-24T09:00:00.000000+00:00"
    assert attempt["submitted_at"] == "2026-10-24T09:00:00.500000+00:00"
    [state] = payload["tables"]["topic_state"]
    assert state["topic_id"] == "cml/vector-operations"
    assert state["mastery"] == "solved"
    assert dict(result.rows) == {**EMPTY_ROWS, "attempt": 1, "topic_state": 1}


def test_the_backup_leaves_no_temporary_file_behind(settings: Settings) -> None:
    write_backup(settings, now=NOW)

    assert sorted(p.name for p in (settings.var_dir / "backups").iterdir()) == [
        "dogma-20261024T093015Z.json"
    ]


def test_a_backup_folder_that_is_a_file_is_a_backup_error(settings: Settings) -> None:
    settings.var_dir.mkdir(parents=True)
    (settings.var_dir / "backups").write_text("not a folder")

    with pytest.raises(StorageError) as caught:
        write_backup(settings, now=NOW)

    assert caught.value.code is StorageErrorCode.BACKUP_WRITE_FAILED


def test_backup_preserves_the_original_sqlite_database_before_migrating(settings: Settings) -> None:
    engine = make_engine(database_path(settings))
    try:
        with engine.begin() as connection:
            config = alembic_config()
            config.attributes["connection"] = connection
            command.upgrade(config, "c3d4e5f6a7b8")
            connection.exec_driver_sql(
                "INSERT INTO topic_state VALUES "
                "('cml/vector-operations', 'solved', NULL, "
                "'2026-10-20T12:01:00.000000+00:00', NULL, NULL)"
            )
        # Keep the engine's pooled connection open: committed rows still live in the WAL.
        assert database_path(settings).with_suffix(".db-wal").stat().st_size > 0
        result = write_backup(settings, now=NOW)
    finally:
        engine.dispose()

    snapshot = result.path.with_suffix(".sqlite3")
    assert snapshot.is_file(), "a pre-migration SQLite backup must exist before opening a session"
    with sqlite3.connect(snapshot) as saved:
        assert saved.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert saved.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "c3d4e5f6a7b8",
        )
        assert saved.execute("SELECT topic_id, mastery FROM topic_state").fetchall() == [
            ("cml/vector-operations", "solved")
        ]
    assert _load(result.path)["tables"]["topic_state"][0]["mastery"] == "solved"


def test_a_failed_snapshot_stops_before_any_migration(settings: Settings) -> None:
    engine = make_engine(database_path(settings))
    try:
        with engine.begin() as connection:
            config = alembic_config()
            config.attributes["connection"] = connection
            command.upgrade(config, "c3d4e5f6a7b8")
        (settings.var_dir / "backups").write_text("not a directory")
        with pytest.raises(StorageError) as caught:
            write_backup(settings, now=NOW)
        assert caught.value.code is StorageErrorCode.BACKUP_WRITE_FAILED
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").one() == (
                "c3d4e5f6a7b8",
            )
    finally:
        engine.dispose()
