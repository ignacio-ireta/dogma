"""The read-only protection snapshot: no creation, no migration, no writes, and loud failures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dogma.core.config import Settings
from dogma.db.engine import DB_FILENAME
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.run_artifacts import protected_runs
from dogma.db.session import upgrade_database


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(repo_root=tmp_path / "repo")


def _db(settings: Settings) -> Path:
    return settings.var_dir / DB_FILENAME


def _insert(path: Path, attempt_id: str, run_dir: str | None) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO attempt (id, problem_id, problem_version, mode, status, public_passed,"
            " public_total, hidden_passed, hidden_total, hints_used, run_dir, started_at,"
            " submitted_at, duration_ms) VALUES (?, 'cml/x/p01', 1, 'run', 'passed', 1, 1, 0, 0,"
            " 0, ?, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:01+00:00', 1000)",
            (attempt_id, run_dir),
        )
    conn.close()


def test_a_missing_database_yields_empty_sets_and_creates_nothing(settings: Settings) -> None:
    assert protected_runs(settings) == (frozenset(), frozenset())
    assert not settings.var_dir.exists()


def test_recorded_ids_and_stored_run_dirs_are_returned(settings: Settings) -> None:
    upgrade_database(settings)
    db = _db(settings)
    _insert(db, "a-1", "var/runs/a-1")
    _insert(db, "rejected-2", None)
    _insert(db, "odd-3", "../../outside")

    ids, paths = protected_runs(settings)
    assert ids == frozenset({"a-1", "rejected-2", "odd-3"})
    assert paths == frozenset({"var/runs/a-1", "../../outside"})


def test_the_snapshot_does_not_change_the_database_file(settings: Settings) -> None:
    upgrade_database(settings)
    db = _db(settings)
    _insert(db, "a-1", "var/runs/a-1")
    before_bytes = db.read_bytes()
    before_stat = db.stat()

    protected_runs(settings)

    assert db.read_bytes() == before_bytes
    assert db.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_the_snapshot_opens_the_file_read_only(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    upgrade_database(settings)
    seen: list[tuple[str, bool]] = []
    real_connect = sqlite3.connect

    def spy(database: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        seen.append((database, bool(kwargs.get("uri"))))
        return real_connect(database, *args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as patched:
        patched.setattr("dogma.db.run_artifacts.sqlite3.connect", spy)
        protected_runs(settings)
    assert len(seen) == 1
    uri, is_uri = seen[0]
    assert is_uri and uri.startswith("file:") and uri.endswith("?mode=ro")


def test_a_write_through_the_snapshot_connection_is_refused(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    upgrade_database(settings)
    db = _db(settings)
    _insert(db, "a-1", "var/runs/a-1")
    real_connect = sqlite3.connect
    refused: list[str] = []

    def probe(database: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(database, *args, **kwargs)  # type: ignore[arg-type]
        try:
            conn.execute("DELETE FROM attempt")
        except sqlite3.OperationalError as exc:
            refused.append(str(exc))
        return conn

    # Scoped to the call: the test's own verification connections must not go through the probe.
    with monkeypatch.context() as patched:
        patched.setattr("dogma.db.run_artifacts.sqlite3.connect", probe)
        protected_runs(settings)
    assert refused, "the connection must not accept writes"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM attempt").fetchone()[0] == 1
    conn.close()


def test_a_database_without_migrations_is_not_migrated(settings: Settings) -> None:
    db = _db(settings)
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE attempt (id TEXT PRIMARY KEY, run_dir TEXT)")
        conn.execute("INSERT INTO attempt VALUES ('a-1', NULL)")
    conn.close()

    assert protected_runs(settings) == (frozenset({"a-1"}), frozenset())
    with sqlite3.connect(db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    conn.close()
    user_tables = {name for name in tables if not name.startswith("sqlite_")}
    assert user_tables == {"attempt"}, "no alembic or other tables may appear"


def test_a_malformed_database_aborts_with_a_storage_error(settings: Settings) -> None:
    db = _db(settings)
    db.parent.mkdir(parents=True)
    db.write_bytes(b"this is not a sqlite database at all" * 200)
    with pytest.raises(StorageError) as info:
        protected_runs(settings)
    assert info.value.code == StorageErrorCode.DB_READ_FAILED
    assert db.read_bytes().startswith(b"this is not")


def test_a_schema_without_the_attempt_columns_aborts(settings: Settings) -> None:
    db = _db(settings)
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE attempt (id TEXT PRIMARY KEY)")
    conn.close()
    with pytest.raises(StorageError) as info:
        protected_runs(settings)
    assert info.value.code == StorageErrorCode.DB_READ_FAILED


def test_a_database_without_an_attempt_table_aborts(settings: Settings) -> None:
    db = _db(settings)
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE something_else (x INTEGER)")
    conn.close()
    with pytest.raises(StorageError) as info:
        protected_runs(settings)
    assert info.value.code == StorageErrorCode.DB_READ_FAILED


def test_an_unopenable_database_path_aborts(settings: Settings) -> None:
    db = _db(settings)
    db.mkdir(parents=True)  # a directory where the database file should be
    with pytest.raises(StorageError) as info:
        protected_runs(settings)
    assert info.value.code == StorageErrorCode.DB_UNAVAILABLE
