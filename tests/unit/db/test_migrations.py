"""Migrations: empty file to head, idempotent re-runs, the models' schema, forward-only."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, inspect

from dogma.core.config import Settings
from dogma.db import (  # noqa: F401
    answer_models,
    mock_models,
    models,
    resolve_models,
    review_models,
    session_models,
)
from dogma.db.base import Base
from dogma.db.engine import make_engine
from dogma.db.errors import StorageError, StorageErrorCode
from dogma.db.migrate import alembic_config, head_revision
from dogma.db.session import database_path, ensure_schema, upgrade_database

INITIAL = "a1b2c3d4e5f6"
I09_HEAD = "b2c3d4e5f6a7"  # cards and reviews
I10_HEAD = "c3d4e5f6a7b8"  # planned sessions
I13_HEAD = "d4e5f6a7b8c9"  # I-13: timed re-solve windows
I18A_HEAD = "e5f6a7b8c9d0"  # I-18a: interview answers and their rubric scores
I18C_HEAD = "f6a7b8c9d0e1"  # I-18c: mock sessions, interview kind for planned sessions
I26_HEAD = "b8c9d0e1f2a3"  # I-26: persisted session target
HEAD = "a9b0c1d2e3f4"  # I-30a: the personal calendar log (calendar_warp)
PREVIOUS_HEAD = I13_HEAD


def _table_names(path: Path) -> set[str]:
    engine = make_engine(path)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _schema_sql(path: Path) -> dict[str, str]:
    """Each object's CREATE statement, whitespace-normalised, minus alembic's own bookkeeping."""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
        ).fetchall()
    finally:
        connection.close()
    return {name: " ".join(sql.split()) for name, sql in rows}


def test_upgrading_an_empty_location_reaches_head(settings: Settings) -> None:
    result = upgrade_database(settings)

    assert result.before is None
    assert result.after == HEAD
    assert {
        "attempt",
        "topic_state",
        "card",
        "card_review",
        "study_session",
        "session_block",
        "resolve_window",
        "answer",
        "calendar_warp",
        "alembic_version",
    } <= _table_names(database_path(settings))


def test_rerunning_the_upgrade_changes_nothing(settings: Settings) -> None:
    upgrade_database(settings)

    again = upgrade_database(settings)

    assert again.before == HEAD
    assert again.after == HEAD
    assert not again.changed


def test_the_migrated_schema_is_the_schema_the_models_describe(
    settings: Settings, tmp_path: Path
) -> None:
    upgrade_database(settings)
    from_models = tmp_path / "models" / "dogma.db"
    engine = make_engine(from_models)
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    assert _schema_sql(database_path(settings)) == _schema_sql(from_models)
    engine = make_engine(database_path(settings))
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            assert compare_metadata(context, Base.metadata) == []
    finally:
        engine.dispose()


def test_there_is_exactly_one_head_and_it_follows_the_initial_revision() -> None:
    script = ScriptDirectory.from_config(alembic_config())

    assert script.get_heads() == [HEAD]
    assert head_revision() == HEAD
    assert script.get_revision(HEAD).down_revision == I26_HEAD
    assert script.get_revision(I26_HEAD).down_revision == "a7b8c9d0e1f2"
    assert script.get_revision(I18C_HEAD).down_revision == I18A_HEAD
    assert script.get_revision(I18A_HEAD).down_revision == I13_HEAD
    assert script.get_revision(I13_HEAD).down_revision == I10_HEAD
    assert script.get_revision(I10_HEAD).down_revision == I09_HEAD
    assert script.get_revision(I09_HEAD).down_revision == INITIAL


def test_upgrading_old_progress_preserves_all_existing_rows_and_schema(settings: Settings) -> None:
    engine = make_engine(database_path(settings))
    config = alembic_config()
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, INITIAL)
            connection.exec_driver_sql(
                "INSERT INTO attempt VALUES "
                "('legacy-attempt', 'cml/vector-operations/p01-vector-projection', 2, "
                "'submit', 'passed', 4, 4, 11, 11, 0, 'var/runs/legacy-attempt', "
                "'2026-10-20T12:00:00.000000+00:00', '2026-10-20T12:01:00.000000+00:00', 60000)"
            )
            connection.exec_driver_sql(
                "INSERT INTO topic_state VALUES "
                "('cml/vector-operations', 'solved', '2026-10-19T12:00:00.000000+00:00', "
                "'2026-10-20T12:01:00.000000+00:00', NULL, NULL)"
            )
            before = {
                table: connection.exec_driver_sql(f"SELECT * FROM {table}").all()  # noqa: S608  fixed table names
                for table in ("attempt", "topic_state")
            }
        old_schema = _schema_sql(database_path(settings))
        result = ensure_schema(engine)
        assert result.before == INITIAL and result.after == HEAD
        with engine.connect() as connection:
            after = {
                table: connection.exec_driver_sql(f"SELECT * FROM {table}").all()  # noqa: S608  fixed table names
                for table in ("attempt", "topic_state")
            }
        assert after == before
        new_schema = _schema_sql(database_path(settings))
        assert all(new_schema[name] == sql for name, sql in old_schema.items())
    finally:
        engine.dispose()


def test_upgrading_from_the_i09_head_preserves_every_existing_row(settings: Settings) -> None:
    engine = make_engine(database_path(settings))
    config = alembic_config()
    tables = ("attempt", "topic_state", "card", "card_review")
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, PREVIOUS_HEAD)
            connection.exec_driver_sql(
                "INSERT INTO attempt VALUES "
                "('legacy-attempt', 'cml/vector-operations/p01-vector-projection', 2, "
                "'submit', 'passed', 4, 4, 11, 11, 0, 'var/runs/legacy-attempt', "
                "'2026-10-20T12:00:00.000000+00:00', '2026-10-20T12:01:00.000000+00:00', 60000)"
            )
            connection.exec_driver_sql(
                "INSERT INTO topic_state VALUES "
                "('cml/vector-operations', 'solved', '2026-10-19T12:00:00.000000+00:00', "
                "'2026-10-20T12:01:00.000000+00:00', NULL, NULL)"
            )
            connection.exec_driver_sql(
                "INSERT INTO card VALUES ('cml/vector-operations#1', 'cml/vector-operations', "
                "'" + "a" * 64 + "', 'Q?', 'A.', 1, 3, 2, 5.5, 4.2, "
                "'2026-10-21T12:00:00.000000+00:00', '2026-10-20T12:01:00.000000+00:00')"
            )
            connection.exec_driver_sql(
                "INSERT INTO card_review VALUES ('review-1', 'cml/vector-operations#1', 3, "
                "'2026-10-20T12:01:00.000000+00:00', '2026-10-20T12:01:00.000000+00:00', "
                "'2026-10-21T12:00:00.000000+00:00', '" + "a" * 64 + "', 'Q?', 'A.')"
            )
            before = {
                name: connection.exec_driver_sql(f"SELECT * FROM {name}").all()  # noqa: S608  fixed names
                for name in tables
            }
        result = ensure_schema(engine)
        assert result.before == PREVIOUS_HEAD and result.after == HEAD
        with engine.connect() as connection:
            after = {
                name: connection.exec_driver_sql(f"SELECT * FROM {name}").all()  # noqa: S608  fixed names
                for name in tables
            }
            new_tables = connection.exec_driver_sql(
                "SELECT COUNT(*) FROM study_session"
            ).scalar_one()
        assert after == before
        assert all(len(rows) == 1 for rows in after.values())
        assert new_tables == 0
    finally:
        engine.dispose()


def test_upgrading_from_the_i10_head_preserves_every_row_in_every_table(settings: Settings) -> None:
    engine = make_engine(database_path(settings))
    config = alembic_config()
    tables = (
        "attempt",
        "topic_state",
        "card",
        "card_review",
        "study_session",
        "session_block",
    )
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, I10_HEAD)
            _seed_every_table(connection)
            before = {name: _rows(connection, name) for name in tables}
        result = ensure_schema(engine)
        assert result.before == I10_HEAD and result.after == HEAD
        with engine.connect() as connection:
            after = {name: _rows(connection, name) for name in tables}
            windows = connection.exec_driver_sql("SELECT COUNT(*) FROM resolve_window").scalar_one()
            index_sql = connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name = 'ix_resolve_window_one_open'"
            ).scalar_one()
        assert after == before
        assert all(len(rows) == 1 for rows in after.values())
        assert windows == 0
        assert "UNIQUE" in index_sql and "status = 'open'" in index_sql
    finally:
        engine.dispose()


def _seed_every_table(connection: Connection) -> None:
    """One row in each table, written the way the I-10 schema holds them."""
    connection.exec_driver_sql(
        "INSERT INTO attempt VALUES "
        "('legacy-attempt', 'cml/vector-operations/p01-vector-projection', 2, "
        "'submit', 'passed', 4, 4, 11, 11, 0, 'var/runs/legacy-attempt', "
        "'2026-10-20T12:00:00.000000+00:00', '2026-10-20T12:01:00.000000+00:00', 60000)"
    )
    connection.exec_driver_sql(
        "INSERT INTO topic_state VALUES "
        "('cml/vector-operations', 'solved', '2026-10-19T12:00:00.000000+00:00', "
        "'2026-10-20T12:01:00.000000+00:00', NULL, NULL)"
    )
    connection.exec_driver_sql(
        "INSERT INTO card VALUES ('cml/vector-operations#1', 'cml/vector-operations', "
        "'" + "a" * 64 + "', 'Q?', 'A.', 1, 3, 2, 5.5, 4.2, "
        "'2026-10-21T12:00:00.000000+00:00', '2026-10-20T12:01:00.000000+00:00')"
    )
    connection.exec_driver_sql(
        "INSERT INTO card_review VALUES ('review-1', 'cml/vector-operations#1', 3, "
        "'2026-10-20T12:01:00.000000+00:00', '2026-10-20T12:01:00.000000+00:00', "
        "'2026-10-21T12:00:00.000000+00:00', '" + "a" * 64 + "', 'Q?', 'A.')"
    )
    connection.exec_driver_sql(
        "INSERT INTO study_session (id, plan_date, slot, kind, energy, status, topic_id, "
        "must_do, first_action, plan_notes, created_at, closed_at, stopped_where, next_step, "
        "blocker, self_answer) VALUES ('session-1', '2026-10-20', 1, 'learn', 'normal', "
        "'closed', 'cml/vector-operations', 'cml/vector-operations/p01-vector-projection', "
        "'Open it', NULL, '2026-10-20T12:00:00.000000+00:00', "
        "'2026-10-20T12:30:00.000000+00:00', 'Practice', 'Submit', NULL, NULL)"
    )
    connection.exec_driver_sql(
        "INSERT INTO session_block (session_id, position, name, minutes, detail) "
        "VALUES ('session-1', 1, 'Learn', 12, 'Read the lesson.')"
    )


def _rows(connection: Connection, name: str) -> list[tuple[object, ...]]:
    rows = connection.exec_driver_sql(f"SELECT * FROM {name}").all()  # noqa: S608  fixed names
    # Compare every legacy column across the additive session-target migration.
    return [tuple(row[:-1] if name == "study_session" and len(row) == 17 else row) for row in rows]


def test_connections_turn_on_foreign_keys_and_wal(settings: Settings) -> None:
    upgrade_database(settings)
    engine = make_engine(database_path(settings))
    try:
        with engine.connect() as connection:
            foreign_keys = connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()
            journal = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()
    finally:
        engine.dispose()

    assert foreign_keys == 1
    assert journal == "wal"


def test_downgrading_is_refused(settings: Settings) -> None:
    upgrade_database(settings)
    engine = make_engine(database_path(settings))
    config = alembic_config()
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            with pytest.raises(RuntimeError, match="forward-only"):
                command.downgrade(config, "base")
    finally:
        engine.dispose()


def test_a_file_that_is_not_a_database_is_a_migration_error(settings: Settings) -> None:
    path = database_path(settings)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not a sqlite database" * 100)

    with pytest.raises(StorageError) as caught:
        upgrade_database(settings)

    assert caught.value.code is StorageErrorCode.DB_MIGRATION_FAILED


def test_a_database_at_an_unknown_revision_is_a_migration_error(settings: Settings) -> None:
    engine = make_engine(database_path(settings))
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
            )
            connection.exec_driver_sql("INSERT INTO alembic_version VALUES ('zzzz0000')")
        with pytest.raises(StorageError) as caught:
            ensure_schema(engine)
    finally:
        engine.dispose()

    assert caught.value.code is StorageErrorCode.DB_MIGRATION_FAILED


def test_upgrading_from_the_i13_head_preserves_every_row_and_adds_only_the_answer_table(
    settings: Settings,
) -> None:
    engine = make_engine(database_path(settings))
    config = alembic_config()
    tables = (
        "attempt",
        "topic_state",
        "card",
        "card_review",
        "study_session",
        "session_block",
        "resolve_window",
    )
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, I13_HEAD)
            _seed_every_table(connection)
            connection.exec_driver_sql(
                "INSERT INTO resolve_window (id, problem_id, problem_version, topic_id, "
                "started_at, budget_s, hints_used, status, attempt_id, closed_at) VALUES "
                "('window-1', 'cml/vector-operations/p01-vector-projection', 2, "
                "'cml/vector-operations', '2026-10-20T12:00:00.000000+00:00', 1200, 0, "
                "'passed', 'legacy-attempt', '2026-10-20T12:01:00.000000+00:00')"
            )
            before_schema = _schema_sql(database_path(settings))
            before = {name: _rows(connection, name) for name in tables}
        result = ensure_schema(engine)
        assert result.before == I13_HEAD and result.after == HEAD
        with engine.connect() as connection:
            after = {name: _rows(connection, name) for name in tables}
            answers = connection.exec_driver_sql("SELECT COUNT(*) FROM answer").scalar_one()
        assert after == before
        assert answers == 0
        new_schema = _schema_sql(database_path(settings))
        assert all(
            new_schema[name] == sql
            for name, sql in before_schema.items()
            if name != "study_session"  # the timed target adds one column, tested separately
        )
        assert "answer" in new_schema
    finally:
        engine.dispose()
