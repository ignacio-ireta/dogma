"""The I-30a migration: calendar_warp appears from the I-26 head, and every existing row stays."""

from __future__ import annotations

import sqlite3

import pytest
from alembic import command
from sqlalchemy import inspect

from dogma.core.config import Settings
from dogma.db.engine import make_engine
from dogma.db.migrate import alembic_config
from dogma.db.session import database_path, ensure_schema

I26_HEAD = "b8c9d0e1f2a3"  # the revision this migration builds on
HEAD = "a9b0c1d2e3f4"  # I-30a: calendar_warp


def test_upgrading_from_the_i26_head_keeps_every_row_and_adds_only_the_calendar_log(
    settings: Settings,
) -> None:
    engine = make_engine(database_path(settings))
    config = alembic_config()
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, I26_HEAD)
            connection.exec_driver_sql(
                "INSERT INTO attempt VALUES "
                "('legacy', 'cml/vector-operations/p01-vector-projection', "
                "2, 'submit', 'passed', 4, 4, 11, 11, 0, 'var/runs/legacy', "
                "'2026-10-20T12:00:00.000000+00:00', '2026-10-20T12:01:00.000000+00:00', 60000)"
            )
            connection.exec_driver_sql(
                "INSERT INTO topic_state VALUES "
                "('cml/vector-operations', 'solved', '2026-10-19T12:00:00.000000+00:00', "
                "'2026-10-20T12:01:00.000000+00:00', NULL, NULL)"
            )
    finally:
        engine.dispose()

    result = ensure_schema(make_engine(database_path(settings)))

    assert result.before == I26_HEAD
    assert result.after == HEAD
    connection = sqlite3.connect(str(database_path(settings)))
    try:
        assert connection.execute("SELECT COUNT(*) FROM attempt").fetchone() == (1,)
        assert connection.execute("SELECT mastery FROM topic_state").fetchone() == ("solved",)
        assert connection.execute("SELECT COUNT(*) FROM calendar_warp").fetchone() == (0,)
    finally:
        connection.close()
    engine = make_engine(database_path(settings))
    try:
        assert "calendar_warp" in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_the_calendar_log_refuses_an_unknown_kind(settings: Settings) -> None:
    ensure_schema(make_engine(database_path(settings)))

    connection = sqlite3.connect(str(database_path(settings)))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO calendar_warp (created_at, kind, offset_weeks, "
                "previous_offset_weeks, week_before, week_after) "
                "VALUES ('2026-11-03T12:00:00.000000+00:00', 'bogus', 1, 0, 2, 1)"
            )
    finally:
        connection.close()
