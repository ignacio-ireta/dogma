"""The I-18c migration: mock sessions, the answer link, and the interview kind, with rows kept."""

from __future__ import annotations

import sqlite3

from alembic import command
from sqlalchemy import inspect

from dogma.core.config import Settings
from dogma.db.engine import make_engine
from dogma.db.migrate import alembic_config
from dogma.db.session import database_path, ensure_schema

PREVIOUS_HEAD = "e5f6a7b8c9d0"  # I-18a: interview answers
I18C_HEAD = "f6a7b8c9d0e1"  # I-18c: mock sessions, interview planned sessions
HEAD = "a9b0c1d2e3f4"  # I-30a: the personal calendar log

SESSION_ROW = (
    "'s-learn', '2026-10-27', 1, 'learn', 'normal', 'closed', 'cml/vector-operations', "
    "'cml/vector-operations/p01-vector-projection', 'Open the problem', NULL, "
    "'2026-10-27T08:00:00.000000+00:00', '2026-10-27T09:00:00.000000+00:00', "
    "'Practice', 'Solve example 2', NULL, NULL"
)
BLOCK_ROW = "1, 's-learn', 1, 'Learn', 12, 'Read the lesson'"
ANSWER_ROW = (
    "'a-old', 'cml/vector-operations', 'interview', 'interview.md#Q1', 'A dot product.', "
    '\'topic-answer\', 1, \'{"accuracy": 3, "structure": 3, "trade-offs": 3, '
    "\"communication\": 4}', NULL, '2026-10-27T09:30:00.000000+00:00'"
)


def _rows(path: object, table: str) -> list[tuple[object, ...]]:
    connection = sqlite3.connect(str(path))
    try:
        return connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()  # noqa: S608  fixed names
    finally:
        connection.close()


def test_upgrading_from_the_i18a_head_keeps_every_row_and_adds_the_new_schema(
    settings: Settings,
) -> None:
    engine = make_engine(database_path(settings))
    config = alembic_config()
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, PREVIOUS_HEAD)
            connection.exec_driver_sql(f"INSERT INTO study_session VALUES ({SESSION_ROW})")  # noqa: S608
            connection.exec_driver_sql(f"INSERT INTO session_block VALUES ({BLOCK_ROW})")  # noqa: S608
            connection.exec_driver_sql(f"INSERT INTO answer VALUES ({ANSWER_ROW})")  # noqa: S608
    finally:
        engine.dispose()
    before = {
        table: _rows(database_path(settings), table)
        for table in ("study_session", "session_block", "answer")
    }

    result = ensure_schema(make_engine(database_path(settings)))

    assert result.before == PREVIOUS_HEAD
    assert result.after == HEAD
    after = {
        table: _rows(database_path(settings), table)
        for table in ("study_session", "session_block", "answer")
    }
    assert [(*row, None) for row in before["study_session"]] == after["study_session"]
    assert before["session_block"] == after["session_block"]
    assert [row[:4] for row in before["answer"]] == [row[:4] for row in after["answer"]]
    engine = make_engine(database_path(settings))
    try:
        names = set(inspect(engine).get_table_names())
        columns = {column["name"] for column in inspect(engine).get_columns("answer")}
        foreign_keys = inspect(engine).get_foreign_keys("session_block")
    finally:
        engine.dispose()
    assert "mock_session" in names
    assert "mock_session_id" in columns
    assert foreign_keys[0]["referred_table"] == "study_session"


def test_after_the_upgrade_a_planned_interview_session_is_accepted(settings: Settings) -> None:
    ensure_schema(make_engine(database_path(settings)))

    connection = sqlite3.connect(str(database_path(settings)))
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"INSERT INTO study_session VALUES ({SESSION_ROW}, NULL)")  # noqa: S608
        connection.execute("UPDATE study_session SET kind = 'interview' WHERE id = 's-learn'")
        connection.commit()
        kind = connection.execute("SELECT kind FROM study_session").fetchone()
    finally:
        connection.close()
    assert kind == ("interview",)
