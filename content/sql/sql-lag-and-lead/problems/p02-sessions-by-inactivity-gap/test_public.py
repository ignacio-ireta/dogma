# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
# The learner's query is read from submission.sql; tests never import solution.sql.
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

COLUMNS = ["visitor_id", "view_id", "viewed_at", "session_number", "starts_session"]
TYPES = ["BIGINT", "BIGINT", "TIMESTAMP", "BIGINT", "BOOLEAN"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE screen_views(visitor_id BIGINT, view_id BIGINT, viewed_at TIMESTAMP)"
        )
        if rows:
            connection.executemany("INSERT INTO screen_views VALUES (?, ?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared screen_views schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the five named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("check the types: session_number must be BIGINT, not a wider integer")
        return cursor.fetchall()


def test_thirty_minutes_exactly_continues_and_thirty_one_minutes_starts_a_session():
    rows = [
        (5, 1, datetime(2026, 4, 10, 10, 0)),
        (5, 2, datetime(2026, 4, 10, 10, 30)),
        (5, 3, datetime(2026, 4, 10, 11, 1)),
    ]
    assert _run(rows) == [
        (5, 1, datetime(2026, 4, 10, 10, 0), 1, True),
        (5, 2, datetime(2026, 4, 10, 10, 30), 1, False),
        (5, 3, datetime(2026, 4, 10, 11, 1), 2, True),
    ], "a gap of exactly 30 minutes continues the session; 31 minutes starts a new one"


def test_equal_timestamps_at_a_boundary_let_the_lower_view_id_open_the_session():
    rows = [
        (6, 11, datetime(2026, 4, 10, 10, 0)),
        (6, 13, datetime(2026, 4, 10, 10, 40)),
        (6, 12, datetime(2026, 4, 10, 10, 40)),
    ]
    assert _run(rows) == [
        (6, 11, datetime(2026, 4, 10, 10, 0), 1, True),
        (6, 12, datetime(2026, 4, 10, 10, 40), 2, True),
        (6, 13, datetime(2026, 4, 10, 10, 40), 2, False),
    ], "equal timestamps must be ordered by view_id, so the lower id opens the session"


def test_empty_table_returns_no_rows_with_the_declared_schema():
    assert _run([]) == [], "an empty screen_views table must return no rows"
