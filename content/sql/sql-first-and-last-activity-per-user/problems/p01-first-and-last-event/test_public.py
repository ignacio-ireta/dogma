"""Public tests: the two worked examples from problem.md and the empty-table rule.

The learner's query is read from submission.sql and run against an in-memory DuckDB table with the
declared schema. The tests grade the returned rows and types, never the query text.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pytest

COLUMNS = ["user_id", "first_event_time", "first_event_type", "last_event_time", "last_event_type"]
TYPES = ["BIGINT", "TIMESTAMP", "VARCHAR", "TIMESTAMP", "VARCHAR"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")


def _run(rows: list[tuple[int, int, datetime, str]]) -> list[tuple[object, ...]]:
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE events(user_id BIGINT, event_id BIGINT, event_time TIMESTAMP, "
            "event_type VARCHAR)"
        )
        if rows:
            connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("the query cannot run against the declared events schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the five named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("keep the id as BIGINT and the times as TIMESTAMP, the types as VARCHAR")
        return cursor.fetchall()


def test_ties_on_first_and_last_time_use_the_smallest_and_largest_event_id() -> None:
    rows = [
        (1, 1, datetime(2026, 1, 1, 10), "login"),
        (1, 2, datetime(2026, 1, 1, 10), "view"),
        (1, 3, datetime(2026, 1, 2, 9), "logout"),
        (1, 4, datetime(2026, 1, 2, 9), "purchase"),
        (2, 5, datetime(2026, 1, 3, 8), "view"),
    ]
    expected = [
        (1, datetime(2026, 1, 1, 10), "login", datetime(2026, 1, 2, 9), "purchase"),
        (2, datetime(2026, 1, 3, 8), "view", datetime(2026, 1, 3, 8), "view"),
    ]
    got = _run(rows)
    assert len(got) == len(expected), "return one row per user, including users with one event"
    for actual, wanted in zip(got, expected, strict=True):
        assert actual[0] == wanted[0], "return all users ordered by user_id ascending"
        assert actual[1] == wanted[1], "first_event_time must be the earliest timestamp"
        assert actual[3] == wanted[3], "last_event_time must be the latest timestamp"
        assert actual[2] == wanted[2], (
            "first_event_type must come from the earliest event, breaking time ties by the "
            "smallest event_id; MIN(event_type) chooses alphabetically instead"
        )
        assert actual[4] == wanted[4], (
            "last_event_type must come from the latest event, breaking time ties by the "
            "largest event_id; MAX(event_type) chooses alphabetically instead"
        )


def test_event_type_is_selected_by_time_rather_than_alphabetically() -> None:
    rows = [
        (3, 90, datetime(2026, 4, 1, 9), "zebra"),
        (3, 2, datetime(2026, 4, 1, 10), "alpha"),
    ]
    [result] = _run(rows)
    assert result[2] == "zebra", (
        "first_event_type comes from the earliest timestamp, not MIN(event_type) or MIN(event_id)"
    )
    assert result[4] == "alpha", (
        "last_event_type comes from the latest timestamp, not MAX(event_type) or MAX(event_id)"
    )


def test_user_with_one_event_is_both_first_and_last() -> None:
    rows = [(9, 4, datetime(2026, 3, 1, 6), "signup")]
    got = _run(rows)
    if got != [(9, datetime(2026, 3, 1, 6), "signup", datetime(2026, 3, 1, 6), "signup")]:
        pytest.fail("a user with a single event must return that event as both first and last")


def test_empty_table_returns_no_rows_with_the_declared_schema() -> None:
    if _run([]) != []:
        pytest.fail("an empty events table must return no rows")
