# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's query is read from submission.sql; tests never read solution.sql.
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

COLUMNS = ["user_id", "streak_days", "streak_start", "streak_end"]
TYPES = ["BIGINT", "BIGINT", "DATE", "DATE"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE visits(user_id BIGINT, visit_date DATE)")
        if rows:
            connection.executemany("INSERT INTO visits VALUES (?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared visits schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the four named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("cast streak_days to BIGINT; dates must stay DATE")
        return cursor.fetchall()


def _same(got, expected, mistake):
    if got != expected:
        pytest.fail(mistake)


def test_worked_example_returns_longest_run_per_user_with_its_dates():
    rows = [
        (1, date(2026, 1, 1)),
        (1, date(2026, 1, 2)),
        (1, date(2026, 1, 2)),
        (1, date(2026, 1, 3)),
        (1, date(2026, 1, 5)),
        (1, date(2026, 1, 6)),
        (2, date(2026, 3, 10)),
        (3, date(2026, 2, 27)),
        (3, date(2026, 2, 28)),
        (3, date(2026, 3, 1)),
        (4, date(2026, 1, 1)),
        (4, date(2026, 1, 2)),
        (4, date(2026, 1, 10)),
        (4, date(2026, 1, 11)),
    ]
    _same(
        _run(rows),
        [
            (1, 3, date(2026, 1, 1), date(2026, 1, 3)),
            (2, 1, date(2026, 3, 10), date(2026, 3, 10)),
            (3, 3, date(2026, 2, 27), date(2026, 3, 1)),
            (4, 2, date(2026, 1, 1), date(2026, 1, 2)),
        ],
        "the worked example differs: check duplicate days, the month-end run and the tie rule",
    )


def test_single_visit_user_is_a_streak_of_one_day():
    _same(
        _run([(9, date(2026, 7, 4))]),
        [(9, 1, date(2026, 7, 4), date(2026, 7, 4))],
        "a user with one visit has a streak of one day that starts and ends on that day",
    )


def test_empty_table_returns_no_rows_with_the_declared_schema():
    _same(_run([]), [], "an empty table must return no rows, with the declared columns and types")
