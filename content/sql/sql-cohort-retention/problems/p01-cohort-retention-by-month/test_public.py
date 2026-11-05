"""Public tests: the worked example from problem.md, a single-event user, and an empty table.

The learner's query is read from submission.sql and run against a fresh in-memory DuckDB table.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import duckdb
import pytest

COLUMNS = ["cohort_month", "period", "active_users", "cohort_size", "retention_rate"]
TYPES = ["DATE", "BIGINT", "BIGINT", "BIGINT", "DOUBLE"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")
# ROUND(x, 4) moves a rate by at most half a unit in the fourth decimal. The margin is DOUBLE noise.
RATE_TOLERANCE = 5.0e-5 + 1.0e-9


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE activity(user_id BIGINT, event_at TIMESTAMP)")
        if rows:
            connection.executemany("INSERT INTO activity VALUES (?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared activity schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the five named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail(
                "column types differ from the contract: cohort_month is DATE, period and the two "
                "counts are BIGINT, retention_rate is DOUBLE (cast the rate if needed)"
            )
        return cursor.fetchall()


def _check(got, expected):
    if [(row[0], row[1]) for row in got] != [(row[0], row[1]) for row in expected]:
        pytest.fail(
            "cohort-period grid differs: keep every period from 0 to the last activity month, "
            "including empty ones, and list only cohort months that have users"
        )
    for given, want in zip(got, expected, strict=True):
        if given[2] != want[2]:
            pytest.fail(
                "active_users differs: count DISTINCT users, not events, per cohort and period; "
                "measure the period in whole calendar months"
            )
        if given[3] != want[3]:
            pytest.fail(
                "cohort_size differs: the denominator is the number of users whose first activity "
                "is in that cohort month"
            )
        if abs(given[4] - want[4]) > RATE_TOLERANCE:
            pytest.fail(
                "retention_rate differs from ROUND(active_users / cohort_size, 4): divide by the "
                "cohort size (not the previous period or the active users) and round to four "
                "decimal places, not two"
            )


def test_worked_example_from_problem_md():
    rows = [
        (1, datetime(2026, 1, 3, 9)),
        (1, datetime(2026, 1, 20, 18, 30)),
        (1, datetime(2026, 3, 7, 8)),
        (1, datetime(2026, 4, 2, 12)),
        (2, datetime(2026, 1, 9, 10)),
        (2, datetime(2026, 1, 9, 11)),
        (3, datetime(2026, 2, 1, 7, 45)),
        (3, datetime(2026, 3, 5, 16)),
        (4, datetime(2026, 2, 27, 21, 10)),
    ]
    expected = [
        (date(2026, 1, 1), 0, 2, 2, 1.0),
        (date(2026, 1, 1), 1, 0, 2, 0.0),
        (date(2026, 1, 1), 2, 1, 2, 0.5),
        (date(2026, 1, 1), 3, 1, 2, 0.5),
        (date(2026, 2, 1), 0, 2, 2, 1.0),
        (date(2026, 2, 1), 1, 1, 2, 0.5),
        (date(2026, 2, 1), 2, 0, 2, 0.0),
    ]
    _check(_run(rows), expected)


def test_single_event_user_is_a_cohort_of_one_with_rate_one_at_period_zero():
    _check(
        _run([(9, datetime(2026, 1, 5))]),
        [(date(2026, 1, 1), 0, 1, 1, 1.0)],
    )


def test_empty_table_returns_no_rows_with_the_declared_schema():
    _check(_run([]), [])
