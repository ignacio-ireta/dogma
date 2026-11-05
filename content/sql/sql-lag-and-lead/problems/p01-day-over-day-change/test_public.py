# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
# The learner's query is read from submission.sql; tests never import solution.sql.
from datetime import date
from pathlib import Path

import duckdb
import pytest

COLUMNS = [
    "account_id",
    "usage_date",
    "api_calls",
    "previous_usage_date",
    "previous_api_calls",
    "calls_change",
]
TYPES = ["BIGINT", "DATE", "BIGINT", "DATE", "BIGINT", "BIGINT"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE account_daily_usage(account_id BIGINT, usage_date DATE, api_calls BIGINT)"
        )
        if rows:
            connection.executemany("INSERT INTO account_daily_usage VALUES (?, ?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared account_daily_usage schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the six named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("check the column types: ids and changes BIGINT, dates DATE")
        return cursor.fetchall()


def test_first_day_per_account_has_null_previous_fields_and_change():
    rows = [
        (1, date(2026, 3, 1), 100),
        (1, date(2026, 3, 2), 150),
        (1, date(2026, 3, 5), 120),
        (2, date(2026, 3, 2), 40),
    ]
    expected = [
        (1, date(2026, 3, 1), 100, None, None, None),
        (1, date(2026, 3, 2), 150, date(2026, 3, 1), 100, 50),
        (1, date(2026, 3, 5), 120, date(2026, 3, 2), 150, -30),
        (2, date(2026, 3, 2), 40, None, None, None),
    ]
    assert _run(rows) == expected, (
        "previous-day fields and change must come from the previous recorded row of the same "
        "account; the first row of each account must be NULL"
    )


def test_empty_table_returns_no_rows_with_the_declared_schema():
    assert _run([]) == [], "an empty usage table must return no rows, not a placeholder row"


def test_single_day_account_has_null_predecessor_not_zero():
    rows = [(9, date(2026, 1, 15), 77)]
    assert _run(rows) == [(9, date(2026, 1, 15), 77, None, None, None)], (
        "an account with one recorded day has no predecessor: keep NULL, do not coalesce to 0"
    )
