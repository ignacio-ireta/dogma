from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import duckdb
import pytest

COLUMNS = ["store_id", "sales_date", "trailing_average_cents", "days_in_window"]
TYPES = ["BIGINT", "DATE", "DOUBLE", "BIGINT"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE daily_sales(store_id BIGINT, sales_date DATE, revenue_cents BIGINT)"
        )
        if rows:
            connection.executemany("INSERT INTO daily_sales VALUES (?, ?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared daily_sales schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the four named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("output types must be BIGINT, DATE, DOUBLE and BIGINT in that order")
        return cursor.fetchall()


def _same(got, expected, mistake):
    if len(got) != len(expected):
        pytest.fail(mistake)
    for g, e in zip(got, expected, strict=True):
        if g[:2] != e[:2] or g[3] != e[3]:
            pytest.fail(mistake)
        if not math.isclose(g[2], e[2], rel_tol=1e-12, abs_tol=1e-9):
            pytest.fail(mistake)


def test_window_skips_closed_days_and_covers_seven_calendar_days():
    rows = [
        (1, date(2026, 1, 1), 100),
        (1, date(2026, 1, 3), 300),
        (1, date(2026, 1, 7), 7),
        (1, date(2026, 1, 9), 50),
        (1, date(2026, 1, 10), 70),
    ]
    expected = [
        (1, date(2026, 1, 1), 100.0, 1),
        (1, date(2026, 1, 3), 200.0, 2),
        (1, date(2026, 1, 7), (100 + 300 + 7) / 3, 3),
        (1, date(2026, 1, 9), (300 + 7 + 50) / 3, 3),
        (1, date(2026, 1, 10), (7 + 50 + 70) / 3, 3),
    ]
    _same(
        _run(rows),
        expected,
        "trailing averages differ: the window must be the seven calendar days ending on the "
        "row's date, and closed days must not count as zero",
    )


def test_single_row_store_averages_only_itself():
    _same(
        _run([(2, date(2026, 2, 28), 999)]),
        [(2, date(2026, 2, 28), 999.0, 1)],
        "a store with one row must average only that row",
    )


def test_empty_table_returns_no_rows_with_the_declared_schema():
    _same(_run([]), [], "an empty table must return no rows with the declared columns and types")
