from __future__ import annotations

import csv
import math
import random
import re
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

COLUMNS = ["store_id", "sales_date", "trailing_average_cents", "days_in_window"]
TYPES = ["BIGINT", "DATE", "DOUBLE", "BIGINT"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")
BASE = date(2026, 1, 1)


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
        result = cursor.fetchall()
        after = connection.execute(
            "SELECT store_id, sales_date, revenue_cents "
            "FROM daily_sales ORDER BY store_id, sales_date"
        ).fetchall()
        if after != sorted(rows, key=lambda row: (row[0], row[1])):
            pytest.fail("query must not modify the input daily_sales table")
        return result


def _same(got, expected, mistake):
    if len(got) != len(expected):
        pytest.fail(mistake)
    for g, e in zip(got, expected, strict=True):
        if g[:2] != e[:2] or g[3] != e[3]:
            pytest.fail(mistake)
        if not math.isclose(g[2], e[2], rel_tol=1e-12, abs_tol=1e-9):
            pytest.fail(mistake)


def _oracle(rows):
    """Independent answer: for each row, average the rows of its store within six days before it."""
    result = []
    for store, when, _ in sorted(rows, key=lambda row: (row[0], row[1])):
        window = [
            revenue
            for other_store, other_when, revenue in rows
            if other_store == store and when - timedelta(days=6) <= other_when <= when
        ]
        result.append((store, when, sum(window) / len(window), len(window)))
    return result


MISTAKE = (
    "trailing averages differ from the calendar-window oracle: the window must be the seven "
    "calendar days ending on the row's date, per store, with closed days neither zero nor counted"
)


def test_matches_calendar_oracle_on_seeded_shops_with_gaps():
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic fixtures, not security.
    for _ in range(25):
        rows = []
        for store in range(1, rng.randrange(2, 5)):
            for offset in range(60):
                if rng.random() < 0.45:  # about half the days are closed
                    continue
                rows.append((store, BASE + timedelta(days=offset), rng.randrange(0, 5000)))
        rng.shuffle(rows)
        _same(_run(rows), _oracle(rows), MISTAKE)


def test_row_exactly_six_days_back_is_in_the_window():
    rows = [(1, BASE, 100), (1, BASE + timedelta(days=6), 300)]
    got = _run(rows)
    _same(got, _oracle(rows), MISTAKE)
    if got[1][3] != 2:
        pytest.fail(
            "a row exactly six days back must be in the window: use six days preceding, "
            "inclusive, as the lower bound"
        )


def test_row_seven_days_back_is_outside_the_window():
    rows = [(1, BASE, 100), (1, BASE + timedelta(days=7), 300)]
    _same(
        _run(rows),
        [(1, BASE, 100.0, 1), (1, BASE + timedelta(days=7), 300.0, 1)],
        "a row seven days back is outside a seven-day window: use six days preceding, not seven",
    )


def test_long_gap_leaves_only_the_current_row_in_the_window():
    rows = [(1, BASE, 100), (1, BASE + timedelta(days=30), 40)]
    _same(
        _run(rows),
        [(1, BASE, 100.0, 1), (1, BASE + timedelta(days=30), 40.0, 1)],
        "a gap longer than the window must leave only the current row: check for zero-filled days",
    )


def test_closed_days_are_not_zero_filled_into_the_average():
    rows = [(1, BASE, 60), (1, BASE + timedelta(days=3), 120)]
    _same(_run(rows), _oracle(rows), MISTAKE)


def test_stores_on_the_same_dates_do_not_share_windows():
    rows = [
        (1, BASE, 10),
        (2, BASE, 1000),
        (1, BASE + timedelta(days=1), 30),
        (2, BASE + timedelta(days=1), 3000),
    ]
    _same(_run(rows), _oracle(rows), "windows mixed across stores: partition by store_id")


def test_single_day_store_has_days_in_window_one():
    rows = [(4, BASE, 777)]
    _same(_run(rows), [(4, BASE, 777.0, 1)], MISTAKE)


def test_window_crosses_month_and_year_boundaries():
    rows = [
        (1, date(2025, 12, 29), 10),
        (1, date(2025, 12, 31), 20),
        (1, date(2026, 1, 2), 30),
        (1, date(2026, 1, 4), 40),
    ]
    _same(_run(rows), _oracle(rows), MISTAKE)


def test_large_revenue_values_keep_the_average_accurate():
    rows = [(1, BASE + timedelta(days=i), 10**12 + i) for i in range(7)]
    _same(_run(rows), _oracle(rows), MISTAKE)


def test_output_order_is_store_then_date_regardless_of_insertion():
    rows = [
        (3, BASE + timedelta(days=2), 5),
        (1, BASE + timedelta(days=5), 6),
        (3, BASE, 7),
        (1, BASE, 8),
    ]
    _same(_run(rows), _oracle(rows), "output must be ordered by store_id, then sales_date")


def test_empty_table_returns_no_rows_with_the_declared_schema():
    _same(_run([]), [], "an empty table must return no rows with the declared columns and types")


def test_fixture_matches_independent_oracle():
    with (
        Path(__file__).with_name("fixtures").joinpath("daily_sales.csv").open(newline="") as handle
    ):
        rows = [
            (
                int(row["store_id"]),
                date.fromisoformat(row["sales_date"]),
                int(row["revenue_cents"]),
            )
            for row in csv.DictReader(handle)
        ]
    _same(_run(rows), _oracle(rows), MISTAKE)


def test_query_text_does_not_hard_code_store_or_date_values():
    code = "\n".join(line.split("--", 1)[0] for line in SQL.splitlines())
    if re.search(r"'2026-|2026-\d\d-\d\d", code):
        pytest.fail("do not hard-code dates in the query: the window must come from each row")
