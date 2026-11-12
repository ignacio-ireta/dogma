"""Hidden tests: your query runs against tables that each test builds itself.

The oracle is an independent pandas computation over the same rows. Each family test states the
mistake it guards against in its first failure line. No message carries a hidden id, timestamp or
expected count.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from dogma_data import load_frame, rng

COLUMNS = ["month", "active_users"]
EVENTS_DDL = "CREATE TABLE events (user_id BIGINT, event_ts TIMESTAMP)"
MONTH_STARTS = [
    datetime(2023, 10, 1),
    datetime(2023, 11, 1),
    datetime(2023, 12, 1),
    datetime(2024, 1, 1),
    datetime(2024, 2, 1),
    datetime(2024, 3, 1),
    datetime(2024, 4, 1),
]
RANGE_MONTHS = [date(2023, 11, 1), date(2023, 12, 1), date(2024, 1, 1), date(2024, 2, 1)]


def _connect(events: Sequence[tuple[int, datetime]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(EVENTS_DDL)
    if events:
        con.executemany("INSERT INTO events VALUES (?, ?)", list(events))
    return con


def _run_learner_query(con: duckdb.DuckDBPyConnection) -> tuple[list[str], list[tuple[Any, ...]]]:
    sql = Path(__file__).with_name("submission.sql").read_text()
    try:
        result = con.execute(sql)
    except duckdb.Error as exc:
        pytest.fail(f"your SQL did not run: check expressions and casts ({type(exc).__name__})")
    if result.description is None:
        pytest.fail("submission.sql returned no result set: finish it with a SELECT")
    columns = [column[0] for column in result.description]
    if [str(column[1]) for column in result.description] != ["DATE", "BIGINT"]:
        pytest.fail("wrong SQL output types: keep dates as DATE and counts as BIGINT")
    return columns, result.fetchall()


def _oracle(events: Sequence[tuple[int, datetime]]) -> list[tuple[date, int]]:
    """Independent answer in pandas: a half-open month window per month, distinct users counted."""
    frame = pd.DataFrame(list(events), columns=["user_id", "event_ts"])
    frame["event_ts"] = pd.to_datetime(frame["event_ts"])
    out: list[tuple[date, int]] = []
    for month_start in pd.date_range("2023-11-01", periods=4, freq="MS"):
        month_end = month_start + pd.DateOffset(months=1)
        in_month = frame[(frame["event_ts"] >= month_start) & (frame["event_ts"] < month_end)]
        out.append((month_start.date(), int(in_month["user_id"].nunique())))
    return out


def _rows_of(columns: list[str], rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    if columns != COLUMNS:
        pytest.fail(
            f"output columns differ: expected {COLUMNS}, got {columns}. "
            "Select month, then active_users, in that order."
        )
    return rows


def _must_equal(got: list[tuple[Any, ...]], want: list[tuple[date, int]], mistake: str) -> None:
    """One assertion per test: the first failing line names the mistake the test guards against."""
    if got != want:
        pytest.fail(mistake)


@pytest.fixture(scope="module")
def event_log_events() -> list[tuple[int, datetime]]:
    """A real activity log, moved forward in time so the range of the statement falls inside it.

    Each event is one invoice line of an online retailer's customer (online_retail): two years
    of real activity, December 2009 to December 2011, with busy and quiet months and many lines
    per customer and minute. Every timestamp is moved forward by exactly 13 years, to December
    2022 through December 2024, so the log holds the four months asked for and also the same
    month numbers of other years. A random nine tenths of the lines is used.
    """
    gen = rng("sql/sql-date-handling/p01-monthly-active-users:event-log:0")
    lines = load_frame("online_retail")
    keep = gen.permutation(len(lines))[: len(lines) * 9 // 10]
    users = lines["Customer ID"].to_numpy()[keep]
    stamps = (pd.to_datetime(lines["InvoiceDate"]) + pd.DateOffset(years=13)).to_numpy()[keep]
    return [
        (int(user_id), pd.Timestamp(stamp).to_pydatetime())
        for user_id, stamp in zip(users, stamps, strict=True)
    ]


def _random_events(rng: np.random.Generator) -> list[tuple[int, datetime]]:
    events: list[tuple[int, datetime]] = []
    for _ in range(int(rng.integers(0, 41))):
        user = int(rng.integers(1, 9))
        if rng.random() < 0.25:
            # A month boundary: the first instant of a month, or the last second before it.
            anchor = MONTH_STARTS[int(rng.integers(0, len(MONTH_STARTS)))]
            ts = anchor if rng.random() < 0.5 else anchor - timedelta(seconds=1)
        else:
            ts = datetime(2023, 10, 1) + timedelta(seconds=int(rng.integers(0, 15_000_000)))
        events.append((user, ts))
    return events


def test_random_tables_match_the_pandas_oracle() -> None:
    # Fixed seeds. The first mismatch fails the test, with a message that names the kind of mistake.
    for seed in range(60):
        events = _random_events(np.random.default_rng(seed))
        want = _oracle(events)
        columns, rows = _run_learner_query(_connect(events))
        rows = _rows_of(columns, rows)
        if len(rows) != len(want):
            if not rows or rows[-1][0] != RANGE_MONTHS[-1]:
                pytest.fail(
                    "the last month of the range is missing: generate_series includes its stop "
                    "value, "
                    "while range() does not"
                )
            pytest.fail(
                "a month with no events is missing from the result: start from a calendar of "
                "months "
                "and LEFT JOIN the events onto it, so an empty month keeps its row"
            )
        if [row[0] for row in rows] != [month for month, _ in want]:
            pytest.fail(
                "the month values or their order differ: months are the first day of each month of "
                "the range, in ascending order"
            )
        for row, (_month, count) in zip(rows, want, strict=True):
            if row[1] != count:
                if row[1] > count:
                    pytest.fail(
                        "a month counts more users than were active in it: count DISTINCT user_id, "
                        "and do not let an event match two months (use < the next month's first "
                        "day)"
                    )
                pytest.fail(
                    "a month counts fewer users than were active in it: an event on the last day "
                    "of "
                    "a month can be lost; compare with the first day of the next month, not the "
                    "last day"
                )


def test_event_log_matches_the_pandas_month_oracle(
    event_log_events: list[tuple[int, datetime]],
) -> None:
    want = _oracle(event_log_events)
    columns, rows = _run_learner_query(_connect(event_log_events))
    rows = _rows_of(columns, rows)
    assert rows == want, (
        "event_log months are wrong: the log spans several years, so match each event on its full "
        "date (month and year) inside the four months only, counting distinct users over the "
        "half-open calendar range"
    )


def test_empty_middle_month_keeps_its_row() -> None:
    events = [(1, datetime(2023, 11, 9, 12, 0)), (2, datetime(2024, 2, 9, 12, 0))]
    columns, rows = _run_learner_query(_connect(events))
    _must_equal(
        _rows_of(columns, rows),
        [(RANGE_MONTHS[0], 1), (RANGE_MONTHS[1], 0), (RANGE_MONTHS[2], 0), (RANGE_MONTHS[3], 1)],
        "a month with no events is missing or wrong: start from the calendar of months and LEFT "
        "JOIN "
        "the events onto it, so an empty month keeps its row with 0",
    )


def test_empty_last_month_is_still_returned() -> None:
    events = [(1, datetime(2023, 11, 20, 8, 0)), (2, datetime(2024, 1, 5, 8, 0))]
    columns, rows = _run_learner_query(_connect(events))
    _must_equal(
        _rows_of(columns, rows),
        [(RANGE_MONTHS[0], 1), (RANGE_MONTHS[1], 0), (RANGE_MONTHS[2], 1), (RANGE_MONTHS[3], 0)],
        "the last month of the range is missing or its count is wrong: the range ends at February "
        "2024 inclusive, and the calendar must include that month even when it has no events",
    )


def test_empty_events_table_returns_four_zero_rows() -> None:
    columns, rows = _run_learner_query(_connect([]))
    _must_equal(
        _rows_of(columns, rows),
        [(month, 0) for month in RANGE_MONTHS],
        "an empty events table must still return one 0 row for each month of the range, not no "
        "rows",
    )


def test_last_second_of_a_month_counts_in_that_month() -> None:
    events = [
        (1, datetime(2023, 11, 30, 23, 59, 59)),
        (2, datetime(2024, 1, 31, 23, 59, 59)),
        (3, datetime(2024, 2, 29, 23, 59, 59)),
    ]
    columns, rows = _run_learner_query(_connect(events))
    _must_equal(
        _rows_of(columns, rows),
        [(RANGE_MONTHS[0], 1), (RANGE_MONTHS[1], 0), (RANGE_MONTHS[2], 1), (RANGE_MONTHS[3], 1)],
        "an event at the last second of a month is not counted in that month: compare with the "
        "first "
        "day of the next month (<), not with the last day of the month",
    )


def test_first_instant_of_next_month_is_not_counted_in_the_previous_month() -> None:
    events = [(8, datetime(2023, 12, 1, 0, 0, 0)), (9, datetime(2024, 2, 1, 0, 0, 0))]
    columns, rows = _run_learner_query(_connect(events))
    _must_equal(
        _rows_of(columns, rows),
        [(RANGE_MONTHS[0], 0), (RANGE_MONTHS[1], 1), (RANGE_MONTHS[2], 0), (RANGE_MONTHS[3], 1)],
        "an event at midnight on the first day of a month is counted in the previous month: "
        "BETWEEN "
        "includes its upper bound; use >= the first day and < the next month's first day",
    )


def test_several_events_by_one_user_in_a_month_count_once() -> None:
    events = [(4, datetime(2024, 1, day, 9, 0)) for day in (2, 9, 16, 23, 30)]
    columns, rows = _run_learner_query(_connect(events))
    _must_equal(
        _rows_of(columns, rows),
        [(RANGE_MONTHS[0], 0), (RANGE_MONTHS[1], 0), (RANGE_MONTHS[2], 1), (RANGE_MONTHS[3], 0)],
        "one user with several events in a month is counted once per event: count DISTINCT "
        "user_id, "
        "not the rows",
    )


def test_a_user_active_in_every_month_counts_once_in_each() -> None:
    events = [
        (5, datetime(2023, 11, 3, 10, 0)),
        (5, datetime(2023, 12, 3, 10, 0)),
        (5, datetime(2024, 1, 3, 10, 0)),
        (5, datetime(2024, 2, 3, 10, 0)),
        (5, datetime(2024, 2, 20, 10, 0)),
    ]
    columns, rows = _run_learner_query(_connect(events))
    _must_equal(
        _rows_of(columns, rows),
        [(month, 1) for month in RANGE_MONTHS],
        "a user active in several months is missing from a month or counted more than once in one",
    )


def test_events_outside_the_range_are_ignored() -> None:
    events = [
        (6, datetime(2023, 10, 31, 23, 59, 59)),
        (6, datetime(2023, 6, 15, 12, 0)),
        (7, datetime(2024, 3, 1, 0, 0, 0)),
        (7, datetime(2024, 9, 9, 9, 9)),
    ]
    columns, rows = _run_learner_query(_connect(events))
    _must_equal(
        _rows_of(columns, rows),
        [(month, 0) for month in RANGE_MONTHS],
        "an event outside the four months was counted: keep only months from November 2023 to "
        "February 2024",
    )


def test_month_column_is_a_date_and_rows_are_in_month_order() -> None:
    events = [(1, datetime(2024, 2, 2, 2, 2)), (2, datetime(2023, 11, 2, 2, 2))]
    columns, rows = _run_learner_query(_connect(events))
    rows = _rows_of(columns, rows)
    if not all(type(row[0]) is date for row in rows):
        pytest.fail("month must be a DATE, not a TIMESTAMP: cast the month with CAST(... AS DATE)")
    if [row[0] for row in rows] != RANGE_MONTHS:
        pytest.fail(
            "the rows are not in month order: ORDER BY month ascending, the answer depends on it"
        )
    _must_equal(
        rows,
        [(RANGE_MONTHS[0], 1), (RANGE_MONTHS[1], 0), (RANGE_MONTHS[2], 0), (RANGE_MONTHS[3], 1)],
        "the counts are wrong for a small table of two users",
    )


def test_user_ids_above_the_32_bit_range_stay_distinct() -> None:
    big = 3_000_000_000  # a valid BIGINT user id, larger than a 32-bit integer can hold
    events = [
        (big, datetime(2023, 12, 5, 9, 0)),
        (big + 1, datetime(2023, 12, 6, 9, 0)),
        (big, datetime(2023, 12, 7, 9, 0)),
    ]
    columns, rows = _run_learner_query(_connect(events))
    # Two different users are active in December 2023; the same user twice counts once.
    _must_equal(
        _rows_of(columns, rows),
        [(RANGE_MONTHS[0], 0), (RANGE_MONTHS[1], 2), (RANGE_MONTHS[2], 0), (RANGE_MONTHS[3], 0)],
        "two different users with large user ids are counted as one, or the query fails on them: "
        "keep user_id as BIGINT and count its DISTINCT values",
    )
