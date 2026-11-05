"""Hidden tests: your query runs against tables that each test builds itself.

The oracle is an independent pandas computation over the same rows. Each family test names the
mistake it guards against in its first failure line. No message carries a hidden id or value.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

COLUMNS = ["user_id", "days_to_second"]
EVENTS_DDL = "CREATE TABLE events (user_id BIGINT, event_ts TIMESTAMP)"
# Large, non-sequential user ids: without an explicit ORDER BY the engine may return rows in any
# order.
USER_IDS = [3, 17, 42, 1_204, 905_113_208, 6_000_000_001]


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
    if [str(column[1]) for column in result.description] != ["BIGINT", "BIGINT"]:
        pytest.fail("wrong SQL output types: user_id and days_to_second must both be BIGINT")
    return columns, result.fetchall()


def _oracle(events: Sequence[tuple[int, datetime]]) -> list[tuple[int, int]]:
    """Independent answer in pandas: sort each user's timestamps, take the first two, count days."""
    frame = pd.DataFrame(list(events), columns=["user_id", "event_ts"])
    frame["event_ts"] = pd.to_datetime(frame["event_ts"])
    out: list[tuple[int, int]] = []
    for user, group in frame.groupby("user_id"):
        stamps = group["event_ts"].sort_values().tolist()
        if len(stamps) >= 2:
            days = (stamps[1].normalize() - stamps[0].normalize()).days
            out.append((int(user), int(days)))
    return sorted(out)


def _must_equal(got: list[tuple[Any, ...]], want: list[tuple[int, int]], mistake: str) -> None:
    """One assertion per test: the first failing line names the mistake the test guards against."""
    if got != want:
        pytest.fail(mistake)


def _columns_or_fail(columns: list[str]) -> None:
    if columns != COLUMNS:
        pytest.fail(
            f"output columns differ: expected {COLUMNS}, got {columns}. Select user_id, then "
            f"days_to_second."
        )


def _random_events(rng: np.random.Generator) -> list[tuple[int, datetime]]:
    events: list[tuple[int, datetime]] = []
    for _ in range(int(rng.integers(0, 31))):
        user = USER_IDS[int(rng.integers(0, len(USER_IDS)))]
        stamp = datetime(2024, 1, 1) + timedelta(seconds=int(rng.integers(0, 200 * 86_400)))
        if rng.random() < 0.2:
            stamp = stamp.replace(
                hour=0, minute=0, second=0
            )  # midnight: a shared or a day-boundary stamp
        events.append((user, stamp))
    return events


def test_random_tables_match_the_pandas_oracle() -> None:
    # Fixed seeds. The first mismatch fails the test, with a message that names the kind of mistake.
    for seed in range(60):
        events = _random_events(np.random.default_rng(seed))
        want = _oracle(events)
        columns, rows = _run_learner_query(_connect(events))
        _columns_or_fail(columns)
        if [row[0] for row in rows] != [user for user, _ in want]:
            pytest.fail(
                "the set of users or their order differs: return users with at least two events, "
                "ordered by user_id ascending"
            )
        for row, (_, days) in zip(rows, want, strict=True):
            if row[1] != days:
                pytest.fail(
                    "a day count differs: take the first and second events in time order, and "
                    "count "
                    "calendar days from their dates (or subtract CAST(... AS DATE) values)"
                )


def test_single_event_users_are_excluded() -> None:
    events = [
        (17, datetime(2024, 3, 1, 9, 0)),
        (17, datetime(2024, 3, 4, 8, 0)),
        (3, datetime(2024, 5, 5, 5, 5)),
    ]
    columns, rows = _run_learner_query(_connect(events))
    _columns_or_fail(columns)
    _must_equal(
        rows,
        [(17, 3)],
        "a user with a single event is returned: an outer or left join keeps users without a "
        "second "
        "event; use an inner join on the second event",
    )


def test_tied_first_timestamp_counts_the_second_event_as_the_same_time() -> None:
    events = [
        (42, datetime(2024, 6, 1, 8, 0)),
        (42, datetime(2024, 6, 1, 8, 0)),
        (42, datetime(2024, 6, 4, 8, 0)),
    ]
    columns, rows = _run_learner_query(_connect(events))
    _columns_or_fail(columns)
    _must_equal(
        rows,
        [(42, 0)],
        "two events with the same timestamp were treated as one event (DISTINCT on the "
        "timestamp): the "
        "second event is the duplicate, so the distance is 0 days",
    )


def test_three_or_more_events_use_the_second_event_not_the_last() -> None:
    events = [
        (1_204, datetime(2024, 2, 10, 12, 0)),
        (1_204, datetime(2024, 2, 11, 12, 0)),
        (1_204, datetime(2024, 2, 20, 12, 0)),
        (1_204, datetime(2024, 3, 2, 12, 0)),
    ]
    columns, rows = _run_learner_query(_connect(events))
    _columns_or_fail(columns)
    _must_equal(
        rows,
        [(1_204, 1)],
        "the distance is measured to the last event rather than to the second event: the first "
        "two events "
        "in time order define the answer",
    )


def test_two_minutes_across_midnight_is_one_calendar_day() -> None:
    events = [
        (6_000_000_001, datetime(2024, 9, 30, 23, 59, 0)),
        (6_000_000_001, datetime(2024, 10, 1, 0, 1, 0)),
    ]
    columns, rows = _run_learner_query(_connect(events))
    _columns_or_fail(columns)
    _must_equal(
        rows,
        [(6_000_000_001, 1)],
        "the distance counts elapsed time, not calendar days: subtract the dates (CAST to DATE), "
        "not the "
        "timestamps, so a span across midnight is one day",
    )


def test_distance_across_the_leap_day_is_counted_in_days() -> None:
    events = [(3, datetime(2023, 12, 31, 18, 0)), (3, datetime(2024, 3, 1, 6, 0))]
    columns, rows = _run_learner_query(_connect(events))
    _columns_or_fail(columns)
    _must_equal(
        rows,
        [(3, 61)],
        "calendar distance across a leap day is wrong: include every crossed calendar date",
    )


def test_direction_is_second_minus_first() -> None:
    events = [(905_113_208, datetime(2024, 4, 9, 9, 0)), (905_113_208, datetime(2024, 4, 13, 9, 0))]
    columns, rows = _run_learner_query(_connect(events))
    _columns_or_fail(columns)
    _must_equal(
        rows,
        [(905_113_208, 4)],
        "the difference has the wrong sign or the wrong operands: days_to_second is the second "
        "event's date "
        "minus the first event's date, so it is never negative",
    )


def test_events_given_out_of_time_order_are_sorted_by_time() -> None:
    events = [
        (17, datetime(2024, 7, 30, 10, 0)),
        (17, datetime(2024, 7, 1, 10, 0)),
        (17, datetime(2024, 7, 2, 10, 0)),
        (17, datetime(2024, 7, 9, 10, 0)),
    ]
    columns, rows = _run_learner_query(_connect(events))
    _columns_or_fail(columns)
    _must_equal(
        rows,
        [(17, 1)],
        "the first and second events were taken in insertion order: order the user's events by "
        "event_ts "
        "before choosing the first and the second",
    )


def test_no_user_with_two_events_returns_zero_rows_with_two_columns() -> None:
    events = [(3, datetime(2024, 1, 5, 5, 5)), (17, datetime(2024, 1, 6, 6, 6))]
    columns, rows = _run_learner_query(_connect(events))
    _columns_or_fail(columns)
    _must_equal(
        rows,
        [],
        "rows were returned for users who each have one event: users with fewer than two events "
        "are not "
        "returned",
    )


def test_empty_events_table_returns_zero_rows_with_two_columns() -> None:
    columns, rows = _run_learner_query(_connect([]))
    _columns_or_fail(columns)
    _must_equal(
        rows,
        [],
        "rows were returned for an empty events table: with no events there is no user with two "
        "events",
    )


def test_large_user_ids_come_back_in_user_id_order() -> None:
    events: list[tuple[int, datetime]] = []
    for user in sorted(USER_IDS, reverse=True):
        events.append((user, datetime(2024, 5, 1, 9, 0)))
        events.append((user, datetime(2024, 5, 2, 9, 0)))
    columns, rows = _run_learner_query(_connect(events))
    _columns_or_fail(columns)
    _must_equal(
        rows,
        [(user, 1) for user in sorted(USER_IDS)],
        "the result is not ordered by user_id ascending: add ORDER BY user_id",
    )
