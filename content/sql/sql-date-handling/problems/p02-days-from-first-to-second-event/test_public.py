"""Public tests: the column names, the users returned, and one check per rule the fixture exercises.

Each test names the mistake it guards against in its failure line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
COLUMNS = ["user_id", "days_to_second"]
EXPECTED = [(201, 3), (202, 1), (204, 0), (205, 1), (206, 59)]


def _load_fixture() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE events AS SELECT * FROM read_csv(?, columns={'user_id': 'BIGINT', "
        "'event_ts': 'TIMESTAMP'}, header=true)",
        [str(FIXTURES / "events.csv")],
    )
    return con


def _run_learner_query(con: duckdb.DuckDBPyConnection) -> tuple[list[str], list[tuple[Any, ...]]]:
    sql = Path(__file__).with_name("submission.sql").read_text()
    try:
        result = con.execute(sql)
    except duckdb.Error as exc:
        pytest.fail(f"your SQL did not run: {exc}")
    if result.description is None:
        pytest.fail("submission.sql returned no result set: finish it with a SELECT")
    columns = [column[0] for column in result.description]
    return columns, result.fetchall()


def _value_of(rows: list[tuple[Any, ...]], user: int) -> Any:
    matches = [row[1] for row in rows if row[0] == user]
    return matches[0] if len(matches) == 1 else None


def test_output_columns_are_user_id_then_days_to_second() -> None:
    columns, _ = _run_learner_query(_load_fixture())
    assert columns == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {columns}. Select user_id, then "
        f"days_to_second."
    )


def test_only_users_with_two_events_are_returned() -> None:
    _, rows = _run_learner_query(_load_fixture())
    users = [row[0] for row in rows]
    assert 203 not in users, (
        "a user with a single event was returned: only users with a second event have a value"
    )
    assert users == [201, 202, 204, 205, 206], (
        "the users returned are wrong: every user with at least two events, and only those, in "
        "user_id order"
    )


def test_tied_first_timestamp_gives_zero_days() -> None:
    _, rows = _run_learner_query(_load_fixture())
    assert _value_of(rows, 204) == 0, (
        "user 204 has two events at the same timestamp, so the second event is on the same day: "
        "two "
        "events with equal timestamps are two events, not one (a DISTINCT on the timestamp is "
        "wrong)"
    )


def test_span_of_two_minutes_across_midnight_is_one_calendar_day() -> None:
    _, rows = _run_learner_query(_load_fixture())
    assert _value_of(rows, 202) == 1, (
        "user 202's events are two minutes apart across midnight and span one calendar day: the "
        "answer "
        "is a DATE difference, not elapsed time floored to whole days"
    )


def test_second_event_not_last_event_sets_the_distance() -> None:
    _, rows = _run_learner_query(_load_fixture())
    assert _value_of(rows, 205) == 1, (
        "user 205 has three events: the distance is to the second event in time order, not to the "
        "last "
        "event (MAX minus MIN is wrong)"
    )


def test_direction_is_second_minus_first_and_spans_the_leap_day() -> None:
    _, rows = _run_learner_query(_load_fixture())
    value = _value_of(rows, 206)
    assert value is not None and value >= 0, (
        "the difference is negative or missing: days_to_second is the second event's date minus "
        "the first "
        "event's date"
    )
    assert value == 59, (
        "the day count across the leap day is wrong: from 2024-01-01 to 2024-02-29 is 59 calendar "
        "days"
    )


def test_fixture_matches_the_hand_checked_answer() -> None:
    _, rows = _run_learner_query(_load_fixture())
    assert rows == EXPECTED, (
        "the day counts differ from the hand-checked answer: one row per user with two events, "
        "with the "
        "calendar days from the first event to the second"
    )
