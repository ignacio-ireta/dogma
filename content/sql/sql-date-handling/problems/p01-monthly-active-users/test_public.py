"""Public tests: the fixture's four months, the column names and types, the month list, the counts.

Each check names the kind of mistake in its first line. Structural checks run before the counts,
so a missing month is reported as such and not as a count that follows from it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
COLUMNS = ["month", "active_users"]
RANGE_MONTHS = [date(2023, 11, 1), date(2023, 12, 1), date(2024, 1, 1), date(2024, 2, 1)]
EXPECTED = [
    (date(2023, 11, 1), 3),
    (date(2023, 12, 1), 0),
    (date(2024, 1, 1), 3),
    (date(2024, 2, 1), 4),
]


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


def _check_columns(columns: list[str]) -> None:
    assert columns == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {columns}. Select month, then "
        f"active_users."
    )


def _check_month_type(rows: list[tuple[Any, ...]]) -> None:
    for row in rows:
        # datetime is a subclass of date, so an exact type check rejects TIMESTAMP values.
        assert type(row[0]) is date, (
            "month must be a DATE (the first day of the month), not a TIMESTAMP: cast the month "
            "with "
            "CAST(... AS DATE), because date_trunc on a DATE returns a TIMESTAMP in DuckDB"
        )


def _check_month_list(rows: list[tuple[Any, ...]]) -> None:
    months = [row[0] for row in rows]
    if RANGE_MONTHS[-1] not in months:
        raise AssertionError(
            "the last month of the range is missing: generate_series includes its stop value, "
            "while "
            "range() does not; the calendar must end at February 2024 inclusive"
        )
    missing = [month for month in RANGE_MONTHS if month not in months]
    if missing:
        raise AssertionError(
            "a month with no events is missing from the result: start from a calendar of months "
            "and "
            "LEFT JOIN the events onto it, so an empty month keeps its row"
        )
    if months != sorted(months):
        raise AssertionError("the rows are not in month order: ORDER BY month ascending")


def _check_counts(rows: list[tuple[Any, ...]]) -> None:
    for got, want in zip(rows, EXPECTED, strict=True):
        if got[1] > want[1]:
            raise AssertionError(
                "a month counts more users than were active in it: count DISTINCT user_id, and let "
                "each event match one month only (>= the first day, < the next month's first day, "
                "not BETWEEN)"
            )
        if got[1] < want[1]:
            raise AssertionError(
                "a month counts fewer users than were active in it: an event on the last day of a "
                "month can be lost; compare with the first day of the next month, not the last day"
            )


def test_output_has_month_then_active_users_columns() -> None:
    columns, _ = _run_learner_query(_load_fixture())
    _check_columns(columns)


def test_month_column_is_a_date_not_a_timestamp() -> None:
    _, rows = _run_learner_query(_load_fixture())
    _check_month_type(rows)


def test_every_month_of_the_range_has_a_row_in_month_order() -> None:
    _, rows = _run_learner_query(_load_fixture())
    _check_month_list(rows)


def test_fixture_matches_the_hand_checked_answer() -> None:
    columns, rows = _run_learner_query(_load_fixture())
    _check_columns(columns)
    _check_month_type(rows)
    _check_month_list(rows)
    _check_counts(rows)


def test_empty_december_keeps_its_row_with_zero() -> None:
    _, rows = _run_learner_query(_load_fixture())
    december = [row for row in rows if row[0] == date(2023, 12, 1)]
    assert december == [(date(2023, 12, 1), 0)], (
        "December 2023 has no events and must still appear once with 0: start from a calendar of "
        "months and LEFT JOIN the events onto it"
    )
