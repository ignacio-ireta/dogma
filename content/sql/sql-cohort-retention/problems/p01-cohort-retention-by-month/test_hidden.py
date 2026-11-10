"""Hidden tests: an independent calendar-month oracle, edge cases and the fixture.

The oracle works on calendar-month indices (year * 12 + month), not on SQL, and it is the only
reference used. Learner code is read from submission.sql; nothing here imports solution.sql.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import duckdb
import pytest

from dogma_data import rng, synth

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
        result = cursor.fetchall()
        after = connection.execute("SELECT COUNT(*) FROM activity").fetchone()[0]
        if after != len(rows):
            pytest.fail("query must not change the activity table")
        return result


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
        if abs(given[4] * 10000 - round(given[4] * 10000)) > 1e-6:
            pytest.fail("retention_rate is not rounded to four decimal places")


def _index(moment: datetime) -> int:
    """Calendar month as an integer: January of year Y is Y * 12 + 0."""
    return moment.year * 12 + moment.month - 1


def _oracle(rows):
    first: dict[int, int] = {}
    last = None
    for user, moment in rows:
        month = _index(moment)
        first[user] = min(first.get(user, month), month)
        last = month if last is None else max(last, month)
    sizes = Counter(first.values())
    active: dict[tuple[int, int], set[int]] = defaultdict(set)
    for user, moment in rows:
        cohort = first[user]
        active[(cohort, _index(moment) - cohort)].add(user)
    out = []
    for cohort in sorted(sizes):
        for period in range(0, last - cohort + 1):
            count = len(active.get((cohort, period), set()))
            size = sizes[cohort]
            out.append(
                (
                    date(cohort // 12, cohort % 12 + 1, 1),
                    period,
                    count,
                    size,
                    count / size,
                )
            )
    return out


def _at(month_index: int, day: int, hour: int = 12) -> datetime:
    """Helper: a moment in the calendar month with index month_index (year * 12 + month - 1)."""
    return datetime(month_index // 12, month_index % 12 + 1, day, hour)


JAN_2025 = 2025 * 12


def test_matches_calendar_month_oracle_on_seeded_random_activity():
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic data, not security.
    for _ in range(25):
        rows = []
        for user in range(1, rng.randrange(2, 30)):
            start = JAN_2025 + rng.randrange(0, 20)
            rows.append((user, _at(start, rng.randrange(1, 29), rng.randrange(24))))
            for _ in range(rng.randrange(0, 6)):
                month = start + rng.randrange(0, 12)
                rows.append((user, _at(month, rng.randrange(1, 29), rng.randrange(24))))
        rng.shuffle(rows)
        _check(
            _run(rows),
            _oracle(rows),
        )


def test_repeated_events_in_one_month_count_each_user_once():
    rows = [
        (1, _at(JAN_2025, 1)),
        (1, _at(JAN_2025, 2)),
        (1, _at(JAN_2025, 3)),
        (1, _at(JAN_2025 + 1, 3)),
        (1, _at(JAN_2025 + 1, 4)),
        (2, _at(JAN_2025, 9)),
    ]
    got = _run(rows)
    assert got[1][2] == 1, (
        "active_users counts events, not users: a user with several events in one month must "
        "count once in that cell"
    )


def test_month_boundary_splits_cohorts_at_midnight_of_the_first():
    rows = [
        (1, datetime(2025, 1, 31, 23, 59, 59)),
        (2, datetime(2025, 2, 1, 0, 0, 0)),
    ]
    got = _run(rows)
    cohorts = {row[0]: row[3] for row in got if row[1] == 0}
    assert cohorts == {date(2025, 1, 1): 1, date(2025, 2, 1): 1}, (
        "the cohort month is cut at the calendar month: 31 January 23:59 and 1 February "
        "00:00 must be in different cohorts"
    )


def test_returning_user_is_counted_only_in_the_periods_they_are_active():
    rows = [
        (1, _at(JAN_2025, 5)),
        (1, _at(JAN_2025 + 2, 5)),
        (2, _at(JAN_2025, 6)),
    ]
    _check(_run(rows), _oracle(rows))


def test_year_boundary_gives_correct_period_offsets():
    rows = [
        (1, _at(2025 * 12 + 10, 4)),  # November 2025
        (2, _at(2025 * 12 + 10, 9)),
        (1, _at(2026 * 12 + 1, 2)),  # February 2026, three months later
    ]
    _check(_run(rows), _oracle(rows))


def test_cohort_without_later_activity_still_reaches_the_last_month():
    rows = [
        (1, _at(JAN_2025, 1)),
        (2, _at(JAN_2025 + 4, 1)),
        (2, _at(JAN_2025 + 6, 1)),
    ]
    got = _run(rows)
    jan_periods = [row[1] for row in got if row[0] == date(2025, 1, 1)]
    assert jan_periods == list(range(0, 7)), (
        "the cohort grid must run to the latest activity month of the whole table, "
        "with a row for every empty period"
    )


def test_long_span_periods_count_calendar_months_not_days():
    # 1461 days from January 2020 to January 2024; days divided by 30 would say 49 months.
    rows = [(1, datetime(2020, 1, 15)), (1, datetime(2024, 1, 10)), (2, datetime(2020, 1, 20))]
    _check(_run(rows), _oracle(rows))


def test_fixture_matches_independent_oracle():
    path = Path(__file__).with_name("fixtures").joinpath("activity.csv")
    with path.open(newline="") as handle:
        rows = [
            (int(row["user_id"]), datetime.fromisoformat(row["event_at"]))
            for row in csv.DictReader(handle)
        ]
    _check(_run(rows), _oracle(rows))


def test_empty_table_returns_no_rows_with_the_declared_schema():
    assert _run([]) == [], "an empty table must return no rows, with the declared columns and types"


def test_duplicated_log_rows_count_each_user_once_in_the_cohort_size():
    gen = rng("sql/sql-cohort-retention/p01-cohort-retention-by-month:dup-log:0")
    log = synth.event_log(
        gen,
        3000,
        n_users=400,
        zipf_a=1.1,
        start="2023-01-01",
        days=1000,
        duplicate_rate=0.1,
    )
    rows = [
        (int(user), moment.item()) for user, moment in zip(log["user_id"], log["ts"], strict=True)
    ]
    _check(_run(rows), _oracle(rows))
