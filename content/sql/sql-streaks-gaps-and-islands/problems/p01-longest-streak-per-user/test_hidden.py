# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from dogma_data import retail_events, rng

COLUMNS = ["user_id", "streak_days", "streak_start", "streak_end"]
TYPES = ["BIGINT", "BIGINT", "DATE", "DATE"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")
BASE = date(2026, 1, 1)
MISTAKE = "longest streak differs from the oracle: check distinct days, run ends and the tie rule"


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE visits(user_id BIGINT, visit_date DATE)")
        if rows:
            connection.executemany("INSERT INTO visits VALUES (?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared visits schema", pytrace=False)
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the four named columns in the stated order", pytrace=False)
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("cast streak_days to BIGINT; dates must stay DATE", pytrace=False)
        result = cursor.fetchall()
        after = connection.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        if after != len(rows):
            pytest.fail("the query must not change the visits table", pytrace=False)
        return result


def _oracle(rows):
    """Independent answer: distinct dates per user, sorted, walked into runs in Python."""
    days: dict[int, set[date]] = {}
    for user, when in rows:
        days.setdefault(user, set()).add(when)
    out = []
    for user in sorted(days):
        runs = []
        for when in sorted(days[user]):
            if runs and when - runs[-1][1] == timedelta(days=1):
                runs[-1][1] = when
                runs[-1][2] += 1
            else:
                runs.append([when, when, 1])
        start, end, length = max(runs, key=lambda run: (run[2], -run[0].toordinal()))
        out.append((user, length, start, end))
    return out


def _same(got, expected):
    if got != expected:
        pytest.fail(MISTAKE, pytrace=False)


def test_matches_the_oracle_on_the_fixture():
    with Path(__file__).with_name("fixtures").joinpath("visits.csv").open(newline="") as handle:
        rows = [
            (int(row["user_id"]), date.fromisoformat(row["visit_date"]))
            for row in csv.DictReader(handle)
        ]
    _same(_run(rows), _oracle(rows))


def test_matches_the_oracle_on_seeded_random_visits_with_duplicates():
    rng = random.Random(4242)  # noqa: S311 -- seeded synthetic data, not security
    for _ in range(30):
        rows = []
        for user in range(1, rng.randrange(2, 7)):
            for _ in range(rng.randrange(1, 25)):
                day = BASE + timedelta(days=rng.randrange(0, 40))
                rows.append((user, day))
                if rng.random() < 0.3:
                    rows.append((user, day))  # a duplicate visit on the same day
        rng.shuffle(rows)
        _same(_run(rows), _oracle(rows))


def test_gap_of_exactly_one_day_breaks_the_streak():
    rows = [
        (1, date(2026, 4, 1)),
        (1, date(2026, 4, 2)),
        (1, date(2026, 4, 4)),
        (1, date(2026, 4, 5)),
    ]
    _same(_run(rows), [(1, 2, date(2026, 4, 1), date(2026, 4, 2))])


def test_duplicate_visits_on_one_day_do_not_make_a_longer_streak():
    rows = [(1, date(2026, 4, 1))] * 5 + [(1, date(2026, 4, 2))]
    _same(_run(rows), [(1, 2, date(2026, 4, 1), date(2026, 4, 2))])


def test_ties_in_length_return_the_earliest_start():
    rows = [
        (1, date(2026, 5, 1)),
        (1, date(2026, 5, 2)),
        (1, date(2026, 5, 10)),
        (1, date(2026, 5, 11)),
    ]
    _same(_run(rows), [(1, 2, date(2026, 5, 1), date(2026, 5, 2))])


def test_longest_run_later_in_the_year_beats_earlier_shorter_runs():
    rows = [(2, date(2026, 1, 1)), (2, date(2026, 1, 9)), (2, date(2026, 6, 1)),
            (2, date(2026, 6, 2)), (2, date(2026, 6, 3)), (2, date(2026, 6, 4))]  # fmt: skip
    _same(_run(rows), [(2, 4, date(2026, 6, 1), date(2026, 6, 4))])


def test_run_across_a_leap_day_and_a_year_end_is_one_streak():
    first, last = date(2027, 12, 31), date(2028, 3, 1)
    rows = [(3, first + timedelta(days=d)) for d in range((last - first).days + 1)]
    length = (last - first).days + 1  # 2028 is a leap year, so February has 29 days
    _same(_run(rows), [(3, length, first, last)])


def test_insertion_order_does_not_change_the_result():
    rows = [(4, BASE + timedelta(days=d)) for d in range(10)] + [(5, BASE)]
    forward = _run(rows)
    backward = _run(list(reversed(rows)))
    _same(backward, forward)


def test_one_row_per_user_even_with_many_equal_length_runs():
    rows = []
    for user in (7, 8):
        for block in range(4):
            start = BASE + timedelta(days=10 * block)
            rows += [(user, start), (user, start + timedelta(days=1))]
    got = _run(rows)
    assert len(got) == 2, "return exactly one row per user, not one per run"
    _same(got, _oracle(rows))


def test_empty_table_returns_no_rows_with_the_declared_schema():
    _same(_run([]), [])


@pytest.fixture(scope="module")
def holiday_burst() -> list[tuple[int, date]]:
    """Real activity across 31 Dec 2028, a leap-year end: skewed users and duplicate rows.

    The rows are invoice lines of an online retailer (online_retail). A user is one product
    family (the first two characters of the stock code) and a visit is a line of that family
    on its invoice date, so a busy user appears many times on one day, is active on runs of up
    to six days (the retailer is closed on Saturdays) and has gaps. The lines are those of
    twenty days of its busy season, 13 November to 2 December 2010, and every date is moved
    forward by the same number of days, so that they run from 20 December 2028 to 8 January
    2029. A random four fifths of the lines is used.
    """
    gen = rng("sql/sql-streaks-gaps-and-islands/p01-longest-streak-per-user:holiday_burst:0")
    events = retail_events()
    busy = (events["ts"] >= "2010-11-13") & (events["ts"] < "2010-12-03")
    events = events[busy]
    events = events.iloc[gen.permutation(len(events))[: len(events) * 4 // 5]]
    families = events["code"].str[:2]
    user_of = {family: number for number, family in enumerate(sorted(families.unique()), 1)}
    shift = date(2028, 12, 31) - date(2010, 11, 24)
    return [
        (user_of[family], moment.date() + shift)
        for family, moment in zip(families, events["ts"], strict=True)
    ]


def test_matches_the_oracle_on_a_holiday_burst_across_a_leap_year_end(holiday_burst):
    got = _run(holiday_burst)
    if got != _oracle(holiday_burst):
        pytest.fail(
            "longest streak differs on dense activity across a leap-year end: check how a run "
            "continues across a year end, where the day count restarts"
        )


def test_run_across_a_leap_year_end_is_one_streak():
    first = date(2028, 12, 29)
    rows = [(5, first + timedelta(days=offset)) for offset in range(6)]  # to 3 January 2029
    _same(_run(rows), [(5, 6, first, date(2029, 1, 3))])
