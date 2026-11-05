# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

COLUMNS = ["user_id", "island_start", "island_end", "island_days"]
TYPES = ["BIGINT", "DATE", "DATE", "BIGINT"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")
BASE = date(2026, 1, 1)
MISTAKE = "streaks differ from the oracle: check gap flags, the running total and the grouping"


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
            pytest.fail("cast island_days to BIGINT; dates must stay DATE", pytrace=False)
        result = cursor.fetchall()
        after = connection.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        if after != len(rows):
            pytest.fail("the query must not change the visits table", pytrace=False)
        return result


def _oracle(rows):
    """Independent answer: distinct dates per user, sorted, split at each gap over one day."""
    days: dict[int, set[date]] = {}
    for user, when in rows:
        days.setdefault(user, set()).add(when)
    out = []
    for user in sorted(days):
        runs: list[list] = []
        for when in sorted(days[user]):
            if runs and when - runs[-1][1] == timedelta(days=1):
                runs[-1][1] = when
                runs[-1][2] += 1
            else:
                runs.append([when, when, 1])
        out.extend((user, start, end, length) for start, end, length in runs)
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
    rng = random.Random(9090)  # noqa: S311 -- seeded synthetic data, not security
    for _ in range(30):
        rows = []
        for user in range(1, rng.randrange(2, 6)):
            for _ in range(rng.randrange(1, 25)):
                day = BASE + timedelta(days=rng.randrange(0, 40))
                rows.append((user, day))
                if rng.random() < 0.3:
                    rows.append((user, day))
        rng.shuffle(rows)
        _same(_run(rows), _oracle(rows))


def test_single_day_streak_has_equal_start_end_and_length_one():
    _same(_run([(1, date(2026, 8, 1))]), [(1, date(2026, 8, 1), date(2026, 8, 1), 1)])


def test_gap_of_exactly_one_day_starts_a_new_streak():
    rows = [(1, date(2026, 4, 1)), (1, date(2026, 4, 2)), (1, date(2026, 4, 4))]
    _same(
        _run(rows),
        [(1, date(2026, 4, 1), date(2026, 4, 2), 2), (1, date(2026, 4, 4), date(2026, 4, 4), 1)],
    )


def test_duplicate_visits_do_not_split_or_extend_a_streak():
    rows = [(1, date(2026, 4, 1))] * 3 + [(1, date(2026, 4, 2))] * 3
    _same(_run(rows), [(1, date(2026, 4, 1), date(2026, 4, 2), 2)])


def test_run_across_a_leap_day_is_one_streak():
    rows = [(3, date(2028, 2, 28)), (3, date(2028, 2, 29)), (3, date(2028, 3, 1))]
    _same(_run(rows), [(3, date(2028, 2, 28), date(2028, 3, 1), 3)])


def test_users_without_visits_do_not_appear_and_order_is_user_then_start():
    rows = [(9, date(2026, 5, 3)), (2, date(2026, 5, 9)), (2, date(2026, 5, 1))]
    _same(
        _run(rows),
        [
            (2, date(2026, 5, 1), date(2026, 5, 1), 1),
            (2, date(2026, 5, 9), date(2026, 5, 9), 1),
            (9, date(2026, 5, 3), date(2026, 5, 3), 1),
        ],
    )


def test_adjacent_days_of_two_users_do_not_join_into_one_streak():
    rows = [
        (1, date(2026, 8, 1)),
        (1, date(2026, 8, 2)),
        (2, date(2026, 8, 3)),
        (2, date(2026, 8, 4)),
    ]
    _same(
        _run(rows),
        [(1, date(2026, 8, 1), date(2026, 8, 2), 2), (2, date(2026, 8, 3), date(2026, 8, 4), 2)],
    )


def test_insertion_order_does_not_change_the_streaks():
    rows = [(4, BASE + timedelta(days=d)) for d in (0, 1, 2, 5, 6)]
    _same(_run(list(reversed(rows))), _run(rows))


def test_empty_table_returns_no_rows_with_the_declared_schema():
    _same(_run([]), [])
