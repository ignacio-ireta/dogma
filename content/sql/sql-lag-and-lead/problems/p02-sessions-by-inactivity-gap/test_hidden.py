# Hidden tests: the oracle is a pandas sort, groupby-shift, gap test and cumulative sum over the
# same rows. Messages name the kind of mistake and never reveal hidden inputs or expected values.
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from dogma_data import rng

COLUMNS = ["visitor_id", "view_id", "viewed_at", "session_number", "starts_session"]
TYPES = ["BIGINT", "BIGINT", "TIMESTAMP", "BIGINT", "BOOLEAN"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")
GAP = timedelta(minutes=30)


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE screen_views(visitor_id BIGINT, view_id BIGINT, viewed_at TIMESTAMP)"
        )
        if rows:
            connection.executemany("INSERT INTO screen_views VALUES (?, ?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared screen_views schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the five named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("check the types: session_number must be BIGINT, not a wider integer")
        result = cursor.fetchall()
        try:
            after = connection.execute(
                "SELECT visitor_id, view_id, viewed_at FROM screen_views ORDER BY view_id"
            ).fetchall()
        except duckdb.Error:
            pytest.fail("query must leave the input table intact")
        if after != sorted(rows, key=lambda row: row[1]):
            pytest.fail("query must not modify the input screen_views table")
        return result


def _same(got, expected):
    if got != expected:
        pytest.fail(
            "session numbers, session-start flags, tie order or output order differ from the "
            "contract"
        )


def _oracle(rows):
    frame = pd.DataFrame(rows, columns=["visitor_id", "view_id", "viewed_at"])
    frame = frame.sort_values(["visitor_id", "viewed_at", "view_id"], kind="stable")
    frame = frame.reset_index(drop=True)
    previous = frame.groupby("visitor_id")["viewed_at"].shift(1)
    starts = previous.isna() | ((frame["viewed_at"] - previous) > GAP)
    sessions = starts.astype(int).groupby(frame["visitor_id"]).cumsum()
    return [
        (int(v), int(w), t.to_pydatetime(), int(s), bool(f))
        for v, w, t, s, f in zip(
            frame["visitor_id"], frame["view_id"], frame["viewed_at"], sessions, starts, strict=True
        )
    ]


def _seeded_rows(seed):
    rng = random.Random(seed)  # noqa: S311 -- seeded synthetic fixtures, not security.
    rows = []
    view_id = 1
    for visitor in rng.sample(range(1, 30), rng.randrange(1, 6)):
        clock = datetime(2026, 4, 1, 8, 0)
        for _ in range(rng.randrange(1, 20)):
            clock += rng.choice(
                [
                    timedelta(0),
                    timedelta(minutes=5),
                    GAP,
                    GAP + timedelta(seconds=1),
                    timedelta(minutes=45),
                    timedelta(hours=7),
                ]
            )
            rows.append((visitor, view_id, clock))
            view_id += 1
    rng.shuffle(rows)
    return rows


def test_matches_pandas_session_oracle_on_seeded_views_with_boundary_gaps():
    for seed in range(25):
        rows = _seeded_rows(seed)
        _same(_run(rows), _oracle(rows))


def test_gap_of_exactly_thirty_minutes_does_not_start_a_session():
    t0 = datetime(2026, 5, 2, 7, 0)
    rows = [(1, 1, t0), (1, 2, t0 + GAP)]
    _same(_run(rows), [(1, 1, t0, 1, True), (1, 2, t0 + GAP, 1, False)])


def test_gap_just_over_thirty_minutes_starts_a_new_session():
    t0 = datetime(2026, 5, 2, 7, 0)
    rows = [(1, 1, t0), (1, 2, t0 + GAP + timedelta(seconds=1))]
    _same(_run(rows), [(1, 1, t0, 1, True), (1, 2, t0 + GAP + timedelta(seconds=1), 2, True)])


def test_tied_timestamps_order_by_view_id_when_choosing_the_session_opener():
    t0 = datetime(2026, 5, 2, 7, 0)
    later = t0 + timedelta(hours=1)
    rows = [(4, 30, later), (4, 29, later), (4, 10, t0)]
    _same(
        _run(rows),
        [
            (4, 10, t0, 1, True),
            (4, 29, later, 2, True),
            (4, 30, later, 2, False),
        ],
    )


def test_session_numbers_restart_at_one_for_each_visitor():
    t0 = datetime(2026, 5, 2, 7, 0)
    rows = [(1, 1, t0), (1, 2, t0 + timedelta(hours=2)), (2, 3, t0)]
    result = _run(rows)
    if [row[3] for row in result if row[0] == 2] != [1]:
        pytest.fail("session numbers must restart at 1 for each visitor", pytrace=False)
    _same(result, _oracle(rows))


def test_single_view_visitor_starts_session_one():
    t0 = datetime(2026, 5, 2, 7, 0)
    _same(_run([(12, 5, t0)]), [(12, 5, t0, 1, True)])


def test_output_order_is_visitor_then_time_then_view_id_not_input_order():
    t0 = datetime(2026, 5, 2, 7, 0)
    rows = [(9, 3, t0), (2, 8, t0 + timedelta(minutes=5)), (2, 7, t0), (9, 1, t0 - GAP)]
    _same(_run(rows), _oracle(rows))


def test_fixture_matches_independent_oracle():
    path = Path(__file__).with_name("fixtures").joinpath("screen_views.csv")
    with path.open(newline="") as handle:
        rows = [
            (
                int(row["visitor_id"]),
                int(row["view_id"]),
                datetime.fromisoformat(row["viewed_at"]),
            )
            for row in csv.DictReader(handle)
        ]
    _same(_run(rows), _oracle(rows))


def test_empty_input_returns_declared_schema_with_no_rows():
    _same(_run([]), [])


def _millisecond_visitor_views(key: str) -> list[tuple[int, int, datetime]]:
    """Visitors with Zipf-skewed view counts and microsecond clocks, with gaps near 30 minutes.

    Most gaps are short. About one gap in seven sits at 30 minutes plus or minus a few microseconds,
    a second, or a fraction of a second, so the boundary is tested at the resolution that app
    logs keep. Rows come back shuffled.
    """
    gen = rng(key)
    visitors = 40
    weights = 1.0 / np.arange(1, visitors + 1, dtype=np.float64) ** 1.2
    sizes = gen.multinomial(1200, weights / weights.sum())
    around = np.array([-1, 0, 1, 1_000, 500_000, 999_999, 1_000_000, 1_000_001], dtype=np.int64)
    rows: list[tuple[int, int, datetime]] = []
    view_id = 1
    for visitor, size in enumerate(sizes, start=1):
        start_us = int(gen.integers(0, 3 * 86_400 * 1_000_000))
        clock = datetime(2026, 4, 1) + timedelta(microseconds=start_us)
        for _ in range(int(size)):
            roll = gen.random()
            if roll < 1 / 6:
                step = GAP + timedelta(microseconds=int(gen.choice(around)))
            elif roll < 0.9:
                step = timedelta(microseconds=int(gen.exponential(4 * 60 * 1_000_000)))
            else:
                step = timedelta(hours=float(gen.uniform(1, 9)))
            clock += step
            rows.append((visitor, view_id, clock))
            view_id += 1
    order = gen.permutation(len(rows))
    return [rows[int(i)] for i in order]


def test_gap_of_thirty_minutes_and_one_millisecond_starts_a_session():
    t0 = datetime(2026, 5, 2, 7, 0, 0, 200_000)
    later = t0 + GAP + timedelta(milliseconds=1)
    got = _run([(1, 1, t0), (1, 2, later)])
    if got != [(1, 1, t0, 1, True), (1, 2, later, 2, True)]:
        pytest.fail(
            "a gap just over thirty minutes is judged in whole seconds: compare the exact gap, "
            "fractions of a second included",
            pytrace=False,
        )


def test_millisecond_clocks_with_skewed_visitors_match_the_oracle():
    rows = _millisecond_visitor_views("sql/sql-lag-and-lead/p02-sessions-by-inactivity-gap:ms:0")
    got = _run(rows)
    if got != _oracle(rows):
        pytest.fail(
            "session starts differ on microsecond timestamps: the gap keeps its fractions of a "
            "second, so judge it as an exact interval",
            pytrace=False,
        )


def test_output_orders_each_visitor_by_time_before_view_id():
    # Visitor 6 has a later view with the lower view id. Sorted by time, the 07:00 view comes
    # first and opens session 1; the view five minutes later is within 30 minutes, so it stays
    # in session 1 and does not open one.
    t0 = datetime(2026, 5, 2, 7, 0)
    later = t0 + timedelta(minutes=5)
    rows = [(6, 2, later), (6, 9, t0)]
    expected = [(6, 9, t0, 1, True), (6, 2, later, 1, False)]
    result = _run(rows)
    if result != expected:
        pytest.fail(
            "each visitor's rows must come out in time order first, and only then by view id",
            pytrace=False,
        )
