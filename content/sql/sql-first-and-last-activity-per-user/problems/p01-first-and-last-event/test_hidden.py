"""Hidden tests: a pandas sort-and-head oracle on seeded and fixture events, and the tie rules.

The oracle sorts the events and takes the first and last row of each user. It uses no window
functions or arg functions, so it does not depend on one query shape. Messages name the kind of
mistake and never print hidden rows or expected results.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

COLUMNS = ["user_id", "first_event_time", "first_event_type", "last_event_time", "last_event_type"]
TYPES = ["BIGINT", "TIMESTAMP", "VARCHAR", "TIMESTAMP", "VARCHAR"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")
BASE = datetime(2026, 1, 1, 0, 0)


def _run(rows: list[tuple[int, int, datetime, str]]) -> list[tuple[Any, ...]]:
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE events(user_id BIGINT, event_id BIGINT, event_time TIMESTAMP, "
            "event_type VARCHAR)"
        )
        if rows:
            connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("the query cannot run against the declared events schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the five named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("keep the id as BIGINT and the times as TIMESTAMP, the types as VARCHAR")
        result = cursor.fetchall()
        after = connection.execute(
            "SELECT user_id, event_id, event_time, event_type FROM events ORDER BY event_id"
        ).fetchall()
        if after != sorted(rows, key=lambda row: row[1]):
            pytest.fail("the query must not change the events table")
        return result


def _oracle(rows: list[tuple[int, int, datetime, str]]) -> list[tuple[Any, ...]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows, columns=["user_id", "event_id", "event_time", "event_type"])
    ascending = frame.sort_values(["user_id", "event_time", "event_id"], kind="mergesort")
    first = ascending.groupby("user_id", sort=True).head(1).set_index("user_id")
    descending = frame.sort_values(
        ["user_id", "event_time", "event_id"], ascending=[True, False, False], kind="mergesort"
    )
    last = descending.groupby("user_id", sort=True).head(1).set_index("user_id")
    result = []
    for user_id in sorted(first.index):
        result.append(
            (
                int(user_id),
                first.loc[user_id, "event_time"].to_pydatetime(),
                first.loc[user_id, "event_type"],
                last.loc[user_id, "event_time"].to_pydatetime(),
                last.loc[user_id, "event_type"],
            )
        )
    return result


def _same(got: list[tuple[Any, ...]], expected: list[tuple[Any, ...]]) -> None:
    if len(got) != len(expected):
        pytest.fail(
            "one row per user is expected: a join back on a MIN or MAX timestamp duplicates tied "
            "events, and LAST_VALUE with the default frame returns a row for every event"
        )
    for row_got, row_want in zip(got, expected, strict=True):
        if row_got[0] != row_want[0]:
            pytest.fail(
                "the rows are out of order or the wrong users appear: order by user_id ascending"
            )
        if row_got[1] != row_want[1] or row_got[3] != row_want[3]:
            pytest.fail(
                "a first or last time is wrong: first is the earliest time, last the latest"
            )
        if row_got[2] != row_want[2]:
            pytest.fail(
                "the first event type is wrong: pick the earliest event_time, and use the smallest "
                "event_id only to break a tie"
            )
        if row_got[4] != row_want[4]:
            pytest.fail(
                "the last event type is wrong: pick the latest event_time, and use the largest "
                "event_id only to break a tie; keep each time and type from the same event"
            )


def _random_events(
    rng: random.Random, count: int, users: int, slots: int
) -> list[tuple[int, int, datetime, str]]:
    """Few distinct timestamps, so ties are common, and ids drawn in any order."""
    ids = rng.sample(range(1, count * 5), count)
    kinds = ["login", "view", "click", "purchase", "logout"]
    rows = []
    for event_id in ids:
        rows.append(
            (
                rng.randrange(1, users + 1),
                event_id,
                BASE + timedelta(hours=rng.randrange(slots)),
                rng.choice(kinds),
            )
        )
    return rows


def test_seeded_tied_events_match_the_pandas_oracle() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic events, not security.
    for _ in range(25):
        rows = _random_events(rng, rng.randint(1, 80), users=6, slots=4)
        _same(_run(rows), _oracle(rows))


def test_fixture_events_match_the_pandas_oracle() -> None:
    with Path(__file__).with_name("fixtures").joinpath("events.csv").open(newline="") as handle:
        rows = [
            (
                int(row["user_id"]),
                int(row["event_id"]),
                datetime.fromisoformat(row["event_time"]),
                row["event_type"],
            )
            for row in csv.DictReader(handle)
        ]
    _same(_run(rows), _oracle(rows))


def test_first_ties_take_the_smallest_event_id() -> None:
    rows = [
        (5, 40, BASE, "late-id"),
        (5, 7, BASE, "early-id"),
        (5, 90, BASE + timedelta(days=1), "later"),
    ]
    _same(_run(rows), _oracle(rows))
    assert _run(rows)[0][2] == "early-id", (
        "the first event type is wrong on a tie: use the smallest event_id"
    )


def test_last_ties_take_the_largest_event_id() -> None:
    rows = [
        (6, 3, BASE, "start"),
        (6, 11, BASE + timedelta(days=2), "middle-id"),
        (6, 30, BASE + timedelta(days=2), "last-id"),
        (6, 25, BASE + timedelta(days=2), "other"),
    ]
    _same(_run(rows), _oracle(rows))
    assert _run(rows)[0][4] == "last-id", (
        "the last event type is wrong on a tie: use the largest event_id"
    )


def test_event_ids_that_disagree_with_time_do_not_define_order() -> None:
    rows = [(8, 100 - i, BASE + timedelta(hours=i), f"step{i}") for i in range(6)]
    _same(_run(rows), _oracle(rows))


def test_user_whose_events_share_one_time_is_handled_by_ids() -> None:
    rows = [(4, eid, BASE, f"e{eid}") for eid in (12, 3, 8)]
    _same(_run(rows), [(4, BASE, "e3", BASE, "e12")])


def test_one_row_per_user_even_with_many_ties() -> None:
    rows = [(2, eid, BASE, "same") for eid in range(1, 21)] + [(3, 99, BASE, "solo")]
    result = _run(rows)
    if len(result) != 2:
        pytest.fail(
            "one row per user is expected: a join back on a MIN or MAX timestamp duplicates "
            "tied events"
        )


def test_large_identifiers_stay_exact_bigint() -> None:
    rows = [(2**60, 2**60 + 1, BASE, "a"), (2**60, 2**60 + 2, BASE, "b")]
    _same(_run(rows), _oracle(rows))


def test_timestamps_keep_their_exact_values() -> None:
    rows = [
        (1, 1, datetime(2026, 12, 31, 23, 59, 59, 999000), "end"),
        (1, 2, datetime(2026, 1, 1, 0, 0, 0, 1), "start"),
    ]
    _same(_run(rows), _oracle(rows))


def test_empty_table_returns_nothing() -> None:
    _same(_run([]), [])


def test_single_event_per_user_is_both_first_and_last() -> None:
    rows = [(u, u * 10, BASE + timedelta(hours=u), f"t{u}") for u in range(1, 6)]
    _same(_run(rows), _oracle(rows))
