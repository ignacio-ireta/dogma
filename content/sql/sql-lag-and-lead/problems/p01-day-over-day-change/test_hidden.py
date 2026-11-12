# Hidden tests: the oracle is a pandas groupby-shift over the same rows. Messages name the
# kind of mistake and never reveal hidden inputs or expected values.
import csv
import functools
import random
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from dogma_data import retail_events

COLUMNS = [
    "account_id",
    "usage_date",
    "api_calls",
    "previous_usage_date",
    "previous_api_calls",
    "calls_change",
]
TYPES = ["BIGINT", "DATE", "BIGINT", "DATE", "BIGINT", "BIGINT"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE account_daily_usage(account_id BIGINT, usage_date DATE, api_calls BIGINT)"
        )
        if rows:
            connection.executemany("INSERT INTO account_daily_usage VALUES (?, ?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared account_daily_usage schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the six named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("check the column types: ids and changes BIGINT, dates DATE")
        result = cursor.fetchall()
        try:
            after = connection.execute(
                "SELECT account_id, usage_date, api_calls FROM account_daily_usage "
                "ORDER BY account_id, usage_date"
            ).fetchall()
        except duckdb.Error:
            pytest.fail("query must leave the input table intact")
        if after != sorted(rows, key=lambda row: (row[0], row[1])):
            pytest.fail("query must not modify the input table")
        return result


def _same(got, expected):
    if got != expected:
        pytest.fail(
            "change, previous-day values, output order or NULL handling differ from the contract"
        )


def _oracle(rows):
    # Object dtype keeps large Python ints exact; a float column would round them.
    frame = pd.DataFrame(rows, columns=["account_id", "usage_date", "api_calls"], dtype=object)
    frame = frame.sort_values(["account_id", "usage_date"], kind="stable").reset_index(drop=True)
    grouped = frame.groupby("account_id", sort=True)
    previous_date = grouped["usage_date"].shift(1)
    previous_calls = grouped["api_calls"].shift(1)
    out = []
    for index in range(len(frame)):
        account, day, calls = (frame.loc[index, name] for name in COLUMNS[:3])
        before = previous_calls.loc[index]
        if pd.isna(before):
            out.append((account, day, calls, None, None, None))
        else:
            out.append((account, day, calls, previous_date.loc[index], before, calls - before))
    return out


@functools.cache
def _daily_units():
    """Real daily usage: the units each customer of an online retailer bought on each day.

    From online_retail. An account is a customer, a usage day is a day with a sale and the
    count is the number of units sold to that customer on that day. The gaps between a
    customer's days are real: a few are one day, most are weeks.
    """
    events = retail_events()
    sold = events[events["quantity"] > 0]
    daily = sold.groupby(["user_id", sold["ts"].dt.date])["quantity"].sum()
    usage = {}
    for (customer, day), units in daily.items():
        usage.setdefault(int(customer), []).append((day, int(units)))
    return usage


def _seeded_rows(seed):
    rng = random.Random(seed)  # noqa: S311 -- seeded choice of real rows, not security.
    usage = _daily_units()
    rows = []
    for account in rng.sample(sorted(usage), rng.randrange(1, 7)):
        days = usage[account]
        first = rng.randrange(len(days))
        for day, units in days[first : first + rng.randrange(1, 15)]:
            rows.append((account, day, units))
    rng.shuffle(rows)
    return rows


def test_matches_pandas_shift_oracle_on_seeded_usage_with_gaps():
    for seed in range(25):
        rows = _seeded_rows(seed)
        _same(_run(rows), _oracle(rows))


def test_previous_row_is_the_previous_recorded_day_not_the_previous_calendar_day():
    rows = [(4, date(2026, 5, 1), 10), (4, date(2026, 5, 9), 25), (4, date(2026, 5, 10), 5)]
    _same(
        _run(rows),
        [
            (4, date(2026, 5, 1), 10, None, None, None),
            (4, date(2026, 5, 9), 25, date(2026, 5, 1), 10, 15),
            (4, date(2026, 5, 10), 5, date(2026, 5, 9), 25, -20),
        ],
    )


def test_output_order_follows_account_then_date_not_insertion_order():
    rows = [
        (2, date(2026, 1, 3), 3),
        (1, date(2026, 1, 2), 2),
        (2, date(2026, 1, 1), 1),
        (1, date(2026, 1, 1), 1),
    ]
    _same(_run(rows), _oracle(rows))


def test_nulls_are_kept_for_every_first_row_of_every_account():
    rows = [(a, date(2026, 6, 1), a * 100) for a in range(1, 6)]
    result = _run(rows)
    for row in result:
        if row[3] is not None or row[4] is not None or row[5] is not None:
            pytest.fail("a first recorded day must have NULL previous fields and change")
    _same(result, _oracle(rows))


def test_decreases_are_negative_changes_from_current_minus_previous():
    rows = [(8, date(2026, 4, 1), 900), (8, date(2026, 4, 2), 300)]
    _same(
        _run(rows),
        [
            (8, date(2026, 4, 1), 900, None, None, None),
            (8, date(2026, 4, 2), 300, date(2026, 4, 1), 900, -600),
        ],
    )


def test_large_bigint_values_are_exact():
    big = 2**60
    rows = [(2**50, date(2026, 1, 1), big), (2**50, date(2026, 1, 2), big + 7)]
    _same(_run(rows), _oracle(rows))


def test_fixture_matches_independent_oracle():
    path = Path(__file__).with_name("fixtures").joinpath("daily_usage.csv")
    with path.open(newline="") as handle:
        rows = [
            (int(row["account_id"]), date.fromisoformat(row["usage_date"]), int(row["api_calls"]))
            for row in csv.DictReader(handle)
        ]
    _same(_run(rows), _oracle(rows))


def test_empty_input_returns_declared_schema_with_no_rows():
    _same(_run([]), [])
