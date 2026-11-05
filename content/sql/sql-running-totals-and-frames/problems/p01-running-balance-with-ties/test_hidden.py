from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest

COLUMNS = [
    "account_id",
    "posting_id",
    "posted_at",
    "amount_cents",
    "position_in_account",
    "balance_cents",
]
TYPES = ["BIGINT", "BIGINT", "TIMESTAMP", "BIGINT", "BIGINT", "BIGINT"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")
BASE = datetime(2026, 1, 1)


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE postings(account_id BIGINT, posting_id BIGINT, "
            "posted_at TIMESTAMP, amount_cents BIGINT)"
        )
        if rows:
            connection.executemany("INSERT INTO postings VALUES (?, ?, ?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared postings schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the six named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("cast balance_cents back to BIGINT; the other columns keep their types")
        result = cursor.fetchall()
        after = connection.execute(
            "SELECT account_id, posting_id, posted_at, amount_cents "
            "FROM postings ORDER BY posting_id"
        ).fetchall()
        if after != sorted([(a, p, t, m) for a, p, t, m in rows], key=lambda row: row[1]):
            pytest.fail("query must not modify the input postings")
        return result


def _same(got, expected, mistake=None):
    if got != expected:
        pytest.fail(
            mistake or "running balances, position order or tie handling differ from the contract"
        )


def _oracle(rows):
    """Independent answer: group by account, sort by (posted_at, posting_id), then accumulate."""
    grouped: dict[int, list[tuple[datetime, int, int]]] = {}
    for account, posting, when, amount in rows:
        grouped.setdefault(account, []).append((when, posting, amount))
    result = []
    for account in sorted(grouped):
        balance = 0
        for position, (when, posting, amount) in enumerate(
            sorted(grouped[account], key=lambda row: (row[0], row[1])), start=1
        ):
            balance += amount
            result.append((account, posting, when, amount, position, balance))
    return result


def test_matches_group_sort_oracle_on_seeded_postings_with_ties():
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic fixtures, not security.
    for _ in range(25):
        rows = []
        for posting in rng.sample(range(1, 500), rng.randrange(1, 60)):
            when = BASE + timedelta(hours=rng.randrange(3))  # three timestamps: many ties
            rows.append((rng.randrange(1, 6), posting, when, rng.randrange(-500, 501)))
        rng.shuffle(rows)
        _same(
            _run(rows),
            _oracle(rows),
            (
                "running balance differs from the group-and-sort oracle: check the partition by "
                "account, the (posted_at, posting_id) order, and that the running sum covers every "
                "earlier row"
            ),
        )


def test_each_posting_in_a_tie_group_gets_its_own_balance():
    when = BASE
    rows = [(1, 30, when, 5), (1, 20, when, -2), (1, 10, when, 7)]
    _same(
        _run(rows),
        _oracle(rows),
        (
            "tied postings must each get their own balance: the order needs posting_id as a "
            "tiebreaker; with posting_id in ORDER BY, the frame can be ROWS or the default"
        ),
    )


def test_reverse_insertion_order_does_not_change_the_balances():
    when = BASE
    rows = [(2, i, when, i) for i in range(9, 0, -1)]
    _same(
        _run(rows),
        _oracle(rows),
        "balances depend on insertion order: the tiebreaker is missing from the ORDER BY",
    )


def test_balances_restart_for_every_account():
    when = BASE
    rows = [
        (1, 1, when, 100),
        (2, 2, when, 40),
        (1, 3, when + timedelta(days=1), 25),
        (2, 4, when + timedelta(days=1), -15),
    ]
    _same(
        _run(rows),
        _oracle(rows),
        "balances must restart for each account: the PARTITION BY account_id is missing or wrong",
    )


def test_negative_running_balance_is_kept_exact():
    when = BASE
    rows = [(5, 1, when, -300), (5, 2, when, -1), (5, 3, when + timedelta(days=2), 1000)]
    _same(
        _run(rows),
        _oracle(rows),
        "negative balances differ: check the sign of amounts and the running sum",
    )


def test_large_amounts_stay_exact_beyond_double_precision():
    when = BASE
    big = 10**16 + 7  # running sums reach about 8 * 10**16, beyond 2**53
    rows = [(8, i, when + timedelta(minutes=i % 2), big + i) for i in range(1, 9)]
    _same(
        _run(rows),
        _oracle(rows),
        "large amounts lost precision or were wrapped: keep the sum in integers and cast to BIGINT",
    )


def test_output_is_ordered_by_account_then_position_not_by_insertion():
    when = BASE
    rows = [
        (3, 1, when, 1),
        (1, 2, when + timedelta(days=1), 2),
        (3, 3, when - timedelta(days=1), 3),
    ]
    _same(
        _run(rows),
        _oracle(rows),
        "output must be ordered by account_id, then position_in_account: add a final ORDER BY",
    )


def test_empty_table_returns_no_rows_with_the_declared_schema():
    _same(_run([]), [], "an empty table must return no rows, with the declared columns and types")


def test_fixture_matches_independent_oracle():
    with Path(__file__).with_name("fixtures").joinpath("postings.csv").open(newline="") as handle:
        rows = [
            (
                int(row["account_id"]),
                int(row["posting_id"]),
                datetime.fromisoformat(row["posted_at"]),
                int(row["amount_cents"]),
            )
            for row in csv.DictReader(handle)
        ]
    _same(
        _run(rows),
        _oracle(rows),
        (
            "running balance differs from the group-and-sort oracle: check the partition by "
            "account, the (posted_at, posting_id) order, and that the running sum covers every "
            "earlier row"
        ),
    )
