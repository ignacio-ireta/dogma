from __future__ import annotations

from datetime import datetime
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
        return cursor.fetchall()


def _same(got, expected, mistake=None):
    if got != expected:
        pytest.fail(
            mistake or "running balances, position order or tie handling differ from the contract"
        )


def test_running_balance_with_a_tie_matches_the_worked_example():
    _same(
        _run(
            [
                (7, 12, datetime(2026, 3, 1, 9), 500),
                (7, 10, datetime(2026, 3, 1, 9), -120),
                (7, 11, datetime(2026, 2, 28, 17), 80),
                (7, 13, datetime(2026, 3, 2, 8), -200),
            ]
        ),
        [
            (7, 11, datetime(2026, 2, 28, 17), 80, 1, 80),
            (7, 10, datetime(2026, 3, 1, 9), -120, 2, -40),
            (7, 12, datetime(2026, 3, 1, 9), 500, 3, 460),
            (7, 13, datetime(2026, 3, 2, 8), -200, 4, 260),
        ],
        "the worked example differs: check the tie order and the running sum for account 7",
    )


def test_single_posting_account_starts_its_own_balance():
    _same(
        _run([(9, 4, datetime(2026, 1, 5), 75)]),
        [(9, 4, datetime(2026, 1, 5), 75, 1, 75)],
        "a single-posting account must start its own balance at its amount",
    )


def test_empty_table_returns_no_rows_with_the_declared_schema():
    _same(_run([]), [], "an empty table must return no rows, with the declared columns and types")
