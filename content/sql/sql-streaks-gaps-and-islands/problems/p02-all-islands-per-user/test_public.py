# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's query is read from submission.sql; tests never read solution.sql.
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

COLUMNS = ["user_id", "island_start", "island_end", "island_days"]
TYPES = ["BIGINT", "DATE", "DATE", "BIGINT"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE visits(user_id BIGINT, visit_date DATE)")
        if rows:
            connection.executemany("INSERT INTO visits VALUES (?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared visits schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the four named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("cast island_days to BIGINT; dates must stay DATE")
        return cursor.fetchall()


def _same(got, expected, mistake):
    if got != expected:
        pytest.fail(mistake)


def test_first_example_splits_a_user_into_two_streaks_at_a_gap():
    rows = [
        (1, date(2026, 1, 1)), (1, date(2026, 1, 2)), (1, date(2026, 1, 2)),
        (1, date(2026, 1, 3)), (1, date(2026, 1, 5)), (1, date(2026, 1, 6)),
        (2, date(2026, 3, 10)),
    ]  # fmt: skip
    _same(
        _run(rows),
        [
            (1, date(2026, 1, 1), date(2026, 1, 3), 3),
            (1, date(2026, 1, 5), date(2026, 1, 6), 2),
            (2, date(2026, 3, 10), date(2026, 3, 10), 1),
        ],
        "streaks differ: a gap of more than one day must end a streak; duplicates count once",
    )


def test_user_with_two_streaks_of_equal_length_returns_both_in_date_order():
    rows = [
        (4, date(2026, 1, 1)),
        (4, date(2026, 1, 2)),
        (4, date(2026, 1, 10)),
        (4, date(2026, 1, 11)),
    ]
    _same(
        _run(rows),
        [
            (4, date(2026, 1, 1), date(2026, 1, 2), 2),
            (4, date(2026, 1, 10), date(2026, 1, 11), 2),
        ],
        "both streaks of an equal-length user must appear, ordered by island_start",
    )


def test_empty_table_returns_no_rows_with_the_declared_schema():
    _same(_run([]), [], "an empty table must return no rows, with the declared columns and types")
