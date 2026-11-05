from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest

COLUMNS = ["entity_id", "event_id", "occurred_at", "payload", "rank_in_entity"]
TYPES = ["BIGINT", "BIGINT", "TIMESTAMP", "VARCHAR", "BIGINT"]
SQL = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")


def _run(rows):
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE events(entity_id BIGINT, event_id BIGINT, occurred_at TIMESTAMP, "
            "kind VARCHAR, payload VARCHAR)"
        )
        if rows:
            connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)", rows)
        try:
            cursor = connection.execute(SQL)
        except duckdb.Error:
            pytest.fail("query cannot run against the declared events schema")
        if [column[0] for column in cursor.description] != COLUMNS:
            pytest.fail("return the five named columns in the stated order")
        if [str(column[1]) for column in cursor.description] != TYPES:
            pytest.fail("output ids and rank must be BIGINT; preserve TIMESTAMP and VARCHAR")
        result = cursor.fetchall()
        try:
            after = connection.execute(
                "SELECT entity_id, event_id, occurred_at, kind, payload "
                "FROM events ORDER BY event_id"
            ).fetchall()
        except duckdb.Error:
            pytest.fail("query must leave the input events table intact")
        if after != sorted(rows, key=lambda row: row[1]):
            pytest.fail("query must not modify the input events")
        return result


def _same(got, expected):
    if got != expected:
        pytest.fail("latest rows, tie-breaking, output order or rank differ from the contract")


def test_two_latest_successes_after_filtering():
    when = datetime(2026, 1, 1)
    rows = [
        (1, 1, when, "success", "old"),
        (1, 2, when + timedelta(days=2), "failure", "ignored"),
        (1, 3, when + timedelta(days=1), "success", "new"),
    ]
    _same(_run(rows), [(1, 3, when + timedelta(days=1), "new", 1), (1, 1, when, "old", 2)])


def test_timestamp_ties_use_descending_event_id_and_preserve_null_payload():
    when = datetime(2026, 1, 2)
    _same(
        _run(
            [
                (2, 8, when, "success", "eight"),
                (2, 9, when, "success", None),
                (2, 10, None, "success", "unknown"),
            ]
        ),
        [(2, 9, when, None, 1), (2, 8, when, "eight", 2)],
    )


def test_empty_table_returns_no_rows_with_the_declared_schema():
    _same(_run([]), [])
