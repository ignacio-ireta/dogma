from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from dogma_data import retail_events

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


def _oracle(rows):
    grouped: dict[int, list[tuple[int, datetime | None, str | None]]] = {}
    for entity, event, when, kind, payload in rows:
        if kind == "success":
            grouped.setdefault(entity, []).append((event, when, payload))
    result: list[tuple[int, int, datetime | None, str | None, int]] = []
    for entity in sorted(grouped):
        ordered = sorted(
            grouped[entity],
            key=lambda row: (row[1] is not None, row[1] or datetime.min, row[0]),
            reverse=True,
        )
        result.extend(
            (entity, event, when, payload, rank)
            for rank, (event, when, payload) in enumerate(ordered[:2], start=1)
        )
    return result


def test_matches_independent_group_sort_oracle_on_seeded_records():
    # Real records (online_retail): a run of consecutive invoice lines of an online retailer.
    # The entity is the customer, the time is the invoice minute, so the lines of an invoice
    # tie, and the payload is the product description as typed. A sold line is a success and a
    # line of a cancellation a failure. The source has no missing time and no hand-typed kind,
    # and the contract has rules for both: one line in five loses its time, and one success in
    # four is typed in capitals.
    rng = random.Random(2026)  # noqa: S311 -- seeded choice of real rows, not security.
    events = retail_events()
    for _ in range(20):
        first = rng.randrange(len(events) - 80)
        picked = events.iloc[first : first + rng.randrange(0, 79)]
        rows = []
        for customer, event, moment, cancelled, label in zip(
            picked["user_id"],
            picked["event_id"],
            picked["ts"],
            picked["cancelled"],
            picked["label"],
            strict=True,
        ):
            when = None if rng.randrange(5) == 0 else moment.to_pydatetime()
            kind = (
                "failure" if cancelled else rng.choice(["success", "success", "success", "SUCCESS"])
            )
            rows.append((int(customer), int(event), when, kind, str(label)))
        rng.shuffle(rows)
        _same(_run(rows), _oracle(rows))


def test_filtering_happens_before_assigning_row_numbers():
    when = datetime(2026, 1, 1)
    rows = [
        (1, i, when + timedelta(days=i), "success" if i <= 2 else "failure", None)
        for i in range(1, 6)
    ]
    _same(_run(rows), _oracle(rows))


def test_equal_timestamps_still_return_at_most_two_rows_per_entity():
    when = datetime(2026, 1, 1)
    rows = [(3, i, when, "success", None) for i in [4, 8, 2, 7, 9]]
    _same(_run(rows), [(3, 9, when, None, 1), (3, 8, when, None, 2)])


def test_null_timestamps_sort_after_known_timestamps_and_tie_by_id():
    when = datetime(2026, 1, 1)
    rows = [
        (1, 20, None, "success", "a"),
        (1, 21, None, "success", "b"),
        (1, 1, when, "success", "known"),
        (2, 30, None, "success", None),
    ]
    _same(_run(rows), _oracle(rows))


def test_output_is_ordered_by_entity_then_rank_not_input_or_global_time():
    when = datetime(2026, 1, 1)
    rows = [
        (7, 10, when, "success", "a"),
        (2, 11, when, "success", "b"),
        (7, 12, when + timedelta(days=1), "success", "c"),
        (2, 13, when - timedelta(days=1), "success", "d"),
    ]
    _same(_run(rows), _oracle(rows))


def test_exact_success_label_and_groups_with_no_success():
    when = datetime(2026, 1, 1)
    rows = [
        (1, 1, when, "SUCCESS", "a"),
        (1, 2, when, "success ", "b"),
        (2, 3, when, "failure", "c"),
        (3, 4, when, "success", None),
    ]
    _same(_run(rows), [(3, 4, when, None, 1)])


def test_fixture_matches_independent_oracle():
    with Path(__file__).with_name("fixtures").joinpath("events.csv").open(newline="") as handle:
        rows = [
            (
                int(row["entity_id"]),
                int(row["event_id"]),
                datetime.fromisoformat(row["occurred_at"]) if row["occurred_at"] else None,
                row["kind"],
                row["payload"] or None,
            )
            for row in csv.DictReader(handle)
        ]
    _same(_run(rows), _oracle(rows))


def test_large_integer_ids_preserve_exact_bigint_values():
    when = datetime(2026, 1, 1)
    rows = [(2**60, 2**60 + i, when, "success", None) for i in range(4)]
    _same(_run(rows), _oracle(rows))
