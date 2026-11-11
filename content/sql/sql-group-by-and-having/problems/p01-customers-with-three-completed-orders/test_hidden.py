"""Hidden tests: your query runs against tables that each test builds itself.

The oracle is an independent pandas computation over the same rows. Each test name says what
behaviour it checks, and each failure message names the likely mistake.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dogma_data import rng, synth

COLUMNS = ["customer_id", "completed_orders", "total_amount"]
ALL_COLUMNS = ["order_id", "customer_id", "order_date", "status", "amount"]
TABLE_DDL = (
    "CREATE TABLE orders (order_id BIGINT, customer_id BIGINT, order_date DATE, "
    "status VARCHAR, amount DOUBLE)"
)
INSERT = "INSERT INTO orders VALUES (?, ?, ?, ?, ?)"
DAY = date(2024, 1, 1)
STATUSES = ["completed", "completed", "completed", "pending", "cancelled", "refunded"]


def _connect(rows: Sequence[tuple[object, ...]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(TABLE_DDL)
    con.executemany(INSERT, rows)
    return con


def _run_learner_query(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = Path(__file__).with_name("submission.sql").read_text()
    try:
        result = con.execute(sql)
    except duckdb.Error as exc:
        pytest.fail(f"your SQL did not run: {exc}")
    if result is None:
        pytest.fail("submission.sql returned no result set: finish it with a SELECT")
    return result.fetchdf()


def _oracle(rows: list[tuple[object, ...]]) -> pd.DataFrame:
    """Independent answer: count completed orders per customer, keep >= 3, sort by total."""
    frame = pd.DataFrame(rows, columns=ALL_COLUMNS)
    done = frame[frame["status"] == "completed"]
    grouped = done.groupby("customer_id")
    summary = pd.DataFrame(
        {
            "completed_orders": grouped.size(),
            "total_amount": grouped["amount"].sum(min_count=1),
        }
    )
    kept = summary[summary["completed_orders"] >= 3].reset_index()
    ranked = kept.sort_values(
        ["total_amount", "customer_id"], ascending=[False, True], na_position="last"
    )
    return ranked[COLUMNS].reset_index(drop=True)


def _clean(value: object) -> object:
    """Make DuckDB and pandas values comparable: NULL -> None, floats rounded to 6 places."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return str(value)


def _rows(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    return [tuple(_clean(v) for v in row) for row in frame.itertuples(index=False, name=None)]


def _same_total(got: Any, want: Any) -> bool:
    if got is None or want is None:
        return got is want
    return math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-6)


def _explain_mismatch(
    got_rows: list[tuple[object, ...]], want_rows: list[tuple[object, ...]]
) -> str | None:
    got_by = {row[0]: row for row in got_rows}
    want_by = {row[0]: row for row in want_rows}
    # Messages name the mistake only: customer ids and expected totals are hidden values.
    for customer in sorted(set(got_by) | set(want_by), key=repr):
        got, want = got_by.get(customer), want_by.get(customer)
        if want is None:
            return (
                "a customer was returned who should not be: count only status = 'completed' "
                "rows and keep groups with COUNT(*) >= 3"
            )
        if got is None:
            return "a qualifying customer is missing: check the threshold (>= 3, not > 3)"
        if got[1] != want[1]:
            return (
                "a customer's completed_orders is wrong: count every completed order with "
                "COUNT(*), including orders whose amount is NULL"
            )
        if not _same_total(got[2], want[2]):
            return (
                "a customer's total_amount is wrong: SUM skips NULL amounts, and an all-NULL "
                "group has a NULL total, not 0"
            )
    return None


def _check_same_result(got: pd.DataFrame, want: pd.DataFrame) -> None:
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}. "
        "Select customer_id, COUNT(*) as completed_orders, SUM(amount) as total_amount."
    )
    got_rows, want_rows = _rows(got), _rows(want)
    assert len(got_rows) == len(want_rows), (
        f"row count differs: your query returned {len(got_rows)} customers. "
        "Check the status filter and the HAVING threshold."
    )
    assert len({row[0] for row in got_rows}) == len(got_rows), (
        "a customer appears more than once: each customer must be one row (GROUP BY customer_id)"
    )
    problem = _explain_mismatch(got_rows, want_rows)
    assert problem is None, problem
    assert got_rows == want_rows, (
        "row order differs: did you ORDER BY total_amount DESC NULLS LAST, customer_id ASC?"
    )


@pytest.fixture(scope="module")
def event_log_orders() -> list[tuple[object, ...]]:
    generated = synth.event_log(
        rng("sql/sql-group-by-and-having/p01-customers-with-three-completed-orders:event-log:0"),
        3_000,
        n_users=100,
        categories=tuple(STATUSES),
        start="2024-01-01",
        days=90,
        amount_median=30.0,
        amount_sigma=1.0,
        dirty_rate=0.15,
    )
    return [
        (
            int(order_id),
            int(customer_id),
            pd.Timestamp(stamp).date(),
            str(status),
            float(amount),
        )
        for order_id, customer_id, stamp, status, amount in zip(
            generated["event_id"],
            generated["user_id"],
            generated["ts"],
            generated["category"],
            generated["amount"],
            strict=True,
        )
    ]


def _random_rows(rng: np.random.Generator, n: int) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for order_id in range(1, n + 1):
        customer_id = int(rng.integers(1, 12))
        status = STATUSES[int(rng.integers(0, len(STATUSES)))]
        # Multiples of 10 make equal totals common, which exercises the customer_id tie-break.
        amount = None if rng.random() < 0.2 else float(rng.integers(0, 8) * 10)
        order_date = DAY
        rows.append((order_id, customer_id, order_date, status, amount))
    shuffled = rng.permutation(len(rows))  # insertion order must not matter
    return [rows[int(i)] for i in shuffled]


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_random_tables_match_the_pandas_oracle(seed: int) -> None:
    rows = _random_rows(np.random.default_rng(seed), 90)
    _check_same_result(_run_learner_query(_connect(rows)), _oracle(rows))


def test_event_log_orders_match_the_pandas_aggregation_oracle(
    event_log_orders: list[tuple[object, ...]],
) -> None:
    _check_same_result(_run_learner_query(_connect(event_log_orders)), _oracle(event_log_orders))


def test_exactly_three_completed_orders_is_kept_and_two_is_not() -> None:
    rows = [
        (1, 1, DAY, "completed", 10.0),
        (2, 1, DAY, "completed", 20.0),
        (3, 1, DAY, "completed", 30.0),
        (4, 2, DAY, "completed", 100.0),
        (5, 2, DAY, "completed", 100.0),
    ]
    got = _run_learner_query(_connect(rows))
    _check_same_result(got, pd.DataFrame([(1, 3, 60.0)], columns=COLUMNS))


def test_non_completed_orders_do_not_count_toward_the_threshold() -> None:
    rows = [
        (1, 1, DAY, "completed", 5.0),
        (2, 1, DAY, "completed", 5.0),
        (3, 1, DAY, "cancelled", 5.0),
        (4, 1, DAY, "cancelled", 5.0),
        (5, 1, DAY, "refunded", 5.0),
        (6, 2, DAY, "completed", 7.0),
        (7, 2, DAY, "completed", 7.0),
        (8, 2, DAY, "completed", 7.0),
        (9, 2, DAY, "pending", 7.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert got["customer_id"].tolist() == [2], (
        f"got customers {got['customer_id'].tolist()}. A customer with only two completed orders "
        "was kept or dropped wrongly: cancelled and refunded rows must be filtered by WHERE "
        "before counting, not counted by HAVING."
    )
    _check_same_result(got, pd.DataFrame([(2, 3, 21.0)], columns=COLUMNS))


def test_status_match_is_exact_not_case_insensitive() -> None:
    rows = [
        (1, 1, DAY, "Completed", 10.0),
        (2, 1, DAY, "COMPLETED", 10.0),
        (3, 1, DAY, "completed", 10.0),
        (4, 1, DAY, "completed", 10.0),
        (5, 2, DAY, "completed", 7.0),
        (6, 2, DAY, "completed", 7.0),
        (7, 2, DAY, "completed", 7.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert got["customer_id"].tolist() == [2], (
        f"got customers {got['customer_id'].tolist()}. status must equal 'completed' exactly: "
        "a customer with only two orders of that exact status must not be kept, so another "
        "capitalisation must not count (no ILIKE or LOWER)."
    )
    _check_same_result(got, pd.DataFrame([(2, 3, 21.0)], columns=COLUMNS))


def test_null_amount_counts_as_an_order_but_sum_skips_it() -> None:
    rows = [
        (1, 1, DAY, "completed", 40.0),
        (2, 1, DAY, "completed", None),
        (3, 1, DAY, "completed", 10.0),
    ]
    got = _run_learner_query(_connect(rows))
    _check_same_result(got, pd.DataFrame([(1, 3, 50.0)], columns=COLUMNS))


def test_group_whose_amounts_are_all_null_has_null_total_and_sorts_last() -> None:
    rows = [
        (1, 1, DAY, "completed", None),
        (2, 1, DAY, "completed", None),
        (3, 1, DAY, "completed", None),
        (4, 2, DAY, "completed", 10.0),
        (5, 2, DAY, "completed", 10.0),
        (6, 2, DAY, "completed", 10.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert got["customer_id"].tolist() == [2, 1], (
        f"got customers {got['customer_id'].tolist()}: "
        "a NULL total sorts after every number (ORDER BY total_amount DESC NULLS LAST)"
    )

    assert pd.isna(got["total_amount"].iloc[1]), (
        "total of an all-NULL group should be NULL, not 0: SUM of nothing is NULL"
    )
    _check_same_result(got, pd.DataFrame([(2, 3, 30.0), (1, 3, None)], columns=COLUMNS))


def test_equal_totals_are_ordered_by_customer_id_even_when_inserted_out_of_order() -> None:
    # Large, non-sequential ids: without an explicit customer_id key the engine returns the
    # tied groups in an arbitrary order, which these ids expose.
    tied_ids = [905_113_208, 17_204, 6_000_000_001, 42, 310_552_910_887]
    rows: list[tuple[object, ...]] = []
    order_id = 1
    for customer in sorted(tied_ids, reverse=True):
        for _ in range(3):
            rows.append((order_id, customer, DAY, "completed", 10.0))
            order_id += 1
    rows += [(order_id + i, 7_001, DAY, "completed", 5.0) for i in range(3)]
    got = _run_learner_query(_connect(rows))
    want = [(c, 3, 30.0) for c in sorted(tied_ids)] + [(7_001, 3, 15.0)]
    _check_same_result(got, pd.DataFrame(want, columns=COLUMNS))


def test_each_customer_is_one_row_with_all_of_its_completed_orders() -> None:
    rows = [(i, 5, DAY, "completed", 2.0) for i in range(1, 51)]
    rows += [(i, 5, DAY, "cancelled", 2.0) for i in range(51, 61)]
    got = _run_learner_query(_connect(rows))
    _check_same_result(got, pd.DataFrame([(5, 50, 100.0)], columns=COLUMNS))


def test_no_customer_qualifies_returns_zero_rows_with_the_three_columns() -> None:
    rows = [
        (1, 1, DAY, "completed", 10.0),
        (2, 1, DAY, "completed", 10.0),
        (3, 2, DAY, "refunded", 10.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert list(got.columns) == COLUMNS, (
        f"columns differ on an empty result: expected {COLUMNS}, got {list(got.columns)}"
    )
    assert len(got) == 0, f"expected no customers, got {len(got)}: check the threshold"
