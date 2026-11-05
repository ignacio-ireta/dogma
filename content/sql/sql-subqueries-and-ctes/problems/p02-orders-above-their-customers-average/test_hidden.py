"""Hidden tests: your query runs against tables that each test builds itself.

The oracle is an independent pandas computation over the same rows. Failure messages name the kind
of mistake; none of them carries an id, a value or an expected answer.
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

COLUMNS = ["order_id", "customer_id", "amount", "customer_average"]
ORDERS_DDL = (
    "CREATE TABLE orders (order_id BIGINT, customer_id BIGINT, order_date DATE, "
    "status VARCHAR, amount DOUBLE)"
)
DAY = date(2024, 1, 1)
STATUSES = ["completed", "completed", "completed", "pending", "cancelled", "refunded", "Completed"]
CUSTOMER_IDS = [1, 2, 3, 17, 42, 1_204, 905_113_208, 6_000_000_001]


def _connect(orders: Sequence[tuple[object, ...]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(ORDERS_DDL)
    if orders:
        con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", list(orders))
    return con


def _run_learner_query(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = Path(__file__).with_name("submission.sql").read_text()
    try:
        result = con.execute(sql)
    except duckdb.Error as exc:
        pytest.fail(f"your SQL did not run: {exc}")
    if result is None:
        pytest.fail("submission.sql returned no result set: finish it with a SELECT")
    frame = result.fetchdf()
    assert list(frame.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(frame.columns)}. "
        "Select order_id, customer_id, amount and customer_average, in that order."
    )
    return frame


def _oracle(orders: Sequence[tuple[object, ...]]) -> pd.DataFrame:
    """Independent answer in pandas: per-customer mean of completed amounts, then the filter."""
    frame = pd.DataFrame(
        orders, columns=["order_id", "customer_id", "order_date", "status", "amount"]
    ).astype({"customer_id": "Int64"})
    done = frame[(frame["status"] == "completed") & frame["customer_id"].notna()]
    means = done.groupby("customer_id")["amount"].mean()  # skipna: NULL amounts are not averaged
    candidates = done[done["amount"].notna()].copy()
    candidates["customer_average"] = candidates["customer_id"].map(means).astype("float64")
    kept = candidates[candidates["amount"] > candidates["customer_average"]]
    ranked = kept.sort_values("order_id")
    return ranked[COLUMNS].reset_index(drop=True)


def _plain(value: object) -> object:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        value = value.item()
    return value


def _same_value(got: Any, want: Any) -> bool:
    if got is None or want is None:
        return got is None and want is None
    return math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-6)


def _rows(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [tuple(_plain(v) for v in row) for row in frame.itertuples(index=False, name=None)]


def _check_same_result(got: pd.DataFrame, want: pd.DataFrame) -> None:
    got_rows, want_rows = _rows(got), _rows(want)
    assert len(got_rows) == len(want_rows), (
        f"row count differs: your query returned {len(got_rows)} rows. Check the candidate filter, "
        "the average (the customer's own completed amounts, NULLs skipped) and the strict "
        "comparison."
    )
    got_ids = [row[0] for row in got_rows]
    want_ids = [row[0] for row in want_rows]
    assert sorted(got_ids, key=repr) == sorted(want_ids, key=repr), (
        "a qualifying order is missing or a non-qualifying one is returned: check the candidate "
        "filter (status exactly 'completed', amount not NULL, customer present) and the strict "
        "comparison against the customer's own average"
    )
    assert got_ids == want_ids, "row order differs: did you ORDER BY order_id ASC?"
    for got_row, want_row in zip(got_rows, want_rows, strict=True):
        assert got_row[1] == want_row[1], (
            "a row carries the wrong customer: customer_id must be the order's own customer"
        )
        assert _same_value(got_row[2], want_row[2]), (
            "amount differs: return the order's own amount for each qualifying order"
        )
        assert _same_value(got_row[3], want_row[3]), (
            "customer_average is wrong: average that customer's completed amounts only, with "
            "NULL amounts skipped, and include the order itself"
        )


def _random_orders(rng: np.random.Generator) -> list[tuple[object, ...]]:
    orders: list[tuple[object, ...]] = []
    for order_id in range(1, int(rng.integers(0, 61)) + 1):
        customer = None if rng.random() < 0.08 else CUSTOMER_IDS[int(rng.integers(0, 8))]
        status = STATUSES[int(rng.integers(0, len(STATUSES)))]
        # Multiples of 0.25 are exact in binary floating point, so equal-to-average cases are exact.
        amount = None if rng.random() < 0.2 else float(rng.integers(0, 801)) / 4.0
        orders.append((order_id, customer, DAY, status, amount))
    return [orders[int(i)] for i in rng.permutation(len(orders))]  # insertion order is free


def test_random_tables_match_the_pandas_oracle() -> None:
    # Fixed seeds, one mismatch reported at a time: the first failure names the mistake.
    for seed in range(60):
        orders = _random_orders(np.random.default_rng(seed))
        got = _run_learner_query(_connect(orders))
        _check_same_result(got, _oracle(orders))


def test_order_equal_to_its_own_customers_average_is_not_returned() -> None:
    orders = [
        (1, 1, DAY, "completed", 10.0),
        (2, 1, DAY, "completed", 20.0),
        (3, 1, DAY, "completed", 30.0),
        (4, 2, DAY, "completed", 45.0),
    ]
    got = _run_learner_query(_connect(orders))
    assert 2 not in got["order_id"].tolist(), (
        "an order equal to its customer's average was returned: the comparison must be strict (>)"
    )
    _check_same_result(got, pd.DataFrame([(3, 1, 30.0, 20.0)], columns=COLUMNS))


def test_average_uses_only_completed_orders_with_a_known_amount() -> None:
    orders = [
        (1, 1, DAY, "completed", 100.0),
        (2, 1, DAY, "completed", 120.0),
        (3, 1, DAY, "pending", 900.0),
        (4, 1, DAY, "refunded", 900.0),
        (5, 1, DAY, "completed", None),
    ]
    got = _run_learner_query(_connect(orders))
    row = got.loc[got["order_id"] == 2]
    assert len(row) == 1 and _same_value(row["customer_average"].iloc[0], 110.0), (
        "customer_average is wrong: average only the completed amounts with a value; pending, "
        "refunded and NULL rows must not move it"
    )
    _check_same_result(got, pd.DataFrame([(2, 1, 120.0, 110.0)], columns=COLUMNS))


def test_null_amount_is_never_returned_and_is_skipped_in_the_average() -> None:
    orders = [
        (1, 2, DAY, "completed", 10.0),
        (2, 2, DAY, "completed", None),
        (3, 2, DAY, "completed", 30.0),
    ]
    got = _run_learner_query(_connect(orders))
    assert 2 not in got["order_id"].tolist(), (
        "an order with no amount was returned: a NULL amount is never above an average"
    )
    _check_same_result(got, pd.DataFrame([(3, 2, 30.0, 20.0)], columns=COLUMNS))


def test_customer_with_a_single_known_amount_gives_zero_rows() -> None:
    orders = [(1, 1, DAY, "completed", 25.0), (2, 2, DAY, "completed", 40.0)]
    got = _run_learner_query(_connect(orders))
    assert len(got) == 0, (
        f"a single completed order equals its own average, so it must not be returned; "
        f"your query returned {len(got)} rows"
    )


def test_order_without_a_customer_is_never_returned_and_joins_no_average() -> None:
    orders = [
        (1, None, DAY, "completed", 500.0),
        (2, 3, DAY, "completed", 10.0),
        (3, 3, DAY, "completed", 30.0),
    ]
    got = _run_learner_query(_connect(orders))
    assert got["customer_id"].notna().all(), (
        "a row without a customer was returned: an order with a NULL customer_id belongs to no "
        "customer, so it is never a candidate"
    )
    _check_same_result(got, pd.DataFrame([(3, 3, 30.0, 20.0)], columns=COLUMNS))


def test_each_qualifying_order_is_its_own_row_not_one_row_per_customer() -> None:
    orders = [
        (1, 4, DAY, "completed", 0.0),
        (2, 4, DAY, "completed", 0.0),
        (3, 4, DAY, "completed", 0.0),
        (4, 4, DAY, "completed", 90.0),
        (5, 4, DAY, "completed", 100.0),
    ]
    got = _run_learner_query(_connect(orders))
    assert len(got) == 2, (
        f"each qualifying order must be its own row: your query returned {len(got)} rows. "
        "Do not GROUP BY customer in the final query"
    )
    _check_same_result(
        got,
        pd.DataFrame([(4, 4, 90.0, 38.0), (5, 4, 100.0, 38.0)], columns=COLUMNS),
    )


def test_output_is_ordered_by_order_id_even_when_inserted_out_of_order() -> None:
    # Each customer has one qualifying order; the ids are large and not sorted by insertion.
    qualifying = [905_113_208, 17_204, 6_000_000_001, 42]
    orders: list[tuple[object, ...]] = []
    for customer, order_id in enumerate(qualifying, start=1):
        orders.append((order_id, customer, DAY, "completed", 80.0))
        for offset in (1_000_000, 2_000_000, 3_000_000):
            orders.append((order_id + offset, customer, DAY, "completed", 0.0))
    got = _run_learner_query(_connect(orders))
    want = [
        (oid, customer, 80.0, 20.0)
        for customer, oid in sorted(enumerate(qualifying, 1), key=lambda p: p[1])
    ]
    _check_same_result(got, pd.DataFrame(want, columns=COLUMNS))


def test_capitalised_completed_is_not_a_candidate_or_part_of_the_average() -> None:
    orders = [
        (1, 9, DAY, "Completed", 1000.0),
        (2, 9, DAY, "completed", 10.0),
        (3, 9, DAY, "completed", 30.0),
    ]
    got = _run_learner_query(_connect(orders))
    _check_same_result(got, pd.DataFrame([(3, 9, 30.0, 20.0)], columns=COLUMNS))


def test_no_order_above_its_customers_average_gives_zero_rows_with_four_columns() -> None:
    orders = [(1, 1, DAY, "completed", 5.0), (2, 1, DAY, "completed", 5.0)]
    got = _run_learner_query(_connect(orders))
    _check_same_result(got, pd.DataFrame([], columns=COLUMNS))
