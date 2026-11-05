"""Hidden tests: your query runs against tables that each test builds itself.

The oracle is an independent pandas computation over the same rows. Each test name says what
behaviour it checks, and each failure message names the likely mistake first.
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

COLUMNS = ["customer_id", "customer_name", "order_count", "total_amount"]
CUSTOMERS_DDL = (
    "CREATE TABLE customers (customer_id BIGINT, customer_name VARCHAR, is_current BOOLEAN)"
)
ORDERS_DDL = (
    "CREATE TABLE orders (order_id BIGINT, customer_id BIGINT, order_date DATE, amount DOUBLE)"
)
DAY = date(2024, 1, 1)
# Repeated names on purpose: two different customers can share a name.
NAMES = ["Oak Yard", "Pine Mill", "Harbor Supply Co", "Harbor Supply Co", "Cedar Works", "Ridge"]


def _connect(
    customers: Sequence[tuple[object, ...]], orders: Sequence[tuple[object, ...]]
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(CUSTOMERS_DDL)
    con.execute(ORDERS_DDL)
    if customers:
        con.executemany("INSERT INTO customers VALUES (?, ?, ?)", customers)
    if orders:
        con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
    return con


def _query(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame | None, str | None]:
    """Run the learner's query. Returns (frame, None), or (None, the mistake). Never raises."""
    sql = Path(__file__).with_name("submission.sql").read_text()
    try:
        result = con.execute(sql)
    except duckdb.Error as exc:
        return None, f"your SQL did not run: {exc}"
    if result is None:
        return None, "submission.sql returned no result set: finish it with a SELECT"
    return result.fetchdf(), None


def _run_learner_query(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    got, problem = _query(con)
    if problem is not None:
        pytest.fail(problem)
    return got


def _oracle(customers: list[tuple[Any, ...]], orders: list[tuple[Any, ...]]) -> pd.DataFrame:
    """Independent answer: current customers, order counts and sums, ties by customer id."""
    cust = pd.DataFrame(customers, columns=["customer_id", "customer_name", "is_current"])
    cust = cust.astype({"customer_id": "int64", "is_current": "bool"})
    ords = pd.DataFrame(orders, columns=["order_id", "customer_id", "order_date", "amount"])
    ords = ords.astype({"order_id": "int64", "customer_id": "int64", "amount": "float64"})
    current = cust[cust["is_current"]]
    counts = ords.groupby("customer_id")["order_id"].count()
    totals = ords.groupby("customer_id")["amount"].sum()
    out = current[["customer_id", "customer_name"]].copy()
    out["order_count"] = out["customer_id"].map(counts).fillna(0).astype("int64")
    out["total_amount"] = out["customer_id"].map(totals).fillna(0.0).astype("float64")
    ranked = out.sort_values(["total_amount", "customer_id"], ascending=[False, True])
    return ranked.reset_index(drop=True)


def _clean(value: object) -> object:
    """Make DuckDB and pandas values comparable: NULL -> None, floats rounded to 6 places."""
    if value is None or value is pd.NaT or value is pd.NA:
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


def _same_number(got: Any, want: Any) -> bool:
    if got is None or want is None:
        return got is want
    return math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-6)


def _explain_mismatch(  # noqa: PLR0911 (one return per kind of mistake, kept flat on purpose)
    got_rows: list[tuple[object, ...]], want_rows: list[tuple[object, ...]]
) -> str | None:
    got_by = {row[0]: row for row in got_rows}
    want_by = {row[0]: row for row in want_rows}
    for customer in sorted(set(got_by) | set(want_by), key=repr):
        got, want = got_by.get(customer), want_by.get(customer)
        if want is None:
            return (
                "a result row is not a current customer: keep only is_current customers "
                "(an order of an unknown customer must never appear)"
            )
        if got is None:
            return (
                "a current customer is missing: keep every current customer, "
                "even with no orders (LEFT JOIN from customers, not an inner join)"
            )
        if got[1] != want[1]:
            return "customer_name differs: return the name on the current version (is_current)"
        if got[2] != want[2]:
            if want[2] == 0 and got[2] != 0:
                return (
                    "a customer with no orders shows a count above zero: count the order key "
                    "(COUNT(o.order_id)), not the joined rows (COUNT(*))"
                )
            return (
                "order_count differs: count matched orders with COUNT(o.order_id); if counts "
                "are too high, old customer versions are repeating orders (filter is_current)"
            )
        if not _same_number(got[3], want[3]):
            if got[3] is None:
                return (
                    "a customer with no orders has a NULL total: wrap the sum in "
                    "COALESCE(SUM(o.amount), 0.0)"
                )
            return (
                "total_amount differs: sums are wrong; check that old customer versions do not "
                "repeat orders and that each order is summed once"
            )
    return None


def _first_problem(got: pd.DataFrame, want: pd.DataFrame, check_order: bool = True) -> str | None:  # noqa: PLR0911 (one return per kind of mistake, kept flat on purpose)
    """Return the first mistake found, or None. One return path keeps failure reports stable."""
    if list(got.columns) != COLUMNS:
        return f"output columns differ: expected {COLUMNS}, got {list(got.columns)}"
    got_rows, want_rows = _rows(got), _rows(want)
    if any(row[0] is None for row in got_rows):
        return (
            "a row has no customer_id (NULL): an order of an unknown customer leaked in. "
            "Start from customers and LEFT JOIN orders onto it, not the other way round"
        )
    if len({row[0] for row in got_rows}) != len(got_rows):
        return "a customer appears more than once: each current customer must be exactly one row"
    problem = _explain_mismatch(got_rows, want_rows)
    if problem is not None:
        return problem
    if len(got_rows) != len(want_rows):
        return "row count differs from the number of current customers: check join and filter"
    if check_order and got_rows != want_rows:
        return (
            "row order differs: ORDER BY total_amount DESC, customer_id ASC (ties by customer_id)"
        )
    return None


def _check_same_result(got: pd.DataFrame, want: pd.DataFrame, check_order: bool = True) -> None:
    problem = _first_problem(got, want, check_order)
    assert problem is None, problem


def _random_tables(
    rng: np.random.Generator, n_customers: int
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    ids = [int(x) for x in rng.choice(np.arange(1, 400), size=n_customers, replace=False)]
    customers: list[tuple[Any, ...]] = []
    for cid in ids:
        if rng.random() < 0.3:  # an older version of the same customer
            customers.append((cid, NAMES[int(rng.integers(0, len(NAMES)))], False))
        customers.append((cid, NAMES[int(rng.integers(0, len(NAMES)))], True))
    unknown = [int(x) for x in rng.choice(np.arange(500, 900), size=3, replace=False)]
    orders: list[tuple[Any, ...]] = []
    for order_id in range(1, int(rng.integers(0, 80)) + 1):
        pool = ids if rng.random() < 0.85 else unknown
        cid = pool[int(rng.integers(0, len(pool)))]
        # Quarters are exact in binary, so every sum is exact and ties are common.
        amount = float(rng.integers(0, 4000)) / 4.0
        orders.append((order_id, cid, DAY, amount))
    customers = [customers[int(i)] for i in rng.permutation(len(customers))]
    orders = [orders[int(i)] for i in rng.permutation(len(orders))]
    return customers, orders


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_random_tables_match_the_pandas_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    customers, orders = _random_tables(rng, int(rng.integers(1, 13)))
    got, problem = _query(_connect(customers, orders))
    if problem is None:
        problem = _first_problem(got, _oracle(customers, orders), check_order=False)
    assert problem is None, problem


def test_customer_without_orders_is_counted_as_zero_not_one() -> None:
    customers = [(1, "Oak Yard", True), (2, "Pine Mill", True)]
    orders = [(1, 1, DAY, 5.0)]
    got = _run_learner_query(_connect(customers, orders))
    _check_same_result(got, _oracle(customers, orders))


def test_every_current_customer_is_kept_when_none_of_them_has_orders() -> None:
    customers = [(cid, NAMES[i], True) for i, cid in enumerate([3, 9, 27, 81, 243])]
    orders = [(1, 500, DAY, 10.0), (2, 501, DAY, 20.0)]  # both orders are for unknown customers
    got = _run_learner_query(_connect(customers, orders))
    assert len(got) == 5, (
        f"a current customer is missing: expected 5 rows, got {len(got)}. "
        "An inner join drops customers with no orders; LEFT JOIN from customers keeps them"
    )
    _check_same_result(got, _oracle(customers, orders))


def test_old_customer_versions_do_not_repeat_orders() -> None:
    customers = [
        (7, "Former name", False),
        (7, "Oak Yard", False),
        (7, "Oak Yard and Co", True),
    ]
    orders = [(i, 7, DAY, 2.5) for i in range(1, 5)]
    got = _run_learner_query(_connect(customers, orders))
    _check_same_result(got, _oracle(customers, orders))


def test_orders_of_unknown_customers_never_appear_as_a_null_customer() -> None:
    customers = [(1, "Oak Yard", True)]
    orders = [(1, 1, DAY, 3.0), (2, 99, DAY, 40.0), (3, 98, DAY, 50.0)]
    got = _run_learner_query(_connect(customers, orders))
    assert got["customer_id"].notna().all(), (
        "a row has no customer_id (NULL): an order of an unknown customer leaked in. "
        "Use LEFT JOIN from customers, not a RIGHT or FULL join"
    )
    _check_same_result(got, _oracle(customers, orders))


def test_ties_are_ordered_by_customer_id_with_large_non_sequential_ids() -> None:
    # DuckDB often returns small sequential group keys in order, so the tie-break is only
    # tested for real when the ids are large and scattered.
    tied = [3_300_000_007, 700_001]
    zero = [905_113_208, 17_204, 6_000_000_001, 42, 310_552_910_887]
    customers = [(cid, "Cedar Works", True) for cid in reversed(zero + tied)]
    orders = [(i + 1, cid, DAY, 10.0) for i, cid in enumerate(reversed(tied))]
    got = _run_learner_query(_connect(customers, orders))
    _check_same_result(got, _oracle(customers, orders))


def test_empty_orders_table_gives_zero_orders_and_zero_totals_for_every_customer() -> None:
    customers = [(1, "Oak Yard", True), (2, "Pine Mill", True), (3, "Ridge", True)]
    got = _run_learner_query(_connect(customers, []))
    _check_same_result(got, _oracle(customers, []))


def test_no_customers_returns_zero_rows_with_the_four_columns() -> None:
    orders = [(1, 1, DAY, 3.0)]
    got = _run_learner_query(_connect([], orders))
    assert list(got.columns) == COLUMNS, (
        f"columns differ on an empty result: expected {COLUMNS}, got {list(got.columns)}"
    )
    assert len(got) == 0, f"orders with no customer row must not produce rows, got {len(got)} rows"


def test_customers_sharing_a_name_are_separate_rows() -> None:
    customers = [(10, "Harbor Supply Co", True), (11, "Harbor Supply Co", True)]
    orders = [(1, 10, DAY, 7.0), (2, 11, DAY, 1.0), (3, 11, DAY, 2.0)]
    got = _run_learner_query(_connect(customers, orders))
    assert len(got) == 2, (
        f"customers with the same name were merged: expected 2 rows, got {len(got)}. "
        "Group by customer_id (and name), never by name alone"
    )
    _check_same_result(got, _oracle(customers, orders))


def test_zero_order_customer_total_is_zero_not_null_next_to_non_zero_totals() -> None:
    customers = [(1, "Oak Yard", True), (2, "Pine Mill", True)]
    orders = [(1, 1, DAY, 12.5)]
    got = _run_learner_query(_connect(customers, orders))
    totals = got.set_index("customer_id")["total_amount"]
    assert not pd.isna(totals.loc[2]), (
        "a customer with no orders has a NULL total: wrap the sum in COALESCE(SUM(o.amount), 0.0)"
    )
    _check_same_result(got, _oracle(customers, orders))
