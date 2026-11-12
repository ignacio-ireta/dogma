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

from dogma_data import retail_events
from dogma_data import rng as data_rng

COLUMNS = ["customer_id", "region", "total_spend", "average_spend"]
CUSTOMERS_DDL = "CREATE TABLE customers (customer_id BIGINT, region VARCHAR)"
ORDERS_DDL = (
    "CREATE TABLE orders (order_id BIGINT, customer_id BIGINT, order_date DATE, "
    "status VARCHAR, amount DOUBLE)"
)
DAY = date(2024, 1, 1)
STATUSES = ["completed", "completed", "completed", "pending", "cancelled", "refunded", "Completed"]
OTHER_STATUSES = ["pending", "refunded", "Completed"]
# Large, non-sequential ids: without an explicit ORDER BY the engine returns groups in an
# arbitrary order, and these ids expose that.
CUSTOMER_IDS = [3, 17, 42, 1_204, 17_204, 905_113_208, 6_000_000_001, 310_552_910_887]


def _connect(
    customers: Sequence[tuple[object, ...]], orders: Sequence[tuple[object, ...]]
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(CUSTOMERS_DDL)
    con.execute(ORDERS_DDL)
    if customers:
        con.executemany("INSERT INTO customers VALUES (?, ?)", list(customers))
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
        "Select customer_id, region, total_spend and average_spend, in that order."
    )
    return frame


def _oracle(
    customers: Sequence[tuple[object, ...]], orders: Sequence[tuple[object, ...]]
) -> pd.DataFrame:
    """Independent answer in pandas: buyers, NULL-skipping totals, average over buyers, filter."""
    cust = pd.DataFrame(customers, columns=["customer_id", "region"]).astype(
        {"customer_id": "Int64"}
    )
    ords = pd.DataFrame(
        orders, columns=["order_id", "customer_id", "order_date", "status", "amount"]
    ).astype({"customer_id": "Int64"})
    done = ords[(ords["status"] == "completed") & ords["customer_id"].notna()]
    totals = (
        done.groupby("customer_id")["amount"].sum(min_count=1).rename("total_spend").reset_index()
    )
    buyers = totals.merge(cust, on="customer_id", how="inner")
    average = buyers["total_spend"].mean()  # skipna: a NULL total is not in the average
    kept = buyers.assign(average_spend=average)
    kept = kept[kept["total_spend"] > average]  # a NaN average keeps nothing
    ranked = kept.sort_values(["total_spend", "customer_id"], ascending=[False, True])
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
    if isinstance(got, str) or isinstance(want, str):
        return got == want
    return math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-6)


def _rows(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [tuple(_plain(v) for v in row) for row in frame.itertuples(index=False, name=None)]


def _explain(got_rows: list[tuple[Any, ...]], want_rows: list[tuple[Any, ...]]) -> str | None:
    got_by = {row[0]: row for row in got_rows}
    want_by = {row[0]: row for row in want_rows}
    for customer in sorted(set(got_by) | set(want_by), key=repr):
        got, want = got_by.get(customer), want_by.get(customer)
        if want is None:
            return (
                "a customer was returned who should not be: check the buyer filter (completed "
                "orders with a customer), the strict > comparison, and the NULL rules"
            )
        if got is None:
            return (
                "a qualifying customer is missing: check the strict comparison (> not >=) and "
                "whether the average is taken over the right buyers"
            )
        if got[1] != want[1]:
            return "a customer's region is wrong: take region from the customers table"
        if not _same_value(got[2], want[2]):
            return (
                "a customer's total_spend is wrong: sum only completed orders, skip NULL amounts, "
                "and keep the total NULL when every completed amount is NULL"
            )
        if not _same_value(got[3], want[3]):
            return (
                "average_spend is wrong: average the buyers' totals with NULL totals skipped, "
                "not the individual order amounts and not every customer"
            )
    return None


def _check_same_result(got: pd.DataFrame, want: pd.DataFrame) -> None:
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}. "
        "Select customer_id, region, total_spend and average_spend, in that order."
    )
    got_rows, want_rows = _rows(got), _rows(want)
    assert len(got_rows) == len(want_rows), (
        f"row count differs: your query returned {len(got_rows)} rows. Check the buyer filter, "
        "the strict comparison and the NULL rules."
    )
    assert len({row[0] for row in got_rows}) == len(got_rows), (
        "a customer appears more than once: the final query must return one row per buyer"
    )
    problem = _explain(got_rows, want_rows)
    assert problem is None, problem
    assert [row[0] for row in got_rows] == [row[0] for row in want_rows], (
        "row order differs: did you ORDER BY total_spend DESC, customer_id ASC?"
    )


def _real_orders(rng: np.random.Generator) -> tuple[list[tuple[object, ...]], dict[int, str]]:
    """Up to sixty real orders, and the country of each customer id that stands in them.

    The orders are invoice lines of eight customers of an online retailer (online_retail), each
    standing behind one of the test's large customer ids. A few customers hold most of the
    lines. A sold line is ``completed`` and a line of a cancellation is ``cancelled``. The
    amount is the size of the line total, rounded to the nearest 0.25: the contract's amounts
    are multiples of 0.25. Three things the source lacks and the contract has rules for are
    placed by the draw: one order in twelve has no customer, one amount in five is removed, and
    one status in five is replaced by another status or another capitalisation.
    """
    events = retail_events()
    real = rng.choice(np.sort(events["user_id"].unique()), size=len(CUSTOMER_IDS), replace=False)
    stand_in = {int(customer): cid for customer, cid in zip(real, CUSTOMER_IDS, strict=True)}
    lines = events[events["user_id"].isin(real)]
    picked = lines.iloc[rng.permutation(len(lines))[: int(rng.integers(0, 61))]]
    country = {
        stand_in[int(customer)]: str(place)
        for customer, place in zip(lines["user_id"], lines["country"], strict=True)
    }
    orders: list[tuple[object, ...]] = []
    for order_id, (customer, cancelled, value) in enumerate(
        zip(picked["user_id"], picked["cancelled"], picked["amount"], strict=True), start=1
    ):
        owner = None if rng.random() < 0.08 else stand_in[int(customer)]
        status = "cancelled" if cancelled else "completed"
        if rng.random() < 0.2:
            status = OTHER_STATUSES[int(rng.integers(0, len(OTHER_STATUSES)))]
        amount = None if rng.random() < 0.2 else round(abs(float(value)) * 4.0) / 4.0
        orders.append((order_id, owner, DAY, status, amount))
    return [
        orders[int(i)] for i in rng.permutation(len(orders))
    ], country  # insertion order is free


def _random_tables(
    rng: np.random.Generator,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    """Real orders, and a customer table that lists three to eight of the eight customers.

    The region of a customer is its real country. An order of a customer that is not listed
    belongs to no customer row.
    """
    orders, country = _real_orders(rng)
    size = int(rng.integers(3, 9))
    ids = [int(i) for i in rng.choice(CUSTOMER_IDS, size=size, replace=False)]
    customers: list[tuple[object, ...]] = [(cid, country[cid]) for cid in ids]
    customers = [customers[int(i)] for i in rng.permutation(len(customers))]
    return customers, orders


def test_random_tables_match_the_pandas_oracle() -> None:
    # Fixed seeds, one mismatch reported at a time: the first failure names the mistake.
    for seed in range(60):
        customers, orders = _random_tables(data_rng(f"above-average-spend-p01:tables:{seed}"))
        got = _run_learner_query(_connect(customers, orders))
        _check_same_result(got, _oracle(customers, orders))


def test_buyer_whose_total_equals_the_average_is_not_returned_but_one_just_above_is() -> None:
    customers = [(1, "north"), (2, "south"), (3, "north"), (4, "south"), (5, "north")]
    orders = [
        (1, 1, DAY, "completed", 100.0),
        (2, 2, DAY, "completed", 150.0),
        (3, 2, DAY, "completed", 50.0),
        (4, 3, DAY, "completed", 300.0),
        (5, 4, DAY, "completed", 200.25),
        (6, 5, DAY, "completed", 199.75),
    ]
    got = _run_learner_query(_connect(customers, orders))
    assert 2 not in got["customer_id"].tolist(), (
        "a buyer whose total equals the average was returned: the comparison must be strict (>)"
    )
    assert 4 in got["customer_id"].tolist(), (
        "a buyer just above the average was dropped: check the comparison direction"
    )
    _check_same_result(
        got,
        pd.DataFrame(
            [(3, "north", 300.0, 200.0), (4, "south", 200.25, 200.0)],
            columns=COLUMNS,
        ),
    )


def test_buyer_whose_completed_amounts_are_all_null_is_left_out_of_the_average() -> None:
    customers = [(1, "north"), (2, "south"), (3, "north")]
    orders = [
        (1, 1, DAY, "completed", 10.0),
        (2, 2, DAY, "completed", 40.0),
        (3, 3, DAY, "completed", None),
        (4, 3, DAY, "completed", None),
    ]
    got = _run_learner_query(_connect(customers, orders))
    average = got["average_spend"].iloc[0] if len(got) else None
    assert average is not None and math.isclose(float(average), 25.0, abs_tol=1e-6), (
        "average_spend is wrong: a buyer whose completed amounts are all NULL has no total, so "
        "it must not count as 0 in the average"
    )
    _check_same_result(got, pd.DataFrame([(2, "south", 40.0, 25.0)], columns=COLUMNS))


def test_customers_with_no_completed_order_are_not_in_the_average() -> None:
    customers = [(1, "north"), (2, "south"), (3, "north"), (4, "south"), (5, "north")]
    orders = [
        (1, 1, DAY, "completed", 10.0),
        (2, 2, DAY, "completed", 30.0),
        (3, 3, DAY, "pending", 500.0),
        (4, 4, DAY, "cancelled", 500.0),
    ]
    got = _run_learner_query(_connect(customers, orders))
    average = got["average_spend"].iloc[0] if len(got) else None
    assert average is not None and math.isclose(float(average), 20.0, abs_tol=1e-6), (
        "average_spend is wrong: customers without a completed order are not buyers, so they "
        "are not in the average and do not count as 0"
    )
    _check_same_result(got, pd.DataFrame([(2, "south", 30.0, 20.0)], columns=COLUMNS))


def test_order_whose_customer_has_no_customer_row_is_not_a_buyer_or_in_the_average() -> None:
    customers = [(1, "north"), (2, "south")]
    orders = [
        (1, 1, DAY, "completed", 10.0),
        (2, 2, DAY, "completed", 30.0),
        (3, 99, DAY, "completed", 500.0),  # customer 99 has no row in customers
    ]
    got = _run_learner_query(_connect(customers, orders))
    average = got["average_spend"].iloc[0] if len(got) else None
    assert average is not None and math.isclose(float(average), 20.0, abs_tol=1e-6), (
        "an order whose customer_id has no row in customers was counted in the average: only "
        "customers in the customers table are buyers"
    )
    _check_same_result(got, pd.DataFrame([(2, "south", 30.0, 20.0)], columns=COLUMNS))


def test_average_is_over_customer_totals_not_over_individual_order_amounts() -> None:
    customers = [(1, "north"), (2, "south"), (3, "north")]
    orders = [(i, 1, DAY, "completed", 5.0) for i in range(1, 5)]
    orders += [(5, 2, DAY, "completed", 100.0), (6, 3, DAY, "completed", 40.0)]
    got = _run_learner_query(_connect(customers, orders))
    assert 3 not in got["customer_id"].tolist(), (
        "a customer below the average of customer totals was returned: average the customer "
        "totals, one value per buyer, not the individual order amounts"
    )
    _check_same_result(got, pd.DataFrame([(2, "south", 100.0, 160.0 / 3.0)], columns=COLUMNS))


def test_orders_without_a_customer_do_not_change_the_answer() -> None:
    customers = [(1, "north"), (2, "south")]
    orders = [
        (1, 1, DAY, "completed", 10.0),
        (2, 2, DAY, "completed", 30.0),
        (3, None, DAY, "completed", 500.0),
        (4, None, DAY, "pending", 40.0),
    ]
    got = _run_learner_query(_connect(customers, orders))
    assert got["customer_id"].notna().all(), (
        "a row without a customer was returned: an order with a NULL customer_id belongs to no "
        "customer, so exclude it explicitly"
    )
    assert len(got) == 1, (
        "a buyer is missing: orders without a customer must not change the average or empty the "
        "result. Check NULL customer ids in the average, and NOT IN lists (they return no rows "
        "when the list contains a NULL)"
    )
    _check_same_result(got, pd.DataFrame([(2, "south", 30.0, 20.0)], columns=COLUMNS))


def test_only_completed_orders_count_toward_a_customers_total() -> None:
    customers = [(1, "north"), (2, "south"), (3, "north"), (4, "south")]
    orders = [
        (1, 1, DAY, "completed", 10.0),
        (2, 1, DAY, "refunded", 1000.0),
        (3, 2, DAY, "pending", 500.0),
        (4, 3, DAY, "Completed", 300.0),
        (5, 4, DAY, "completed", 30.0),
    ]
    got = _run_learner_query(_connect(customers, orders))
    _check_same_result(got, _oracle(customers, orders))


def test_tied_totals_are_ordered_by_customer_id_even_when_inserted_out_of_order() -> None:
    tied_ids = [905_113_208, 17_204, 6_000_000_001, 42, 310_552_910_887]
    customers = [(cid, "north") for cid in sorted(tied_ids, reverse=True)]
    customers.append((7_001, "south"))
    orders: list[tuple[object, ...]] = []
    order_id = 1
    for cid in sorted(tied_ids, reverse=True):
        for _ in range(3):
            orders.append((order_id, cid, DAY, "completed", 100.0))
            order_id += 1
    orders.append((order_id, 7_001, DAY, "completed", 1.0))
    got = _run_learner_query(_connect(customers, orders))
    want = [(cid, "north", 300.0, (5 * 300.0 + 1.0) / 6.0) for cid in sorted(tied_ids)]
    _check_same_result(got, pd.DataFrame(want, columns=COLUMNS))


def test_no_buyer_above_the_average_returns_zero_rows_with_the_four_columns() -> None:
    customers = [(1, "north"), (2, "south")]
    orders = [(1, 1, DAY, "completed", 50.0), (2, 2, DAY, "completed", 50.0)]
    got = _run_learner_query(_connect(customers, orders))
    _check_same_result(got, pd.DataFrame([], columns=COLUMNS))


def test_no_buyer_with_a_known_total_returns_zero_rows_with_the_four_columns() -> None:
    customers = [(1, "north"), (2, "south")]
    orders = [(1, 1, DAY, "completed", None), (2, 2, DAY, "completed", None)]
    got = _run_learner_query(_connect(customers, orders))
    assert list(got.columns) == COLUMNS, (
        f"columns differ on an empty result: expected {COLUMNS}, got {list(got.columns)}"
    )
    assert len(got) == 0, (
        "a row was returned although no buyer has a known total: the average is NULL, and "
        "nothing compares above NULL"
    )


def test_a_buyer_with_many_orders_is_one_row_with_all_its_completed_orders() -> None:
    customers = [(1, "north"), (2, "south")]
    orders = [(i, 1, DAY, "completed", 2.0) for i in range(1, 51)]
    orders += [(i, 1, DAY, "cancelled", 2.0) for i in range(51, 61)]
    orders.append((61, 2, DAY, "completed", 10.0))
    got = _run_learner_query(_connect(customers, orders))
    _check_same_result(got, pd.DataFrame([(1, "north", 100.0, 55.0)], columns=COLUMNS))
