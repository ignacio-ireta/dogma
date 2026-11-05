"""Public tests: the fixture's answer, output columns, the threshold, and the total tie-break."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "orders.csv"
COLUMNS = ["customer_id", "completed_orders", "total_amount"]


def _load_fixture() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE orders AS SELECT * FROM read_csv_auto(?)", [str(FIXTURE)])
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


def _oracle(orders: pd.DataFrame) -> pd.DataFrame:
    """Independent answer, computed with pandas from the same rows."""
    done = orders[orders["status"] == "completed"]
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
    for customer in sorted(set(got_by) | set(want_by), key=repr):
        got, want = got_by.get(customer), want_by.get(customer)
        if want is None:
            return (
                f"customer {customer} should not appear: count only status = 'completed' rows "
                "and keep groups with COUNT(*) >= 3"
            )
        if got is None:
            return f"customer {customer} is missing: check the threshold (>= 3, not > 3)"
        if got[1] != want[1]:
            return (
                f"customer {customer}: completed_orders is {got[1]}, expected {want[1]}. "
                "Count every completed order with COUNT(*), including orders whose amount is NULL"
            )
        if not _same_total(got[2], want[2]):
            return (
                f"customer {customer}: total_amount is {got[2]}, expected {want[2]}. "
                "SUM skips NULL amounts; the total is the sum of the known amounts"
            )
    return None


def _check_same_result(got: pd.DataFrame, want: pd.DataFrame) -> None:
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}. "
        "Select customer_id, COUNT(*) as completed_orders, SUM(amount) as total_amount."
    )
    got_rows, want_rows = _rows(got), _rows(want)
    assert len(got_rows) == len(want_rows), (
        f"row count differs: expected {len(want_rows)} customers, got {len(got_rows)}. "
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


def test_customers_with_three_or_more_completed_orders_match_the_pandas_answer() -> None:
    fixture = pd.read_csv(FIXTURE)
    _check_same_result(_run_learner_query(_load_fixture()), _oracle(fixture))


def test_output_has_exactly_the_three_requested_columns_in_order() -> None:
    got = _run_learner_query(_load_fixture())
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}"
    )


def test_only_completed_orders_count_and_a_null_amount_still_counts_as_an_order() -> None:
    got = _run_learner_query(_load_fixture())
    ids = set(got["customer_id"].tolist())
    assert not {203, 205, 209} & ids, (
        "customers 203, 205 and 209 have fewer than three completed orders and must not appear"
    )
    row = got.loc[got["customer_id"] == 201]
    assert len(row) == 1 and int(row["completed_orders"].iloc[0]) == 4, (
        "customer 201 has four completed orders, one with a NULL amount: "
        "use COUNT(*), not COUNT(amount)"
    )


def test_equal_totals_are_ordered_by_customer_id_ascending() -> None:
    got = _run_learner_query(_load_fixture())
    customers = got["customer_id"].tolist()
    assert customers.index(202) < customers.index(210), (
        "customers 202 and 210 both total 300.00: order ties by customer_id ASC"
    )
