"""Public tests: the fixture's answer, the column list, the strict comparison and the tie-break."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd  # type: ignore[import-untyped]
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
COLUMNS = ["customer_id", "region", "total_spend", "average_spend"]
EXPECTED = [
    (305, "north", 300.0, 180.0),
    (306, "south", 300.0, 180.0),
    (303, "north", 240.0, 180.0),
]


def _load_fixture() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE customers AS SELECT * FROM read_csv_auto(?)",
        [str(FIXTURES / "customers.csv")],
    )
    con.execute(
        "CREATE TABLE orders AS SELECT * FROM read_csv_auto(?)", [str(FIXTURES / "orders.csv")]
    )
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


def _plain(value: object) -> object:
    """A DuckDB or pandas value as a plain Python value; NULL becomes None."""
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


def test_fixture_returns_the_three_customers_above_the_average_in_order() -> None:
    got = _run_learner_query(_load_fixture())
    rows = [tuple(_plain(v) for v in row) for row in got.itertuples(index=False, name=None)]
    assert [row[0] for row in rows] == [row[0] for row in EXPECTED], (
        "wrong customers or wrong order: expected customers 305, 306 and 303, in that order. "
        "Check the strict comparison, the buyer filter, and the total_spend DESC order"
    )
    for got_row, want_row in zip(rows, EXPECTED, strict=True):
        assert all(_same_value(g, w) for g, w in zip(got_row, want_row, strict=True)), (
            f"customer {want_row[0]}: a value differs. total_spend sums completed amounts only, "
            "and average_spend is the average of the buyers' totals"
        )


def test_output_has_exactly_the_four_requested_columns_in_order() -> None:
    got = _run_learner_query(_load_fixture())
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}"
    )


def test_customer_whose_total_equals_the_average_is_not_returned() -> None:
    got = _run_learner_query(_load_fixture())
    assert 307 not in got["customer_id"].tolist(), (
        "customer 307 totals exactly the average (180.00) and must not be returned: "
        "the comparison is strict, so use >, not >="
    )


def test_tied_totals_are_ordered_by_customer_id_ascending() -> None:
    got = _run_learner_query(_load_fixture())
    customers = got["customer_id"].tolist()
    assert 305 in customers and 306 in customers, (
        "customers 305 and 306 both belong in the result: check the buyer filter and the average"
    )
    assert customers.index(305) < customers.index(306), (
        "customers 305 and 306 both total 300.00: order ties by customer_id ascending"
    )
