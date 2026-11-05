"""Public tests: the fixture's answer, the column list, the strict comparison and the NULL rules."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd  # type: ignore[import-untyped]
import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "orders.csv"
COLUMNS = ["order_id", "customer_id", "amount", "customer_average"]
EXPECTED = [
    (3, 501, 80.0, 50.0),
    (7, 503, 30.0, 20.0),
    (10, 504, 120.0, 110.0),
    (13, 505, 15.0, 35.0 / 3.0),
]


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
    frame = result.fetchdf()
    assert list(frame.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(frame.columns)}. "
        "Select order_id, customer_id, amount and customer_average, in that order."
    )
    return frame


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


def test_fixture_returns_the_four_qualifying_orders_in_order_id_order() -> None:
    got = _run_learner_query(_load_fixture())
    ids = [_plain(v) for v in got["order_id"].tolist()]
    assert ids == [row[0] for row in EXPECTED], (
        "wrong orders or wrong order: expected orders 3, 7, 10 and 13, in that order. Compare each "
        "order with its own customer's average, not one global average; check the strict "
        "comparison and ORDER BY order_id"
    )
    for got_row, want_row in zip(got.itertuples(index=False, name=None), EXPECTED, strict=True):
        values = [_plain(v) for v in got_row]
        assert all(_same_value(g, w) for g, w in zip(values, want_row, strict=True)), (
            f"order {want_row[0]}: a value differs. amount is the order's own amount, and "
            "customer_average is the average of its customer's completed amounts, NULLs skipped"
        )


def test_output_has_exactly_the_four_requested_columns_in_order() -> None:
    got = _run_learner_query(_load_fixture())
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}"
    )


def test_order_equal_to_its_customers_average_and_single_order_customer_are_not_returned() -> None:
    got = _run_learner_query(_load_fixture())
    ids = got["order_id"].tolist()
    assert 2 not in ids, (
        "order 2 equals its customer's average (50.00) and must not be returned: the comparison "
        "is strict, so use >, not >="
    )
    assert 4 not in ids, (
        "order 4 is customer 502's only completed order, so it equals its own average and must "
        "not be returned"
    )


def test_null_amount_is_skipped_in_the_average_and_never_returned() -> None:
    got = _run_learner_query(_load_fixture())
    assert 6 not in got["order_id"].tolist(), (
        "order 6 has no amount, so it cannot be above an average and must not be returned"
    )
    row = got.loc[got["order_id"] == 7]
    assert len(row) == 1 and _same_value(row["customer_average"].iloc[0], 20.0), (
        "customer 503 has amounts 10, NULL and 30: the average is 20, because the NULL is skipped "
        "and not counted as 0"
    )
