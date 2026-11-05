"""Public tests: the worked example from problem.md, string dates against datetimes, and a customer
whose orders all total zero. The hidden tests build their own tables."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest
from submission import customer_order_features

COLUMNS = [
    "order_id",
    "customer_id",
    "order_date",
    "amount",
    "order_seq",
    "share_of_customer_total",
    "days_since_previous_order",
    "customer_running_total",
]
EXPECTED_ORDER = [2, 7, 4, 9, 5]
EXPECTED_SEQ = [1, 2, 3, 1, 2]
EXPECTED_SHARE = [20 / 70, 40 / 70, 10 / 70, 1.0, 0.0]
EXPECTED_DAYS = [math.nan, 0.0, 17.0, math.nan, 58.0]
EXPECTED_RUNNING = [20.0, 60.0, 70.0, 30.0, 30.0]


def _example_orders(as_strings: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "order_id": [4, 7, 2, 9, 5],
            "customer_id": [10, 10, 10, 20, 20],
            "order_date": ["2024-02-01", "2024-01-15", "2024-01-15", "2024-01-03", "2024-03-01"],
            "amount": [10.0, 40.0, 20.0, 30.0, 0.0],
        }
    ).astype({"order_id": "int64", "customer_id": "int64", "amount": "float64"})
    if not as_strings:
        frame["order_date"] = pd.to_datetime(frame["order_date"])
    return frame


def _run(orders: pd.DataFrame) -> pd.DataFrame:
    try:
        return customer_order_features(orders)
    except Exception as exc:
        pytest.fail(
            f"customer_order_features raised {type(exc).__name__}: check the inputs and the "
            "output contract in problem.md"
        )


def _assert_column(got: pd.DataFrame, name: str, want: list[Any], rule: str) -> None:
    for index, (value, expected) in enumerate(zip(got[name].tolist(), want, strict=True)):
        if isinstance(expected, float) and math.isnan(expected):
            ok = pd.isna(value)
        else:
            ok = not pd.isna(value) and math.isclose(
                float(value), expected, rel_tol=1e-9, abs_tol=1e-9
            )
        assert ok, f"{name} differs in output row {index}: {rule}"


def test_worked_example_matches_problem_md() -> None:
    got = _run(_example_orders(as_strings=False))
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}"
    )
    assert got["order_id"].tolist() == EXPECTED_ORDER, (
        "row order differs: sort by customer_id, order_date, then order_id (order 2 comes before "
        "order 7 on the same date)"
    )
    assert str(got["order_seq"].dtype) == "int64", (
        f"order_seq dtype differs: expected int64, got {got['order_seq'].dtype}; build it with "
        "groupby(...).cumcount() + 1"
    )
    _assert_column(
        got, "order_seq", EXPECTED_SEQ, "number each customer's orders 1, 2, 3 in sort order"
    )
    _assert_column(
        got,
        "share_of_customer_total",
        EXPECTED_SHARE,
        "divide each amount by its customer's total over all of their orders",
    )
    _assert_column(
        got,
        "days_since_previous_order",
        EXPECTED_DAYS,
        "take the gap to the previous order of the same customer; NaN for a first order",
    )
    _assert_column(
        got,
        "customer_running_total",
        EXPECTED_RUNNING,
        "the running total restarts for each customer and includes the current order",
    )


def test_string_dates_give_the_same_features_as_datetimes() -> None:
    from_strings = _run(_example_orders(as_strings=True))
    from_datetimes = _run(_example_orders(as_strings=False))
    assert from_strings.equals(from_datetimes), (
        "string dates must be parsed to the same result as datetime values"
    )


def test_customer_whose_orders_all_total_zero_has_nan_share() -> None:
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "customer_id": [30, 30, 40],
            "order_date": pd.to_datetime(["2024-05-01", "2024-05-04", "2024-05-02"]),
            "amount": [0.0, 0.0, 6.0],
        }
    ).astype({"order_id": "int64", "customer_id": "int64", "amount": "float64"})
    got = _run(orders).set_index("order_id")
    assert pd.isna(got.loc[1, "share_of_customer_total"]) and pd.isna(
        got.loc[2, "share_of_customer_total"]
    ), (
        "share_of_customer_total is not NaN for a customer whose total is 0.0: 0.0 / 0.0 is NaN, "
        "so do not replace it with 0.0 or raise"
    )
    assert math.isclose(float(got.loc[3, "share_of_customer_total"]), 1.0), (
        "a customer with one non-zero order has share 1.0 for that order"
    )
