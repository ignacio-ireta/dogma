"""Public tests: the worked example, unknown customers, duplicates and NaN amounts."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from submission import monthly_revenue_by_region


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "customer_id": [10, 10, 20, 99, 20, 10],
            "order_date": [
                "2024-01-05",
                "2024-02-10",
                "2024-01-20",
                "2024-01-22",
                "2024-02-02",
                "2024-02-14",
            ],
            "amount": [100.0, 50.0, 30.0, 40.0, np.nan, 25.0],
        }
    )


def _customers() -> pd.DataFrame:
    return pd.DataFrame({"customer_id": [10, 20], "region": ["North", "South"]})


def _expected() -> pd.DataFrame:
    return pd.DataFrame(
        [[100.0, 75.0], [30.0, 0.0], [40.0, 0.0]],
        index=pd.Index(["North", "South", "unknown"], name="region"),
        columns=pd.Index(["2024-01", "2024-02"], name="month"),
        dtype="float64",
    )


def test_worked_example_matches_the_table_in_problem_md() -> None:
    got = monthly_revenue_by_region(_orders(), _customers())
    assert isinstance(got, pd.DataFrame), "the result must be a pandas DataFrame"
    assert got.index.name == "region" and got.columns.name == "month", (
        "index and columns must be named 'region' and 'month'"
    )
    assert list(got.index) == ["North", "South", "unknown"], (
        "region labels differ: regions are sorted ascending, and orders from customers missing "
        "from the table form their own 'unknown' region (an inner merge drops them)"
    )
    assert list(got.columns) == ["2024-01", "2024-02"], (
        "month labels differ: use 'YYYY-MM' strings, sorted ascending"
    )
    assert np.allclose(got.to_numpy(), _expected().to_numpy(), rtol=1e-9, atol=1e-9), (
        "revenue differs: sum the known amounts per region and month; missing cells are 0.0"
    )
    assert all(dtype == np.float64 for dtype in got.dtypes), (
        "a value column is not float64: cast amounts to float64"
    )


def test_orders_from_unknown_customers_go_to_the_unknown_region() -> None:
    got = monthly_revenue_by_region(_orders(), _customers())
    assert "unknown" in got.index, (
        "orders from customers that are not in the customers table must form the 'unknown' region; "
        "an inner merge drops them"
    )
    assert "2024-01" in got.columns, (
        "the January column is missing: months come from the known order dates"
    )
    assert got.loc["unknown", "2024-01"] == 40.0, (
        "the unknown region's January revenue is wrong: it should be the order from customer 99"
    )


def test_nan_amount_is_dropped_so_the_cell_is_zero_not_nan() -> None:
    got = monthly_revenue_by_region(_orders(), _customers())
    assert not got.isna().to_numpy().any(), (
        "NaN cells remain: a region-month cell with no known revenue must be 0.0"
    )
    assert "2024-02" in got.columns, "the February column is missing from the result"
    assert got.loc["South", "2024-02"] == 0.0, (
        "South's only February order has an unknown amount: its cell must be 0.0"
    )


def test_region_whose_only_order_has_no_amount_has_no_row() -> None:
    orders = pd.DataFrame(
        {
            "order_id": [1, 2],
            "customer_id": [10, 30],
            "order_date": ["2024-01-05", "2024-03-01"],
            "amount": [100.0, np.nan],
        }
    )
    customers = pd.DataFrame({"customer_id": [10, 30], "region": ["North", "East"]})
    got = monthly_revenue_by_region(orders, customers)
    assert list(got.index) == ["North"], (
        "a region with only unknown amounts must not appear: drop NaN amounts, do not fill them"
    )
    assert list(got.columns) == ["2024-01"], (
        "a month whose only order has an unknown amount must not appear as a column"
    )


def test_duplicate_customer_ids_raise_value_error() -> None:
    customers = pd.DataFrame({"customer_id": [10, 10], "region": ["North", "South"]})
    with pytest.raises(ValueError):
        monthly_revenue_by_region(_orders(), customers)
