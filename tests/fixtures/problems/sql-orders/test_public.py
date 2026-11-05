from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ORDERS = pd.read_csv(HERE / "fixtures" / "orders.csv")


def _run(orders: pd.DataFrame) -> pd.DataFrame:
    sql = Path(__file__).with_name("submission.sql").read_text(encoding="utf-8")
    con = duckdb.connect(":memory:")
    try:
        con.register("orders", orders)
        return con.execute(sql).df()
    finally:
        con.close()


def _expected(orders: pd.DataFrame) -> pd.DataFrame:
    large = orders[orders["amount"] >= 10]
    grouped = large.groupby("country", as_index=False)["amount"].sum()
    return grouped.rename(columns={"amount": "total"}).sort_values("country", ignore_index=True)


def test_output_has_exactly_country_and_total_columns():
    got = _run(ORDERS)
    assert list(got.columns) == ["country", "total"], (
        f"columns should be ['country', 'total'], got {list(got.columns)}"
    )


def test_total_per_country_counts_orders_of_at_least_ten():
    got = _run(ORDERS).sort_values("country", ignore_index=True)
    expected = _expected(ORDERS)
    assert got["country"].tolist() == expected["country"].tolist(), (
        f"countries should be {expected['country'].tolist()}, got {got['country'].tolist()}"
    )
    np.testing.assert_allclose(
        got["total"].to_numpy(dtype=float),
        expected["total"].to_numpy(dtype=float),
        err_msg="per-country totals of orders with amount >= 10 are wrong",
    )
