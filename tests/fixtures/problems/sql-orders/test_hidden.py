from pathlib import Path

import duckdb
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


def test_amount_exactly_ten_is_included():
    got = _run(ORDERS)
    fr = got.loc[got["country"] == "FR", "total"].tolist()
    assert fr == [10.0], f"the order of exactly 10 must count (threshold is inclusive), got {fr}"


def test_countries_without_qualifying_orders_are_absent():
    got = _run(ORDERS)
    assert "DE" not in set(got["country"]), "DE has only orders below 10 and must not appear"


def test_empty_orders_table_gives_no_rows():
    got = _run(ORDERS.iloc[0:0])
    assert len(got) == 0, f"no orders should give no rows, got {len(got)}"
