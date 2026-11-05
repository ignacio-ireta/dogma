"""Public tests: the fixture answer, output columns, the version rule and unknown customers."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
COLUMNS = ["customer_id", "customer_name", "order_count", "total_amount"]


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
    return result.fetchdf()


def _oracle() -> pd.DataFrame:
    """Independent answer, computed with pandas straight from the CSV files."""
    customers = pd.read_csv(FIXTURES / "customers.csv")
    orders = pd.read_csv(FIXTURES / "orders.csv")
    current = customers[customers["is_current"]]
    counts = orders.groupby("customer_id")["order_id"].count()
    totals = orders.groupby("customer_id")["amount"].sum()
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


def _check_same_result(got: pd.DataFrame, want: pd.DataFrame) -> None:
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}. "
        "Select customer_id, customer_name, COUNT(o.order_id) and COALESCE(SUM(o.amount), 0.0)."
    )
    got_rows, want_rows = _rows(got), _rows(want)
    assert len(got_rows) == len(want_rows), (
        "row count differs: expect one row per current customer. Keep customers with no orders "
        "(LEFT JOIN from customers) and drop the older versions (WHERE is_current)."
    )
    assert len({row[0] for row in got_rows}) == len(got_rows), (
        "a customer appears more than once: older versions are repeating orders; filter "
        "is_current before the join"
    )
    by_id = {row[0]: row for row in want_rows}
    for row in got_rows:
        expected = by_id.get(row[0])
        assert expected is not None, (
            "a customer appears that is not a current customer: keep only is_current rows"
        )
        assert row[1] == expected[1], (
            "customer_name differs: return the name on the current version"
        )
        if row[2] != expected[2] and expected[2] == 0:
            raise AssertionError(
                "a customer with no orders shows a count above zero: count the order key "
                "(COUNT(o.order_id)), not the joined rows (COUNT(*))"
            )
        assert row[2] == expected[2], (
            "order_count differs: count matched orders with COUNT(o.order_id), and do not "
            "count orders of unknown customers"
        )
        assert row[3] is not None or expected[3] is None, (
            "a customer with no orders has a NULL total: wrap the sum in COALESCE(..., 0.0)"
        )
        assert _close(row[3], expected[3]), (
            "total_amount differs: sum only the matched orders' amounts, once each"
        )
    assert got_rows == want_rows, "row order differs: ORDER BY total_amount DESC, customer_id ASC"


def _close(got: object, want: object) -> bool:
    if got is None or want is None:
        return got is want
    return abs(float(got) - float(want)) <= 1e-6  # type: ignore[arg-type]


def test_fixture_totals_match_the_pandas_answer() -> None:
    _check_same_result(_run_learner_query(_load_fixture()), _oracle())


def test_output_has_the_four_columns_in_order() -> None:
    got = _run_learner_query(_load_fixture())
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}"
    )


def test_customers_without_orders_show_zero_orders_and_a_zero_total() -> None:
    got = _run_learner_query(_load_fixture())
    row = got.loc[got["customer_id"] == 108]
    assert len(row) == 1, "customer 108 has no orders and must still appear once"
    assert int(row["order_count"].iloc[0]) == 0, (
        "a customer with no orders shows a non-zero count: count the order key, not the rows"
    )
    total = row["total_amount"].iloc[0]
    assert not pd.isna(total) and float(total) == 0.0, (
        "a customer with no orders shows a NULL total: wrap the sum in COALESCE(..., 0.0)"
    )


def test_customer_with_two_versions_appears_once_under_its_current_name() -> None:
    got = _run_learner_query(_load_fixture())
    rows = got.loc[got["customer_id"] == 101]
    assert len(rows) == 1, (
        "customer 101 has two versions and must appear once: filter is_current before the join"
    )
    assert rows["customer_name"].iloc[0] == "Bluefin Bakery & Cafe", (
        "customer_name is the name from an old version: return the current version's name"
    )
    assert int(rows["order_count"].iloc[0]) == 3, (
        "customer 101 has three orders; the count is doubled when old versions are joined"
    )


def test_orders_from_unknown_customers_are_excluded() -> None:
    con = _load_fixture()
    got = _run_learner_query(con)
    all_amounts = float(con.execute("SELECT SUM(amount) FROM orders").fetchone()[0])  # type: ignore[index]
    unknown = float(
        con.execute(
            "SELECT COALESCE(SUM(amount), 0.0) FROM orders WHERE customer_id NOT IN "
            "(SELECT customer_id FROM customers)"
        ).fetchone()[0]  # type: ignore[index]
    )
    assert abs(float(got["total_amount"].sum()) - (all_amounts - unknown)) <= 1e-6, (
        "orders of customers that are not in the customers table leaked into the totals: "
        "start from customers and LEFT JOIN orders onto it"
    )
    assert not got["customer_id"].isin([113, 120]).any(), (
        "an order of a customer id with no customer row appears in the result: "
        "start from customers and LEFT JOIN orders onto it"
    )
