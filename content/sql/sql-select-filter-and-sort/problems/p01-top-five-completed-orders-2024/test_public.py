"""Public tests: the fixture's answer, the output columns, a tie at the cutoff, NULL amounts."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "orders.csv"
COLUMNS = ["order_id", "customer_id", "order_date", "amount"]


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


def _oracle_top_five(fixture: Path) -> pd.DataFrame:
    """Independent answer, computed with pandas straight from the CSV."""
    orders = pd.read_csv(fixture, parse_dates=["order_date"])
    qualifying = orders[
        (orders["status"] == "completed")
        & (orders["order_date"].dt.year == 2024)
        & orders["amount"].notna()
    ]
    ranked = qualifying.sort_values(["amount", "order_id"], ascending=[False, True])
    return ranked[COLUMNS].head(5).reset_index(drop=True)


def _clean(value: object) -> object:
    """Make DuckDB and pandas values comparable: NULL -> None, dates -> ISO text."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, date):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
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
        "Select exactly these four columns, in this order (not SELECT *)."
    )
    got_rows, want_rows = _rows(got), _rows(want)
    assert len(got_rows) == len(want_rows), (
        f"row count differs: expected {len(want_rows)} rows, got {len(got_rows)}. "
        "Check the status, 2024 and NULL-amount filters, and the LIMIT 5."
    )
    assert sorted(got_rows, key=repr) == sorted(want_rows, key=repr), (
        f"wrong orders: got {got_rows}, expected {want_rows}. "
        "Check status = 'completed', order_date in 2024 (both ends), amount IS NOT NULL."
    )
    assert got_rows == want_rows, "row order differs: did you ORDER BY amount DESC, order_id ASC?"


def test_top_five_match_the_independent_pandas_answer_on_fixture() -> None:
    con = _load_fixture()
    _check_same_result(_run_learner_query(con), _oracle_top_five(FIXTURE))


def test_output_has_exactly_the_four_requested_columns_in_order() -> None:
    got = _run_learner_query(_load_fixture())
    assert list(got.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(got.columns)}. "
        "Name and order the columns exactly as the problem asks."
    )


def test_tie_at_the_cutoff_keeps_the_lowest_order_id() -> None:
    # Four completed 2024 orders are 250.00 and only one place is left: order 7 wins.
    got = _run_learner_query(_load_fixture())
    ids = got["order_id"].tolist()
    assert ids[-1] == 7, (
        f"tie at 250.00: expected order 7 in the last place, got order {ids[-1]}. "
        "Break amount ties with order_id ASC."
    )
    assert not {8, 18, 19} & set(ids), "tied orders 8, 18 and 19 should not reach the top five"


def test_null_amount_orders_are_never_returned() -> None:
    got = _run_learner_query(_load_fixture())
    assert got["amount"].notna().all(), "a NULL amount was returned: filter amount IS NOT NULL"
    assert not {9, 25, 32} & set(got["order_id"].tolist()), (
        "orders 9, 25 and 32 have no amount and must not be ranked"
    )
