"""Hidden tests: your query runs against tables that each test builds itself.

A query that hard-codes the fixture's answer fails here. The oracle is an independent pandas
computation over the same rows. Each test name says what behaviour it checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dogma_data import retail_events
from dogma_data import rng as data_rng

COLUMNS = ["order_id", "customer_id", "order_date", "amount"]
ALL_COLUMNS = ["order_id", "customer_id", "order_date", "status", "amount"]
TABLE_DDL = (
    "CREATE TABLE orders (order_id BIGINT, customer_id BIGINT, order_date DATE, "
    "status VARCHAR, amount DOUBLE)"
)
INSERT = "INSERT INTO orders VALUES (?, ?, ?, ?, ?)"
STATUSES = ["completed", "completed", "pending", "cancelled", "refunded"]
OTHER_STATUSES = ["pending", "refunded", "Completed", "COMPLETED"]
BOUNDARY_DATES = [date(2023, 12, 31), date(2024, 1, 1), date(2024, 12, 31), date(2025, 1, 1)]


def _connect(rows: Sequence[tuple[object, ...]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(TABLE_DDL)
    con.executemany(INSERT, rows)
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


def _oracle(rows: list[tuple[object, ...]]) -> pd.DataFrame:
    """Independent answer: completed, dated in 2024, amount not NULL, top five by amount."""
    frame = pd.DataFrame(rows, columns=ALL_COLUMNS)
    frame["order_date"] = pd.to_datetime(frame["order_date"])
    keep = (
        (frame["status"] == "completed")
        & (frame["order_date"].dt.year == 2024)
        & frame["amount"].notna()
    )
    ranked = frame.loc[keep, COLUMNS].sort_values(["amount", "order_id"], ascending=[False, True])
    return ranked.head(5).reset_index(drop=True)


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
        f"row count differs: your query returned {len(got_rows)} rows. "
        "Check the status, 2024 and NULL-amount filters, and the LIMIT 5."
    )
    assert sorted(got_rows, key=repr) == sorted(want_rows, key=repr), (
        "wrong set of orders: check status = 'completed', order_date in 2024 (both ends), "
        "and amount IS NOT NULL, then ORDER BY amount DESC before the LIMIT."
    )
    assert got_rows == want_rows, "row order differs: did you ORDER BY amount DESC, order_id ASC?"


def _random_rows(rng: np.random.Generator, n: int) -> list[tuple[object, ...]]:
    """n real orders: invoice lines of an online retailer (online_retail), in shuffled order.

    The order id is the line's number, the customer is the real customer and the amount is the
    line total: a product sells at a few price points, so equal amounts are common, which
    exercises the order_id tie-break. A sold line is ``completed`` and a line of a cancellation
    is ``cancelled``, with a negative amount. The real dates run from December 2009 to December
    2011 and are moved forward by 14 years, so that 2010 becomes 2024. Three things the source
    lacks and the contract has rules for are placed by the draw: one date in five is moved to a
    year boundary, one amount in seven is removed, and one status in seven is replaced by
    another status or another capitalisation.
    """
    events = retail_events()
    picked = events.iloc[rng.permutation(len(events))[:n]]
    rows: list[tuple[object, ...]] = []
    for order_id, customer_id, moment, cancelled, value in zip(
        picked["event_id"],
        picked["user_id"],
        picked["ts"],
        picked["cancelled"],
        picked["amount"],
        strict=True,
    ):
        if rng.random() < 0.2:
            order_date = BOUNDARY_DATES[int(rng.integers(0, len(BOUNDARY_DATES)))]
        else:
            order_date = date(moment.year + 14, moment.month, moment.day)
        status = "cancelled" if cancelled else "completed"
        if rng.random() < 1.0 / 7.0:
            status = OTHER_STATUSES[int(rng.integers(0, len(OTHER_STATUSES)))]
        amount = None if rng.random() < 1.0 / 7.0 else float(value)
        rows.append((int(order_id), int(customer_id), order_date, status, amount))
    return rows


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_random_tables_match_the_independent_pandas_oracle(seed: int) -> None:
    rows = _random_rows(data_rng(f"top-five-completed-p01:random-tables:{seed}"), 120)
    _check_same_result(_run_learner_query(_connect(rows)), _oracle(rows))


def test_year_boundaries_include_2024_first_and_last_day_only() -> None:
    rows = [
        (1, 1, date(2023, 12, 31), "completed", 900.0),  # last day of 2023: out
        (2, 2, date(2024, 1, 1), "completed", 800.0),  # first day of 2024: in
        (3, 3, date(2024, 12, 31), "completed", 700.0),  # last day of 2024: in
        (4, 4, date(2025, 1, 1), "completed", 600.0),  # first day of 2025: out
    ]
    got = _run_learner_query(_connect(rows))
    assert sorted(got["order_id"].tolist()) == [2, 3], (
        f"year filter wrong: got orders {sorted(got['order_id'].tolist())}. "
        "Use order_date >= DATE '2024-01-01' AND order_date < DATE '2025-01-01'."
    )


def test_only_status_exactly_completed_qualifies() -> None:
    rows = [
        (1, 1, date(2024, 2, 1), "pending", 999.0),
        (2, 2, date(2024, 2, 2), "cancelled", 998.0),
        (3, 3, date(2024, 2, 3), "refunded", 997.0),
        (4, 4, date(2024, 2, 4), "processing", 996.0),
        (5, 5, date(2024, 2, 5), "completed", 10.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert got["order_id"].tolist() == [5], (
        f"status filter wrong: got orders {got['order_id'].tolist()}. "
        "Only status = 'completed' qualifies; pending, cancelled, refunded and processing do not."
    )


def test_status_match_is_exact_not_case_insensitive() -> None:
    rows = [
        (1, 1, date(2024, 2, 1), "Completed", 999.0),
        (2, 2, date(2024, 2, 2), "COMPLETED", 998.0),
        (3, 3, date(2024, 2, 3), "completed", 10.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert got["order_id"].tolist() == [3], (
        f"status match wrong: got orders {got['order_id'].tolist()}. status must equal "
        "'completed' exactly, so another capitalisation does not qualify (no ILIKE or LOWER)."
    )


def test_zero_amount_is_a_real_amount_and_qualifies() -> None:
    rows = [
        (1, 1, date(2024, 3, 1), "completed", 0.0),
        (2, 2, date(2024, 3, 2), "completed", 5.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert got["order_id"].tolist() == [2, 1], (
        f"got orders {got['order_id'].tolist()}. A zero amount is known, so its order qualifies "
        "and sorts last. The only amount test is amount IS NOT NULL, not amount > 0."
    )


def test_null_amount_orders_are_not_returned_when_few_orders_qualify() -> None:
    rows = [
        (1, 1, date(2024, 3, 1), "completed", 50.0),
        (2, 2, date(2024, 3, 2), "completed", None),
        (3, 3, date(2024, 3, 3), "completed", 75.0),
        (4, 4, date(2024, 3, 4), "completed", None),
        (5, 5, date(2024, 3, 5), "completed", 20.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert got["amount"].notna().all(), (
        "a NULL amount was returned: NULL is not a value to rank; add amount IS NOT NULL"
    )
    assert got["order_id"].tolist() == [3, 1, 5], (
        f"wrong orders returned: got {got['order_id'].tolist()}. Check the amount ordering "
        "and make sure NULL-amount orders are excluded."
    )


def test_ties_at_the_cutoff_keep_the_lowest_order_ids() -> None:
    # Inserted in reverse id order so that insertion order cannot explain a correct answer.
    rows = [(i, 1, date(2024, 5, 1), "completed", 50.0) for i in range(15, 9, -1)]
    rows += [
        (1, 2, date(2024, 5, 2), "completed", 200.0),
        (2, 3, date(2024, 5, 3), "completed", 200.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert got["order_id"].tolist() == [1, 2, 10, 11, 12], (
        f"tie-break wrong: got {got['order_id'].tolist()}. "
        "Within equal amounts, order_id ASC decides which orders make the cut."
    )


def test_fewer_than_five_qualifying_orders_returns_all_of_them() -> None:
    rows = [
        (7, 1, date(2024, 6, 1), "completed", 30.0),
        (8, 2, date(2024, 6, 2), "pending", 500.0),
        (9, 3, date(2024, 6, 3), "completed", 90.0),
        (10, 4, date(2024, 6, 4), "completed", 60.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert got["order_id"].tolist() == [9, 10, 7], (
        f"got {got['order_id'].tolist()}. All qualifying orders are returned, sorted by amount "
        "descending. A LIMIT must not pad or drop rows."
    )


def test_no_qualifying_orders_returns_zero_rows_with_the_four_columns() -> None:
    rows = [
        (1, 1, date(2023, 5, 1), "completed", 100.0),
        (2, 2, date(2024, 5, 2), "refunded", 100.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert list(got.columns) == COLUMNS, (
        f"columns differ on an empty result: expected {COLUMNS}, got {list(got.columns)}"
    )
    assert len(got) == 0, f"expected no rows, got {len(got)}: check every filter"
