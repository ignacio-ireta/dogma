"""Hidden tests: random order tables and interview-prep orders checked against an independent
DuckDB window-function computation, plus edge cases. Each test name says what it checks, and each
failure message names the likely mistake first."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import customer_order_features

from dogma_data import retail_invoices

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
DTYPES = {
    "order_id": "int64",
    "customer_id": "int64",
    "amount": "float64",
    "order_seq": "int64",
    "share_of_customer_total": "float64",
    "days_since_previous_order": "float64",
    "customer_running_total": "float64",
}
DTYPE_HINTS = {
    "order_id": "keep order_id as int64, as in the input",
    "customer_id": "keep customer_id as int64, as in the input",
    "amount": "keep amount as float64, as in the input",
    "order_seq": "build it with groupby(...).cumcount() + 1, which is int64; rank() gives float64",
    "share_of_customer_total": (
        "cast the share to float64; it must be float64 even for an empty input"
    ),
    "days_since_previous_order": (
        "cast the day difference with astype('float64'); dt.days is int64 for an empty frame"
    ),
    "customer_running_total": "use groupby(...).cumsum() and cast the result to float64",
}
DAY0 = date(2024, 1, 1)
Row = tuple[int, int, date, float]  # (order_id, customer_id, order date, amount)


def _frame(rows: Sequence[Row], as_strings: bool = False) -> pd.DataFrame:
    dates = [row[2].isoformat() for row in rows] if as_strings else [row[2] for row in rows]
    frame = pd.DataFrame(
        {
            "order_id": [row[0] for row in rows],
            "customer_id": [row[1] for row in rows],
            "order_date": dates,
            "amount": [row[3] for row in rows],
        }
    ).astype({"order_id": "int64", "customer_id": "int64", "amount": "float64"})
    if not as_strings:
        frame["order_date"] = pd.to_datetime(frame["order_date"])
    return frame


def _oracle(rows: Sequence[Row]) -> list[tuple[Any, ...]]:
    """Independent answer from DuckDB window functions, in the contract's row order."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE o (order_id BIGINT, customer_id BIGINT, order_date DATE, amount DOUBLE)"
    )
    if rows:
        con.executemany("INSERT INTO o VALUES (?, ?, ?, ?)", list(rows))
    return con.execute(
        """
        SELECT order_id, customer_id, order_date, amount,
               ROW_NUMBER() OVER w AS order_seq,
               CASE WHEN SUM(amount) OVER (PARTITION BY customer_id) = 0 THEN NULL
                    ELSE amount / SUM(amount) OVER (PARTITION BY customer_id) END AS share,
               order_date - LAG(order_date) OVER w AS days,
               SUM(amount) OVER (
                   PARTITION BY customer_id ORDER BY order_date, order_id
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               ) AS running
        FROM o
        WINDOW w AS (PARTITION BY customer_id ORDER BY order_date, order_id)
        ORDER BY customer_id, order_date, order_id
        """
    ).fetchall()


def _is_missing(value: object) -> bool:
    return value is None or bool(pd.isna(value))


def _same(got: object, want: object) -> bool:
    if _is_missing(got) or _is_missing(want):
        return _is_missing(got) and _is_missing(want)
    return math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-9)  # type: ignore[arg-type]


def _all_same(values: pd.Series, wants: Sequence[object]) -> bool:
    got = values.tolist()
    return len(got) == len(wants) and all(_same(g, w) for g, w in zip(got, wants, strict=True))


def _dtype_problem(got: pd.DataFrame) -> str | None:
    for name, dtype in DTYPES.items():
        if str(got[name].dtype) != dtype:
            actual = got[name].dtype
            return f"{name} dtype differs: expected {dtype}, got {actual}. {DTYPE_HINTS[name]}"
    if not pd.api.types.is_datetime64_any_dtype(got["order_date"]):
        return "order_date is not a datetime column: parse it with pd.to_datetime"
    return None


def _first_problem(got: object, rows: Sequence[Row]) -> str | None:  # noqa: PLR0911
    """Return the first mistake found, or None. One return path keeps failure reports stable."""
    if not isinstance(got, pd.DataFrame):
        return "the result is not a pandas DataFrame: return the feature frame"
    if list(got.columns) != COLUMNS:
        return f"output columns differ: expected {COLUMNS}, got {list(got.columns)}"
    problem = _dtype_problem(got)
    if problem is not None:
        return problem
    if not got.index.equals(pd.RangeIndex(len(got))):
        return "index differs: return a fresh 0..n-1 index (reset_index(drop=True) after sorting)"
    want = _oracle(rows)
    if len(got) != len(want):
        return "row count differs: every input order must appear exactly once in the result"
    got_keys = list(zip(got["order_id"].tolist(), got["customer_id"].tolist(), strict=True))
    want_keys = [(row[0], row[1]) for row in want]
    if got_keys != want_keys:
        return (
            "rows are out of order: sort by customer_id, order_date, then order_id, before any "
            "per-customer step; order_id breaks ties on the same date"
        )
    if [pd.Timestamp(v).date() for v in got["order_date"]] != [row[2] for row in want]:
        return "order_date values differ: parse the input dates and keep their days"
    if not _all_same(got["amount"], [row[3] for row in want]):
        return "amount values differ: keep the input amounts unchanged"
    if not _all_same(got["order_seq"], [row[4] for row in want]):
        return (
            "order_seq differs: number each customer's orders 1, 2, 3, ... in order_date then "
            "order_id order; a count taken in input row order does not do this"
        )
    if not _all_same(got["share_of_customer_total"], [row[5] for row in want]):
        return (
            "share_of_customer_total differs: divide each amount by its customer's total over all "
            "of their orders, and give NaN when that total is 0.0"
        )
    if not _all_same(got["days_since_previous_order"], [row[6] for row in want]):
        return (
            "days_since_previous_order differs: take the gap to the previous order of the same "
            "customer (grouped shift), and use NaN for a first order"
        )
    if not _all_same(got["customer_running_total"], [row[7] for row in want]):
        return (
            "customer_running_total differs: the running total restarts for each customer and "
            "includes the current order (grouped cumsum), accumulated in float64; a float32 "
            "running sum loses cents on large totals"
        )
    return None


def _run(orders: pd.DataFrame) -> tuple[Any, str | None]:
    try:
        return customer_order_features(orders), None
    except Exception as exc:  # report the crash as a broken contract rule, not a traceback
        return None, (
            f"customer_order_features raised {type(exc).__name__} on a valid orders frame: "
            "check the inputs and the output contract in problem.md"
        )


def _check(rows: Sequence[Row], as_strings: bool = False) -> None:
    got, problem = _run(_frame(rows, as_strings=as_strings))
    if problem is None:
        problem = _first_problem(got, rows)
    assert problem is None, problem


@pytest.fixture(scope="module")
def interview_orders() -> list[Row]:
    """Real orders: every invoice of 60 customers of an online retailer over two years.

    The rows come from online_retail, one per invoice. The order id is the invoice number, the
    date is the invoice date and the amount is the invoice total. A cancellation is a real
    refund with a negative amount. Half the customers ordered four times or fewer and one
    ordered 91 times, and many customers have several orders on one day.
    """
    invoices = retail_invoices()
    return [
        (int(invoice.lstrip("C")), int(customer_id), pd.Timestamp(moment).date(), float(amount))
        for invoice, customer_id, moment, amount in invoices[
            ["invoice", "customer_id", "invoice_ts", "amount"]
        ].itertuples(index=False, name=None)
    ]


def _random_rows(rng: np.random.Generator) -> list[Row]:
    customers = [
        int(x) for x in rng.choice(np.arange(1, 1000), size=int(rng.integers(1, 6)), replace=False)
    ]
    order_ids = [
        int(x)
        for x in rng.choice(np.arange(1, 10_000), size=int(rng.integers(0, 40)), replace=False)
    ]
    rows: list[Row] = []
    for order_id in order_ids:
        customer = customers[int(rng.integers(0, len(customers)))]
        # A 60-day window makes same-day ties common. Quarters are exact in binary.
        day = DAY0 + timedelta(days=int(rng.integers(0, 60)))
        amount = 0.0 if rng.random() < 0.1 else float(rng.integers(0, 200)) / 4.0
        rows.append((order_id, customer, day, amount))
    return [rows[int(i)] for i in rng.permutation(len(rows))]


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000), as_strings=st.booleans())
def test_random_orders_match_the_duckdb_window_oracle(seed: int, as_strings: bool) -> None:
    _check(_random_rows(np.random.default_rng(seed)), as_strings=as_strings)


def test_interview_prep_orders_match_the_duckdb_window_oracle(
    interview_orders: list[Row],
) -> None:
    _check(interview_orders, as_strings=True)


def test_rows_are_sorted_by_customer_then_date_then_order_id() -> None:
    rows: list[Row] = [
        (5, 2, DAY0 + timedelta(days=60), 4.0),
        (1, 1, DAY0 + timedelta(days=8), 1.0),
        (3, 2, DAY0 + timedelta(days=1), 2.0),
        (2, 1, DAY0 + timedelta(days=8), 8.0),
        (4, 1, DAY0 + timedelta(days=31), 16.0),
    ]
    _check(rows)


def test_same_day_orders_are_numbered_by_order_id() -> None:
    day = date(2024, 1, 5)
    rows: list[Row] = [(9, 4, day, 3.0), (2, 4, day, 5.0), (7, 4, day, 1.0)]
    _check(rows)


def test_sequence_gap_and_running_total_restart_for_each_customer() -> None:
    d1, d2, d3, d4 = date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 9), date(2024, 2, 2)
    rows: list[Row] = [
        (1, 1, d1, 10.0),
        (2, 2, d1, 20.0),
        (3, 1, d2, 5.0),
        (4, 2, d3, 1.0),
        (5, 1, d4, 7.0),
    ]
    _check(rows)


def test_gap_is_nan_for_a_first_order_and_zero_for_a_same_day_order() -> None:
    day = date(2024, 1, 5)
    rows: list[Row] = [(1, 3, day, 2.0), (2, 3, day, 3.0), (3, 4, date(2024, 1, 6), 1.0)]
    got, problem = _run(_frame(rows))
    assert problem is None, problem
    gaps = got.set_index("order_id")["days_since_previous_order"]
    assert pd.isna(gaps.loc[1]), (
        "days_since_previous_order is not NaN for a customer's first order: keep the shift NaN"
    )
    assert float(gaps.loc[2]) == 0.0, (
        "days_since_previous_order is not 0.0 for two orders on the same date: subtract the dates"
    )
    _check(rows)


def test_customer_whose_orders_total_zero_has_nan_share() -> None:
    day = date(2024, 1, 2)
    rows: list[Row] = [(1, 7, day, 0.0), (2, 7, day + timedelta(days=3), 0.0), (3, 8, day, 5.0)]
    got, problem = _run(_frame(rows))
    assert problem is None, problem
    shares = got.set_index("order_id")["share_of_customer_total"]
    assert pd.isna(shares.loc[1]) and pd.isna(shares.loc[2]), (
        "share_of_customer_total is not NaN for a customer whose total is 0.0: 0.0 / 0.0 is NaN, "
        "so do not replace it with 0.0 or raise"
    )
    _check(rows)


def test_refunds_that_cancel_spend_have_nan_shares() -> None:
    rows: list[Row] = [
        (1, 7, DAY0, 8.0),
        (2, 7, DAY0 + timedelta(days=1), -8.0),
        (3, 8, DAY0, -2.0),
        (4, 8, DAY0 + timedelta(days=2), 5.0),
    ]
    _check(rows)


def test_string_and_datetime_dates_give_the_same_features() -> None:
    rows: list[Row] = [
        (1, 1, date(2024, 12, 31), 2.0),
        (2, 1, date(2025, 1, 2), 4.0),
        (3, 2, date(2024, 2, 14), 1.0),
    ]
    _check(rows, as_strings=False)
    _check(rows, as_strings=True)


def test_empty_orders_keep_the_same_columns_and_dtypes() -> None:
    got, problem = _run(_frame([]))
    assert problem is None, problem
    assert list(got.columns) == COLUMNS, (
        f"output columns differ on an empty input: expected {COLUMNS}, got {list(got.columns)}"
    )
    assert len(got) == 0, f"an empty input must give no rows, got {len(got)}"
    dtype_problem = _dtype_problem(got)
    assert dtype_problem is None, f"empty input: {dtype_problem}"


def test_every_column_has_its_contract_dtype() -> None:
    rows: list[Row] = [(1, 1, DAY0, 2.0), (2, 1, DAY0 + timedelta(days=1), 0.0), (3, 2, DAY0, 1.5)]
    got, problem = _run(_frame(rows))
    assert problem is None, problem
    dtype_problem = _dtype_problem(got)
    assert dtype_problem is None, dtype_problem


def test_result_has_a_fresh_zero_based_index() -> None:
    rows: list[Row] = [(5, 2, DAY0, 4.0), (1, 1, DAY0 + timedelta(days=3), 1.0), (3, 2, DAY0, 2.0)]
    got, problem = _run(_frame(rows))
    assert problem is None, problem
    assert got.index.equals(pd.RangeIndex(len(got))), (
        "index differs: return a fresh 0..n-1 index (reset_index(drop=True) after sorting)"
    )


def test_input_orders_are_not_mutated() -> None:
    rows: list[Row] = [(5, 2, DAY0 + timedelta(days=5), 4.0), (1, 1, DAY0, 1.0), (3, 2, DAY0, 2.0)]
    for as_strings in (False, True):
        orders = _frame(rows, as_strings=as_strings)
        before = orders.copy(deep=True)
        _run(orders)
        assert list(orders.columns) == list(before.columns), (
            "orders gained or lost a column: work on a copy, do not add columns to the input"
        )
        assert orders.equals(before), (
            "orders was changed in place: do not sort, parse or assign into the input frame"
        )
