"""Hidden tests: hand-built edge cases, plus interview-prep frames checked against a pandas oracle.

The hand-built cases use an independent oracle: DuckDB SQL for the joins and sums, then a plain
Python pivot. The interview-prep test uses a pandas merge and groupby on the same frames. Each
failure message names the likely mistake first.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import monthly_revenue_by_region

from dogma_data import load_frame

REGIONS = ["North", "South", "East", "West"]
ORDER_COLUMNS = ["order_id", "customer_id", "order_date", "amount"]


def _oracle(
    order_rows: list[tuple[Any, ...]], customer_rows: list[tuple[Any, ...]]
) -> pd.DataFrame:
    """Independent answer: DuckDB joins and sums, then a plain Python pivot."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE o (order_id BIGINT, customer_id BIGINT, order_date VARCHAR, amount DOUBLE)"
    )
    con.execute("CREATE TABLE c (customer_id BIGINT, region VARCHAR)")
    if order_rows:
        con.executemany("INSERT INTO o VALUES (?, ?, ?, ?)", order_rows)
    if customer_rows:
        con.executemany("INSERT INTO c VALUES (?, ?)", customer_rows)
    rows = con.execute(
        """
        SELECT COALESCE(c.region, 'unknown') AS region,
               strftime(CAST(o.order_date AS DATE), '%Y-%m') AS month,
               SUM(o.amount) AS revenue
        FROM o LEFT JOIN c ON c.customer_id = o.customer_id
        WHERE o.amount IS NOT NULL
        GROUP BY 1, 2
        """
    ).fetchall()
    regions = sorted({r for r, _, _ in rows})
    months = sorted({m for _, m, _ in rows})
    cells = {(r, m): float(v) for r, m, v in rows}
    matrix = [[cells.get((r, m), 0.0) for m in months] for r in regions]
    return pd.DataFrame(
        matrix,
        index=pd.Index(regions, name="region"),
        columns=pd.Index(months, name="month"),
        dtype="float64",
    )


def _frames(
    order_rows: list[tuple[Any, ...]],
    customer_rows: list[tuple[Any, ...]],
    amount_dtype: str = "float64",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.DataFrame(order_rows, columns=ORDER_COLUMNS)
    orders = orders.astype({"order_id": "int64", "customer_id": "int64", "amount": amount_dtype})
    customers = pd.DataFrame(customer_rows, columns=["customer_id", "region"])
    customers = customers.astype({"customer_id": "int64"})
    return orders, customers


def _try(orders: pd.DataFrame, customers: pd.DataFrame) -> tuple[Any, str | None]:
    """Return (result, None), or (None, the crash as a mistake). Never raises."""
    try:
        return monthly_revenue_by_region(orders, customers), None
    except Exception as exc:  # report the learner's crash as a failure, with its type
        return None, f"your function raised {type(exc).__name__}: {exc}"


def _run(orders: pd.DataFrame, customers: pd.DataFrame, hint: str = "") -> Any:
    got, problem = _try(orders, customers)
    if problem is not None:
        pytest.fail(problem + (f". {hint}" if hint else ""))
    return got


def _first_problem(got: Any, want: pd.DataFrame) -> str | None:  # noqa: PLR0911 (one return per kind of mistake, kept flat on purpose)
    """Return the first mistake found, or None. One return path keeps failure reports stable."""
    if not isinstance(got, pd.DataFrame):
        return "the result is not a pandas DataFrame: return the wide table"
    if got.index.name != "region" or got.columns.name != "month":
        return "index or columns name differs: name the index 'region' and the columns 'month'"
    if got.shape != want.shape:
        return (
            "the number of regions or months differs: a region or month with no known revenue "
            "must add no row or column (drop NaN amounts, do not fill them). Orders from unknown "
            "customers, and from customers with no region, belong to the 'unknown' region; do not "
            "drop them"
        )
    if list(got.index) != list(want.index):
        return "region labels or their order differ: sort regions ascending as strings"
    if list(got.columns) != list(want.columns):
        return "month labels or their order differ: months are 'YYYY-MM' strings, sorted ascending"
    if any(dtype != np.float64 for dtype in got.dtypes):
        return "a value column is not float64: cast amounts to float64 before grouping"
    if got.isna().to_numpy().any():
        return "NaN cells remain: a region-month cell with no known revenue must be 0.0, not NaN"
    if not np.allclose(got.to_numpy(), want.to_numpy(), rtol=1e-9, atol=1e-9):
        return (
            "revenue differs in some region-month cell: sum the known amounts per region and "
            "month, counting each order once (a repeated merge key doubles it)"
        )
    return None


def _pandas_oracle(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Expected monthly revenue from a pandas many-to-one merge and groupby."""
    merged = orders.merge(customers, how="left", on="customer_id", validate="many_to_one")
    merged["region"] = merged["region"].fillna("unknown")
    known = merged.dropna(subset=["amount"]).copy()
    if known.empty:
        return pd.DataFrame(
            index=pd.Index([], name="region"),
            columns=pd.Index([], name="month"),
            dtype="float64",
        )
    known["month"] = pd.to_datetime(known["order_date"]).dt.strftime("%Y-%m")
    result = known.groupby(["region", "month"])["amount"].sum().unstack("month", fill_value=0.0)
    result = result.sort_index().sort_index(axis=1).astype("float64")
    result.index.name = "region"
    result.columns.name = "month"
    return result


@pytest.fixture(scope="module")
def interview_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_orders = load_frame("orders")
    orders = source_orders[["order_id", "customer_id", "order_date", "amount"]].copy()
    orders["order_id"] = orders["order_id"].str[1:].astype("int64")
    orders["customer_id"] = orders["customer_id"].str[1:].astype("int64")
    orders["amount"] = orders["amount"].astype("float64")

    source_customers = load_frame("customers")
    customers = source_customers[["customer_id", "city"]].rename(columns={"city": "region"}).copy()
    customers["customer_id"] = customers["customer_id"].str[1:].astype("int64")
    customers = customers.drop_duplicates("customer_id", keep="first").reset_index(drop=True)
    return orders, customers


def _random_rows(
    rng: np.random.Generator,
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    n_customers = int(rng.integers(1, 9))
    ids = [int(x) for x in rng.choice(np.arange(1, 60), size=n_customers, replace=False)]
    customer_rows = [(cid, REGIONS[int(rng.integers(0, len(REGIONS)))]) for cid in ids]
    unknown = [int(x) for x in rng.choice(np.arange(100, 130), size=3, replace=False)]
    order_rows: list[tuple[Any, ...]] = []
    for order_id in range(1, int(rng.integers(0, 60)) + 1):
        pool = ids if rng.random() < 0.8 else unknown
        cid = pool[int(rng.integers(0, len(pool)))]
        day = date(2024, 1, 1) + timedelta(days=int(rng.integers(0, 600)))
        amount = None if rng.random() < 0.2 else float(rng.integers(0, 4000)) / 4.0
        order_rows.append((order_id, cid, day.isoformat(), amount))
    return order_rows, customer_rows


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_random_tables_match_an_independent_sql_computation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    order_rows, customer_rows = _random_rows(rng)
    orders, customers = _frames(order_rows, customer_rows)
    if rng.random() < 0.5:  # both date forms must give the same months
        orders["order_date"] = pd.to_datetime(orders["order_date"])
    got, problem = _try(orders, customers)
    if problem is None:
        problem = _first_problem(got, _oracle(order_rows, customer_rows))
    assert problem is None, problem


def test_interview_prep_frames_match_the_pandas_merge_oracle(
    interview_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    orders, customers = interview_frames
    got, problem = _try(orders.copy(deep=True), customers.copy(deep=True))
    if problem is None:
        problem = _first_problem(got, _pandas_oracle(orders, customers))
    assert problem is None, problem


def test_unknown_customers_are_kept_in_their_own_region() -> None:
    order_rows = [
        (1, 1, "2024-03-01", 10.0),
        (2, 77, "2024-03-02", 5.0),
        (3, 78, "2024-04-01", 2.0),
    ]
    orders, customers = _frames(order_rows, [(1, "North")])
    got = _run(orders, customers)
    problem = _first_problem(got, _oracle(order_rows, [(1, "North")]))
    assert problem is None, problem
    assert got.loc["unknown", "2024-03"] == 5.0, (
        "orders from customers missing from the table must add to the 'unknown' region"
    )


def test_duplicate_customer_ids_raise_value_error_instead_of_doubling_revenue() -> None:
    orders, _ = _frames([(1, 5, "2024-01-10", 9.0)], [])
    customers = pd.DataFrame({"customer_id": [5, 5], "region": ["North", "South"]})
    try:
        monthly_revenue_by_region(orders, customers)
    except ValueError:
        return
    pytest.fail(
        "duplicate customer ids were accepted: pass validate='many_to_one' to the merge so a "
        "duplicate raises, instead of repeating the order"
    )


def test_region_with_only_unknown_amounts_has_no_row_and_no_column() -> None:
    order_rows = [
        (1, 1, "2024-01-05", 10.0),
        (2, 2, "2024-05-09", None),
        (3, 2, "2024-05-10", None),
    ]
    customer_rows = [(1, "North"), (2, "East")]
    orders, customers = _frames(order_rows, customer_rows)
    got = _run(orders, customers)
    problem = _first_problem(got, _oracle(order_rows, customer_rows))
    assert problem is None, problem
    assert "East" not in got.index and "2024-05" not in got.columns, (
        "a region whose only orders have unknown amounts must not appear: drop NaN amounts "
        "before grouping, do not fill them with zero"
    )


def test_missing_region_month_cells_are_float_zero_not_nan() -> None:
    order_rows = [(1, 1, "2024-01-05", 10.0), (2, 2, "2024-02-05", 4.0)]
    customer_rows = [(1, "North"), (2, "South")]
    orders, customers = _frames(order_rows, customer_rows)
    got = _run(orders, customers)
    problem = _first_problem(got, _oracle(order_rows, customer_rows))
    assert problem is None, problem
    assert got.loc["North", "2024-02"] == 0.0, "a missing region-month cell must be 0.0, not NaN"


def test_integer_amounts_give_float64_cells() -> None:
    order_rows = [(1, 1, "2024-01-05", 10), (2, 1, "2024-01-06", 7), (3, 2, "2024-02-01", 3)]
    customer_rows = [(1, "North"), (2, "South")]
    orders, customers = _frames(order_rows, customer_rows, amount_dtype="int64")
    got = _run(orders, customers)
    problem = _first_problem(got, _oracle(order_rows, customer_rows))
    assert problem is None, problem


def test_string_and_datetime_dates_give_the_same_months_in_calendar_order() -> None:
    order_rows = [
        (1, 1, "2024-12-31", 1.0),
        (2, 1, "2025-01-02", 2.0),
        (3, 2, "2024-02-14", 4.0),
    ]
    customer_rows = [(1, "North"), (2, "South")]
    strings, customers = _frames(order_rows, customer_rows)
    dates = strings.copy()
    dates["order_date"] = pd.to_datetime(dates["order_date"])
    want = _oracle(order_rows, customer_rows)
    for orders in (strings, dates):
        got = _run(orders, customers)
        problem = _first_problem(got, want)
        assert problem is None, problem
    assert list(_run(dates, customers).columns) == ["2024-02", "2024-12", "2025-01"], (
        "months must be sorted in calendar order across the year boundary"
    )


def test_all_amounts_unknown_gives_an_empty_frame_with_the_index_and_column_names() -> None:
    order_rows = [(1, 1, "2024-01-05", None), (2, 2, "2024-01-06", None)]
    orders, customers = _frames(order_rows, [(1, "North"), (2, "South")])
    got = _run(
        orders,
        customers,
        hint="with no known amounts the result is an empty frame, not an error",
    )
    assert got.shape == (0, 0), (
        f"an empty result must have no rows and no columns, got shape {got.shape}"
    )
    assert got.index.name == "region" and got.columns.name == "month", (
        "an empty result keeps the names: index 'region', columns 'month'"
    )


def test_inputs_are_not_mutated() -> None:
    order_rows = [(1, 1, "2024-01-05", 10.0), (2, 99, "2024-01-06", None)]
    orders, customers = _frames(order_rows, [(1, "North")])
    orders_before, customers_before = orders.copy(deep=True), customers.copy(deep=True)
    _run(orders, customers)
    assert list(orders.columns) == list(orders_before.columns), (
        "orders gained or lost a column: work on a copy, do not add columns to the input"
    )
    assert orders.equals(orders_before), (
        "orders was changed in place: do not drop, fill or reorder rows of the input"
    )
    assert customers.equals(customers_before), (
        "customers was changed in place: do not modify the input table"
    )
