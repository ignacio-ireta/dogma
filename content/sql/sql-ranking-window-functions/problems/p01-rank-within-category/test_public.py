"""Public tests: the fixture's exact result, the column list, the tie-break and the NULL rule.

The learner's query is read from submission.sql and run on the fixture. Tests never import solution.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd  # type: ignore[import-untyped]
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
COLUMNS = ["category", "product_id", "revenue", "row_number", "rank", "dense_rank"]
EXPECTED = [
    ("books", 101, 250.0, 1, 1, 1),
    ("books", 103, 250.0, 2, 1, 1),
    ("books", 102, 180.5, 3, 3, 2),
    ("books", 105, 90.0, 4, 4, 3),
    ("books", 104, None, 5, 5, 4),
    ("books", 106, None, 6, 5, 4),
    ("games", 210, 320.0, 1, 1, 1),
    ("games", 211, 320.0, 2, 1, 1),
    ("games", 213, 40.0, 3, 3, 2),
    ("games", 212, None, 4, 4, 3),
    ("tools", 308, 15.0, 1, 1, 1),
    ("tools", 905113208, 15.0, 2, 1, 1),
    ("tools", 306, None, 3, 3, 2),
    ("tools", 307, None, 4, 3, 2),
    ("toys", 305, 75.25, 1, 1, 1),
]


def _load_fixture() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE products AS SELECT * FROM read_csv_auto(?)",
        [str(FIXTURES / "products.csv")],
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
    frame = result.fetchdf()
    assert list(frame.columns) == COLUMNS, (
        f"output columns differ: expected {COLUMNS}, got {list(frame.columns)}. Select category, "
        "product_id, revenue, row_number, rank and dense_rank, in that order."
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
    if isinstance(got, str) or isinstance(want, str):
        return got == want
    return math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-9)


def test_fixture_returns_every_product_with_three_ranks_in_order() -> None:
    got = _run_learner_query(_load_fixture())
    rows = [tuple(_plain(v) for v in row) for row in got.itertuples(index=False, name=None)]
    assert len(rows) == len(EXPECTED), (
        f"row count differs: expected {len(EXPECTED)} rows, one per product, got {len(rows)}"
    )
    for position, (got_row, want_row) in enumerate(zip(rows, EXPECTED, strict=True), start=1):
        assert all(_same_value(g, w) for g, w in zip(got_row, want_row, strict=True)), (
            f"row {position} differs: check the partition, the revenue order with NULL last, the "
            "tie-break by product_id for row_number, and the ORDER BY of the output"
        )


def test_output_has_exactly_the_six_columns_in_order() -> None:
    got = _run_learner_query(_load_fixture())
    assert list(got.columns) == COLUMNS, f"output columns differ: expected {COLUMNS}"


def test_equal_revenues_get_different_row_numbers_by_product_id() -> None:
    got = _run_learner_query(_load_fixture())
    tools = got[got["category"] == "tools"].reset_index(drop=True)
    assert [int(v) for v in tools["product_id"][:2]] == [308, 905113208], (
        "two products tie at 15.00: ROW_NUMBER breaks the tie by product_id ascending"
    )
    assert [int(v) for v in tools["rank"][:2]] == [1, 1], (
        "a tie must share RANK: do not add product_id to the ORDER BY of RANK or DENSE_RANK"
    )


def test_null_revenue_ranks_last_and_shares_its_rank() -> None:
    got = _run_learner_query(_load_fixture())
    books = got[got["category"] == "books"].reset_index(drop=True)
    assert [int(v) for v in books["product_id"][4:]] == [104, 106], (
        "NULL revenue must rank after every number, so the two NULL rows come last in books"
    )
    assert [int(v) for v in books["rank"][4:]] == [5, 5], (
        "two NULL revenues are peers: they share RANK (and DENSE_RANK)"
    )
