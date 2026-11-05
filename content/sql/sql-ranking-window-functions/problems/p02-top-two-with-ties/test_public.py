"""Public tests: the fixture's exact result, a tie at the cut, and NULL ranks.

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
COLUMNS = ["category", "product_id", "revenue", "dense_rank"]
EXPECTED = [
    ("audio", 11, 500.0, 1),
    ("audio", 12, 500.0, 1),
    ("audio", 13, 300.0, 2),
    ("audio", 14, 300.0, 2),
    ("audio", 905113208, 300.0, 2),
    ("cameras", 31, 900.0, 1),
    ("cameras", 33, 450.0, 2),
    ("lenses", 41, 75.0, 1),
    ("lenses", 42, 75.0, 1),
    ("lenses", 43, 75.0, 1),
    ("lenses", 44, 60.0, 2),
    ("mounts", 53, None, 1),
    ("mounts", 54, None, 1),
    ("mounts", 55, None, 1),
    ("video", 23, 80.0, 1),
    ("video", 24, 80.0, 1),
    ("video", 25, 10.0, 2),
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
        "product_id, revenue and dense_rank, in that order."
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


def test_fixture_returns_the_top_two_dense_ranks_including_ties() -> None:
    got = _run_learner_query(_load_fixture())
    rows = [tuple(_plain(v) for v in row) for row in got.itertuples(index=False, name=None)]
    assert len(rows) == len(EXPECTED), (
        f"row count differs: expected {len(EXPECTED)} rows, got {len(rows)}. Keep every product "
        "whose dense rank is at most 2, including all products tied at the cut"
    )
    for position, (got_row, want_row) in enumerate(zip(rows, EXPECTED, strict=True), start=1):
        assert all(_same_value(g, w) for g, w in zip(got_row, want_row, strict=True)), (
            f"row {position} differs: check the dense rank (not row_number or rank), the NULL "
            "order, and the output order"
        )


def test_tie_at_the_cut_brings_in_every_tied_product() -> None:
    got = _run_learner_query(_load_fixture())
    audio = got[got["category"] == "audio"]
    assert sorted(int(v) for v in audio["product_id"]) == [11, 12, 13, 14, 905113208], (
        "three products share dense rank 2 in audio, and all three belong to the top two"
    )


def test_null_revenue_is_not_kept_when_the_top_two_are_filled() -> None:
    got = _run_learner_query(_load_fixture())
    video = got[got["category"] == "video"]
    assert 21 not in video["product_id"].tolist() and 22 not in video["product_id"].tolist(), (
        "video's NULL revenues rank after 80.00 and 10.00, which fill the top two dense ranks"
    )
