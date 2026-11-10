"""Hidden tests: your query runs against tables that each test builds itself.

The oracle is an independent Python set computation over the same rows. Each test name says what
behaviour it checks, and each failure message names the likely mistake first.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dogma_data import rng, synth

SHIPMENTS_DDL = (
    "CREATE TABLE shipments (shipment_id BIGINT, warehouse_code VARCHAR, shipped_on DATE)"
)
RETURNS_DDL = "CREATE TABLE returns (return_id BIGINT, warehouse_code VARCHAR, returned_on DATE)"
CODES = ["BOS-1", "DEN-2", "bos-1", "LAX-3", "ORD-4", "RNO-7", "SEA-5", "PHX-6"]


def _connect(
    shipments: Sequence[tuple[Any, ...]], returns: Sequence[tuple[Any, ...]]
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(SHIPMENTS_DDL)
    con.execute(RETURNS_DDL)
    if shipments:
        con.executemany(
            "INSERT INTO shipments VALUES (?, ?, DATE '2024-01-01')", [row[:2] for row in shipments]
        )
    if returns:
        con.executemany(
            "INSERT INTO returns VALUES (?, ?, DATE '2024-01-01')", [row[:2] for row in returns]
        )
    return con


def _query(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame | None, str | None]:
    """Run the learner's query. Returns (frame, None), or (None, the mistake). Never raises."""
    sql = Path(__file__).with_name("submission.sql").read_text()
    try:
        result = con.execute(sql)
    except duckdb.Error as exc:
        return None, f"your SQL did not run: {exc}"
    if result is None:
        return None, "submission.sql returned no result set: finish it with a SELECT"
    return result.fetchdf(), None


def _run_learner_query(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    got, problem = _query(con)
    if problem is not None:
        pytest.fail(problem)
    return got


def _oracle(shipments: Sequence[tuple[Any, ...]], returns: Sequence[tuple[Any, ...]]) -> list[str]:
    """Independent answer: a Python set over both lists, NULL codes dropped, sorted."""
    codes = {row[1] for row in shipments} | {row[1] for row in returns}
    return sorted(code for code in codes if code is not None)


def _first_problem(  # noqa: PLR0911 (one return per kind of mistake, kept flat on purpose)
    got: pd.DataFrame,
    shipments: Sequence[tuple[Any, ...]],
    returns: Sequence[tuple[Any, ...]],
) -> str | None:
    """Return the first mistake found, or None. One return path keeps failure reports stable."""
    if list(got.columns) != ["warehouse_code"]:
        return f"output column differs: expected ['warehouse_code'], got {list(got.columns)}"
    codes = got["warehouse_code"].tolist()
    if not all(isinstance(code, str) for code in codes):
        return (
            "a NULL warehouse code was returned: filter WHERE warehouse_code IS NOT NULL "
            "in each branch before the union"
        )
    want = _oracle(shipments, returns)
    if set(codes) - set(want):
        return (
            "a code appears that is in neither table: check the two branches of the union "
            "and do not change codes with UPPER, LOWER or TRIM"
        )
    if set(want) - set(codes):
        return (
            "a code from one table is missing: keep codes from both tables (UNION), not only "
            "codes present in both (JOIN); compare codes exactly, with no UPPER, LOWER or TRIM"
        )
    if len(codes) != len(set(codes)):
        return "a code appears more than once: UNION removes duplicates, UNION ALL keeps them"
    if codes != want:
        return "rows are not sorted: add ORDER BY warehouse_code ASC after the union"
    return None


def _check_same_codes(
    got: pd.DataFrame,
    shipments: Sequence[tuple[Any, ...]],
    returns: Sequence[tuple[Any, ...]],
) -> None:
    problem = _first_problem(got, shipments, returns)
    assert problem is None, problem


@pytest.fixture(scope="module")
def event_log_tables() -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    generated = synth.event_log(
        rng("sql/sql-joins-and-unions/p02-warehouses-with-a-shipment-or-a-return:event-log:0"),
        1_200,
        n_users=300,
        categories=tuple(CODES),
        start="2024-01-01",
        days=45,
        dirty_rate=0.2,
    )
    shipments: list[tuple[Any, ...]] = []
    returns: list[tuple[Any, ...]] = []
    for i, event_id in enumerate(generated["event_id"]):
        row = (int(event_id), str(generated["category"][i]), date(2024, 1, 1))
        (shipments if i % 2 == 0 else returns).append(row)
    return shipments, returns


def test_event_log_dirty_categories_match_the_pandas_union_oracle(
    event_log_tables: tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]],
) -> None:
    shipments, returns = event_log_tables
    got = _run_learner_query(_connect(shipments, returns))
    shipment_codes = pd.DataFrame(shipments, columns=["id", "warehouse_code", "day"])[
        "warehouse_code"
    ]
    return_codes = pd.DataFrame(returns, columns=["id", "warehouse_code", "day"])["warehouse_code"]
    want = (
        pd.concat([shipment_codes, return_codes], ignore_index=True)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    assert got["warehouse_code"].tolist() == want, (
        "the event_log category keys must be unioned exactly as stored, including case and spaces"
    )


def _random_rows(rng: np.random.Generator, n: int, first_id: int) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for i in range(n):
        if rng.random() < 0.15:
            code = None
        else:
            code = CODES[int(rng.integers(0, len(CODES)))]
        rows.append((first_id + i, code))
    return rows


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_random_tables_match_the_python_set_answer(seed: int) -> None:
    rng = np.random.default_rng(seed)
    shipments = _random_rows(rng, int(rng.integers(0, 15)), 1)
    returns = _random_rows(rng, int(rng.integers(0, 15)), 1)
    got, problem = _query(_connect(shipments, returns))
    if problem is None:
        problem = _first_problem(got, shipments, returns)
    assert problem is None, problem


def test_codes_that_appear_in_only_one_table_are_kept() -> None:
    shipments = [(1, "DEN-2", None), (2, "ORD-4", None)]
    returns = [(1, "RNO-7", None), (2, "PHX-6", None)]
    got = _run_learner_query(_connect(shipments, returns))
    _check_same_codes(got, shipments, returns)


def test_codes_in_both_tables_appear_once() -> None:
    shipments = [(1, "BOS-1", None), (2, "BOS-1", None), (3, "LAX-3", None)]
    returns = [(1, "BOS-1", None), (2, "LAX-3", None), (3, "LAX-3", None)]
    got = _run_learner_query(_connect(shipments, returns))
    _check_same_codes(got, shipments, returns)


def test_null_codes_are_never_returned_even_when_both_tables_have_them() -> None:
    shipments = [(1, None, None), (2, "SEA-5", None)]
    returns = [(1, None, None), (2, None, None)]
    got = _run_learner_query(_connect(shipments, returns))
    _check_same_codes(got, shipments, returns)


def test_codes_that_differ_only_by_case_stay_distinct() -> None:
    shipments = [(1, "bos-1", None)]
    returns = [(1, "BOS-1", None)]
    got = _run_learner_query(_connect(shipments, returns))
    assert sorted(got["warehouse_code"].tolist()) == ["BOS-1", "bos-1"], (
        "codes that differ only by case must stay distinct: compare and return them exactly"
    )


def test_result_is_sorted_ascending_when_rows_arrive_out_of_order() -> None:
    codes = ["SEA-5", "BOS-1", "RNO-7", "DEN-2", "PHX-6", "LAX-3", "ORD-4", "bos-1"]
    shipments = [(i + 1, code, None) for i, code in enumerate(codes)]
    returns = [(i + 1, code, None) for i, code in enumerate(reversed(codes))]
    got = _run_learner_query(_connect(shipments, returns))
    _check_same_codes(got, shipments, returns)


def test_one_empty_table_returns_the_other_tables_codes_once() -> None:
    shipments = [(1, "DEN-2", None), (2, "DEN-2", None), (3, "LAX-3", None)]
    got = _run_learner_query(_connect(shipments, []))
    _check_same_codes(got, shipments, [])


def test_both_tables_empty_returns_zero_rows_with_the_one_column() -> None:
    got = _run_learner_query(_connect([], []))
    assert list(got.columns) == ["warehouse_code"], (
        f"column differs on an empty result: expected ['warehouse_code'], got {list(got.columns)}"
    )
    assert len(got) == 0, f"expected no codes, got {len(got)} rows"
