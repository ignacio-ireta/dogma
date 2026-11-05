"""Public tests: the fixture's answer, the single column, NULL codes and the sort order."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd  # type: ignore[import-untyped]
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_fixture() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE shipments AS SELECT * FROM read_csv_auto(?)",
        [str(FIXTURES / "shipments.csv")],
    )
    con.execute(
        "CREATE TABLE returns AS SELECT * FROM read_csv_auto(?)", [str(FIXTURES / "returns.csv")]
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


def _oracle() -> list[str]:
    """Independent answer: a Python set over both CSV files, NULL codes dropped, sorted."""
    shipments = pd.read_csv(FIXTURES / "shipments.csv", dtype=str, keep_default_na=False)
    returns = pd.read_csv(FIXTURES / "returns.csv", dtype=str, keep_default_na=False)
    codes = set(shipments["warehouse_code"]) | set(returns["warehouse_code"])
    return sorted(code for code in codes if code != "")


def test_fixture_codes_match_the_python_set_answer() -> None:
    got = _run_learner_query(_load_fixture())
    assert got["warehouse_code"].tolist() == _oracle(), (
        "the codes differ from the expected list: keep codes from both tables (UNION, not JOIN), "
        "drop NULL codes, and sort ascending"
    )


def test_output_is_one_column_named_warehouse_code() -> None:
    got = _run_learner_query(_load_fixture())
    assert list(got.columns) == ["warehouse_code"], (
        f"output column differs: expected ['warehouse_code'], got {list(got.columns)}"
    )


def test_each_code_appears_once_even_when_both_tables_have_it() -> None:
    got = _run_learner_query(_load_fixture())
    codes = got["warehouse_code"].tolist()
    assert len(codes) == len(set(codes)), (
        "a code appears more than once: UNION removes duplicates; UNION ALL keeps them"
    )
    assert codes.count("BOS-1") == 1, (
        "BOS-1 is in both tables and must appear once: use UNION, not UNION ALL"
    )


def test_null_codes_are_not_returned() -> None:
    got = _run_learner_query(_load_fixture())
    assert got["warehouse_code"].notna().all(), (
        "a NULL warehouse code was returned: filter WHERE warehouse_code IS NOT NULL in each branch"
    )
