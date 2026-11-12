"""Hidden tests: pandas oracle on random tables, tie and NULL rules, ids, and empty input.

Each test builds its own products table. Failure messages name the kind of mistake; none carries an
id, a value or an expected answer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from scipy.special import ndtr

from dogma_data import retail_events, rng, sample_rows

COLUMNS = ["category", "product_id", "revenue", "row_number", "rank", "dense_rank"]
PRODUCTS_DDL = "CREATE TABLE products (product_id BIGINT, category VARCHAR, revenue DOUBLE)"
CATEGORIES = ["books", "games", "tools", "toys"]
# Revenues repeat on purpose (ties), and include 0.0 next to NULL.
REVENUES: list[float | None] = [None, 0.0, 10.0, 10.0, 25.5, 99.0, 25.5, None]
# Large, non-sequential ids. 9 and 17 sort differently as text and as numbers.
ID_POOL = [3, 9, 17, 42, 101, 1_204, 17_204, 905_113_208, 6_000_000_001, 310_552_910_887] + [
    1_000 + 7 * i for i in range(30)
]


def _real_normal(gen, size=None, loc=0.0, scale=1.0):
    """Real numbers where this test used to draw from a normal distribution.

    Standardised breast_cancer measurements (mean 0 and standard deviation 1 over the dataset,
    skewed, with a few values far out), drawn by ``gen``. A vector is one real measurement
    over that many tumours, a matrix is real tumours by real measurements, and any other
    shape is filled from whole rows. ``loc`` and ``scale`` shift and stretch them as before.
    """
    shape = () if size is None else tuple(int(side) for side in np.atleast_1d(size))
    total = int(np.prod(shape)) if shape else 1
    if len(shape) == 1 and 1 <= shape[0] <= 569:
        values = sample_rows("breast_cancer", gen, shape[0], 1, standardize=True)[0][:, 0]
    elif len(shape) == 2 and 1 <= shape[0] <= 569 and 1 <= shape[1] <= 30:
        values = sample_rows("breast_cancer", gen, shape[0], shape[1], standardize=True)[0]
    else:
        rows = max(1, min(569, -(-total // 30)))
        drawn = sample_rows("breast_cancer", gen, rows, standardize=True)[0]
        values = np.resize(drawn.ravel(), total).reshape(shape)
    values = values * scale + loc
    return values if shape else float(values)


def _real_unit(gen, size=None):
    """Real numbers between 0 and 1 where this test used to draw uniform ones.

    Each is the share of a standard normal that lies below a standardised breast_cancer
    measurement, so the values keep the order, the ties and the skew of real measurements.
    """
    return ndtr(_real_normal(gen, size))


def _real_ints(gen, low, high, size=None):
    """Real whole numbers from ``low`` up to ``high - 1``: the band a real measurement falls in.

    The range is cut into ``high - low`` bands of equal normal probability, and each value is
    the band of one standardised measurement. Real measurements are skewed, so the bands are
    not equally full, and neighbouring values tie as real categories and grades do.
    """
    bands = np.floor(_real_unit(gen, size) * (high - low))
    return (low + np.minimum(bands, high - low - 1)).astype(np.int64)


def _real_shares(gen, alpha, size=None):
    """Real probability vectors: positive real numbers (see `_real_unit`) divided by their sum."""
    k = int(np.size(alpha))
    shape = (k,) if size is None else (int(size), k)
    weights = _real_unit(gen, shape) + 1e-3
    return weights / weights.sum(axis=-1, keepdims=True)


def _fail(why: str) -> None:
    raise AssertionError(why)


def _connect(rows: Sequence[tuple[Any, ...]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(PRODUCTS_DDL)
    if rows:
        con.executemany("INSERT INTO products VALUES (?, ?, ?)", list(rows))
    return con


def _run_learner_query(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = Path(__file__).with_name("submission.sql").read_text()
    try:
        result = con.execute(sql)
    except duckdb.Error as exc:
        _fail(f"your SQL did not run: {exc}")
    if result is None:
        _fail("submission.sql returned no result set: finish it with a SELECT")
    frame = result.fetchdf()
    if list(frame.columns) != COLUMNS:
        _fail(
            f"output columns differ: expected {COLUMNS}, got {list(frame.columns)}. Select "
            "category, product_id, revenue, row_number, rank and dense_rank, in that order."
        )
    return frame


def _oracle(rows: Sequence[tuple[Any, ...]]) -> pd.DataFrame:
    """Independent answer in pandas: one rank method per column, ties broken by product_id."""
    frame = pd.DataFrame(rows, columns=["product_id", "category", "revenue"]).astype(
        {"product_id": "int64", "category": "object", "revenue": "float64"}
    )
    frame = frame.sort_values("product_id", kind="stable")  # 'first' then breaks ties by id
    by_category = frame.groupby("category")["revenue"]
    frame["row_number"] = by_category.rank(method="first", ascending=False, na_option="bottom")
    frame["rank"] = by_category.rank(method="min", ascending=False, na_option="bottom")
    frame["dense_rank"] = by_category.rank(method="dense", ascending=False, na_option="bottom")
    for column in ("row_number", "rank", "dense_rank"):
        frame[column] = frame[column].astype("int64")
    ordered = frame.sort_values(["category", "row_number"], kind="stable")
    return ordered[COLUMNS].reset_index(drop=True)


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


def _rows(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [tuple(_plain(v) for v in row) for row in frame.itertuples(index=False, name=None)]


def _check_same_result(got: pd.DataFrame, want: pd.DataFrame) -> None:
    if list(got.columns) != COLUMNS:
        _fail(f"output columns differ: expected {COLUMNS}, got {list(got.columns)}")
    got_rows, want_rows = _rows(got), _rows(want)
    if len(got_rows) != len(want_rows):
        _fail(
            f"row count differs: your query returned {len(got_rows)} rows. Return one row per "
            "product, including single-product and all-NULL categories."
        )
    for got_row, want_row in zip(got_rows, want_rows, strict=True):
        if (got_row[0], got_row[1]) != (want_row[0], want_row[1]):
            _fail(
                "a product is in the wrong row: order by category ascending, then row_number "
                "ascending, and break revenue ties by product_id ascending"
            )
        if not _same_value(got_row[2], want_row[2]):
            _fail("revenue differs: take revenue from products, and keep NULL as NULL")
        if not _same_value(got_row[3], want_row[3]):
            _fail(
                "row_number differs: ROW_NUMBER must break revenue ties by product_id ascending "
                "and place NULL revenue last"
            )
        if not _same_value(got_row[4], want_row[4]):
            _fail(
                "rank differs: RANK ties equal revenues (order by revenue only), gives the next "
                "rank after a tie, and treats NULL revenues as peers that sort last"
            )
        if not _same_value(got_row[5], want_row[5]):
            _fail(
                "dense_rank differs: DENSE_RANK ties equal revenues and does not skip the next "
                "rank, and treats NULL revenues as peers that sort last"
            )


def _random_rows(rng: np.random.Generator) -> list[tuple[Any, ...]]:
    size = int(rng.integers(0, 26))
    ids = rng.choice(ID_POOL, size=size, replace=False)
    rows: list[tuple[Any, ...]] = []
    for product_id in ids:
        category = CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]
        revenue = REVENUES[int(rng.integers(0, len(REVENUES)))]
        rows.append((int(product_id), category, revenue))
    return [rows[int(i)] for i in rng.permutation(len(rows))]  # insertion order is free


def test_random_tables_match_the_pandas_oracle() -> None:
    for seed in range(60):
        rows = _random_rows(np.random.default_rng(seed))
        got = _run_learner_query(_connect(rows))
        _check_same_result(got, _oracle(rows))


@pytest.fixture(scope="module")
def event_log_products() -> list[tuple[Any, ...]]:
    """Real rows of an online retailer (online_retail): 1,500 invoice lines.

    The id is the line's number, the category is the product description exactly as typed and
    the revenue is the line total. A product sells at a few price points, so equal revenues
    inside a category are common, and a return has a negative revenue. Descriptions carry
    capitals, leading spaces and doubled spaces, and seven products are stored under spellings
    that differ only in spaces: those are separate categories. No two real spellings differ
    only in letter case, and the contract has a rule for that, so one line in seven has its
    description put in lower case here.
    """
    events = retail_events()
    gen = rng("sql/sql-ranking-window-functions/p01-rank-within-category:event-log:0")
    picked = events.iloc[gen.permutation(len(events))[:1500]]
    lowered = _real_unit(gen, len(picked)) < 1.0 / 7.0
    return [
        (int(product_id), str(category).lower() if lower else str(category), float(revenue))
        for product_id, category, revenue, lower in zip(
            picked["event_id"], picked["label"], picked["amount"], lowered, strict=True
        )
    ]
    return [
        (int(product_id), str(category), float(revenue))
        for product_id, category, revenue in zip(
            picked["event_id"], picked["label"], picked["amount"], strict=True
        )
    ]


def test_event_log_amounts_and_dirty_categories_match_the_pandas_oracle(
    event_log_products: list[tuple[Any, ...]],
) -> None:
    got = _run_learner_query(_connect(event_log_products))
    _check_same_result(got, _oracle(event_log_products))


def test_ties_are_broken_by_numeric_product_id_not_by_text() -> None:
    rows = [(17, "books", 50.0), (9, "books", 50.0), (1_204, "books", 20.0)]
    got = _run_learner_query(_connect(rows))
    books = got[got["category"] == "books"].reset_index(drop=True)
    assert [int(v) for v in books["product_id"]] == [9, 17, 1_204], (
        "the tie-break must compare product ids numerically rather than lexicographically"
    )
    _check_same_result(got, _oracle(rows))


def test_zero_revenue_ranks_above_null_revenue() -> None:
    rows = [(1, "tools", None), (2, "tools", 0.0), (3, "tools", None)]
    got = _run_learner_query(_connect(rows))
    assert [int(v) for v in got["product_id"]] == [2, 1, 3], (
        "0.0 is a number and ranks above every NULL: NULL revenue sorts last"
    )
    _check_same_result(got, _oracle(rows))


def test_null_revenues_are_peers_for_rank_and_dense_rank() -> None:
    rows = [(5, "games", 30.0), (6, "games", None), (7, "games", None), (8, "games", 10.0)]
    got = _run_learner_query(_connect(rows))
    nulls = got[got["revenue"].isna()].reset_index(drop=True)
    assert [int(v) for v in nulls["rank"]] == [3, 3], (
        "NULL revenues are peers: their rank must follow all non-NULL rows"
    )
    assert [int(v) for v in nulls["dense_rank"]] == [3, 3], (
        "NULL revenues are peers: their dense rank must follow the distinct non-NULL values"
    )
    _check_same_result(got, _oracle(rows))


def test_all_null_category_is_one_peer_group_ordered_by_product_id() -> None:
    rows = [(42, "toys", None), (3, "toys", None), (101, "toys", None)]
    got = _run_learner_query(_connect(rows))
    assert [int(v) for v in got["product_id"]] == [3, 42, 101], (
        "in an all-NULL category the row numbers follow product_id"
    )
    assert [int(v) for v in got["rank"]] == [1, 1, 1], (
        "all revenues are NULL and so are peers: every row gets rank 1"
    )
    _check_same_result(got, _oracle(rows))


def test_single_product_category_is_rank_one() -> None:
    rows = [(17, "solo", 12.0), (9, "books", 5.0), (42, "books", 7.0)]
    got = _run_learner_query(_connect(rows))
    solo = got[got["category"] == "solo"].reset_index(drop=True)
    assert [int(v) for v in solo["row_number"]] == [1], "a one-product category has row number 1"
    assert [int(v) for v in solo["rank"]] == [1], "a one-product category has rank 1"
    _check_same_result(got, _oracle(rows))


def test_rank_skips_after_a_tie_and_dense_rank_does_not() -> None:
    rows = [(1, "tools", 10.0), (2, "tools", 10.0), (3, "tools", 5.0)]
    got = _run_learner_query(_connect(rows))
    last = got[got["product_id"] == 3].iloc[0]
    assert int(last["rank"]) == 3 and int(last["dense_rank"]) == 2, (
        "after tied revenues, RANK skips positions while DENSE_RANK advances once"
    )
    _check_same_result(got, _oracle(rows))


def test_large_non_sequential_ids_tie_and_order_correctly() -> None:
    rows = [
        (310_552_910_887, "games", 70.0),
        (6_000_000_001, "games", 70.0),
        (905_113_208, "games", 70.0),
        (17_204, "games", 5.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert [int(v) for v in got["product_id"]] == [
        905_113_208,
        6_000_000_001,
        310_552_910_887,
        17_204,
    ], "products tied in revenue must be ordered by their numeric product_id"
    _check_same_result(got, _oracle(rows))


def test_empty_table_returns_zero_rows_with_the_six_columns() -> None:
    got = _run_learner_query(_connect([]))
    assert list(got.columns) == COLUMNS, "an empty table must still give the six columns"
    assert len(got) == 0, "an empty table has no products, so there are no rows"


def test_every_product_appears_exactly_once() -> None:
    rows = [(i, CATEGORIES[i % 4], REVENUES[i % len(REVENUES)]) for i in range(1, 41)]
    got = _run_learner_query(_connect(rows))
    assert len(got) == len(rows) and got["product_id"].nunique() == len(rows), (
        "one row per product: a join or a filter has duplicated or dropped some"
    )
    _check_same_result(got, _oracle(rows))
