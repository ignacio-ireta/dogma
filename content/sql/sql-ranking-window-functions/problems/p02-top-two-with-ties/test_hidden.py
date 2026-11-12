"""Hidden tests: ties at the cut, NULL ranks, all-NULL and single-product categories, and ids.

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

COLUMNS = ["category", "product_id", "revenue", "dense_rank"]
PRODUCTS_DDL = "CREATE TABLE products (product_id BIGINT, category VARCHAR, revenue DOUBLE)"
CATEGORIES = ["audio", "video", "toys"]
REVENUES: list[float | None] = [None, 0.0, 10.0, 10.0, 25.5, 99.0, 25.5, 99.0, None]
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
            "category, product_id, revenue and dense_rank, in that order."
        )
    return frame


def _oracle(rows: Sequence[tuple[Any, ...]]) -> pd.DataFrame:
    """Independent answer in pandas: dense ranks per category, NULLs last, keep rank <= 2."""
    frame = pd.DataFrame(rows, columns=["product_id", "category", "revenue"]).astype(
        {"product_id": "int64", "category": "object", "revenue": "float64"}
    )
    frame["dense_rank"] = (
        frame.groupby("category")["revenue"]
        .rank(method="dense", ascending=False, na_option="bottom")
        .astype("int64")
    )
    kept = frame[frame["dense_rank"] <= 2]
    ordered = kept.sort_values(["category", "dense_rank", "product_id"], kind="stable")
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
            f"row count differs: your query returned {len(got_rows)} rows. Keep every product "
            "whose dense rank is at most 2, including the ties at the cut."
        )
    for got_row, want_row in zip(got_rows, want_rows, strict=True):
        if (got_row[0], got_row[1]) != (want_row[0], want_row[1]):
            _fail(
                "a product is in the wrong row: the kept products must be the ones with dense rank "
                "<= 2, ordered by category, dense_rank, then product_id"
            )
        if not _same_value(got_row[2], want_row[2]):
            _fail("revenue differs: take revenue from products, and keep NULL as NULL")
        if not _same_value(got_row[3], want_row[3]):
            _fail(
                "dense_rank differs: use DENSE_RANK, with NULL revenue last and NULLs as peers "
                "of each other"
            )


def _random_rows(rng: np.random.Generator) -> list[tuple[Any, ...]]:
    size = int(rng.integers(0, 22))
    ids = rng.choice(ID_POOL, size=size, replace=False)
    rows = [
        (
            int(product_id),
            CATEGORIES[int(rng.integers(0, len(CATEGORIES)))],
            REVENUES[int(rng.integers(0, len(REVENUES)))],
        )
        for product_id in ids
    ]
    return [rows[int(i)] for i in rng.permutation(len(rows))]


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
    gen = rng("sql/sql-ranking-window-functions/p02-top-two-with-ties:event-log:0")
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


def test_tie_at_the_cut_brings_in_more_than_two_rows() -> None:
    rows = [(1, "audio", 90.0), (2, "audio", 60.0), (3, "audio", 60.0), (4, "audio", 60.0)]
    got = _run_learner_query(_connect(rows))
    assert sorted(int(v) for v in got["product_id"]) == [1, 2, 3, 4], (
        "all products tied at the retained dense-rank boundary must be returned"
    )
    _check_same_result(got, _oracle(rows))


def test_row_number_would_drop_the_tied_products() -> None:
    rows = [(1, "audio", 70.0), (2, "audio", 70.0), (3, "audio", 40.0)]
    got = _run_learner_query(_connect(rows))
    assert sorted(int(v) for v in got["product_id"]) == [1, 2, 3], (
        "tied top revenues share one dense rank; the next distinct revenue is still retained"
    )
    _check_same_result(got, _oracle(rows))


def test_rank_would_skip_past_the_second_place_after_a_tie() -> None:
    rows = [(10, "video", 100.0), (11, "video", 100.0), (12, "video", 90.0), (13, "video", 80.0)]
    got = _run_learner_query(_connect(rows))
    assert 12 in got["product_id"].tolist() and 13 not in got["product_id"].tolist(), (
        "keep products at the top two distinct revenue values, including ties a"
        "nd excluding lower values"
    )
    _check_same_result(got, _oracle(rows))


def test_null_revenue_ranks_below_every_number_and_is_peer_of_other_nulls() -> None:
    rows = [(5, "toys", 30.0), (6, "toys", None), (7, "toys", None), (8, "toys", 10.0)]
    got = _run_learner_query(_connect(rows))
    assert sorted(int(v) for v in got["product_id"]) == [5, 8], (
        "when two distinct non-NULL revenue values fill the retained ranks, NUL"
        "L revenues must be excluded"
    )
    _check_same_result(got, _oracle(rows))


def test_null_revenues_are_kept_when_nothing_else_fills_the_top_two() -> None:
    rows = [(5, "toys", 30.0), (6, "toys", None), (7, "toys", None)]
    got = _run_learner_query(_connect(rows))
    assert sorted(int(v) for v in got["product_id"]) == [5, 6, 7], (
        "with one number, the NULL peers take dense rank 2 and are kept, both of them"
    )
    _check_same_result(got, _oracle(rows))


def test_all_null_category_keeps_every_product() -> None:
    rows = [(42, "toys", None), (3, "toys", None), (101, "toys", None), (9, "audio", 5.0)]
    got = _run_learner_query(_connect(rows))
    toys = got[got["category"] == "toys"]
    assert sorted(int(v) for v in toys["product_id"]) == [3, 42, 101], (
        "every product in an all-NULL category is at dense rank 1, so every product is kept"
    )
    _check_same_result(got, _oracle(rows))


def test_single_product_category_is_kept_at_rank_one() -> None:
    rows = [(17, "video", 12.0), (9, "audio", 5.0), (42, "audio", 7.0), (3, "audio", 6.0)]
    got = _run_learner_query(_connect(rows))
    video = got[got["category"] == "video"]
    assert [int(v) for v in video["dense_rank"]] == [1], "a one-product category is at dense rank 1"
    _check_same_result(got, _oracle(rows))


def test_ties_are_ordered_by_numeric_product_id_within_the_rank() -> None:
    rows = [(17, "audio", 50.0), (9, "audio", 50.0), (1_204, "audio", 20.0)]
    got = _run_learner_query(_connect(rows))
    assert [int(v) for v in got["product_id"]] == [9, 17, 1_204], (
        "rows of one dense rank must be ordered by product_id numerically rathe"
        "r than lexicographically"
    )
    _check_same_result(got, _oracle(rows))


def test_large_non_sequential_ids_are_kept_and_ordered_correctly() -> None:
    rows = [
        (905_113_208, "toys", 70.0),
        (6_000_000_001, "toys", 70.0),
        (310_552_910_887, "toys", 40.0),
        (17_204, "toys", 5.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert [int(v) for v in got["product_id"]] == [905_113_208, 6_000_000_001, 310_552_910_887], (
        "dense ranks 1, 1 and 2 are kept, ordered by dense rank and then numeric product_id"
    )
    _check_same_result(got, _oracle(rows))


def test_empty_table_returns_zero_rows_with_the_four_columns() -> None:
    got = _run_learner_query(_connect([]))
    assert list(got.columns) == COLUMNS, "an empty table must still give the four columns"
    assert len(got) == 0, "an empty table has no products, so there are no rows"


def test_output_order_is_category_then_dense_rank_then_product_id() -> None:
    rows = [
        (30, "video", 5.0),
        (20, "audio", 9.0),
        (10, "audio", 9.0),
        (40, "video", 7.0),
        (50, "audio", 3.0),
    ]
    got = _run_learner_query(_connect(rows))
    assert [int(v) for v in got["product_id"]] == [10, 20, 50, 40, 30], (
        "the output is ordered by category, then dense_rank, then product_id"
    )
    _check_same_result(got, _oracle(rows))
