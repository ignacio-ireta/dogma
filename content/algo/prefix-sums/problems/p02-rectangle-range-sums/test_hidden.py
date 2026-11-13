from __future__ import annotations

import signal
from copy import deepcopy

import numpy as np
import pytest
from scipy.special import ndtr
from submission import rectangle_sums

from dogma_data import rng as data_rng
from dogma_data import sample_rows


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


def _call(grid, queries):
    try:
        result = rectangle_sums(grid, queries)
    except TimeoutError:
        pytest.fail("rectangle query batch is too slow: build one prefix table before the queries")
    except Exception as exc:
        pytest.fail(f"valid rectangle input raised {type(exc).__name__}")
    if not isinstance(result, list) or any(type(value) is not int for value in result):
        pytest.fail("return a list of exact Python integer sums")
    return result


def _equal(got, expected):
    if got != expected:
        pytest.fail("rectangle sum differs: use inclusive corners and four prefix lookups")


def _reject(grid, queries, error):
    try:
        rectangle_sums(grid, queries)
    except error:
        return
    except Exception as exc:
        pytest.fail(f"invalid rectangle input raised the wrong type: {type(exc).__name__}")
    pytest.fail("invalid rectangle input must raise its contracted exception")


def test_seeded_small_grids_match_direct_nested_sums():
    rng = data_rng("rectangle-sums-p02:brute-force")
    for _ in range(30):
        rows, columns = (int(v) for v in _real_ints(rng, 1, 9, 2))
        # A real image: the top-left rows by columns pixels of a handwritten digit (8 by 8,
        # intensities 0 to 16), with 8 subtracted so that the grid holds both signs. Blank
        # background makes long runs of equal cells.
        image = sample_rows("digits", rng, 1)[0].reshape(8, 8)[:rows, :columns] - 8.0
        grid = [[int(v) for v in row] for row in image]
        queries = []
        for _ in range(20):
            top, bottom = sorted(int(v) for v in rng.integers(0, rows, size=2))
            left, right = sorted(int(v) for v in rng.integers(0, columns, size=2))
            queries.append((top, left, bottom, right))
        expected = [
            sum(grid[r][c] for r in range(top, bottom + 1) for c in range(left, right + 1))
            for top, left, bottom, right in queries
        ]
        _equal(_call(grid, queries), expected)


def test_padding_handles_top_left_and_single_row_or_column():
    _equal(_call([[2, -1, 5]], [(0, 0, 0, 0), (0, 0, 0, 2), (0, 1, 0, 2)]), [2, 6, 4])
    _equal(_call([[2], [-1], [5]], [(0, 0, 2, 0), (1, 0, 1, 0)]), [6, -1])


def test_empty_dimensions_need_empty_queries():
    _equal(_call([], []), [])
    _equal(_call([[]], []), [])
    _equal(_call([[], []], []), [])
    _reject([], [(0, 0, 0, 0)], ValueError)
    _reject([[]], [(0, 0, 0, 0)], ValueError)


def test_query_order_and_duplicate_queries_are_preserved():
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    _equal(_call(grid, [[2, 1, 2, 2], [0, 0, 2, 2], [2, 1, 2, 2]]), [17, 45, 17])


def test_arbitrary_precision_integers_are_not_cast_to_float_or_fixed_width():
    huge = 10**100
    _equal(_call([[huge, -huge], [1, 2]], [(0, 0, 1, 1), (0, 0, 0, 0)]), [3, huge])


def test_grid_and_query_records_are_not_mutated():
    grid = [[2, -3], [8, 1]]
    queries = [[0, 0, 1, 1], [0, 1, 1, 1]]
    original_grid, original_queries = deepcopy(grid), deepcopy(queries)
    _call(grid, queries)
    if grid != original_grid or queries != original_queries:
        pytest.fail("rectangle queries must not mutate the grid or query records")


@pytest.mark.parametrize(
    ("grid", "queries", "error"),
    [
        ([[1], [2, 3]], [], ValueError),
        ([[True]], [], TypeError),
        ([[1.0]], [], TypeError),
        ([[np.int64(1)]], [], TypeError),
        (((1, 2),), [], TypeError),
        ([(1, 2)], [], TypeError),
        ([[1]], (), TypeError),
        ([[1]], [(0, 0, 0)], ValueError),
        ([[1]], [(0, 0, 0, 0, 0)], ValueError),
        ([[1]], [(False, 0, 0, 0)], TypeError),
        ([[1]], [(0, 0, 0.0, 0)], TypeError),
        ([[1]], [0], TypeError),
        ([[1]], [(-1, 0, 0, 0)], ValueError),
        ([[1]], [(0, 0, 1, 0)], ValueError),
        ([[1, 2]], [(0, 1, 0, 0)], ValueError),
    ],
    ids=[
        "ragged",
        "bool-cell",
        "float-cell",
        "numpy-cell",
        "grid-container",
        "row-container",
        "queries-container",
        "short-query",
        "long-query",
        "bool-bound",
        "float-bound",
        "query-container",
        "negative-bound",
        "past-edge",
        "reversed-bound",
    ],
)
def test_invalid_shapes_types_and_bounds_are_rejected(grid, queries, error):
    _reject(grid, queries, error)


def test_large_batch_reuses_one_prefix_table():
    rows = columns = 300
    grid = [[1] * columns for _ in range(rows)]
    queries = [(i % 19, i % 23, rows - 1 - i % 7, columns - 1 - i % 11) for i in range(50_000)]
    expected = [(bottom - top + 1) * (right - left + 1) for top, left, bottom, right in queries]

    def expired(_signum, _frame):
        raise TimeoutError("rectangle batch exceeded its budget")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 4.0)
    try:
        got = _call(grid, queries)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    _equal(got, expected)


def test_query_that_is_not_a_tuple_or_list_raises_type_error():
    grid = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    query = {0: "top", 1: "left", 2: "bottom", 3: "right"}
    try:
        rectangle_sums(grid, [query])
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(
            f"a query of the wrong container type raised {type(exc).__name__}, not TypeError"
        )
    pytest.fail("each query must be a tuple or list: a dict query must raise TypeError")


def test_negative_left_bound_is_rejected_with_value_error():
    grid = [[1, 2], [3, 4]]
    try:
        rectangle_sums(grid, [(0, -1, 0, 0)])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a negative left column raised {type(exc).__name__}, not ValueError")
    pytest.fail(
        "rectangle bounds must lie inside the grid: a negative left column must raise ValueError"
    )


def test_unsorted_grid_rows_and_queries_keep_their_order():
    grid = [[5, -3, 2], [1, 8, -4]]
    queries = [[1, 0, 1, 2], [0, 1, 1, 1], [0, 0, 0, 0]]
    _call(grid, queries)
    if grid != [[5, -3, 2], [1, 8, -4]]:
        pytest.fail("the grid was reordered or changed: never sort or write into the input")
    if queries != [[1, 0, 1, 2], [0, 1, 1, 1], [0, 0, 0, 0]]:
        pytest.fail("the query list was reordered or changed: never sort the caller's queries")
