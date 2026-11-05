from __future__ import annotations

import signal
from copy import deepcopy

import numpy as np
import pytest
from submission import rectangle_sums


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
    rng = np.random.default_rng(421)
    for _ in range(30):
        rows, columns = (int(v) for v in rng.integers(1, 9, size=2))
        grid = [[int(v) for v in row] for row in rng.integers(-20, 21, size=(rows, columns))]
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
