import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from submission import transpose

INTS = st.integers(min_value=-1000, max_value=1000)


@st.composite
def rectangular_matrices(draw, min_cols=0):
    """A matrix as a list of rows: n_rows rows, each of length n_cols."""
    n_cols = draw(st.integers(min_value=min_cols, max_value=6))
    n_rows = draw(st.integers(min_value=1, max_value=6))
    return [draw(st.lists(INTS, min_size=n_cols, max_size=n_cols)) for _ in range(n_rows)]


def _must_raise_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_matches_numpy_transpose_on_random_rectangular_matrices():
    rng = np.random.default_rng(0)
    for n_rows, n_cols in [(1, 1), (1, 5), (5, 1), (3, 4), (4, 3), (6, 6), (2, 0)]:
        matrix = rng.integers(-50, 50, size=(n_rows, n_cols)).tolist()
        expected = np.array(matrix).T.tolist()
        assert transpose(matrix) == expected, (
            "values differ from numpy's transpose: check that result[j][i] == matrix[i][j]"
        )


def test_float_values_are_copied_exactly():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(3, 4)).tolist()
    assert transpose(matrix) == np.array(matrix).T.tolist(), (
        "float entries changed: copy each value as it is, without int() or a float32 cast"
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(rectangular_matrices())
def test_shape_swaps_rows_and_columns(matrix):
    n_rows, n_cols = len(matrix), len(matrix[0])
    out = transpose(matrix)
    assert len(out) == n_cols, (
        "shape mismatch: the transpose should have one row per column of the input"
    )
    assert all(len(row) == n_rows for row in out), (
        "shape mismatch: each row of the transpose should have one entry per row of the input"
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(rectangular_matrices(min_cols=1))
def test_transposing_twice_returns_the_original(matrix):
    assert transpose(transpose(matrix)) == matrix, (
        "transpose(transpose(A)) should equal A: some entry is in the wrong place"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=5), st.integers(min_value=0, max_value=5))
def test_ragged_rows_always_raise_value_error(len_a, len_b):
    assume(len_a != len_b)
    matrix = [[1] * len_a, [2] * len_b]
    _must_raise_value_error(
        lambda: transpose(matrix),
        "ragged rows must raise ValueError; zip(*rows) would silently truncate to the shortest row",
    )


def test_flat_list_raises_value_error():
    _must_raise_value_error(
        lambda: transpose([1, 2, 3]),
        "a flat list is a vector, not a list of rows: raise ValueError instead of guessing",
    )


def test_empty_and_zero_width_matrices_return_empty_list():
    assert transpose([]) == [], "the empty matrix (0, 0) should transpose to []"
    assert transpose([[], [], []]) == [], (
        "a zero-width matrix should transpose to []: a result with no columns has no rows"
    )


def test_single_row_becomes_single_column():
    assert transpose([[1, 2, 3]]) == [[1], [2], [3]], (
        "a single row should become a single column: one row per entry"
    )


def test_returns_list_of_lists_not_tuples():
    out = transpose([[1, 2], [3, 4]])
    assert isinstance(out, list) and all(isinstance(row, list) for row in out), (
        "return a list of lists; tuples (what zip(*rows) yields) do not match the contract"
    )


def test_input_matrix_is_not_mutated():
    matrix = [[1, 2, 3], [4, 5, 6]]
    before = [row[:] for row in matrix]
    transpose(matrix)
    assert matrix == before, "input matrix was modified: transpose must build a new result"


def test_result_does_not_share_rows_with_input():
    matrix = [[1, 2], [3, 4]]
    out = transpose(matrix)
    out[0][0] = 99
    assert matrix == [[1, 2], [3, 4]], (
        "result shares row objects with the input: editing the output changed the input"
    )


def test_tuple_rows_are_accepted_as_rows():
    # The Contract allows each row to be a list or a tuple. Here both rows are tuples,
    # and the transpose must still take column j from every row: (1, 4), (2, 5), (3, 6).
    matrix = [(1, 2, 3), (4, 5, 6)]
    try:
        out = transpose(matrix)
    except ValueError as exc:
        pytest.fail(f"rows given as tuples are valid matrix rows; raised ValueError: {exc}")
    assert out == [[1, 4], [2, 5], [3, 6]], (
        "rows given as tuples are valid matrix rows: transpose them the same as lists"
    )


def test_int_entries_stay_ints():
    # Entries are copied unchanged. 1.0 == 1 in Python, so equality cannot see a float
    # conversion: compare the types of the entries instead.
    out = transpose([[1, 2, 3], [4, 5, 6]])
    assert all(type(x) is int for row in out for x in row), (
        "int entries must stay ints in the transpose: copy each value, do not convert it to float"
    )
