import re

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import matmul

SMALL = st.integers(min_value=-9, max_value=9)
MISMATCHED_SHAPES = [
    ((2, 3), (2, 3)),
    ((3, 2), (3, 2)),
    ((1, 2), (3, 1)),
    ((2, 2), (3, 5)),
    ((4, 1), (2, 4)),
]


@st.composite
def int_matrices(draw, max_dim=4):
    """A random (m, n) matrix of small integers, as a list of rows."""
    m = draw(st.integers(min_value=1, max_value=max_dim))
    n = draw(st.integers(min_value=1, max_value=max_dim))
    return [draw(st.lists(SMALL, min_size=n, max_size=n)) for _ in range(m)]


@st.composite
def product_pairs(draw, max_dim=4):
    """Random A (m, n) and B (n, p), so that AB is defined."""
    m = draw(st.integers(min_value=1, max_value=max_dim))
    n = draw(st.integers(min_value=1, max_value=max_dim))
    p = draw(st.integers(min_value=1, max_value=max_dim))
    A = [draw(st.lists(SMALL, min_size=n, max_size=n)) for _ in range(m)]
    B = [draw(st.lists(SMALL, min_size=p, max_size=p)) for _ in range(n)]
    return A, B


@st.composite
def product_triples(draw, max_dim=3):
    """Random A (m, n), B (n, p) and C (p, q), so that both (AB)C and A(BC) are defined."""
    m = draw(st.integers(min_value=1, max_value=max_dim))
    n = draw(st.integers(min_value=1, max_value=max_dim))
    p = draw(st.integers(min_value=1, max_value=max_dim))
    q = draw(st.integers(min_value=1, max_value=max_dim))
    A = [draw(st.lists(SMALL, min_size=n, max_size=n)) for _ in range(m)]
    B = [draw(st.lists(SMALL, min_size=p, max_size=p)) for _ in range(n)]
    C = [draw(st.lists(SMALL, min_size=q, max_size=q)) for _ in range(p)]
    return A, B, C


def _assert_close(actual, desired, rtol=1e-7, atol=0.0, err_msg=""):
    """Like np.testing.assert_allclose, but the message is the first line of the failure."""
    got = np.asarray(actual, dtype=np.float64)
    want = np.asarray(desired, dtype=np.float64)
    if want.ndim == 0:
        want = np.broadcast_to(want, got.shape)
    if got.shape != want.shape:
        raise AssertionError(f"{err_msg}: shape {got.shape}, expected {want.shape}")
    if not np.allclose(got, want, rtol=rtol, atol=atol):
        worst = float(np.max(np.abs(got - want)))
        raise AssertionError(f"{err_msg} (largest difference {worst:.3g})")


def _assert_equal(actual, desired, err_msg=""):
    """Exact array equality, with the message first, as in _assert_close."""
    if not np.array_equal(np.asarray(actual), np.asarray(desired)):
        raise AssertionError(err_msg)


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def _names_shape(message, shape):
    return re.search(rf"\({shape[0]},\s*{shape[1]}\)", message) is not None


def test_matches_numpy_matmul_on_random_integer_matrices():
    rng = np.random.default_rng(0)
    for m, n, p in [(1, 1, 1), (1, 4, 1), (4, 1, 3), (3, 3, 3), (2, 5, 4), (6, 7, 3)]:
        A = rng.integers(-5, 6, size=(m, n)).tolist()
        B = rng.integers(-5, 6, size=(n, p)).tolist()
        expected = (np.array(A) @ np.array(B)).tolist()
        assert matmul(A, B) == expected, (
            "integer product differs from numpy's @: "
            "check the shared index t and which row and column each entry reads"
        )


def test_matches_numpy_matmul_on_random_float_matrices():
    rng = np.random.default_rng(1)
    for m, n, p in [(2, 3, 4), (5, 1, 5), (1, 6, 1), (6, 6, 6)]:
        A = rng.normal(size=(m, n)).tolist()
        B = rng.normal(size=(n, p)).tolist()
        _assert_close(
            np.array(matmul(A, B), dtype=float),
            np.array(A) @ np.array(B),
            rtol=1e-9,
            atol=1e-9,
            err_msg="float product differs from numpy's @: check the summed index and the shape",
        )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(int_matrices())
def test_identity_matrices_are_neutral_on_both_sides(matrix):
    m, n = len(matrix), len(matrix[0])
    assert matmul(np.eye(m, dtype=int).tolist(), matrix) == matrix, (
        "I_m @ A should equal A: check the identity-case indexing (row i, column j)"
    )
    assert matmul(matrix, np.eye(n, dtype=int).tolist()) == matrix, (
        "A @ I_n should equal A: check that the result keeps A's row count"
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(product_pairs())
def test_result_has_rows_of_a_and_columns_of_b(pair):
    A, B = pair
    out = matmul(A, B)
    assert len(out) == len(A), "shape mismatch: the product should have one row per row of A"
    assert all(len(row) == len(B[0]) for row in out), (
        "shape mismatch: each row of the product should have one entry per column of B"
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(product_pairs())
def test_transpose_of_product_is_product_of_transposes_in_reverse_order(pair):
    A, B = pair
    At = np.array(A).T.tolist()
    Bt = np.array(B).T.tolist()
    lhs = np.array(matmul(A, B)).T.tolist()
    rhs = matmul(Bt, At)
    assert lhs == rhs, "(AB)^T should equal B^T A^T: the factor order must reverse"


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(product_triples())
def test_associativity_holds_exactly_for_small_integer_matrices(triple):
    A, B, C = triple
    left = matmul(matmul(A, B), C)
    right = matmul(A, matmul(B, C))
    assert left == right, (
        "(AB)C should equal A(BC) exactly for integers: check the summed index in each product"
    )


@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    MISMATCHED_SHAPES,
    ids=[
        "same-shape-wide",
        "same-shape-tall",
        "row-by-column-mismatch",
        "square-by-wider",
        "column-by-short-wide",
    ],
)
def test_inner_dimension_mismatch_raises_and_names_both_shapes(a_shape, b_shape):
    A = [[1] * a_shape[1] for _ in range(a_shape[0])]
    B = [[1] * b_shape[1] for _ in range(b_shape[0])]
    exc = _raised_value_error(
        lambda: matmul(A, B),
        "mismatched inner dimensions make the product undefined: matmul must raise ValueError",
    )
    message = str(exc)
    assert _names_shape(message, a_shape) and _names_shape(message, b_shape), (
        f"the ValueError should name both shapes as tuples, like (2, 3) @ (2, 2); got: {message}"
    )


def test_ragged_input_raises_value_error():
    _raised_value_error(
        lambda: matmul([[1, 2], [3]], [[1], [1]]),
        "ragged rows must raise ValueError; truncating the short row gives a wrong product",
    )


def test_flat_list_input_raises_value_error():
    _raised_value_error(
        lambda: matmul([1, 2, 3], [[1], [2], [3]]),
        "a flat list is a vector, not a list of rows: raise ValueError instead of guessing",
    )


def test_zero_width_inputs_give_empty_results():
    assert matmul([], []) == [], "[] @ [] is (0, 0) @ (0, 0), which should give []"
    assert matmul([[], []], []) == [[], []], (
        "zero-width inputs: the inner dimension is 0 on both sides, so each output row is empty"
    )


def test_returns_a_list_of_row_lists():
    out = matmul([[1, 2]], [[3], [4]])
    assert isinstance(out, list) and all(isinstance(row, list) for row in out), (
        "return a list of row lists, as the contract says, not a tuple or an array"
    )


def test_integer_inputs_give_integer_entries_and_float_inputs_give_floats():
    ints = matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    assert all(type(x) is int for row in ints for x in row), (
        "integer inputs should give integer entries: the contract keeps ints as ints"
    )
    floats = matmul([[1.5, 2.0]], [[2], [3]])
    assert all(type(x) is float for row in floats for x in row), (
        "a float input should give float entries"
    )


def test_row_times_column_is_not_the_entrywise_product():
    assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]], (
        "each entry is a row of A dotted with a column of B, not an entrywise product"
    )


def test_result_is_ab_not_ba():
    A = [[1, 2], [3, 4]]
    B = [[0, 1], [1, 0]]
    assert matmul(A, B) == [[2, 1], [4, 3]], (
        "matmul(A, B) must compute A times B: the operands look swapped"
    )


def test_inputs_are_not_mutated():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[7, 8], [9, 10], [11, 12]]
    left_before = [row[:] for row in left]
    right_before = [row[:] for row in right]
    matmul(left, right)
    assert left == left_before, "input A was modified in place"
    assert right == right_before, "input B was modified in place"
