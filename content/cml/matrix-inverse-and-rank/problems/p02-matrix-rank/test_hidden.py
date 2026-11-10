import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import matrix_rank

SEEDS = st.integers(min_value=0, max_value=10**6)


def _with_rank(seed, m, n, r):
    """An m x n matrix of rank exactly r (a product of random m x r and r x n factors)."""
    rng = np.random.default_rng(seed)
    if r == 0:
        return np.zeros((m, n))
    return rng.normal(size=(m, r)) @ rng.normal(size=(r, n))


def _must_raise_value_error(call, mistake):
    """Fail with the kind of mistake, instead of pytest's generic 'DID NOT RAISE'."""
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from exc
    raise AssertionError(mistake)


@pytest.mark.parametrize(
    ("m", "n", "r", "seed"),
    [
        (4, 4, 4, 1),
        (5, 3, 2, 2),
        (3, 6, 3, 3),
        (6, 6, 3, 4),
        (8, 5, 4, 5),
        (1, 4, 1, 6),
        (7, 7, 0, 7),
    ],
    ids=list("abcdefg"),
)
def test_matches_numpy_rank_on_matrices_of_known_rank(m, n, r, seed):
    A = _with_rank(seed, m, n, r)
    got = matrix_rank(A)
    assert got == r, f"rank mismatch for a {m}x{n} matrix of rank {r}: got {got}"
    assert got == np.linalg.matrix_rank(A), "disagrees with the NumPy reference rank"


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=8))
def test_full_rank_random_rectangular_matrix_has_rank_min_m_n(seed, m, n):
    A = np.random.default_rng(seed).normal(size=(m, n))
    assert matrix_rank(A) == min(m, n), (
        f"a random {m}x{n} matrix has full rank {min(m, n)}: check the pivot skipping"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, st.integers(min_value=1, max_value=7), st.integers(min_value=1, max_value=7))
def test_rank_of_transpose_equals_rank(seed, m, n):
    rng = np.random.default_rng(seed)
    A = _with_rank(seed, m, n, int(rng.integers(0, min(m, n) + 1)))
    assert matrix_rank(A.T) == matrix_rank(A), "rank(A^T) must equal rank(A)"


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, st.integers(min_value=2, max_value=7), st.integers(min_value=1, max_value=7))
def test_row_permutation_and_nonzero_row_scaling_keep_the_rank(seed, m, n):
    rng = np.random.default_rng(seed)
    A = _with_rank(seed, m, n, int(rng.integers(1, min(m, n) + 1)))
    scales = rng.choice([-3.0, -0.5, 0.25, 2.0], size=(m, 1))
    perm = rng.permutation(m)
    assert matrix_rank((scales * A)[perm]) == matrix_rank(A), (
        "permuting rows or scaling them by nonzero factors must not change the rank"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, st.integers(min_value=2, max_value=7), st.integers(min_value=1, max_value=7))
def test_adding_a_combination_of_rows_does_not_increase_the_rank(seed, m, n):
    rng = np.random.default_rng(seed)
    A = _with_rank(seed, m, n, int(rng.integers(1, min(m, n) + 1)))
    combo = 1.5 * A[0] - 2.0 * A[1]
    stacked = np.vstack([A, combo])
    assert matrix_rank(stacked) == matrix_rank(A), (
        "a row that is a combination of existing rows must not raise the rank"
    )


@pytest.mark.parametrize(("m", "n", "seed"), [(3, 3, 11), (4, 2, 12), (2, 5, 13)], ids=list("abc"))
def test_outer_product_has_rank_one(m, n, seed):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=m)
    v = rng.normal(size=n)
    assert matrix_rank(np.outer(u, v)) == 1, "an outer product of nonzero vectors has rank 1"


@pytest.mark.parametrize("n", [1, 2, 3, 6], ids=list("abcd"))
def test_identity_has_full_rank(n):
    assert matrix_rank(np.eye(n)) == n, f"the {n}x{n} identity must have rank {n}"


def test_zero_matrix_of_any_shape_has_rank_zero():
    assert matrix_rank(np.zeros((4, 1))) == 0, "a zero column has rank 0"
    assert matrix_rank(np.zeros((1, 4))) == 0, "a zero row has rank 0"


def test_integer_input_gives_the_same_rank():
    assert matrix_rank([[2, 0, 1], [4, 0, 2], [1, 1, 1]]) == 2, (
        "integer input should be handled exactly like float input"
    )


def test_negative_tolerance_raises_value_error():
    _must_raise_value_error(
        lambda: matrix_rank(np.eye(2), tol=-1.0),
        "a negative tol must raise ValueError: check tol < 0 before using it",
    )


def test_one_dimensional_input_raises_value_error():
    _must_raise_value_error(
        lambda: matrix_rank([1.0, 2.0, 3.0]),
        "a 1-D input must raise ValueError: check ndim before counting columns",
    )


def test_empty_matrix_raises_value_error():
    _must_raise_value_error(
        lambda: matrix_rank(np.zeros((0, 3))),
        "an empty matrix must raise ValueError: check for zero rows or columns first",
    )


def test_nan_entry_raises_value_error():
    _must_raise_value_error(
        lambda: matrix_rank([[1.0, np.nan], [0.0, 1.0]]),
        "a NaN entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_rank_one_with_rounding_residue_is_not_counted_as_full_rank():
    # A rank-2 3x3 product. Rounding leaves residue of order eps times the matrix size, and a
    # tolerance without the safety margin counts that residue as a third pivot.
    A = _with_rank(81, 3, 3, 2)
    assert matrix_rank(A) == 2, (
        "a rank-2 matrix was counted as full rank: the default tolerance is too small to absorb "
        "rounding residue (keep the safety factor of 1000)"
    )


def test_nearly_collinear_pair_is_rank_one_under_the_default_tolerance():
    # The second pivot, about 2e-13, lies below the default tolerance of about 9e-13 for this matrix
    # (Frobenius norm 2), so the second column adds nothing. A smaller safety factor would count it.
    assert matrix_rank(np.array([[1.0, 1.0], [1.0, 1.0 + 2e-13]])) == 1, (
        "a nearly collinear pair was counted as full rank: the default tolerance is "
        "1000 * max(m, n) * eps * ||A||_F, and its margin must absorb the residue"
    )


def test_clearly_nonzero_second_pivot_is_kept_by_the_default_tolerance():
    # The default tolerance is about 4e-13 here, far below the second pivot of 1e-11.
    assert matrix_rank(np.diag([1.0, 1e-11])) == 2, (
        "a second pivot well above rounding residue was dropped: the default tolerance is too "
        "large, so keep the stated factor of 1000 and no larger"
    )


def test_zero_leading_entry_needs_a_row_swap():
    assert matrix_rank([[0.0, 1.0], [1.0, 0.0]]) == 2, (
        "a zero in the first pivot position needs a row swap: take the largest entry among "
        "the rows not yet used as pivots"
    )


def test_tiny_and_huge_scales_keep_the_rank():
    assert matrix_rank(1e-12 * np.eye(3)) == 3, (
        "a matrix with tiny entries has full rank: the default tolerance must scale with ||A||_F"
    )
    assert matrix_rank(1e12 * np.eye(3)) == 3, (
        "a matrix with huge entries has full rank: the default tolerance must scale with ||A||_F"
    )


def test_zero_tolerance_is_allowed_and_skips_only_exact_zeros():
    assert matrix_rank(np.zeros((2, 2)), tol=0.0) == 0, (
        "tol = 0 is allowed: a pivot of exactly zero must be skipped, not divided by"
    )
    assert matrix_rank(np.eye(2), tol=0.0) == 2, "tol = 0 must still count nonzero pivots"


def test_rank_is_returned_as_an_integer_not_a_float():
    rank = matrix_rank(np.eye(2))
    assert isinstance(rank, (int, np.integer)) and not isinstance(rank, bool), (
        f"the rank must be an int, got {type(rank).__name__}"
    )


def test_input_matrix_is_not_mutated():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    A_copy = A.copy()
    matrix_rank(A)
    np.testing.assert_array_equal(A, A_copy, err_msg="input matrix was modified in place")


def test_tall_nearly_collinear_pair_is_rank_one_under_the_default_tolerance():
    # Ten rows, two columns. The second column differs from the first by 5e-12 in one row. The
    # default tolerance, about 1e-11 here, scales with max(m, n) and ||A||_F and absorbs that
    # residue. A margin scaled by min(m, n), by max|A| or without the row count would count it.
    c = np.ones(10)
    d = c.copy()
    d[-1] += 5e-12
    assert matrix_rank(np.column_stack([c, d])) == 1, (
        "a tall nearly collinear pair was counted as full rank: the default tolerance must scale "
        "with max(m, n) and with the Frobenius norm, so its margin must absorb the residue"
    )


def test_pivot_equal_to_an_explicit_tolerance_is_treated_as_zero():
    # The rule is to skip a column whose magnitude is at or below tol, so a tie is skipped too.
    assert matrix_rank(np.diag([1.0, 0.5]), tol=0.5) == 1, (
        "a pivot equal to tol was counted: skip a column when its magnitude is <= tol"
    )


def test_matrix_with_no_columns_raises_value_error():
    # The statement rejects empty input, and a matrix with rows but no columns is empty too.
    _must_raise_value_error(
        lambda: matrix_rank(np.zeros((3, 0))),
        "a matrix with no columns is empty and must raise ValueError: check the size before"
        " counting pivots",
    )


def test_rank_keeps_counting_after_a_skipped_zero_column():
    # The second column is negligible, so it is skipped, but the third column is independent.
    # A rank count that stops at the first skipped column returns 1 instead of 2.
    assert matrix_rank([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) == 2, (
        "a rank was undercounted: a column with no usable pivot is skipped, and the columns"
        " after it must still be examined"
    )


def test_default_tolerance_keeps_a_pivot_clearly_above_its_stated_threshold():
    # For this 2x2 matrix the default tolerance is about 4.4e-13, and the second pivot 7e-13 lies
    # above it, so the pivot must be counted.
    assert matrix_rank(np.diag([1.0, 7e-13])) == 2, (
        "a pivot above the default tolerance 1000 * max(m, n) * eps * ||A||_F was dropped: the"
        " safety factor is too large"
    )
