import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import gauss_jordan_inverse

from dogma_data import rng as data_rng
from dogma_data import sample_rows

SEEDS = st.integers(min_value=0, max_value=10**6)


def _well_conditioned(seed, n):
    """A square block of real data shifted by n*I: nonsingular, with a determinant far from zero.

    The block is n rows by n standardised measurements (correlated real columns): wine, or
    breast_cancer when more than 13 columns are asked for. The shift keeps the condition
    number small, as it did for the random matrices this replaces.
    """
    name = "wine" if n <= 13 else "breast_cancer"
    gen = data_rng(f"gauss-jordan-p01:block:{seed}:{n}")
    block, _ = sample_rows(name, gen, n, n, standardize=True)
    return block + n * np.eye(n)


def _must_raise_value_error(call, mistake, match=None):
    """Fail with the kind of mistake, instead of pytest's generic 'DID NOT RAISE'."""
    try:
        call()
    except ValueError as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(f"{mistake} (the message should mention '{match}')") from exc
        return
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from exc
    raise AssertionError(mistake)


@pytest.mark.parametrize(
    ("n", "seed"),
    [(1, 1), (2, 2), (3, 3), (5, 4), (8, 5)],
    ids=list("abcde"),
)
def test_matches_numpy_inverse_on_random_well_conditioned_matrices(n, seed):
    A = _well_conditioned(seed, n)
    np.testing.assert_allclose(
        gauss_jordan_inverse(A),
        np.linalg.inv(A),
        rtol=1e-9,
        atol=1e-9,
        err_msg=f"inverse differs from the reference for a {n}x{n} matrix: check the pivot "
        "row operations and that the right block is kept in step",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, st.integers(min_value=1, max_value=7))
def test_product_with_inverse_is_identity(seed, n):
    A = _well_conditioned(seed, n)
    np.testing.assert_allclose(
        A @ gauss_jordan_inverse(A),
        np.eye(n),
        atol=1e-9,
        err_msg="A @ inverse(A) should be the identity: the inverse is wrong or transposed",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, st.integers(min_value=1, max_value=7))
def test_inverting_twice_returns_the_original_matrix(seed, n):
    A = _well_conditioned(seed, n)
    np.testing.assert_allclose(
        gauss_jordan_inverse(gauss_jordan_inverse(A)),
        A,
        rtol=1e-9,
        atol=1e-9,
        err_msg="inverse(inverse(A)) should give A back",
    )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(SEEDS, st.integers(min_value=1, max_value=6), st.sampled_from([-2.0, 0.5, 10.0]))
def test_transpose_and_scaling_identities(seed, n, c):
    A = _well_conditioned(seed, n)
    inv_A = gauss_jordan_inverse(A)
    np.testing.assert_allclose(
        gauss_jordan_inverse(A.T),
        inv_A.T,
        rtol=1e-9,
        atol=1e-9,
        err_msg="inverse(A transposed) should equal inverse(A) transposed",
    )
    np.testing.assert_allclose(
        gauss_jordan_inverse(c * A),
        inv_A / c,
        rtol=1e-9,
        atol=1e-9,
        err_msg="inverse(c * A) should equal inverse(A) / c",
    )


def test_matrix_with_zero_leading_entry_is_inverted_correctly():
    A = np.array([[0.0, 2.0, 1.0], [3.0, 4.0, 0.0], [1.0, 0.0, 5.0]])
    np.testing.assert_allclose(
        gauss_jordan_inverse(A),
        np.linalg.inv(A),
        rtol=1e-9,
        atol=1e-9,
        err_msg="a zero entry in the first pivot position needs a row swap before dividing",
    )


def test_cyclic_permutation_matrix_inverse_is_its_transpose():
    P = np.roll(np.eye(4), 1, axis=0)
    np.testing.assert_allclose(
        gauss_jordan_inverse(P),
        P.T,
        atol=1e-12,
        err_msg="a permutation matrix's inverse is its transpose: check the row swaps",
    )


def test_badly_scaled_but_nonsingular_matrix_is_not_rejected():
    A = np.diag([1e-3, 1.0, 1e3])
    np.testing.assert_allclose(
        gauss_jordan_inverse(A),
        np.diag([1e3, 1.0, 1e-3]),
        rtol=1e-9,
        err_msg="a condition number of 1e6 is fine: the singularity test must be relative",
    )


def test_exactly_singular_three_by_three_raises_value_error():
    _must_raise_value_error(
        lambda: gauss_jordan_inverse([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "an exactly singular matrix must raise ValueError: test each pivot against the threshold",
        match="singular",
    )


def test_zero_matrix_raises_value_error():
    _must_raise_value_error(
        lambda: gauss_jordan_inverse(np.zeros((3, 3))),
        "the zero matrix must raise ValueError: its pivots are all zero",
        match="singular",
    )


def test_nearly_singular_matrix_is_rejected_by_relative_threshold():
    _must_raise_value_error(
        lambda: gauss_jordan_inverse([[1.0, 1.0], [1.0, 1.0 + 1e-14]]),
        "a nearly singular matrix must raise ValueError: the pivot is below 1e-12 * max|A|",
        match="singular",
    )


def test_pivot_below_the_relative_threshold_is_rejected():
    # The second pivot is 5e-13 with max|A| about 1, which is below the 1e-12 * max|A| threshold.
    _must_raise_value_error(
        lambda: gauss_jordan_inverse([[1.0, 1.0], [1.0, 1.0 + 5e-13]]),
        "a numerically singular matrix (pivot below 1e-12 * max|A|) was not rejected: the "
        "threshold is too fine",
        match="singular",
    )


def test_badly_scaled_diagonal_with_a_pivot_above_the_threshold_is_accepted():
    # Condition number 1e11 but every pivot is far above 1e-12 * max|A|, so the contract accepts it.
    np.testing.assert_allclose(
        gauss_jordan_inverse(np.diag([1e-11, 1.0])),
        np.diag([1e11, 1.0]),
        rtol=1e-9,
        err_msg="a nonsingular diagonal matrix was rejected: the singularity threshold is too "
        "coarse, so it must sit at 1e-12 * max|A| and no higher",
    )


def test_scaled_down_nonsingular_matrix_is_not_rejected():
    # Tiny entries, but the matrix is well conditioned. An absolute threshold would reject it.
    try:
        inverse = gauss_jordan_inverse(1e-13 * np.eye(2))
    except ValueError as exc:
        raise AssertionError(
            "a tiny but nonsingular matrix was rejected: the singularity threshold must be "
            "relative to max|A|, not an absolute number"
        ) from exc
    np.testing.assert_allclose(
        inverse,
        1e13 * np.eye(2),
        rtol=1e-9,
        err_msg="the inverse of 1e-13 * I is 1e13 * I: check the scaling of the pivots",
    )


def test_small_leading_pivot_is_avoided_by_partial_pivoting():
    # The first column's largest entry is in row 2. Eliminating with the tiny 1e-10 pivot
    # multiplies rounding error by about 1e10, so the inverse misses the reference by ~1e-6,
    # far outside the tolerance. Partial pivoting keeps the error near machine precision.
    A = np.array([[1e-10, 2.0, -1.0], [1.5, 0.5, 0.25], [-0.75, 1.0, 2.0]])
    np.testing.assert_allclose(
        gauss_jordan_inverse(A),
        np.linalg.inv(A),
        rtol=1e-9,
        atol=1e-9,
        err_msg="inaccurate inverse: choose the largest-magnitude entry in each column as pivot",
    )


def test_one_by_one_matrix_inverts_to_reciprocal():
    np.testing.assert_allclose(
        gauss_jordan_inverse([[4.0]]),
        [[0.25]],
        rtol=1e-12,
        err_msg="the inverse of a 1x1 matrix is its reciprocal: divide the pivot row by its pivot",
    )
    np.testing.assert_allclose(
        gauss_jordan_inverse([[-2]]),
        [[-0.5]],
        rtol=1e-12,
        err_msg="the inverse of a 1x1 matrix is its reciprocal: divide the pivot row by its pivot",
    )


def test_one_dimensional_input_raises_value_error():
    _must_raise_value_error(
        lambda: gauss_jordan_inverse([2.0, 4.0]),
        "a 1-D input must raise ValueError: check ndim before reading the shape",
    )


def test_empty_matrix_raises_value_error():
    _must_raise_value_error(
        lambda: gauss_jordan_inverse(np.zeros((0, 0))),
        "an empty matrix must raise ValueError: check for zero rows before eliminating",
    )


def test_nan_entry_raises_value_error():
    _must_raise_value_error(
        lambda: gauss_jordan_inverse([[1.0, np.nan], [0.0, 1.0]]),
        "a NaN entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_integer_input_returns_float64_inverse():
    inv = gauss_jordan_inverse([[2, 0], [0, 4]])
    assert inv.dtype == np.float64, f"dtype should be float64 for integer input, got {inv.dtype}"
    np.testing.assert_allclose(
        inv, [[0.5, 0.0], [0.0, 0.25]], rtol=1e-12, err_msg="integer input gives the wrong inverse"
    )


def test_input_matrix_is_not_mutated():
    A = np.array([[4.0, 7.0], [2.0, 6.0]])
    A_copy = A.copy()
    gauss_jordan_inverse(A)
    np.testing.assert_array_equal(A, A_copy, err_msg="input matrix was modified in place")


def test_pivot_just_above_the_relative_threshold_is_accepted():
    # max|A| is 1, so the threshold is 1e-12 and this pivot of 1.5e-12 lies above it.
    A = np.diag([1.0, 1.5e-12])
    try:
        inverse = gauss_jordan_inverse(A)
    except ValueError as exc:
        raise AssertionError(
            "a pivot above 1e-12 * max|A| was rejected as singular: the threshold must sit at"
            " 1e-12 * max|A| exactly"
        ) from exc
    np.testing.assert_allclose(
        inverse,
        np.diag([1.0, 1.0 / 1.5e-12]),
        rtol=1e-9,
        err_msg="the inverse of a nonsingular diagonal matrix is wrong: check the pivots",
    )


def test_pivot_below_threshold_is_rejected_when_the_matrix_has_a_zero_entry():
    # The smallest entry is zero, but the largest is 1, so the pivot 1e-13 is below 1e-12 * max|A|.
    _must_raise_value_error(
        lambda: gauss_jordan_inverse(np.array([[1.0, 0.0], [0.0, 1e-13]])),
        "a pivot below 1e-12 * max|A| was accepted: the threshold must scale with the largest"
        " entry of A, not the smallest",
        match="singular",
    )


def test_all_negative_singular_matrix_raises_value_error():
    _must_raise_value_error(
        lambda: gauss_jordan_inverse([[-1.0, -2.0], [-2.0, -4.0]]),
        "a singular matrix with negative entries must raise ValueError: the singularity threshold"
        " uses the largest absolute entry, which must be taken as a magnitude",
        match="singular",
    )


def test_partial_pivoting_chooses_the_largest_magnitude_not_the_largest_value():
    # The first column holds 1e-13 and -1. Choosing by magnitude swaps in the -1 row; choosing by
    # signed value would keep the tiny pivot and wrongly reject a nonsingular matrix.
    A = np.array([[1e-13, 1.0], [-1.0, 0.0]])
    try:
        inverse = gauss_jordan_inverse(A)
    except ValueError as exc:
        raise AssertionError(
            "a nonsingular matrix was rejected: partial pivoting must pick the row whose entry has"
            " the largest absolute value"
        ) from exc
    np.testing.assert_allclose(
        inverse,
        np.linalg.inv(A),
        rtol=1e-9,
        atol=1e-9,
        err_msg="inaccurate inverse: choose the largest-magnitude entry in each column as pivot",
    )


def test_tuple_of_tuples_is_inverted_like_a_list():
    # The Contract says A is any 2-D array-like. A tuple of tuples is one, so it must be accepted.
    A = ((4.0, 7.0), (2.0, 6.0))
    try:
        inv = gauss_jordan_inverse(A)
    except ValueError as exc:
        raise AssertionError(
            "a tuple of tuples is a valid 2-D array-like: it must be inverted, not refused"
        ) from exc
    np.testing.assert_allclose(
        inv,
        np.linalg.inv(np.array(A, dtype=np.float64)),
        rtol=1e-9,
        atol=1e-9,
        err_msg="inverse of a tuple of tuples differs from the reference: check the row operations",
    )


def test_tiny_entries_of_a_float64_input_are_left_unchanged():
    # A float64 array passed in must come back unchanged, including an entry below 1e-300.
    A = np.array([[1.0, 1e-310], [0.0, 1.0]])
    A_copy = A.copy()
    gauss_jordan_inverse(A)
    np.testing.assert_array_equal(
        A,
        A_copy,
        err_msg="input matrix was modified in place: a tiny entry was changed",
    )
