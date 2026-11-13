import itertools
import signal
from contextlib import contextmanager

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from submission import determinant

from dogma_data import rng as data_rng
from dogma_data import sample_rows

SEEDS = st.integers(min_value=0, max_value=10**6)
SIZES = st.integers(min_value=1, max_value=6)


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


def _well_conditioned(seed, n):
    """A square block of real data shifted by n*I: nonsingular, with a determinant far from zero.

    The block is n rows by n standardised measurements (correlated real columns): wine, or
    breast_cancer when more than 13 columns are asked for. The shift keeps the condition
    number small, as it did for the random matrices this replaces.
    """
    name = "wine" if n <= 13 else "breast_cancer"
    gen = data_rng(f"determinant-p01:block:{seed}:{n}")
    block, _ = sample_rows(name, gen, n, n, standardize=True)
    return block + n * np.eye(n)


@contextmanager
def _finishes_within(seconds, mistake):
    """Fail with the kind of mistake if the block runs too long (Unix signal timer)."""

    def _too_slow(signum, frame):
        raise AssertionError(mistake)

    previous = signal.signal(signal.SIGALRM, _too_slow)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


SLOW = "too slow: use LU-style elimination (O(n^3)), not cofactor expansion (O(n!))"


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
    [(1, 1), (2, 2), (3, 3), (5, 4), (8, 5), (12, 6)],
    ids=["case-a", "case-b", "case-c", "case-d", "case-e", "case-f"],
)
def test_matches_numpy_determinant_on_random_matrices(n, seed):
    A = _well_conditioned(seed, n)
    with _finishes_within(5, f"{n}x{n} matrix: {SLOW}"):
        value = determinant(A)
    np.testing.assert_allclose(
        value,
        np.linalg.det(A),
        rtol=1e-9,
        err_msg=f"determinant differs from the reference for a {n}x{n} matrix: check the sign "
        "bookkeeping for row swaps and the pivot product",
    )


def test_thirty_by_thirty_matrix_matches_numpy():
    A = _well_conditioned(seed=7, n=30)
    with _finishes_within(5, f"30x30 matrix: {SLOW}"):
        value = determinant(A)
    np.testing.assert_allclose(
        value,
        np.linalg.det(A),
        rtol=1e-9,
        err_msg="n=30 must use elimination, not cofactor expansion (which never finishes)",
    )


@settings(max_examples=50, deadline=None, report_multiple_bugs=False)
@given(SEEDS, SIZES)
def test_determinant_of_a_product_is_product_of_determinants(seed, n):
    A = _well_conditioned(seed, n)
    B = _well_conditioned(seed + 1, n)
    np.testing.assert_allclose(
        determinant(A @ B),
        determinant(A) * determinant(B),
        rtol=1e-9,
        err_msg="det(AB) must equal det(A) det(B): check the multiplication order",
    )


@settings(max_examples=50, deadline=None, report_multiple_bugs=False)
@given(SEEDS, SIZES)
def test_transpose_has_the_same_determinant(seed, n):
    A = _well_conditioned(seed, n)
    np.testing.assert_allclose(
        determinant(A.T),
        determinant(A),
        rtol=1e-9,
        err_msg="det(A transposed) must equal det(A)",
    )


@settings(max_examples=50, deadline=None, report_multiple_bugs=False)
@given(SEEDS, SIZES)
def test_swapping_two_rows_flips_the_sign(seed, n):
    if n < 2:
        return
    A = _well_conditioned(seed, n)
    order = [1, 0, *range(2, n)]
    swapped = A[order]
    np.testing.assert_allclose(
        determinant(swapped),
        -determinant(A),
        rtol=1e-9,
        err_msg="swapping two rows must negate the determinant: the sign was not tracked",
    )


@settings(max_examples=50, deadline=None, report_multiple_bugs=False)
@given(SEEDS, SIZES, st.sampled_from([-2.5, 0.5, 3.0]))
def test_scaling_one_row_scales_the_determinant(seed, n, c):
    A = _well_conditioned(seed, n)
    scaled = A.copy()
    scaled[0] *= c
    np.testing.assert_allclose(
        determinant(scaled),
        c * determinant(A),
        rtol=1e-9,
        err_msg="scaling one row by c must multiply the determinant by c",
    )


@settings(max_examples=50, deadline=None, report_multiple_bugs=False)
@given(SEEDS, SIZES, st.sampled_from([-2.0, 0.5, 3.0]))
def test_scaling_the_whole_matrix_multiplies_by_c_to_the_n(seed, n, c):
    A = _well_conditioned(seed, n)
    np.testing.assert_allclose(
        determinant(c * A),
        (c**n) * determinant(A),
        rtol=1e-9,
        err_msg="det(cA) must equal c^n det(A) for an n x n matrix",
    )


@settings(max_examples=50, deadline=None, report_multiple_bugs=False)
@given(SEEDS, SIZES)
def test_adding_a_multiple_of_one_row_to_another_keeps_the_determinant(seed, n):
    if n < 2:
        return
    A = _well_conditioned(seed, n)
    combined = A.copy()
    combined[1] += -1.75 * A[0]
    np.testing.assert_allclose(
        determinant(combined),
        determinant(A),
        rtol=1e-9,
        err_msg="a row operation of the form row += k * other row must leave det unchanged",
    )


@pytest.mark.parametrize(
    "n", [1, 2, 3, 4, 5], ids=["case-a", "case-b", "case-c", "case-d", "case-e"]
)
def test_upper_and_lower_triangular_matrices_give_product_of_diagonal(n):
    rng = np.random.default_rng(100 + n)
    diagonal = (0.5 + (3.0 - 0.5) * _real_unit(rng, n)) * rng.choice([-1.0, 1.0], size=n)
    upper = np.triu(_real_normal(rng, (n, n)), k=1) + np.diag(diagonal)
    lower = upper.T.copy()
    expected = float(np.prod(diagonal))
    np.testing.assert_allclose(
        determinant(upper),
        expected,
        rtol=1e-9,
        err_msg="an upper-triangular determinant is the product of its diagonal",
    )
    np.testing.assert_allclose(
        determinant(lower),
        expected,
        rtol=1e-9,
        err_msg="a lower-triangular determinant is the diagonal product (pivoting may swap rows)",
    )


def test_all_permutation_matrices_of_size_four_give_their_sign():
    for perm in itertools.permutations(range(4)):
        inversions = sum(1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j])
        expected = -1.0 if inversions % 2 else 1.0
        P = np.eye(4)[list(perm)]
        np.testing.assert_allclose(
            determinant(P),
            expected,
            atol=1e-12,
            err_msg="a permutation matrix's determinant is its sign: count the row swaps and flip "
            "the sign once per swap",
        )


def test_tiny_nonzero_pivots_are_not_treated_as_zero():
    np.testing.assert_allclose(
        determinant(1e-13 * np.eye(2)),
        1e-26,
        rtol=1e-9,
        err_msg="a tiny but nonzero pivot is not zero: only an exact zero gives determinant 0",
    )


def test_matrix_with_repeated_row_has_exact_zero_determinant():
    assert abs(determinant([[1, 2, 3], [4, 5, 6], [1, 2, 3]])) <= 1e-12, (
        "a repeated row makes the matrix singular: the pivot for the last row must vanish"
    )


def test_rank_deficient_random_matrix_has_numerically_zero_determinant():
    rng = np.random.default_rng(21)
    A = _real_normal(rng, (4, 2)) @ _real_normal(rng, (2, 4))
    assert abs(determinant(A)) <= 1e-9 * np.linalg.norm(A) ** 4, (
        "a rank-2 4x4 matrix must have determinant zero, up to rounding"
    )


def test_zero_by_zero_matrix_has_determinant_one():
    assert determinant(np.zeros((0, 0))) == 1.0, "the empty matrix has determinant 1 by convention"


def test_one_by_one_matrix_returns_its_single_entry():
    np.testing.assert_allclose(
        determinant([[-3.5]]),
        -3.5,
        rtol=1e-12,
        err_msg="a 1x1 matrix has determinant equal to its single entry",
    )


def test_integer_input_returns_the_correct_value():
    value = determinant([[2, 0], [0, 3]])
    assert abs(value - 6.0) <= 1e-12, (
        f"integer input gave {value}: convert to float before elimination and check the product"
    )


def test_one_dimensional_input_raises_value_error():
    _must_raise_value_error(
        lambda: determinant([1.0, 2.0, 3.0]),
        "a 1-D input must raise ValueError: check ndim before reading the shape",
    )


def test_tall_rectangular_matrix_raises_value_error():
    _must_raise_value_error(
        lambda: determinant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "a non-square matrix must raise ValueError: check that rows equal columns",
        match="square",
    )


def test_nan_entry_raises_value_error():
    _must_raise_value_error(
        lambda: determinant([[1.0, np.nan], [0.0, 1.0]]),
        "a NaN entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_input_matrix_is_not_mutated():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    A_copy = A.copy()
    determinant(A)
    np.testing.assert_array_equal(A, A_copy, err_msg="input matrix was modified in place")


def test_tiny_leading_entry_still_uses_the_largest_pivot():
    # Expand along the first row: det = 1e-16*(1*2 - 1*1) - 1*(1*2 - 1*1) + 1*(1*1 - 1*1)
    #                                   = -1 + 1e-16.
    # Taking the tiny 1e-16 as pivot loses the digits of the later rows, so partial pivoting
    # (swap in the row with the largest entry in column 0) is what keeps the answer near -1.
    A = [[1e-16, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]]
    assert abs(determinant(A) + 1.0) <= 1e-9, (
        "a tiny leading entry must not be the pivot: take the largest entry in the column"
    )


def test_infinite_entry_raises_value_error():
    _must_raise_value_error(
        lambda: determinant([[1.0, np.inf], [0.0, 1.0]]),
        "an infinite entry must raise ValueError: validate with np.isfinite, which rejects inf",
    )


def test_result_is_a_float_and_not_an_array():
    value = determinant([[1, 2], [3, 4]])
    assert isinstance(value, float), (
        "the determinant must be a float: an array that holds one number is another type"
    )
