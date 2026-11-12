import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from submission import power_iteration

from dogma_data import rng as data_rng
from dogma_data import sample_rows

SEEDS = st.integers(min_value=0, max_value=10**6)
SIZES = st.integers(min_value=1, max_value=8)
SIGNS = st.sampled_from([-1.0, 1.0])


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


def _real_axes(gen, n):
    """The principal axes of real data: eigenvectors of the covariance of n wine measurements.

    Sixty wines and n standardised columns are drawn; the columns of the result are the
    orthonormal directions along which those real measurements vary.
    """
    X, _ = sample_rows("wine", gen, 60, n, standardize=True)
    return np.linalg.eigh(np.atleast_2d(np.cov(X, rowvar=False)))[1]


def _symmetric_with_gap(seed, n, dominant_sign=1.0):
    """Symmetric Q diag(λ) Qᵀ with dominant |λ| = 3.0 and all other |λ| <= 1.5.

    Q holds the principal axes of real wine data. The spectrum is set by the test, because the
    convergence the contract promises depends on the gap between the first two eigenvalues.
    """
    rng = data_rng(f"power-iteration-p01:matrix:{seed}:{n}")
    Q = _real_axes(rng, n)
    eigenvalues = rng.choice([-1.0, 1.0], size=n) * (0.2 + (1.5 - 0.2) * _real_unit(rng, n))
    eigenvalues[0] = dominant_sign * 3.0
    return Q @ np.diag(eigenvalues) @ Q.T


def _reference_dominant_pair(A):
    """Dominant eigenpair from NumPy's symmetric eigensolver, chosen by largest |eigenvalue|."""
    values, vectors = np.linalg.eigh(A)
    k = int(np.argmax(np.abs(values)))
    return values[k], vectors[:, k]


def _sign_pivot(vector):
    """Index of the first entry within a relative 1e-9 of the largest magnitude."""
    magnitudes = np.abs(vector)
    return int(np.flatnonzero(magnitudes >= np.max(magnitudes) * (1 - 1e-9))[0])


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
    ("n", "seed", "sign"),
    [(1, 1, 1.0), (2, 2, -1.0), (3, 3, 1.0), (4, 4, -1.0), (6, 5, 1.0), (8, 6, -1.0)],
    ids=["case-a", "case-b", "case-c", "case-d", "case-e", "case-f"],
)
def test_matches_eigh_dominant_pair_on_random_symmetric_matrices(n, seed, sign):
    A = _symmetric_with_gap(seed, n, dominant_sign=sign)
    value, vector = power_iteration(A)
    ref_value, ref_vector = _reference_dominant_pair(A)
    assert abs(value - ref_value) <= 1e-9 * max(1.0, abs(ref_value)), (
        "dominant eigenvalue differs from the reference: check the Rayleigh quotient v.Av on the "
        "final unit vector"
    )
    cosine = abs(float(vector @ ref_vector))
    assert abs(cosine - 1.0) <= 1e-9, (
        "eigenvector direction differs from the reference: check the iteration and the final "
        "normalisation"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, SIZES, SIGNS)
def test_returned_pair_is_an_eigenpair_with_unit_eigenvector(seed, n, sign):
    A = _symmetric_with_gap(seed, n, dominant_sign=sign)
    value, vector = power_iteration(A)
    assert abs(float(vector @ vector) - 1.0) <= 1e-12, "the eigenvector must have unit length"
    residual = A @ vector - value * vector
    assert np.sqrt(residual @ residual) <= 1e-8 * max(1.0, abs(value)), (
        "A v should equal λ v for the returned pair: the iteration did not converge"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, SIZES, SIGNS)
def test_largest_entry_of_eigenvector_is_positive(seed, n, sign):
    A = _symmetric_with_gap(seed, n, dominant_sign=sign)
    _, vector = power_iteration(A)
    assert vector[_sign_pivot(vector)] > 0, (
        "sign convention violated: the largest-magnitude entry of the eigenvector must be positive"
    )


def test_negative_dominant_eigenvalue_keeps_its_sign():
    A = _symmetric_with_gap(seed=9, n=4, dominant_sign=-1.0)
    value, _ = power_iteration(A)
    assert value == pytest.approx(-3.0, abs=1e-8), (
        "a negative dominant eigenvalue must keep its sign: returning |λ| loses it"
    )


def test_dominant_eigenvalue_is_chosen_by_magnitude_not_by_value():
    value, vector = power_iteration(np.diag([-5.0, 4.0]))
    assert value == pytest.approx(-5.0, abs=1e-9), (
        "returned the largest algebraic eigenvalue: the dominant one has the largest |value|"
    )
    np.testing.assert_allclose(
        np.abs(vector),
        [1.0, 0.0],
        atol=1e-9,
        err_msg="the eigenvector returned is not for the largest-magnitude eigenvalue",
    )


def test_start_orthogonal_to_dominant_eigenvector_does_not_stall():
    A = np.array([[2.0, -1.0], [-1.0, 2.0]])
    value, vector = power_iteration(A)
    assert value == pytest.approx(3.0, abs=1e-9), (
        "stalled on the wrong eigenpair: the start vector had no component along the dominant "
        "eigenvector. Use a generic start, not a constant vector"
    )
    np.testing.assert_allclose(
        np.abs(vector),
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        atol=1e-9,
        err_msg="the returned eigenvector belongs to the wrong eigenvalue: check the start vector",
    )


def test_slow_gap_converges_within_the_default_iteration_budget():
    rng = np.random.default_rng(31)
    Q, _ = np.linalg.qr(_real_normal(rng, (3, 3)))
    A = Q @ np.diag([1.0, 0.9, 0.5]) @ Q.T
    value, _ = power_iteration(A)
    assert value == pytest.approx(1.0, abs=1e-8), (
        "did not converge within the default budget: check the residual stopping rule and that "
        "the loop runs up to num_iters updates"
    )


def test_one_by_one_matrix_returns_its_entry_and_the_unit_vector():
    value, vector = power_iteration([[-2.5]])
    assert value == pytest.approx(-2.5, abs=1e-12), "the eigenvalue of a 1x1 matrix is its entry"
    np.testing.assert_allclose(
        vector, [1.0], atol=1e-12, err_msg="the unit eigenvector of 1x1 is [1]"
    )


def test_zero_matrix_returns_zero_eigenvalue_and_unit_vector_of_any_size():
    for n in (1, 4):
        value, vector = power_iteration(np.zeros((n, n)))
        assert value == 0.0, f"zero {n}x{n} matrix must have eigenvalue 0.0"
        assert abs(float(vector @ vector) - 1.0) <= 1e-12, "the vector must have unit length"


def test_non_square_matrix_raises_value_error():
    _must_raise_value_error(
        lambda: power_iteration(np.ones((2, 3))),
        "a non-square matrix must raise ValueError: check the shape before iterating",
    )


def test_small_asymmetry_is_rejected_rather_than_averaged():
    _must_raise_value_error(
        lambda: power_iteration([[1.0, 1e-3], [0.0, 1.0]]),
        "a small asymmetry must raise ValueError: compare A with A transposed, not average it",
        match="symmetric",
    )


def test_nan_entry_raises_value_error():
    _must_raise_value_error(
        lambda: power_iteration([[1.0, np.nan], [np.nan, 1.0]]),
        "a NaN entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


@pytest.mark.parametrize("num_iters", [0, -5], ids=["case-a", "case-b"])
def test_non_positive_iteration_budget_raises_value_error(num_iters):
    _must_raise_value_error(
        lambda: power_iteration(np.eye(2), num_iters=num_iters),
        "num_iters below 1 must raise ValueError: check the budget before iterating",
    )


@pytest.mark.parametrize("tol", [0.0, -1e-6, float("nan")], ids=["case-a", "case-b", "case-c"])
def test_non_positive_or_nan_tolerance_raises_value_error(tol):
    _must_raise_value_error(
        lambda: power_iteration(np.eye(2), tol=tol),
        "tol must be a positive number: check tol > 0, which also rejects NaN",
    )


def test_exhausted_budget_returns_the_current_estimate_without_error():
    # Eigenvalues +1 and -1 have equal magnitude, so this iteration never converges and
    # every update uses up the budget.
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    try:
        value, vector = power_iteration(A, num_iters=5)
    except Exception as exc:
        raise AssertionError(
            f"running out of iterations must return the current estimate, not raise "
            f"({type(exc).__name__})"
        ) from exc
    assert np.isfinite(value) and abs(float(vector @ vector) - 1.0) <= 1e-12, (
        "an exhausted budget must still return a finite eigenvalue and a unit vector"
    )


def test_repeated_calls_on_the_same_matrix_return_identical_results():
    A = _symmetric_with_gap(seed=5, n=4)
    first_value, first_vector = power_iteration(A)
    second_value, second_vector = power_iteration(A)
    assert first_value == second_value and np.array_equal(first_vector, second_vector), (
        "the start vector must be fixed: the same input must give the same output every call"
    )


def test_tie_in_magnitude_resolves_the_sign_to_the_first_entry():
    # The eigenvector for 3 is (1, -1)/sqrt(2): both entries have equal magnitude, so the
    # convention picks the first entry, which must be positive.
    _, vector = power_iteration([[2.0, -1.0], [-1.0, 2.0]])
    assert vector[0] > 0 and vector[1] < 0, (
        "sign convention: with equal magnitudes the first entry must be positive"
    )


def test_integer_input_returns_float_pair():
    value, vector = power_iteration([[2, 1], [1, 2]])
    assert isinstance(value, float), f"eigenvalue should be a Python float, got {type(value)}"
    assert vector.dtype == np.float64, f"eigenvector dtype should be float64, got {vector.dtype}"
    assert value == pytest.approx(3.0, abs=1e-9), "integer input must give the same eigenvalue"


def test_input_matrix_is_not_mutated():
    A = _symmetric_with_gap(seed=12, n=3)
    A_copy = A.copy()
    power_iteration(A)
    np.testing.assert_array_equal(A, A_copy, err_msg="input matrix was modified in place")


def test_a_single_iteration_is_an_allowed_budget():
    value, vector = power_iteration([[2.0, 1.0], [1.0, 2.0]], num_iters=1)
    assert isinstance(value, float) and np.isfinite(value), (
        "num_iters = 1 is a valid budget: one update must return a finite eigenvalue estimate"
    )
    assert abs(float(vector @ vector) - 1.0) <= 1e-12, (
        "num_iters = 1 is a valid budget: the returned eigenvector must still have unit length"
    )


def test_inputs_that_are_not_square_matrices_raise_value_error():
    for bad in (np.array([1.0, 2.0, 3.0]), np.ones((2, 2, 2)), np.array(2.0)):
        try:
            power_iteration(bad)
        except ValueError:
            continue
        except Exception as exc:
            pytest.fail(
                "an input that is not a square matrix must raise ValueError; the function "
                f"raised {type(exc).__name__}"
            )
        pytest.fail("an input that is not a square matrix must raise ValueError, not return a pair")


def test_symmetry_tolerance_scales_with_large_entries():
    A = np.array([[1e6, 4e-4], [0.0, 2e5]])
    value, _ = power_iteration(A)
    assert value == pytest.approx(1e6, rel=1e-6), (
        "a rounding-level asymmetry is judged against the size of the largest entry, so a large "
        "matrix with a small asymmetry must be accepted"
    )


def test_symmetry_tolerance_scales_with_large_negative_entries():
    A = np.array([[-1e6, 4e-4], [0.0, -2e5]])
    value, _ = power_iteration(A)
    assert value == pytest.approx(-1e6, rel=1e-6), (
        "the symmetry tolerance depends on the largest entry by magnitude, not by value: a large "
        "negative matrix with a small asymmetry must be accepted"
    )


def test_symmetry_tolerance_is_at_least_one_for_small_entries():
    try:
        power_iteration(np.array([[0.5, 7e-10], [0.0, 0.25]]))
    except ValueError as exc:
        pytest.fail(
            "the symmetry tolerance is 1e-9 times max(1, largest entry): a small matrix with an "
            f"asymmetry below 1e-9 must be accepted ({exc})"
        )


def test_float64_input_with_tiny_entries_is_not_mutated():
    # Symmetric float64 matrix whose off-diagonal entries are tiny but not zero. A copy-free
    # implementation that cleans small entries would write into the caller's array.
    A = np.array([[2.0, 1e-13, 0.5], [1e-13, 3.0, 0.0], [0.5, 0.0, 1.0]], dtype=np.float64)
    A_copy = A.copy()
    power_iteration(A)
    np.testing.assert_array_equal(
        A,
        A_copy,
        err_msg="the input matrix was modified in place: tiny entries must be left as given",
    )


def test_stop_tolerance_grows_with_a_large_positive_eigenvalue():
    # The residual test is relative to the eigenvalue when it is bigger than one. Once that
    # test is met the iteration must stop, so spare budget cannot change the answer.
    A = np.diag([3.0, 2.7])
    early_value, early_vector = power_iteration(A, num_iters=10, tol=0.05)
    late_value, late_vector = power_iteration(A, num_iters=60, tol=0.05)
    assert early_value == late_value and np.array_equal(early_vector, late_vector), (
        "the iteration kept going after the residual was already within the tolerance: for a "
        "large eigenvalue the tolerance should grow with the size of the eigenvalue"
    )


def test_stop_tolerance_uses_the_size_of_a_negative_eigenvalue():
    # For any unit vector the residual norm is at most 0.05, and |lambda| is at least 2.9, so the
    # stopping threshold is at least 0.058. The stop therefore holds after the first update, for
    # every start vector. A tolerance that uses the sign gets 0.02 and keeps iterating.
    A = np.diag([-3.0, -2.9])
    early_value, early_vector = power_iteration(A, num_iters=1, tol=0.02)
    late_value, late_vector = power_iteration(A, num_iters=200, tol=0.02)
    assert early_value == late_value and np.array_equal(early_vector, late_vector), (
        "the iteration kept going after the residual was already within the tolerance for a "
        "negative eigenvalue: the tolerance should follow the size of the eigenvalue, not its sign"
    )
