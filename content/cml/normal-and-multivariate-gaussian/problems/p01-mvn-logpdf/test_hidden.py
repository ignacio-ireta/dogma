import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from submission import mvn_logpdf

SEEDS = st.integers(min_value=0, max_value=10**6)
DIMS = st.integers(min_value=1, max_value=5)


def _spd(rng, d):
    A = rng.normal(size=(d, d))
    return A @ A.T / d + 0.2 * np.eye(d)


def _logpdf(X, mean, cov, mistake):
    try:
        return mvn_logpdf(X, mean, cov)
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, st.integers(min_value=1, max_value=10), DIMS)
def test_matches_scipy_multivariate_normal_on_random_inputs(seed, n, d):
    rng = np.random.default_rng(seed)
    cov = _spd(rng, d)
    mean = rng.normal(size=d)
    X = rng.normal(size=(n, d)) * 2.0
    expected = stats.multivariate_normal(mean=mean, cov=cov).logpdf(X)
    np.testing.assert_allclose(
        _logpdf(X, mean, cov, "a valid normal density was rejected or failed"),
        np.atleast_1d(expected),
        rtol=1e-9,
        atol=1e-9,
        err_msg="log-density differs from scipy.stats.multivariate_normal: check the constant, "
        "the log-determinant sign and the quadratic form",
    )


def test_one_dimensional_closed_form():
    # Normal with mean 0.5 and standard deviation 2.5, evaluated at -2 and 1.
    x = np.array([[-2.0], [1.0]])
    sigma2 = 2.5**2
    expected = -0.5 * np.log(2 * np.pi * sigma2) - (x[:, 0] - 0.5) ** 2 / (2 * sigma2)
    np.testing.assert_allclose(
        _logpdf(x, [0.5], [[sigma2]], "a one-dimensional normal was rejected or failed"),
        expected,
        rtol=1e-12,
        err_msg="the one-dimensional normal density is -1/2 log(2 pi s^2) - (x - m)^2 / (2 s^2)",
    )


def test_dimension_sixty_with_underflowing_determinant_keeps_its_log_density():
    # Eigenvalues evenly spaced from 1e-7 to 1e-6: det(cov) is about 1e-380, below the float64
    # range, so it underflows to 0.0.
    rng = np.random.default_rng(5)
    d = 60
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    cov = Q @ np.diag(np.linspace(1e-7, 1e-6, d)) @ Q.T
    cov = (cov + cov.T) / 2
    x = rng.multivariate_normal(np.zeros(d), cov, size=3)
    expected = stats.multivariate_normal(mean=np.zeros(d), cov=cov).logpdf(x)
    np.testing.assert_allclose(
        _logpdf(x, np.zeros(d), cov, "a high-dimensional density was rejected or failed"),
        expected,
        rtol=1e-9,
        atol=1e-9,
        err_msg="the log-determinant underflowed: sum the logs of the Cholesky diagonal instead "
        "of multiplying eigenvalues or taking the determinant",
    )


def test_dimension_sixty_with_overflowing_determinant_keeps_its_log_density():
    # Eigenvalues evenly spaced from 1e6 to 1e7: det(cov) is about 1e400, above the float64
    # range, so it overflows to inf.
    rng = np.random.default_rng(6)
    d = 60
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    cov = Q @ np.diag(np.linspace(1e6, 1e7, d)) @ Q.T
    cov = (cov + cov.T) / 2
    x = rng.multivariate_normal(np.zeros(d), cov, size=3)
    expected = stats.multivariate_normal(mean=np.zeros(d), cov=cov).logpdf(x)
    np.testing.assert_allclose(
        _logpdf(x, np.zeros(d), cov, "a high-dimensional density was rejected or failed"),
        expected,
        rtol=1e-9,
        atol=1e-9,
        err_msg="the log-determinant overflowed: sum the logs of the Cholesky diagonal instead "
        "of multiplying eigenvalues or taking the determinant",
    )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(SEEDS, st.integers(min_value=1, max_value=4))
def test_affine_change_of_variables_shifts_log_density_by_log_abs_det(seed, d):
    # If Y = A X, then density(Y) = density(X) / |det A|, so the log-densities differ by log|det A|.
    rng = np.random.default_rng(seed)
    cov = _spd(rng, d)
    mean = rng.normal(size=d)
    A = rng.normal(size=(d, d)) + 3.0 * np.eye(d)
    X = rng.normal(size=(4, d))
    X_mapped = X @ A.T  # built before any call, so the call below cannot change it
    base = _logpdf(X, mean, cov, "a valid density was rejected or failed")
    mapped = _logpdf(X_mapped, A @ mean, A @ cov @ A.T, "a valid transformed density was rejected")
    np.testing.assert_allclose(
        mapped,
        base - np.log(abs(np.linalg.det(A))),
        rtol=1e-9,
        atol=1e-9,
        err_msg="the change-of-variables identity failed: check the log-determinant and the "
        "quadratic form together",
    )


def test_asymmetric_covariance_raises_value_error():
    _raised_value_error(
        lambda: mvn_logpdf([[0.0, 0.0]], [0.0, 0.0], [[2.0, 0.3], [0.0, 1.0]]),
        "an asymmetric covariance must raise ValueError: check symmetry before factoring",
    )


def test_indefinite_covariance_raises_value_error():
    _raised_value_error(
        lambda: mvn_logpdf([[0.0, 0.0]], [0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]]),
        "an indefinite covariance (negative eigenvalue) must raise ValueError",
    )


def test_shape_mismatches_raise_value_error():
    _raised_value_error(
        lambda: mvn_logpdf([[0.0, 0.0]], [0.0], np.eye(2)),
        "a mean with the wrong length must raise ValueError",
    )
    _raised_value_error(
        lambda: mvn_logpdf([[0.0, 0.0]], [0.0, 0.0], np.eye(3)),
        "a covariance that is not (d, d) must raise ValueError",
    )


def test_one_dimensional_input_raises_value_error():
    _raised_value_error(
        lambda: mvn_logpdf([0.0, 0.0], [0.0, 0.0], np.eye(2)),
        "X must be 2-D with one row per observation: raise ValueError",
    )


def test_nan_input_raises_value_error():
    _raised_value_error(
        lambda: mvn_logpdf([[np.nan, 0.0]], [0.0, 0.0], np.eye(2)),
        "a NaN entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_infinite_input_raises_value_error():
    _raised_value_error(
        lambda: mvn_logpdf([[np.inf, 0.0]], [0.0, 0.0], np.eye(2)),
        "an infinite entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_non_finite_mean_raises_value_error():
    _raised_value_error(
        lambda: mvn_logpdf([[0.0, 0.0]], [np.nan, 0.0], np.eye(2)),
        "a NaN mean must raise ValueError: validate every input, not only X",
    )


def test_zero_observations_raise_value_error():
    _raised_value_error(
        lambda: mvn_logpdf(np.zeros((0, 2)), [0.0, 0.0], np.eye(2)),
        "no observations means no density values: raise ValueError",
    )


def test_rounding_level_asymmetry_is_accepted_as_symmetric():
    # The asymmetry (1e-13) is far inside the contract's tolerance of 1e-10 * max(1, max |cov|).
    symmetric = mvn_logpdf([[0.5, -1.0]], [0.0, 0.0], [[2.0, 0.3], [0.3, 1.0]])
    rounded = _logpdf(
        [[0.5, -1.0]],
        [0.0, 0.0],
        [[2.0, 0.3 + 1e-13], [0.3, 1.0]],
        "a covariance that is symmetric up to rounding was rejected or failed",
    )
    np.testing.assert_allclose(
        rounded,
        symmetric,
        rtol=1e-9,
        err_msg="a covariance that differs from its transpose only by rounding must be accepted",
    )


def test_single_observation_returns_array_of_shape_one():
    result = _logpdf([[1.0, -1.0]], [0.0, 0.0], np.eye(2), "a single observation was rejected")
    assert isinstance(result, np.ndarray), "return a NumPy array, not a scalar"
    assert result.shape == (1,), (
        f"shape mismatch: one row of X must give one density value, got shape {result.shape}"
    )


def test_integer_input_returns_float64_array_of_length_n():
    result = _logpdf(
        [[1, 0], [0, 2], [1, 1]], [0, 0], [[2, 0], [0, 1]], "integer input was rejected"
    )
    assert result.dtype == np.float64, (
        f"dtype should be float64 for integer input, got {result.dtype}"
    )
    assert result.shape == (3,), (
        f"shape mismatch: each row of X must give one density value, got shape {result.shape}"
    )


def test_inputs_are_not_mutated():
    X = np.array([[1.0, 2.0], [0.5, -1.0]])
    mean = np.array([0.2, 0.1])
    cov = np.array([[2.0, 0.3], [0.3, 1.0]])
    X0, m0, c0 = X.copy(), mean.copy(), cov.copy()
    mvn_logpdf(X, mean, cov)
    np.testing.assert_array_equal(
        X, X0, err_msg="X was modified in place: subtract the mean from a copy"
    )
    np.testing.assert_array_equal(mean, m0, err_msg="mean was modified in place")
    np.testing.assert_array_equal(cov, c0, err_msg="cov was modified in place")


def test_infinite_covariance_entry_raises_value_error():
    _raised_value_error(
        lambda: mvn_logpdf([[0.0, 0.0]], [0.0, 0.0], [[np.inf, 0.0], [0.0, 1.0]]),
        "an infinite entry in cov must raise ValueError: validate every input for finiteness",
    )


def test_asymmetry_just_above_tolerance_is_rejected():
    # Asymmetry 1.5e-10 exceeds 1e-10 * max(1, max |cov|) = 1e-10, so the matrix is not symmetric.
    _raised_value_error(
        lambda: mvn_logpdf([[0.0, 0.0]], [0.0, 0.0], [[1.0, 1.5e-10], [0.0, 1.0]]),
        "a covariance whose entries differ by more than the tolerance must raise ValueError",
    )


def test_asymmetry_within_tolerance_for_large_covariance_is_accepted():
    # The tolerance scales with max |cov| = 1e4, so asymmetry 1e-7 (below 1e-6) is accepted.
    result = _logpdf(
        [[0.0, 0.0]],
        [0.0, 0.0],
        [[1e4, 5.0], [5.0 + 1e-7, 2.0]],
        "a covariance with large entries and rounding-level asymmetry was rejected",
    )
    assert np.all(np.isfinite(result)), "a valid covariance must give a finite log-density"
