import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from submission import gaussian_mle

SEEDS = st.integers(min_value=0, max_value=10**6)
ROWS = st.integers(min_value=1, max_value=12)
COLS = st.integers(min_value=1, max_value=4)


def _random_matrix(seed, n, d):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def _mle(X, mistake):
    try:
        return gaussian_mle(X)
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
@given(SEEDS, ROWS, COLS)
def test_mean_is_the_column_mean(seed, n, d):
    X = _random_matrix(seed, n, d)
    mean, _ = _mle(X.copy(), "a valid data matrix was rejected or failed")
    np.testing.assert_allclose(
        mean, X.mean(axis=0), rtol=1e-12, atol=1e-12, err_msg="the MLE mean is the column mean"
    )


_SHAPES = [(n, d) for n in range(1, 9) for d in (1, 2, 3)]


@pytest.mark.parametrize(
    ("n", "d"), _SHAPES, ids=[f"case-{index:02d}" for index in range(len(_SHAPES))]
)
def test_covariance_matches_numpy_with_divisor_n(n, d):
    X = _random_matrix(10 * n + d, n, d)
    _, cov = _mle(X.copy(), "a valid data matrix was rejected or failed")
    np.testing.assert_allclose(
        cov,
        np.atleast_2d(np.cov(X, rowvar=False, bias=True)),
        rtol=1e-9,
        atol=1e-12,
        err_msg="the MLE covariance divides by n (bias=True), not by n - 1",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, st.integers(min_value=2, max_value=12), COLS)
def test_mle_covariance_is_n_minus_one_over_n_times_the_unbiased_covariance(seed, n, d):
    X = _random_matrix(seed, n, d)
    _, cov = _mle(X.copy(), "a valid data matrix was rejected or failed")
    unbiased = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    np.testing.assert_allclose(
        cov,
        (n - 1) / n * unbiased,
        rtol=1e-9,
        atol=1e-12,
        err_msg="the MLE covariance should be (n - 1) / n times the ddof=1 covariance",
    )


def test_returned_parameters_maximise_the_log_likelihood():
    rng = np.random.default_rng(7)
    A = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.4], [0.2, 0.0, 1.0]])
    X = rng.normal(size=(20, 3)) @ A.T + np.array([1.0, -2.0, 0.5])
    mean, cov = _mle(X.copy(), "a valid data matrix was rejected or failed")

    def loglik(m, S):
        return stats.multivariate_normal(mean=m, cov=S).logpdf(X).sum()

    best = loglik(mean, cov)
    for factor in (0.95, 0.99, 1.01, 1.05):
        assert loglik(mean, factor * cov) <= best + 1e-9, (
            "the returned covariance is not the likelihood maximiser: a nearby scaling is higher"
        )
    spread = np.sqrt(np.diag(cov))
    for shift in (-0.05, 0.05):
        assert loglik(mean + shift * spread, cov) <= best + 1e-9, (
            "the returned mean is not the likelihood maximiser: a nearby shift scores higher"
        )


def test_single_observation_returns_its_point_and_zero_covariance():
    mean, cov = _mle([[2.0, -1.0, 0.5]], "one observation was rejected or failed")
    np.testing.assert_allclose(
        mean, [2.0, -1.0, 0.5], rtol=1e-12, err_msg="the mean of one point is the point"
    )
    np.testing.assert_allclose(
        cov,
        np.zeros((3, 3)),
        atol=1e-12,
        err_msg="one observation gives zero covariance, not NaN: divide by n, which is 1",
    )


def test_large_common_offset_does_not_destroy_the_covariance():
    base = np.random.default_rng(4).integers(0, 10, size=(20, 2)).astype(np.float64)
    _, cov = _mle(base + 1e9, "a valid offset matrix was rejected or failed")
    np.testing.assert_allclose(
        cov,
        np.cov(base, rowvar=False, bias=True),
        rtol=1e-9,
        atol=1e-9,
        err_msg="a large common offset destroyed the covariance: centre before taking products, "
        "not E[xy] - E[x]E[y]",
    )


def test_one_variable_gives_one_by_one_covariance():
    mean, cov = _mle([[1], [2], [4]], "one variable was rejected or failed")
    assert mean.shape == (1,), (
        f"shape mismatch: the mean must have one entry per variable, got shape {mean.shape}"
    )
    assert cov.shape == (1, 1), (
        "shape mismatch: the covariance must be square with one row "
        f"and column per variable, got {cov.shape}"
    )
    np.testing.assert_allclose(
        cov,
        [[14 / 9]],
        rtol=1e-12,
        err_msg="the one-variable MLE is the biased variance: divide by n",
    )


def test_one_dimensional_input_raises_value_error():
    _raised_value_error(
        lambda: gaussian_mle([1.0, 2.0, 3.0]),
        "a 1-D input is not a data matrix: raise ValueError (rows are observations)",
    )


def test_nan_entry_raises_value_error():
    _raised_value_error(
        lambda: gaussian_mle([[1.0, np.nan], [2.0, 3.0]]),
        "a NaN entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_integer_input_returns_float64_arrays_of_the_right_shapes():
    mean, cov = _mle([[1, 0], [0, 2], [1, 1]], "integer input was rejected or failed")
    assert mean.dtype == np.float64 and cov.dtype == np.float64, (
        "return float64 arrays for integer input"
    )
    assert mean.shape == (2,) and cov.shape == (2, 2), (
        "shape mismatch: the mean needs one entry per variable and "
        f"the covariance a square matrix, got {mean.shape}, {cov.shape}"
    )


def test_zero_rows_raise_value_error():
    _raised_value_error(
        lambda: gaussian_mle(np.zeros((0, 3))),
        "no observations means no mean: raise ValueError before averaging",
    )


def test_zero_columns_raise_value_error():
    _raised_value_error(
        lambda: gaussian_mle(np.zeros((4, 0))),
        "no variables means no covariance matrix: raise ValueError",
    )


def test_inputs_are_not_mutated():
    X = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 4.0]])
    X_before = X.copy()
    gaussian_mle(X)
    np.testing.assert_array_equal(
        X, X_before, err_msg="the input matrix was modified in place: centre a copy, not X"
    )
