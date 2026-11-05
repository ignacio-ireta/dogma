import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import covariance_matrix

SEEDS = st.integers(min_value=0, max_value=10**6)
ROWS = st.integers(min_value=2, max_value=12)
COLS = st.integers(min_value=1, max_value=4)


def _random_matrix(seed, n, d):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def _covariance(X, ddof, mistake):
    try:
        return covariance_matrix(X, ddof=ddof)
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


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS, st.sampled_from([0, 1]))
def test_matches_numpy_cov_on_random_matrices(seed, n, d, ddof):
    X = _random_matrix(seed, n, d)
    np.testing.assert_allclose(
        _covariance(X, ddof, "a valid data matrix was rejected or failed"),
        np.atleast_2d(np.cov(X, rowvar=False, ddof=ddof)),
        rtol=1e-9,
        atol=1e-12,
        err_msg="values differ from numpy.cov(X, rowvar=False, ddof=ddof): check the centring, "
        "the transpose and the divisor",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS)
def test_result_is_symmetric_and_positive_semidefinite(seed, n, d):
    C = _covariance(_random_matrix(seed, n, d), 1, "a valid data matrix was rejected or failed")
    np.testing.assert_allclose(
        C, C.T, atol=1e-12, err_msg="the covariance matrix must be symmetric: check the transpose"
    )
    eigenvalues = np.linalg.eigvalsh((C + C.T) / 2)
    assert eigenvalues.min() >= -1e-10 * max(1.0, np.abs(C).max()), (
        "a covariance matrix cannot have a negative eigenvalue: check the products"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS)
def test_shifting_every_variable_leaves_covariance_unchanged(seed, n, d):
    X = _random_matrix(seed, n, d)
    offset = np.random.default_rng(seed + 1).normal(scale=50.0, size=d)
    np.testing.assert_allclose(
        _covariance(X + offset, 1, "a valid shifted matrix was rejected or failed"),
        _covariance(X, 1, "a valid data matrix was rejected or failed"),
        rtol=1e-9,
        atol=1e-9,
        err_msg="adding a constant to a column must not change covariances: centre each column",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS, st.sampled_from([-3.0, 0.5]))
def test_scaling_data_scales_covariance_by_square_of_factor(seed, n, d, factor):
    X = _random_matrix(seed, n, d)
    np.testing.assert_allclose(
        _covariance(factor * X, 1, "a valid scaled matrix was rejected or failed"),
        factor**2 * _covariance(X, 1, "a valid data matrix was rejected or failed"),
        rtol=1e-9,
        atol=1e-12,
        err_msg="scaling X by s must scale the covariance by s^2, whatever the sign of s",
    )


def test_large_common_offset_does_not_destroy_precision():
    # Integer data plus 1e9 is exact in float64, so the covariance equals the one without offset.
    base = np.random.default_rng(3).integers(0, 10, size=(20, 2)).astype(np.float64)
    expected = np.cov(base, rowvar=False)
    result = _covariance(base + 1e9, 1, "a valid offset matrix was rejected or failed")
    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-9,
        atol=1e-9,
        err_msg="a large common offset destroyed the covariance: centre the columns before "
        "taking products, not E[xy] - E[x]E[y]",
    )


def test_single_variable_returns_one_by_one_matrix():
    result = _covariance([[1.0], [2.0], [4.0]], 1, "one variable was rejected or failed")
    assert result.shape == (1, 1), (
        "shape mismatch: a single variable needs a square matrix with one entry, "
        f"got {result.shape}"
    )
    np.testing.assert_allclose(
        result,
        [[np.var([1.0, 2.0, 4.0], ddof=1)]],
        rtol=1e-12,
        err_msg="the variance of a single column is the 1 x 1 covariance matrix",
    )


def test_collinear_columns_give_exact_rank_one_covariance():
    x = np.array([1.0, 3.0, 4.0, 8.0])
    X = np.column_stack([x, 2.0 * x])
    v = np.var(x, ddof=1)
    np.testing.assert_allclose(
        _covariance(X, 1, "a valid collinear matrix was rejected or failed"),
        [[v, 2 * v], [2 * v, 4 * v]],
        rtol=1e-12,
        err_msg="for Y = 2X the covariance must be [[v, 2v], [2v, 4v]] with v = Var(X)",
    )


@pytest.mark.parametrize(("n", "ddof"), [(1, 1), (2, 2)], ids=["one-row", "rows-equal-ddof"])
def test_too_few_observations_for_ddof_raise_value_error(n, ddof):
    _raised_value_error(
        lambda: covariance_matrix(np.ones((n, 2)), ddof=ddof),
        "n <= ddof leaves no positive divisor: raise ValueError before dividing",
    )


def test_negative_ddof_raises_value_error():
    _raised_value_error(
        lambda: covariance_matrix(_random_matrix(0, 5, 2), ddof=-1),
        "a negative ddof is not a valid divisor: raise ValueError",
    )


def test_one_dimensional_input_raises_value_error():
    _raised_value_error(
        lambda: covariance_matrix([1.0, 2.0, 3.0]),
        "a 1-D input is not a data matrix: raise ValueError (rows are observations)",
    )


def test_nan_entry_raises_value_error():
    _raised_value_error(
        lambda: covariance_matrix([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]),
        "a NaN entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_empty_matrix_raises_value_error():
    _raised_value_error(
        lambda: covariance_matrix(np.zeros((0, 3))),
        "an empty matrix has no covariance: raise ValueError",
    )


def test_infinite_entry_raises_value_error():
    _raised_value_error(
        lambda: covariance_matrix([[1.0, 2.0], [np.inf, 3.0], [4.0, 5.0]]),
        "an infinite entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_matrix_without_variables_raises_value_error():
    _raised_value_error(
        lambda: covariance_matrix(np.zeros((3, 0))),
        "a matrix with no variables has no covariance matrix: raise ValueError",
    )


def test_fractional_ddof_raises_value_error():
    _raised_value_error(
        lambda: covariance_matrix(_random_matrix(0, 5, 2), ddof=0.5),
        "ddof must be a whole number: a fractional divisor offset is not a valid ddof",
    )


def test_integer_input_returns_float64_matrix_of_shape_d_by_d():
    result = _covariance(
        [[1, 0, 2], [3, 1, 2], [5, 4, 9]], 1, "a valid integer matrix was rejected"
    )
    assert isinstance(result, np.ndarray), "return a NumPy array, not a list"
    assert result.shape == (3, 3), (
        "shape mismatch: the result must be a square matrix with one row and column per "
        f"variable, got {result.shape}"
    )
    assert result.dtype == np.float64, (
        f"dtype should be float64 for integer input, got {result.dtype}"
    )


def test_inputs_are_not_mutated():
    X = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 4.0]])
    X_before = X.copy()
    covariance_matrix(X)
    np.testing.assert_array_equal(
        X, X_before, err_msg="the input matrix was modified in place: centre a copy, not X"
    )


def test_single_observation_with_ddof_zero_is_a_zero_matrix():
    result = covariance_matrix([[1.0, 2.0]], ddof=0)
    assert result.shape == (2, 2), "one observation with ddof 0 still gives a d x d matrix"
    np.testing.assert_allclose(
        result,
        np.zeros((2, 2)),
        atol=0.0,
        err_msg="one observation has no spread: with ddof 0 the divisor is 1, so C is zero, "
        "and only n <= ddof is an error",
    )
