import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import correlation_matrix

from dogma_data import rng, sample_rows

SEEDS = st.integers(min_value=0, max_value=10**6)
ROWS = st.integers(min_value=2, max_value=15)
COLS = st.integers(min_value=1, max_value=4)


def _random_matrix(seed, n, d):
    """n rows and d columns of standardised wine: real, correlated measurements.

    Real measurements are rounded, so a small draw can repeat one value down a whole column.
    Such a draw is skipped: these tests are about columns that vary.
    """
    gen = rng(f"correlation-p02:data:{seed}")
    while True:
        X, _ = sample_rows("wine", gen, n, d, standardize=True)
        if np.all(np.ptp(X, axis=0) > 0.0):
            return X


def _correlation(X, mistake):
    try:
        return correlation_matrix(X)
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
@given(SEEDS, ROWS, COLS)
def test_matches_numpy_corrcoef_on_random_matrices(seed, n, d):
    X = _random_matrix(seed, n, d)
    with np.errstate(all="ignore"):
        expected = np.atleast_2d(np.corrcoef(X, rowvar=False))
    np.testing.assert_allclose(
        _correlation(X, "a valid data matrix was rejected or failed"),
        expected,
        rtol=1e-9,
        atol=1e-12,
        err_msg="values differ from numpy.corrcoef(X, rowvar=False): check the centring, the "
        "divisor and the standard deviations",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS)
def test_unit_diagonal_symmetry_and_bounds(seed, n, d):
    R = _correlation(_random_matrix(seed, n, d), "a valid data matrix was rejected or failed")
    np.testing.assert_allclose(
        np.diag(R), 1.0, atol=1e-9, err_msg="a non-constant column must correlate 1 with itself"
    )
    np.testing.assert_allclose(R, R.T, atol=1e-12, err_msg="the matrix must be symmetric")
    assert np.all(np.abs(R) <= 1.0), (
        "a correlation outside [-1, 1] is a rounding overshoot: clip the result"
    )


def test_perfect_negative_linear_relation_gives_minus_one():
    x = np.array([0.5, 1.5, 2.0, 4.0, 7.0])
    X = np.column_stack([x, -3.0 * x + 2.0])
    np.testing.assert_allclose(
        _correlation(X, "a valid linear pair was rejected or failed"),
        [[1.0, -1.0], [-1.0, 1.0]],
        atol=1e-12,
        err_msg="a column that is a negative multiple of another must correlate exactly -1",
    )


@settings(max_examples=50, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS, st.sampled_from([1e-200, 1e-100, 1e100, 1e200]))
def test_rescaling_a_column_does_not_change_correlations(seed, n, d, factor):
    base = _random_matrix(seed, n, d)
    scaled = base.copy()
    scaled[:, 0] *= factor
    with np.errstate(all="ignore"):
        expected = np.atleast_2d(np.corrcoef(base, rowvar=False))
    np.testing.assert_allclose(
        _correlation(scaled, "a valid rescaled matrix was rejected or failed"),
        expected,
        rtol=1e-9,
        atol=1e-9,
        err_msg="correlation is unitless: rescale columns before squaring so no value overflows "
        "or underflows",
    )


def test_constant_column_gives_nan_row_and_column_and_keeps_the_rest():
    base = _random_matrix(11, 6, 3)
    X = base.copy()
    X[:, 1] = 2.5
    with np.errstate(all="ignore"):
        expected = np.corrcoef(X, rowvar=False)
    result = _correlation(X, "a matrix with one constant column was rejected or failed")
    assert np.isnan(result[1]).all() and np.isnan(result[:, 1]).all(), (
        "a constant column's whole row and column must be NaN, including its diagonal"
    )
    keep = [0, 2]
    np.testing.assert_allclose(
        result[np.ix_(keep, keep)],
        expected[np.ix_(keep, keep)],
        rtol=1e-9,
        err_msg="the other columns must keep their correlations when one column is constant",
    )


def test_large_offset_column_with_small_spread_is_not_constant():
    # Spread 3 around 1e6: a relative tolerance such as np.isclose(max, min) would call it constant.
    X = np.column_stack([1e6 + np.arange(4.0), [1.0, 2.0, 4.0, 3.0]])
    with np.errstate(all="ignore"):
        expected = np.corrcoef(X, rowvar=False)
    result = _correlation(X, "a column with a large offset and spread 3 was rejected or failed")
    assert not np.isnan(result).any(), (
        "a column whose entries differ is not constant, however large its offset: do not compare "
        "max and min with a relative tolerance"
    )
    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-9,
        atol=1e-12,
        err_msg="a column with a large offset must keep its correlations: centre and rescale it",
    )


def test_entries_that_differ_only_in_the_last_digits_are_not_constant():
    # 1e6 and 1e6 + 1e-7 are different float64 values, so the column is not constant. A relative
    # tolerance tight enough to call it constant would return NaN here.
    X = np.column_stack([[1e6, 1e6, 1e6 + 1e-7], [1.0, 2.0, 4.0]])
    result = _correlation(X, "a column with entries that differ in the last digits was rejected")
    assert np.isfinite(result).all(), (
        "entries that differ are not a constant column: do not return NaN for them, even at a "
        "large offset"
    )


def test_repeated_value_that_is_not_exact_in_float_is_still_constant():
    # mean([0.1] * 3) is not exactly 0.1, so a variance test would miss this constant column.
    X = np.array([[0.1, 1.0], [0.1, 2.0], [0.1, 4.0]])
    result = _correlation(X, "a constant column of repeated 0.1 was rejected or failed")
    assert np.isnan(result[0]).all() and np.isnan(result[:, 0]).all(), (
        "a column with equal entries is constant even when its mean is not exactly that value: "
        "test for equality of the entries"
    )


def test_all_constant_matrix_is_all_nan():
    result = _correlation(np.full((4, 2), 3.0), "an all-constant matrix was rejected or failed")
    assert result.shape == (2, 2), (
        "shape mismatch: the result must be a square matrix with one "
        f"row and column per variable, got {result.shape}"
    )
    assert np.isnan(result).all(), "every correlation involves a constant column, so all are NaN"


def test_one_row_raises_value_error():
    _raised_value_error(
        lambda: correlation_matrix([[1.0, 2.0]]),
        "one observation has no correlation: raise ValueError when n < 2",
    )


def test_one_dimensional_input_raises_value_error():
    _raised_value_error(
        lambda: correlation_matrix([1.0, 2.0, 3.0]),
        "a 1-D input is not a data matrix: raise ValueError",
    )


def test_nan_entry_raises_value_error():
    _raised_value_error(
        lambda: correlation_matrix([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]),
        "a NaN entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_infinite_entry_raises_value_error():
    _raised_value_error(
        lambda: correlation_matrix([[1.0, 2.0], [np.inf, 3.0], [4.0, 5.0]]),
        "an infinite entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_matrix_without_variables_raises_value_error():
    _raised_value_error(
        lambda: correlation_matrix(np.zeros((3, 0))),
        "a matrix with no variables has no correlation matrix: raise ValueError",
    )


def test_tiny_but_nonconstant_column_is_not_treated_as_constant():
    # The column spans 2e-10 in absolute terms but is not constant: it must keep its correlations.
    X = np.column_stack([[0.0, 0.0, 1e-10, 2e-10], [1.0, 2.0, 4.0, 3.0]])
    with np.errstate(all="ignore"):
        expected = np.corrcoef(X, rowvar=False)
    result = _correlation(X, "a column with a tiny spread was rejected or failed")
    assert not np.isnan(result).any(), (
        "a column whose entries differ is not constant: compare entries exactly, do not use an "
        "absolute tolerance on the spread"
    )
    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-9,
        atol=1e-12,
        err_msg="a column with a small absolute spread must keep its correlations: rescale "
        "before testing the spread",
    )


def test_empty_matrix_raises_value_error():
    _raised_value_error(
        lambda: correlation_matrix(np.zeros((0, 2))),
        "an empty matrix has no correlation: raise ValueError",
    )


def test_integer_input_returns_float64_matrix_of_shape_d_by_d():
    result = _correlation([[1, 0, 2], [3, 1, 2], [5, 4, 9]], "a valid integer matrix was rejected")
    assert isinstance(result, np.ndarray), "return a NumPy array, not a list"
    assert result.shape == (3, 3), (
        "shape mismatch: the result must be a square matrix with one "
        f"row and column per variable, got {result.shape}"
    )
    assert result.dtype == np.float64, (
        f"dtype should be float64 for integer input, got {result.dtype}"
    )


def test_inputs_are_not_mutated():
    X = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 4.0]])
    X_before = X.copy()
    correlation_matrix(X)
    np.testing.assert_array_equal(
        X, X_before, err_msg="the input matrix was modified in place: rescale a copy, not X"
    )


def test_column_spanning_many_orders_of_magnitude_has_a_finite_correlation():
    X = np.column_stack([[1e-200, 1.0, 2.0], [1.0, 2.0, 4.0]])
    expected = np.corrcoef(np.column_stack([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]]), rowvar=False)[0, 1]
    result = correlation_matrix(X)
    assert np.isfinite(result).all(), (
        "a column whose entries span many orders of magnitude is still a valid column: its "
        "correlation is defined, so the result must be finite"
    )
    np.testing.assert_allclose(
        result[0, 1],
        expected,
        rtol=1e-9,
        atol=1e-12,
        err_msg="scale each column by its largest magnitude, not its smallest, before squaring",
    )
