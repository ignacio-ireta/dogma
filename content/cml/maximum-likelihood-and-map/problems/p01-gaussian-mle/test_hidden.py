import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from scipy.special import ndtr
from submission import gaussian_mle

from dogma_data import rng, sample_rows

SEEDS = st.integers(min_value=0, max_value=10**6)
ROWS = st.integers(min_value=1, max_value=12)
COLS = st.integers(min_value=1, max_value=4)


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


def _random_matrix(seed, n, d):
    """n rows and d columns of raw wine: real measurements on very different scales."""
    X, _ = sample_rows("wine", rng(f"gaussian-mle-p01:data:{seed}"), n, d)
    return X


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
    X = _real_normal(rng, (20, 3)) @ A.T + np.array([1.0, -2.0, 0.5])
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
    base = _real_ints(np.random.default_rng(4), 0, 10, (20, 2)).astype(np.float64)
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


def test_infinite_entry_raises_value_error():
    _raised_value_error(
        lambda: gaussian_mle([[1.0, np.inf], [2.0, 3.0]]),
        "an infinite entry must raise ValueError: the Contract rejects NaN or infinity",
    )


def test_read_only_input_arrays_are_accepted_and_left_alone():
    X = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 4.0]])
    for array in (X,):
        array.setflags(write=False)
    try:
        gaussian_mle(X)
    except Exception as exc:
        if "read-only" in str(exc):
            pytest.fail(
                "an input array was written to in place: inputs must not be mutated, so work "
                "on a copy",
                pytrace=False,
            )
        pytest.fail(
            "a valid call failed when its input arrays were read-only: a correct solution "
            "only reads its inputs",
            pytrace=False,
        )
