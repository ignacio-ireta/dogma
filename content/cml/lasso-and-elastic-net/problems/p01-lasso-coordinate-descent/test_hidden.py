import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from sklearn.linear_model import Lasso
from submission import lasso_cd

from dogma_data import rng as data_rng
from dogma_data import sample_rows


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


def _data(seed, n, d, sparsity=0.5, kind="independent"):
    """Semi-synthetic: real features, a known sparse truth and mild noise.

    The columns are standardised real measurements, some strongly correlated: diabetes, or
    breast_cancer when more than ten columns are asked for. They are rescaled to unequal
    scales and centred on the drawn rows. The target is built from a known sparse coefficient
    vector so the tests can reason about which coefficients must vanish.
    """
    rng = data_rng(f"lasso-cd-p01:data:{seed}:{n}:{d}")
    name = "diabetes" if d <= 10 else "breast_cancer"
    X, _ = sample_rows(name, rng, n, d, standardize=True)
    X = X * (0.3 + (3.0 - 0.3) * _real_unit(rng, d))
    if kind == "correlated":
        X[:, 1] = X[:, 0] + 0.1 * _real_normal(rng, n)
    X = X - X.mean(axis=0)
    beta = np.where(_real_unit(rng, d) < sparsity, 2.0 * _real_normal(rng, d), 0.0)
    y = X @ beta + 0.5 * _real_normal(rng, n)
    y = y - y.mean()
    return X, y


def _alpha_max(X, y):
    """The smallest penalty at which every coefficient is zero: max |X_j^T y| / n."""
    return float(np.max(np.abs(X.T @ y)) / X.shape[0])


def _kkt_violation(X, y, w, alpha):
    """Largest violation of the lasso optimality conditions (an independent check on the "
    "solution)."""
    g = X.T @ (y - X @ w) / X.shape[0]
    active = w != 0
    per_active = np.abs(g - alpha * np.sign(w))  # must be 0 where w_j != 0
    per_zero = np.maximum(np.abs(g) - alpha, 0.0)  # must be 0 where w_j == 0
    return float(np.max(np.where(active, per_active, per_zero)))


def _call(X, y, alpha, **kwargs):
    """Call the learner's function on valid input, naming the mistake when the call fails."""
    try:
        return lasso_cd(X, y, alpha, **kwargs)
    except ValueError:
        pytest.fail(
            "a valid input was rejected with ValueError: check the shape and parameter rules"
        )
    except Exception as exc:
        pytest.fail(
            f"a valid input raised {type(exc).__name__}: return a float64 coefficient array of "
            "shape (d,)"
        )


def _checked(X, y, alpha, **kwargs):
    """Fit on copies, so no test corrupts another's arrays, and check the return type and shape."""
    w = _call(np.array(X, copy=True), np.array(y, copy=True), alpha, **kwargs)
    if not isinstance(w, np.ndarray):
        pytest.fail("the coefficients must be a NumPy array")
    if w.dtype != np.float64:
        pytest.fail("the coefficients must be a float64 array")
    if w.shape != (np.shape(X)[1],):
        pytest.fail("the coefficients must be a 1-D array with one entry per column of X")
    return w


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("seed", "n", "d", "fraction", "kind"),
    [
        (0, 60, 5, 0.3, "independent"),
        (1, 80, 8, 0.1, "independent"),
        (2, 50, 6, 0.6, "correlated"),
        (3, 20, 30, 0.2, "independent"),
        (4, 40, 4, 0.9, "independent"),
    ],
    ids=["moderate", "weak-penalty", "correlated", "wide", "strong-penalty"],
)
def test_matches_scikit_learn_lasso_without_intercept(seed, n, d, fraction, kind):
    X, y = _data(seed, n, d, kind=kind)
    alpha = fraction * _alpha_max(X, y)
    ref = Lasso(alpha=alpha, fit_intercept=False, tol=1e-14, max_iter=10**6).fit(X, y).coef_
    w = _checked(X, y, alpha, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        w,
        ref,
        atol=1e-7,
        err_msg="coefficients differ from sklearn Lasso(fit_intercept=False): check the soft "
        "threshold "
        "(alpha, not 2 alpha), the 1/n in rho and z, and the division by z_j",
    )


@pytest.mark.parametrize("seed", [5, 6, 7, 8], ids=["first", "second", "third", "fourth"])
def test_kkt_conditions_hold_at_the_default_tolerance(seed):
    X, y = _data(seed, 60, 6, sparsity=0.4)
    alpha = 0.2 * _alpha_max(X, y)
    w = _checked(X, y, alpha)
    assert np.count_nonzero(w) >= 1, "the fixture should have some active coefficients"
    violation = _kkt_violation(X, y, w, alpha)
    assert violation < 1e-6, (
        "the returned point is not optimal: active coefficients must satisfy "
        "(1/n) X_j^T (y - Xw) = alpha sign(w_j), and inactive ones |(1/n) X_j^T (y - Xw)| <= alpha"
    )


@pytest.mark.parametrize("factor", [1.000001, 3.0], ids=["just-above", "far-above"])
def test_alpha_at_or_above_alpha_max_gives_all_zeros(factor):
    X, y = _data(9, 40, 5)
    w = _checked(X, y, factor * _alpha_max(X, y))
    assert np.all(w == 0.0), (
        "alpha at or above max |X_j^T y| / n must set every coefficient to exactly zero"
    )


def test_just_below_alpha_max_exactly_one_coefficient_is_nonzero():
    X, y = _data(10, 40, 5)
    w = _checked(X, y, 0.999 * _alpha_max(X, y), tol=1e-12, max_iter=5_000)
    assert np.count_nonzero(w) == 1, (
        "just below alpha_max only the column with the largest |X_j^T y| may be active: "
        "the others must be exactly zero"
    )


def test_l1_norm_does_not_increase_as_alpha_grows():
    X, y = _data(11, 50, 6)
    amax = _alpha_max(X, y)
    norms = [
        np.sum(np.abs(_checked(X, y, a, tol=1e-12, max_iter=5_000)))
        for a in amax * np.linspace(0.02, 1.0, 20)
    ]
    for larger_alpha_norm, smaller_alpha_norm in zip(norms[1:], norms[:-1], strict=True):
        assert larger_alpha_norm <= smaller_alpha_norm + 1e-8, (
            "the l1 norm of the solution must not increase as alpha grows: check the soft threshold"
        )


def test_tiny_alpha_recovers_least_squares_on_full_rank_data():
    X, y = _data(12, 50, 4, sparsity=1.0)
    ls = np.linalg.lstsq(X, y, rcond=None)[0]
    w = _checked(X, y, 1e-9, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        w,
        ls,
        rtol=1e-6,
        atol=1e-7,
        err_msg="as alpha goes to 0 the lasso must approach ordinary least squares",
    )


def test_nearly_duplicate_columns_match_scikit_learn():
    rng = np.random.default_rng(31)
    x = _real_normal(rng, 60)
    X = np.column_stack([x, x + 0.01 * _real_normal(rng, 60), _real_normal(rng, 60)])
    X = X - X.mean(axis=0)
    y = 2.0 * x + 0.3 * _real_normal(rng, 60)
    y = y - y.mean()
    alpha = 0.05 * _alpha_max(X, y)
    ref = Lasso(alpha=alpha, fit_intercept=False, tol=1e-14, max_iter=10**6).fit(X, y).coef_
    # Cyclic updates crawl along nearly collinear columns, so this case needs a larger sweep budget.
    w = _checked(X, y, alpha, tol=1e-10, max_iter=50_000)
    np.testing.assert_allclose(
        w,
        ref,
        atol=1e-6,
        err_msg="near-duplicate columns disagree with sklearn: sweep the coordinates in turn, each "
        "update must use the newest coefficients and the residual must be updated after every step",
    )


def test_no_intercept_is_fitted_on_uncentred_data():
    X, y = _data(13, 50, 3)
    X_raw = X + 5.0
    y_raw = y + 10.0
    alpha = 0.1 * _alpha_max(X_raw, y_raw)
    ref = Lasso(alpha=alpha, fit_intercept=False, tol=1e-14, max_iter=10**6).fit(X_raw, y_raw).coef_
    w = _checked(X_raw, y_raw, alpha, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        w,
        ref,
        atol=1e-7,
        err_msg="coefficients differ from sklearn on uncentred data: fit y ~ X w exactly, with no "
        "centring and no intercept",
    )


def test_zero_column_gets_a_zero_coefficient_without_nan():
    X, y = _data(14, 40, 4)
    X_zero = X.copy()
    X_zero[:, 2] = 0.0
    alpha = 0.2 * _alpha_max(X, y)
    w = _checked(X_zero, y, alpha, tol=1e-12, max_iter=5_000)
    assert np.all(np.isfinite(w)), "a zero column must not produce NaN or inf"
    assert w[2] == 0.0, "a column with no variation has no effect and its coefficient must stay 0"
    keep = [0, 1, 3]
    ref = (
        Lasso(alpha=alpha, fit_intercept=False, tol=1e-14, max_iter=10**6).fit(X[:, keep], y).coef_
    )
    np.testing.assert_allclose(
        w[keep],
        ref,
        atol=1e-7,
        err_msg="the other coefficients must match the fit without the zero column",
    )


@pytest.mark.parametrize("c", [0.5, 2.0, 10.0], ids=["half", "double", "ten-times"])
def test_scaling_y_and_alpha_together_scales_the_solution(c):
    X, y = _data(15, 40, 5)
    alpha = 0.3 * _alpha_max(X, y)
    base = _checked(X, y, alpha, tol=1e-12, max_iter=5_000)
    scaled = _checked(X, c * y, c * alpha, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        scaled,
        c * base,
        atol=1e-7,
        err_msg="lasso(X, c y, c alpha) must equal c * lasso(X, y, alpha): the objective scales by "
        "c^2",
    )


@settings(max_examples=25, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(0, 10**6), fraction=st.floats(0.05, 0.95))
def test_random_problems_agree_with_scikit_learn(seed, fraction):
    X, y = _data(seed, 30, 4)
    alpha = fraction * _alpha_max(X, y)
    ref = Lasso(alpha=alpha, fit_intercept=False, tol=1e-14, max_iter=10**6).fit(X, y).coef_
    w = _checked(X, y, alpha, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        w,
        ref,
        atol=1e-7,
        err_msg="a random problem disagrees with sklearn Lasso(fit_intercept=False)",
    )


def test_inputs_are_not_mutated():
    X, y = _data(16, 25, 3)
    X_before, y_before = X.copy(), y.copy()
    _call(X, y, 0.1 * _alpha_max(X, y))
    np.testing.assert_array_equal(X, X_before, err_msg="X was modified in place")
    np.testing.assert_array_equal(
        y, y_before, err_msg="y was modified in place: keep a residual copy"
    )


def test_integer_inputs_give_the_same_solution_as_floats():
    X = np.array([[1, 0], [0, 1], [2, 1], [1, 3], [3, 2]])
    y = np.array([2, 1, 4, 5, 6])
    w_int = _checked(X, y, 0.5)
    w_flt = _checked(X.astype(float), y.astype(float), 0.5)
    np.testing.assert_allclose(
        w_int,
        w_flt,
        rtol=1e-12,
        err_msg="integer inputs gave a different solution from the same values as floats",
    )


@pytest.mark.parametrize(
    ("X", "y", "alpha", "kwargs"),
    [
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 0.1, {}),
        (_real_normal(np.random.default_rng(20), (5, 2)), np.ones(4), 0.1, {}),
        (np.array([[1.0, np.nan], [2.0, 1.0], [3.0, 0.0]]), np.ones(3), 0.1, {}),
        (_real_normal(np.random.default_rng(21), (5, 2)), np.ones(5), -0.1, {}),
        (_real_normal(np.random.default_rng(22), (5, 2)), np.ones(5), np.nan, {}),
        (_real_normal(np.random.default_rng(23), (5, 2)), np.ones(5), 0.1, {"max_iter": 0}),
        (_real_normal(np.random.default_rng(24), (5, 2)), np.ones(5), 0.1, {"tol": 0.0}),
        (_real_normal(np.random.default_rng(25), (3, 2)), np.ones((3, 1)), 0.1, {}),
        (_real_normal(np.random.default_rng(26), (5, 2)), np.ones(5), 0.1, {"tol": np.nan}),
    ],
    ids=[
        "1-D-X",
        "length-mismatch",
        "nan-in-X",
        "negative-alpha",
        "nan-alpha",
        "zero-iterations",
        "zero-tol",
        "column-target",
        "nan-tol",
    ],
)
def test_invalid_input_raises_value_error(X, y, alpha, kwargs):
    _expect_value_error(
        lambda: lasso_cd(X, y, alpha, **kwargs),
        "an invalid argument was accepted: check the shapes, the finiteness, alpha >= 0, "
        "max_iter >= 1 and tol > 0",
    )


def test_one_sweep_respects_the_budget_and_uses_the_latest_coefficient():
    w = _checked([[1.0, 1.0], [1.0, 2.0]], [1.0, 3.0], 0.2, max_iter=1)
    np.testing.assert_allclose(
        w,
        [1.8, 0.24],
        rtol=1e-12,
        atol=1e-12,
        err_msg="one sweep from zero must use cyclic updates and stop at the iteration budget",
    )


@pytest.mark.parametrize(
    ("X", "y"),
    [(np.empty((0, 2)), []), (np.empty((3, 0)), np.ones(3))],
    ids=["no-samples", "no-features"],
)
def test_empty_design_dimensions_raise_value_error(X, y):
    _expect_value_error(
        lambda: lasso_cd(X, y, 0.1),
        "empty design dimensions must raise ValueError before any reduction",
    )


def test_large_tolerance_stops_after_the_first_sweep():
    w = _checked([[1.0, 1.0], [1.0, 2.0]], [1.0, 3.0], 0.2, max_iter=50, tol=10.0)
    np.testing.assert_allclose(
        w,
        [1.8, 0.24],
        rtol=1e-12,
        atol=1e-12,
        err_msg="a sweep with change below tol must end the loop early, even with budget left",
    )


def test_exactly_zero_correlation_gives_exact_zero_not_nan():
    # sign(0) is 0, so a correlation of exactly zero must give a zero coefficient. Computing the
    # sign as rho / |rho| divides 0 by 0 and returns NaN, which the existing tests never reach.
    cases = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros(2)),
        (np.array([[1.0], [-1.0]]), np.array([1.0, 1.0])),
    ]
    for X, y in cases:
        w = _checked(X, y, 0.5, tol=1e-12, max_iter=5_000)
        if not np.all(np.isfinite(w)):
            pytest.fail(
                "a zero correlation gave NaN or inf: the sign of an exact zero is 0, so the "
                "update must not divide by |rho|"
            )
        if np.any(w != 0.0):
            pytest.fail(
                "a zero correlation must give an exact zero coefficient, not a small nonzero value"
            )


def test_mismatched_target_length_raises_even_when_every_column_is_zero():
    # With no variation in X no coordinate is ever updated, so only the length check can raise.
    _expect_value_error(
        lambda: lasso_cd(np.zeros((4, 2)), np.ones(3), 0.1),
        "y with a length different from the rows of X must raise ValueError",
    )


def test_zero_penalty_is_accepted_and_gives_least_squares():
    rng = np.random.default_rng(40)
    X = _real_normal(rng, (20, 3))
    y = _real_normal(rng, 20)
    w = _checked(X, y, 0.0, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        w,
        np.linalg.lstsq(X, y, rcond=None)[0],
        atol=1e-7,
        err_msg="alpha = 0 is a valid penalty and must give the least-squares coefficients",
    )


def test_stop_rule_uses_the_largest_true_coefficient_change():
    # The largest change after sweep 2 is 0.5 < 0.6, so the run stops at [1.5, 0.75].
    w = _checked([[1.0, 1.0], [0.0, 1.0]], [2.0, 1.0], 0.0, tol=0.6, max_iter=100)
    np.testing.assert_allclose(
        w,
        [1.5, 0.75],
        rtol=1e-12,
        atol=1e-12,
        err_msg="the stopping test must use the size of each coefficient change, not its sum",
    )


def test_partly_non_finite_target_raises_value_error():
    _expect_value_error(
        lambda: lasso_cd([[1.0], [2.0], [3.0]], [1.0, np.nan, 2.0], 0.1, max_iter=5),
        "a target with any non-finite entry must raise ValueError",
    )


def test_zero_column_does_not_stop_the_sweep_before_later_columns():
    # Column 0 is all zeros and column 1 is not. The sweep must still update column 1.
    w = _checked([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], [1.0, 2.0, 3.0], 0.0, tol=1e-12, max_iter=5)
    np.testing.assert_allclose(
        w,
        [0.0, 1.0],
        atol=1e-10,
        err_msg="a zero column must be skipped, not end the sweep: later columns still update",
    )


def test_a_single_row_of_data_is_valid_and_gives_the_closed_form_fit():
    # One row and one column: the objective is (1/2)(4 - 2w)^2 + 0.5|w|. For w > 0 its slope is
    # -2(4 - 2w) + 0.5, which is zero at w = 1.875. The fit must not be refused.
    w = _checked([[2.0]], [4.0], 0.5, tol=1e-12, max_iter=100)
    np.testing.assert_allclose(
        w,
        [1.875],
        atol=1e-9,
        err_msg="a single row is valid input and must give the one-row soft-threshold fit",
    )


def test_an_integer_tolerance_is_a_valid_real_number():
    # One column, so the fit is the worked example's 25/14. An integer tol of 1 is a finite real
    # number greater than zero, so the call must run and return that coefficient.
    w = _checked([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0], 1.0, tol=1, max_iter=100)
    np.testing.assert_allclose(
        w,
        [25.0 / 14.0],
        atol=1e-9,
        err_msg="an integer tol is a finite real number greater than zero and must be accepted",
    )


def test_large_entries_are_fitted_in_the_callers_own_units():
    # One column with entries 100, 200, 300 and y = 2, 4, 6. Here n = 3, x'y = 2800, x'x = 140000,
    # so the lasso fit is (x'y / n - alpha) / (x'x / n) = (2800/3 - 1) / (140000/3) = 2797/140000.
    w = _checked([[100.0], [200.0], [300.0]], [2.0, 4.0, 6.0], 1.0, tol=1e-12, max_iter=100)
    np.testing.assert_allclose(
        w,
        [2797.0 / 140000.0],
        atol=1e-10,
        err_msg="a column with large entries must be fitted as given and its coefficient returned "
        "in the caller's units",
    )
