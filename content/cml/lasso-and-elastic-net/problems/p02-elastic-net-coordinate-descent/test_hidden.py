import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.linear_model import ElasticNet
from submission import elastic_net_cd


def _data(seed, n, d, sparsity=0.5):
    """Centred data with unequal column scales, a sparse truth and mild noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) * rng.uniform(0.3, 3.0, size=d)
    X = X - X.mean(axis=0)
    beta = np.where(rng.random(d) < sparsity, 2.0 * rng.normal(size=d), 0.0)
    y = X @ beta + 0.5 * rng.normal(size=n)
    return X, y - y.mean()


def _alpha_max(X, y, l1_ratio):
    """Smallest alpha with every coefficient zero: max |X_j^T y| / (n * l1_ratio)."""
    return float(np.max(np.abs(X.T @ y)) / (X.shape[0] * l1_ratio))


def _kkt_violation(X, y, w, alpha, l1_ratio):
    """Largest violation of the elastic net optimality conditions, checked independently."""
    g = X.T @ (y - X @ w) / X.shape[0]
    active = w != 0
    per_active = np.abs(g - alpha * l1_ratio * np.sign(w) - alpha * (1 - l1_ratio) * w)
    per_zero = np.maximum(np.abs(g) - alpha * l1_ratio, 0.0)
    return float(np.max(np.where(active, per_active, per_zero)))


def _call(X, y, alpha, l1_ratio, **kwargs):
    """Call the learner's function on valid input, naming the mistake when the call fails."""
    try:
        return elastic_net_cd(X, y, alpha, l1_ratio, **kwargs)
    except ValueError:
        pytest.fail(
            "a valid input was rejected with ValueError: check the shape and parameter rules"
        )
    except Exception as exc:
        pytest.fail(
            f"a valid input raised {type(exc).__name__}: return a float64 coefficient array of "
            "shape (d,)"
        )


def _checked(X, y, alpha, l1_ratio, **kwargs):
    """Fit on copies, so no test corrupts another's arrays, and check the return type and shape."""
    w = _call(np.array(X, copy=True), np.array(y, copy=True), alpha, l1_ratio, **kwargs)
    if not isinstance(w, np.ndarray):
        pytest.fail("the coefficients must be a NumPy array")
    if w.dtype != np.float64:
        pytest.fail("the coefficients must be a float64 array")
    if w.shape != (np.shape(X)[1],):
        pytest.fail("the coefficients must be a 1-D array with one entry per column of X")
    return w


def _sk_enet(X, y, alpha, l1_ratio):
    """scikit-learn's elastic net without an intercept, at a tight tolerance (the reference)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, tol=1e-14, max_iter=10**6
        )
        return model.fit(X, y).coef_


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("seed", "l1_ratio", "fraction"),
    [(0, 0.1, 0.3), (1, 0.5, 0.2), (2, 0.9, 0.5), (3, 1.0, 0.4), (4, 0.7, 0.8)],
    ids=["mostly-ridge", "balanced", "mostly-lasso", "pure-lasso", "strong-penalty"],
)
def test_matches_scikit_learn_elastic_net_without_intercept(seed, l1_ratio, fraction):
    X, y = _data(seed, 60, 6)
    alpha = fraction * _alpha_max(X, y, l1_ratio)
    ref = _sk_enet(X, y, alpha, l1_ratio)  # computed before the learner runs
    w = _checked(X, y, alpha, l1_ratio, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        w,
        ref,
        atol=1e-7,
        err_msg="coefficients differ from sklearn ElasticNet: the threshold is alpha * l1_ratio "
        "and the "
        "denominator is z_j + alpha * (1 - l1_ratio), with the 1/n in rho and z",
    )


@pytest.mark.parametrize("seed", [5, 6, 7], ids=["first", "second", "third"])
def test_kkt_conditions_hold_at_the_default_tolerance(seed):
    X, y = _data(seed, 60, 6, sparsity=0.4)
    alpha, l1_ratio = 0.2 * _alpha_max(X, y, 0.6), 0.6
    w = _checked(X, y, alpha, l1_ratio)
    assert np.count_nonzero(w) >= 1, "the fixture should have some active coefficients"
    violation = _kkt_violation(X, y, w, alpha, l1_ratio)
    assert violation < 1e-6, (
        "the returned point is not optimal: an active coefficient must satisfy (1/n) X_j^T (y - "
        "Xw) = "
        "alpha * l1_ratio * sign(w_j) + alpha * (1 - l1_ratio) * w_j, and an inactive one "
        "|(1/n) X_j^T (y - Xw)| <= alpha * l1_ratio"
    )


@pytest.mark.parametrize("alpha", [0.3, 2.0], ids=["moderate", "strong"])
def test_ridge_limit_matches_the_closed_form(alpha):
    X, y = _data(8, 40, 4)
    n = X.shape[0]
    closed_form = np.linalg.solve(X.T @ X / n + alpha * np.eye(4), X.T @ y / n)
    w = _checked(X, y, alpha, 0.0, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        w,
        closed_form,
        atol=1e-7,
        err_msg="with l1_ratio = 0 the problem is ridge: solve (X^T X / n + alpha I) w = X^T y / n",
    )


def test_duplicate_columns_share_their_weight_evenly():
    rng = np.random.default_rng(9)
    x = rng.normal(size=50)
    x = x - x.mean()
    y = 2.0 * x + 0.1 * rng.normal(size=50)
    y = y - y.mean()
    X = np.column_stack([x, x])
    w = _checked(X, y, 0.1, 0.5, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        w[0],
        w[1],
        atol=1e-9,
        err_msg="identical columns must get identical coefficients: the L2 term makes the "
        "minimiser "
        "unique and spreads the weight evenly (the grouping effect)",
    )
    assert w[0] > 0.0, "the shared weight must be non-zero for this signal"


def test_nearly_duplicate_columns_share_their_weight():
    rng = np.random.default_rng(10)
    x = rng.normal(size=60)
    X = np.column_stack([x, x + 1e-3 * rng.normal(size=60)])
    X = X - X.mean(axis=0)
    y = 2.0 * x + 0.1 * rng.normal(size=60)
    y = y - y.mean()
    w = _checked(X, y, 0.1, 0.5, tol=1e-12, max_iter=5_000)
    assert abs(w[0] - w[1]) < 1e-3 * max(1.0, abs(w).max()), (
        "nearly identical columns must get nearly identical coefficients under the elastic net"
    )


def test_zero_column_with_pure_lasso_penalty_gives_zero_without_nan():
    X, y = _data(11, 40, 3)
    X_zero = np.column_stack([X, np.zeros(40)])
    w = _checked(X_zero, y, 0.1, 1.0, tol=1e-12, max_iter=5_000)
    assert np.all(np.isfinite(w)), "a zero column with no L2 term must not produce NaN or inf"
    assert w[3] == 0.0, "a zero column has no effect on the fit, so its coefficient must stay 0"


def test_zero_column_with_mixed_penalty_gives_zero():
    X, y = _data(12, 40, 3)
    X_zero = np.column_stack([X, np.zeros(40)])
    w = _checked(X_zero, y, 0.1, 0.5, tol=1e-12, max_iter=5_000)
    assert w[3] == 0.0, "a zero column must keep a zero coefficient under any mixing ratio"


@pytest.mark.parametrize("l1_ratio", [0.5, 1.0], ids=["mixed", "lasso"])
def test_all_zero_at_or_above_the_elastic_net_threshold(l1_ratio):
    X, y = _data(13, 40, 5)
    w = _checked(X, y, 1.000001 * _alpha_max(X, y, l1_ratio), l1_ratio)
    assert np.all(w == 0.0), (
        "alpha at or above max |X_j^T y| / (n l1_ratio) must set every coefficient to exactly zero"
    )


@pytest.mark.parametrize(
    ("c", "l1_ratio"),
    [(0.5, 0.5), (3.0, 0.5), (3.0, 1.0), (0.5, 0.1)],
    ids=["half-mixed", "triple-mixed", "triple-lasso", "half-ridgy"],
)
def test_scaling_the_data_by_c_and_alpha_by_c_squared_leaves_the_solution_unchanged(c, l1_ratio):
    # Multiplying X and y by c scales the squared error by c^2, so alpha must scale by c^2 to keep
    # the same minimiser. Both penalty terms follow this rule, whatever the mixing ratio.
    X, y = _data(14, 40, 4)
    alpha = 0.2 * _alpha_max(X, y, 0.5)
    base = _checked(X, y, alpha, l1_ratio, tol=1e-12, max_iter=5_000)
    scaled = _checked(c * X, c * y, c * c * alpha, l1_ratio, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        scaled,
        base,
        atol=1e-7,
        err_msg="scaling X and y by c must scale alpha by c^2 and leave the coefficients unchanged",
    )


@settings(max_examples=20, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(0, 10**6), l1_ratio=st.floats(0.05, 1.0), fraction=st.floats(0.05, 0.9))
def test_random_problems_agree_with_scikit_learn(seed, l1_ratio, fraction):
    X, y = _data(seed, 30, 4)
    alpha = fraction * _alpha_max(X, y, l1_ratio)
    ref = _sk_enet(X, y, alpha, l1_ratio)
    w = _checked(X, y, alpha, l1_ratio, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        w,
        ref,
        atol=1e-7,
        err_msg="a random problem disagrees with sklearn ElasticNet(fit_intercept=False)",
    )


@pytest.mark.parametrize(
    ("alpha", "l1_ratio", "kwargs"),
    [
        (0.1, -0.1, {}),
        (0.1, 1.5, {}),
        (0.1, np.nan, {}),
        (-0.1, 0.5, {}),
        (0.1, 0.5, {"max_iter": 0}),
        (0.1, 0.5, {"tol": 0.0}),
    ],
    ids=[
        "ratio-negative",
        "ratio-above-one",
        "ratio-nan",
        "negative-alpha",
        "zero-iterations",
        "zero-tol",
    ],
)
def test_invalid_parameters_raise_value_error(alpha, l1_ratio, kwargs):
    X, y = _data(15, 20, 3)
    _expect_value_error(
        lambda: elastic_net_cd(X, y, alpha, l1_ratio, **kwargs),
        "an invalid parameter was accepted: alpha >= 0, 0 <= l1_ratio <= 1, max_iter >= 1 and tol "
        "> 0",
    )


def test_invalid_shapes_raise_value_error():
    _expect_value_error(
        lambda: elastic_net_cd(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 0.1, 0.5),
        "a 1-D X was accepted: X must be 2-D, one row per sample",
    )


def test_inputs_are_not_mutated():
    X, y = _data(16, 25, 3)
    X_before, y_before = X.copy(), y.copy()
    _call(X, y, 0.1, 0.5)
    np.testing.assert_array_equal(X, X_before, err_msg="X was modified in place")
    np.testing.assert_array_equal(
        y, y_before, err_msg="y was modified in place: keep a residual copy"
    )


def test_integer_inputs_give_the_same_solution_as_floats():
    X = np.array([[1, 0], [0, 1], [2, 1], [1, 3], [3, 2]])
    y = np.array([2, 1, 4, 5, 6])
    w_int = _checked(X, y, 0.5, 0.5)
    w_flt = _checked(X.astype(float), y.astype(float), 0.5, 0.5)
    np.testing.assert_allclose(
        w_int,
        w_flt,
        rtol=1e-12,
        err_msg="integer inputs gave a different solution from the same values as floats",
    )


def test_one_sweep_respects_the_budget_and_uses_the_latest_coefficient():
    w = _checked([[1.0, 1.0], [1.0, 2.0]], [1.0, 3.0], 0.2, 0.5, max_iter=1)
    np.testing.assert_allclose(
        w,
        [19.0 / 11, 89.0 / 286],
        rtol=1e-12,
        atol=1e-12,
        err_msg="one sweep from zero must use cyclic updates and stop at the iteration budget",
    )


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.empty((0, 2)), []),
        (np.empty((3, 0)), np.ones(3)),
        (np.ones((3, 2)), np.ones(2)),
        (np.ones((3, 2)), np.ones((3, 1))),
        (np.array([[np.nan], [1.0]]), [1.0, 2.0]),
        (np.ones((2, 1)), [1.0, np.inf]),
    ],
    ids=[
        "no-samples",
        "no-features",
        "length-mismatch",
        "column-target",
        "nan-feature",
        "inf-target",
    ],
)
def test_empty_misaligned_and_nonfinite_data_raise_value_error(X, y):
    _expect_value_error(
        lambda: elastic_net_cd(X, y, 0.1, 0.5),
        "invalid data must raise ValueError: require nonempty aligned finite X and y",
    )


def test_large_tolerance_stops_after_the_first_sweep():
    w = _checked([[1.0, 1.0], [1.0, 2.0]], [1.0, 3.0], 0.2, 0.5, max_iter=50, tol=10.0)
    np.testing.assert_allclose(
        w,
        [19.0 / 11, 89.0 / 286],
        rtol=1e-12,
        atol=1e-12,
        err_msg="a sweep with change below tol must end the loop early, even with budget left",
    )


def test_mismatched_target_length_raises_when_every_column_is_zero_and_pure_lasso():
    # With l1_ratio = 1 and a zero column the denominator is zero, so no coordinate is updated and
    # only the length check can raise.
    _expect_value_error(
        lambda: elastic_net_cd(np.zeros((4, 2)), np.ones(3), 0.1, 1.0),
        "y with a length different from the rows of X must raise ValueError",
    )


def test_zero_penalty_is_accepted_and_gives_least_squares():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    w = _checked(X, y, 0.0, 0.5, tol=1e-12, max_iter=5_000)
    np.testing.assert_allclose(
        w,
        np.linalg.lstsq(X, y, rcond=None)[0],
        atol=1e-7,
        err_msg="alpha = 0 is a valid penalty and must give the least-squares coefficients",
    )


def test_zero_column_with_pure_lasso_does_not_end_the_sweep_before_later_columns():
    # Column 0 is all zeros and the penalty is pure L1, so its denominator is zero: it is skipped.
    # Column 1 must still be updated in the same sweep.
    X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])
    alpha = 0.1
    z1 = np.sum(X[:, 1] ** 2) / 3
    expected_w1 = (np.sum(X[:, 1] * y) / 3 - alpha) / z1
    w = _checked(X, y, alpha, 1.0, tol=1e-12, max_iter=5)
    np.testing.assert_allclose(
        w,
        [0.0, expected_w1],
        rtol=1e-10,
        atol=1e-12,
        err_msg="a skipped zero column must not end the sweep: later columns still update",
    )
