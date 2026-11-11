import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.linear_model import Ridge
from submission import ridge_fit


def _data(seed, n, d):
    """Random design, a linear signal with an intercept, and mild noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = rng.normal(size=d)
    y = 1.5 + X @ beta + 0.3 * rng.normal(size=n)
    return X, y


def _call(X, y, alpha, fit_intercept=True):
    """Call the learner's function on valid input, naming the mistake when the call fails."""
    try:
        result = ridge_fit(X, y, alpha, fit_intercept=fit_intercept)
        if not isinstance(result, tuple) or len(result) != 2:
            pytest.fail("return a two-item tuple containing the intercept and coefficients")
        return result
    except ValueError:
        pytest.fail("a valid input was rejected with ValueError: check the rank and shape rules")
    except Exception as exc:
        pytest.fail(
            f"a valid input raised {type(exc).__name__}: return a float intercept and a (d,) coef "
            "array"
        )


def _checked(X, y, alpha, fit_intercept=True):
    """Fit on copies (so no test corrupts another's arrays) and check the return types."""
    intercept, coef = _call(
        np.array(X, copy=True), np.array(y, copy=True), alpha, fit_intercept=fit_intercept
    )
    if not isinstance(intercept, float):
        pytest.fail("the intercept must be a Python float, not an array: return float(...)")
    if not isinstance(coef, np.ndarray) or coef.dtype != np.float64:
        pytest.fail("coef must be a float64 NumPy array")
    if coef.shape != (np.shape(X)[1],):
        pytest.fail("coef must be a 1-D array with one entry per feature, not a column vector")
    return intercept, coef


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("n", "d", "seed", "alpha"),
    [(30, 1, 0, 0.5), (50, 6, 1, 2.0), (80, 3, 2, 25.0), (40, 4, 3, 1e-3), (60, 2, 4, 300.0)],
    ids=["small", "wide-features", "large", "weak-penalty", "strong-penalty"],
)
def test_matches_scikit_learn_ridge_with_intercept(n, d, seed, alpha):
    X, y = _data(seed, n, d)
    ref = Ridge(alpha=alpha).fit(X, y)  # oracle computed before the learner runs
    intercept, coef = _checked(X, y, alpha)
    np.testing.assert_allclose(
        coef,
        ref.coef_,
        rtol=1e-7,
        atol=1e-9,
        err_msg="coefficients differ from sklearn Ridge: solve (Xc^T Xc + alpha I) beta = Xc^T yc "
        "on the centred data",
    )
    np.testing.assert_allclose(
        intercept,
        ref.intercept_,
        rtol=1e-7,
        atol=1e-9,
        err_msg="intercept differs from sklearn Ridge: it must be mean(y) - mean(X) @ coef",
    )


@pytest.mark.parametrize(
    ("n", "d", "seed", "alpha"),
    [(30, 2, 5, 0.7), (60, 4, 6, 10.0)],
    ids=["two-features", "four-features"],
)
def test_matches_scikit_learn_ridge_without_intercept(n, d, seed, alpha):
    X, y = _data(seed, n, d)
    ref = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
    intercept, coef = _checked(X, y, alpha, fit_intercept=False)
    assert intercept == 0.0, "without an intercept the returned intercept must be exactly 0.0"
    np.testing.assert_allclose(
        coef,
        ref.coef_,
        rtol=1e-7,
        atol=1e-9,
        err_msg="coefficients differ from sklearn Ridge(fit_intercept=False): solve "
        "(X^T X + alpha I) beta = X^T y on the uncentred data",
    )


def test_rank_deficient_design_matches_scikit_learn_when_alpha_is_positive():
    rng = np.random.default_rng(7)
    a = rng.normal(size=30)
    b = rng.normal(size=30)
    X = np.column_stack([a, 2.0 * a, b, a + b])
    y = 0.5 + a - b + 0.1 * rng.normal(size=30)
    ref = Ridge(alpha=2.0).fit(X, y)
    intercept, coef = _checked(X, y, 2.0)
    np.testing.assert_allclose(
        coef,
        ref.coef_,
        rtol=1e-7,
        atol=1e-9,
        err_msg="a dependent design with alpha > 0 has a unique ridge solution: it must match "
        "sklearn Ridge",
    )
    np.testing.assert_allclose(
        intercept,
        ref.intercept_,
        rtol=1e-7,
        atol=1e-9,
        err_msg="intercept differs from sklearn Ridge",
    )


@pytest.mark.parametrize("seed", [8, 9, 10], ids=["first", "second", "third"])
def test_matches_the_svd_closed_form(seed):
    X, y = _data(seed, 25, 4)
    alpha = 1.3
    xc = X - X.mean(axis=0)
    yc = y - y.mean()
    U, s, Vt = np.linalg.svd(xc, full_matrices=False)
    expected_coef = Vt.T @ ((s / (s**2 + alpha)) * (U.T @ yc))
    expected_intercept = y.mean() - X.mean(axis=0) @ expected_coef
    intercept, coef = _checked(X, y, alpha)
    np.testing.assert_allclose(
        coef,
        expected_coef,
        rtol=1e-9,
        atol=1e-12,
        err_msg="coefficients differ from the SVD form sum_i s_i / (s_i^2 + alpha) (u_i . yc) v_i",
    )
    np.testing.assert_allclose(
        intercept,
        expected_intercept,
        rtol=1e-9,
        atol=1e-12,
        err_msg="intercept differs from the SVD form: mean(y) - mean(X) @ coef",
    )


def test_alpha_near_zero_recovers_least_squares_on_full_rank_data():
    X, y = _data(11, 40, 3)
    design = np.column_stack([np.ones(40), X])
    ls = np.linalg.lstsq(design, y, rcond=None)[0]
    intercept, coef = _checked(X, y, 1e-10)
    np.testing.assert_allclose(
        [intercept, *coef],
        ls,
        rtol=1e-6,
        atol=1e-8,
        err_msg="as alpha goes to 0 the ridge fit must approach ordinary least squares",
    )


def test_coefficient_norm_shrinks_monotonically_as_alpha_grows():
    X, y = _data(12, 40, 5)
    norms = []
    for alpha in np.logspace(-3, 4, 25):
        _, coef = _checked(X, y, alpha)
        norms.append(np.linalg.norm(coef))
    for smaller, larger in itertools.pairwise(norms):
        assert larger <= smaller + 1e-12, (
            "the coefficient norm must not grow as alpha grows: the penalty shrinks coefficients"
        )


def test_very_large_alpha_sends_coefficients_to_zero_and_intercept_to_the_mean():
    X, y = _data(13, 40, 3)
    intercept, coef = _checked(X, y, 1e10)
    np.testing.assert_allclose(
        coef,
        np.zeros(3),
        atol=1e-7,
        err_msg="a huge penalty must shrink every coefficient towards zero",
    )
    np.testing.assert_allclose(
        intercept,
        y.mean(),
        atol=1e-6,
        err_msg="with all slopes at zero the intercept must be the mean of y: it is not penalised",
    )


@pytest.mark.parametrize("alpha", [1.0, 1e4], ids=["moderate", "very-strong"])
def test_shifting_y_moves_only_the_intercept(alpha):
    X, y = _data(14, 35, 3)
    i0, c0 = _checked(X, y, alpha)
    i1, c1 = _checked(X, y + 3.0, alpha)
    np.testing.assert_allclose(
        c1,
        c0,
        rtol=1e-9,
        atol=1e-12,
        err_msg="shifting y must not change the coefficients: the intercept is not penalised, "
        "so it absorbs the shift",
    )
    np.testing.assert_allclose(
        i1 - i0,
        3.0,
        rtol=1e-9,
        atol=1e-9,
        err_msg="shifting y by c must move the intercept by exactly c",
    )


def test_shifting_every_feature_moves_only_the_intercept():
    X, y = _data(15, 35, 3)
    i0, c0 = _checked(X, y, 2.0)
    i1, c1 = _checked(X + 5.0, y, 2.0)
    np.testing.assert_allclose(
        c1,
        c0,
        rtol=1e-9,
        atol=1e-12,
        err_msg="shifting the features must not change the coefficients: the design is centred",
    )
    np.testing.assert_allclose(
        i1,
        i0 - 5.0 * c0.sum(),
        rtol=1e-9,
        atol=1e-9,
        err_msg="shifting every feature by c must move the intercept by -c * sum(coef)",
    )


@settings(max_examples=30, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(0, 10**6), s=st.floats(0.01, 100.0))
def test_rescaling_a_feature_with_alpha_times_square_rescales_its_coefficient(seed, s):
    X, y = _data(seed, 30, 3)
    alpha = 1.7
    i0, c0 = _checked(X, y, alpha)
    i1, c1 = _checked(s * X, y, alpha * s * s)
    np.testing.assert_allclose(
        c1 * s,
        c0,
        rtol=1e-8,
        atol=1e-10,
        err_msg="ridge(s X, y, s^2 alpha) must return coef / s: the penalty is in squared units",
    )
    np.testing.assert_allclose(
        i1, i0, rtol=1e-8, atol=1e-10, err_msg="rescaling X must not change the intercept"
    )


def test_alpha_zero_on_a_rank_deficient_design_raises_value_error():
    rng = np.random.default_rng(16)
    a = rng.normal(size=20)
    X = np.column_stack([a, 3.0 * a])
    y = rng.normal(size=20)
    _expect_value_error(
        lambda: ridge_fit(X, y, 0.0),
        "a rank-deficient design with alpha = 0 was accepted: there is no unique least-squares "
        "solution, so raise ValueError",
    )


def test_constant_column_with_intercept_at_alpha_zero_raises_value_error():
    rng = np.random.default_rng(17)
    X = np.column_stack([np.full(20, 4.0), rng.normal(size=20)])
    y = rng.normal(size=20)
    _expect_value_error(
        lambda: ridge_fit(X, y, 0.0, fit_intercept=True),
        "a constant column was accepted at alpha = 0 with an intercept: centring makes it zero, so "
        "the centred design is rank deficient",
    )


def test_constant_column_without_intercept_at_alpha_zero_is_least_squares():
    rng = np.random.default_rng(18)
    X = np.column_stack([np.ones(20), rng.normal(size=20)])
    y = 0.5 + 2.0 * X[:, 1] + 0.1 * rng.normal(size=20)
    _, coef = _checked(X, y, 0.0, fit_intercept=False)
    np.testing.assert_allclose(
        coef,
        np.linalg.lstsq(X, y, rcond=None)[0],
        rtol=1e-8,
        err_msg="without an intercept a constant column is a valid full-rank feature",
    )


@pytest.mark.parametrize("alpha", [-1.0, np.nan, np.inf], ids=["negative", "nan", "infinite"])
def test_invalid_alpha_raises_value_error(alpha):
    X, y = _data(19, 10, 2)
    _expect_value_error(
        lambda: ridge_fit(X, y, alpha),
        "an alpha that is not a finite number >= 0 was accepted: raise ValueError",
    )


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
        (np.random.default_rng(20).normal(size=(5, 2)), np.ones(4)),
        (np.random.default_rng(21).normal(size=(5, 2)), np.ones((5, 1))),
    ],
    ids=["1-D-X", "length-mismatch", "2-D-y"],
)
def test_invalid_shapes_raise_value_error(X, y):
    _expect_value_error(
        lambda: ridge_fit(X, y, 1.0),
        "an invalid shape was accepted: X must be 2-D and y must be 1-D with one entry per row",
    )


@pytest.mark.parametrize("which", ["X", "y"], ids=["nan-in-X", "inf-in-y"])
def test_non_finite_data_raises_value_error(which):
    X, y = _data(22, 12, 2)
    if which == "X":
        X = X.copy()
        X[2, 0] = np.nan
    else:
        y = y.copy()
        y[0] = np.inf
    _expect_value_error(
        lambda: ridge_fit(X, y, 1.0),
        "non-finite data was accepted: reject NaN and infinity before solving",
    )


def test_inputs_are_not_mutated():
    X, y = _data(23, 25, 3)
    X_before, y_before = X.copy(), y.copy()
    _call(X, y, 0.9)
    np.testing.assert_array_equal(X, X_before, err_msg="X was modified in place: centre a copy")
    np.testing.assert_array_equal(y, y_before, err_msg="y was modified in place: centre a copy")


def test_integer_inputs_give_the_same_fit_as_floats():
    X = np.array([[1, 0], [0, 1], [2, 1], [1, 3], [3, 2]])
    y = np.array([2, 1, 4, 5, 6])
    i_int, c_int = _checked(X, y, 1.5)
    i_flt, c_flt = _checked(X.astype(float), y.astype(float), 1.5)
    np.testing.assert_allclose(
        [i_int, *c_int],
        [i_flt, *c_flt],
        rtol=1e-12,
        err_msg="integer inputs gave a different fit from the same values as floats: "
        "convert to float64 first",
    )


@pytest.mark.parametrize(
    ("X", "y"),
    [(np.empty((0, 2)), []), (np.empty((3, 0)), np.ones(3))],
    ids=["no-samples", "no-features"],
)
def test_empty_design_dimensions_raise_value_error(X, y):
    _expect_value_error(
        lambda: ridge_fit(X, y, 1.0),
        "an empty design dimension was accepted: require at least one sample and one feature",
    )


def test_single_sample_with_positive_penalty_has_zero_slopes_and_target_intercept():
    intercept, coef = _checked([[2.0, -3.0]], [4.0], 1.0)
    assert intercept == 4.0, "a one-sample centered fit must retain the target as intercept"
    np.testing.assert_array_equal(
        coef,
        [0.0, 0.0],
        err_msg="a one-sample centered penalised design has zero slopes",
    )


def test_alpha_zero_gives_exact_least_squares_on_a_nearly_collinear_design():
    n = 20
    x1 = np.arange(n) / 10.0
    x2 = x1 + 0.01 * np.sin(np.arange(n))
    X = np.column_stack([x1, x2])
    y = 1.0 + 2.0 * x1 + 0.5 * x2 + 0.1 * np.cos(3 * np.arange(n))
    # Exact least squares through the centred design, from the SVD (no normal equations).
    xc = X - X.mean(axis=0)
    yc = y - y.mean()
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    beta = vt.T @ ((u.T @ yc) / s)
    intercept_expected = y.mean() - X.mean(axis=0) @ beta
    intercept, coef = _checked(X, y, 0.0)
    np.testing.assert_allclose(
        coef,
        beta,
        rtol=1e-9,
        atol=1e-9,
        err_msg="alpha zero must give exact least squares, not a small penalty in its place",
    )
    np.testing.assert_allclose(
        intercept,
        intercept_expected,
        rtol=1e-9,
        atol=1e-9,
        err_msg="alpha zero must give exact least squares, not a small penalty in its place",
    )


def test_alpha_zero_with_intercept_checks_the_rank_after_centring():
    a = np.array([0.1, 0.2, 0.3, 0.4])
    X = np.column_stack([a, a + 0.7])
    y = np.array([1.0, 2.0, 2.5, 4.0])
    _expect_value_error(
        lambda: ridge_fit(X, y, 0.0, fit_intercept=True),
        "alpha zero with an intercept accepted a design whose columns are dependent once their "
        "means are removed",
    )
