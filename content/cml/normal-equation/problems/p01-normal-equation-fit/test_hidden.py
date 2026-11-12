import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.linear_model import LinearRegression
from submission import fit_linear_regression

from dogma_data import load, rng, sample_rows


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


def _data(seed, n, d):
    """Semi-synthetic: real standardised diabetes columns, a known linear signal and mild noise.

    The columns are correlated real measurements. A draw whose design ``[1 | X]`` is not of full
    rank (a handful of patients can share one sex) is skipped: these tests are about designs
    the contract accepts.
    """
    gen = rng(f"cml/normal-equation/p01-normal-equation-fit:data:{seed}")
    while True:
        X, _ = sample_rows("diabetes", gen, n, d, standardize=True)
        if np.linalg.matrix_rank(np.column_stack([np.ones(n), X])) == d + 1:
            break
    beta = _real_normal(gen, d)
    y = 1.5 + X @ beta + 0.3 * _real_normal(gen, n)
    return X, y


@pytest.fixture(scope="module")
def raw_diabetes():
    data = load("diabetes")  # raw: BMI near 26, serum levels in the hundreds, binary sex coded 1/2
    return np.array(data.X, dtype=np.float64), np.array(data.y, dtype=np.float64)


@pytest.fixture(scope="module")
def stackloss():
    data = load("stackloss")  # 21 plant days, one observation with a large studentized residual
    return np.array(data.X, dtype=np.float64), np.array(data.y, dtype=np.float64)


@pytest.fixture(scope="module")
def longley():
    data = load("longley")  # macroeconomic series in very different units, condition about 5e9
    return np.array(data.X, dtype=np.float64), np.array(data.y, dtype=np.float64)


def _fit(X, y, fit_intercept=True):
    """Call the learner's function and check the return types, naming the mistake if one is off."""
    try:
        intercept, coef = fit_linear_regression(X, y, fit_intercept=fit_intercept)
    except ValueError as exc:
        pytest.fail(
            f"a well-posed fit raised ValueError({exc}): check the normal equations and shapes"
        )
    if not isinstance(intercept, float):
        pytest.fail("intercept must be a Python float, not an array: return float(beta[0])")
    if np.shape(coef) != (np.shape(X)[1],):
        pytest.fail("coef must be a 1-D array with one entry per feature, not a column vector")
    return intercept, coef


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("n", "d", "seed"),
    [(30, 1, 0), (50, 3, 1), (80, 5, 2), (40, 2, 3)],
    ids=list("abcd"),
)
def test_matches_scikit_learn_with_intercept(n, d, seed):
    X, y = _data(seed, n, d)
    intercept, coef = _fit(X, y)
    ref = LinearRegression(fit_intercept=True).fit(X, y)
    np.testing.assert_allclose(
        intercept,
        ref.intercept_,
        rtol=1e-8,
        atol=1e-10,
        err_msg="intercept differs from LinearRegression: check the column of ones",
    )
    np.testing.assert_allclose(
        coef,
        ref.coef_,
        rtol=1e-8,
        atol=1e-10,
        err_msg="coefficients differ from LinearRegression: check the normal equations",
    )


@pytest.mark.parametrize(("n", "d", "seed"), [(30, 2, 4), (60, 4, 5)], ids=list("ab"))
def test_matches_scikit_learn_without_intercept(n, d, seed):
    X, y = _data(seed, n, d)
    intercept, coef = _fit(X, y, fit_intercept=False)
    ref = LinearRegression(fit_intercept=False).fit(X, y)
    assert intercept == 0.0, "without an intercept the returned intercept must be 0.0"
    np.testing.assert_allclose(
        coef,
        ref.coef_,
        rtol=1e-8,
        atol=1e-10,
        err_msg="coefficients differ from LinearRegression(fit_intercept=False)",
    )


@pytest.mark.parametrize("seed", [6, 7, 8], ids=list("abc"))
def test_matches_numpy_least_squares_on_the_augmented_design(seed):
    X, y = _data(seed, 25, 3)
    design = np.hstack([np.ones((25, 1)), X])
    expected = np.linalg.lstsq(design, y, rcond=None)[0]
    intercept, coef = _fit(X, y)
    np.testing.assert_allclose(
        [intercept, *coef],
        expected,
        rtol=1e-8,
        atol=1e-10,
        err_msg="fit differs from the least-squares solution of [1 | X] beta = y",
    )


@pytest.mark.parametrize("seed", [9, 10], ids=list("ab"))
def test_residuals_are_orthogonal_to_every_design_column(seed):
    X, y = _data(seed, 40, 3)
    intercept, coef = _fit(X, y)
    residual = y - intercept - X @ coef
    np.testing.assert_allclose(
        X.T @ residual,
        np.zeros(3),
        atol=1e-8,
        err_msg="residuals must be orthogonal to each feature column (normal equations)",
    )
    np.testing.assert_allclose(
        residual.sum(),
        0.0,
        atol=1e-8,
        err_msg="residuals must sum to zero when an intercept is fitted",
    )


def test_squared_error_gradient_vanishes_at_the_returned_fit():
    X, y = _data(11, 30, 2)
    intercept, coef = _fit(X, y)
    params = np.array([intercept, *coef])

    def loss(p):
        return 0.5 * np.sum((y - p[0] - X @ p[1:]) ** 2)

    h = 1e-6
    numeric = np.array([(loss(params + h * e) - loss(params - h * e)) / (2 * h) for e in np.eye(3)])
    np.testing.assert_allclose(
        numeric,
        np.zeros(3),
        atol=1e-6,
        err_msg="the squared-error gradient is not zero at the fit: it is not a minimiser",
    )


def test_matches_least_squares_on_raw_diabetes_scales(raw_diabetes):
    X, y = raw_diabetes
    intercept, coef = _fit(X, y)
    expected = np.linalg.lstsq(np.hstack([np.ones((len(y), 1)), X]), y, rcond=None)[0]
    np.testing.assert_allclose(
        [intercept, *coef],
        expected,
        rtol=1e-7,
        atol=1e-9,
        err_msg="coefficients differ from least squares on real features with very different "
        "scales: solve the normal equations on the design [1 | X], not on centred or rescaled "
        "copies that drop the intercept",
    )


def test_matches_least_squares_on_stackloss(stackloss):
    X, y = stackloss
    intercept, coef = _fit(X, y)
    expected = np.linalg.lstsq(np.hstack([np.ones((len(y), 1)), X]), y, rcond=None)[0]
    np.testing.assert_allclose(
        [intercept, *coef],
        expected,
        rtol=1e-7,
        atol=1e-9,
        err_msg="coefficients differ from least squares on a small real design: check the "
        "intercept column and the normal equations",
    )


def test_full_rank_real_design_with_large_condition_is_fitted(longley):
    # The design is full rank by NumPy's default rule, so it must be fitted. Its condition number
    # is about 5e9, so only fitted values are compared: the coefficients carry about 8 digits.
    X, y = longley
    intercept, coef = _fit(X, y)
    design = np.hstack([np.ones((len(y), 1)), X])
    fitted_ls = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.array([intercept, *coef])
    np.testing.assert_allclose(
        fitted,
        fitted_ls,
        rtol=1e-7,
        atol=1e-7 * float(np.max(np.abs(fitted_ls))),
        err_msg="a full-rank real design was rejected or fitted inaccurately: the rank rule is "
        "NumPy's matrix_rank with its default tolerance, not a limit on the condition number",
    )


def test_noise_free_line_is_recovered_exactly():
    X = np.column_stack([np.linspace(-2, 2, 9), np.linspace(0, 1, 9) ** 2])
    y = 2.0 + X @ np.array([1.5, -3.0])
    intercept, coef = _fit(X, y)
    np.testing.assert_allclose(
        intercept, 2.0, atol=1e-9, err_msg="noise-free line: the intercept must be recovered"
    )
    np.testing.assert_allclose(
        coef, [1.5, -3.0], atol=1e-9, err_msg="noise-free line: the slopes must be recovered"
    )


@pytest.mark.parametrize(
    "make_design",
    [
        lambda x, z: np.column_stack([x, 3.0 * x]),
        lambda x, z: np.column_stack([x, z, x + z]),
        lambda x, z: np.column_stack([x, np.zeros_like(x)]),
        lambda x, z: np.column_stack([x, np.ones_like(x)]),
        lambda x, z: np.column_stack([x, z, x - 2.0 * z, z]),
    ],
    ids=["scaled-duplicate", "column-sum", "zero-column", "constant-column", "dependent-four"],
)
def test_rank_deficient_design_raises_value_error(make_design):
    rng = np.random.default_rng(12)
    x = _real_normal(rng, 12)
    z = _real_normal(rng, 12)
    X = make_design(x, z)
    y = _real_normal(rng, 12)
    _expect_value_error(
        lambda: fit_linear_regression(X, y),
        "a rank-deficient design was accepted: check the rank of [1 | X] before solving, "
        "since a plain solve can return huge meaningless coefficients",
    )


def test_more_features_than_rows_raises_value_error():
    rng = np.random.default_rng(13)
    _expect_value_error(
        lambda: fit_linear_regression(_real_normal(rng, (3, 4)), _real_normal(rng, 3)),
        "fewer rows than columns was accepted: an underdetermined design has no unique fit",
    )


def test_constant_column_is_fine_without_an_intercept():
    rng = np.random.default_rng(14)
    x = _real_normal(rng, 20)
    X = np.column_stack([np.ones(20), x])
    y = 0.5 + 2.0 * x + 0.1 * _real_normal(rng, 20)
    intercept, coef = _fit(X, y, fit_intercept=False)
    expected = np.linalg.lstsq(X, y, rcond=None)[0]
    assert intercept == 0.0, "a constant feature without fit_intercept must return intercept 0.0"
    np.testing.assert_allclose(
        coef,
        expected,
        rtol=1e-8,
        err_msg="a constant feature is valid when no separate intercept is fitted",
    )


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
        (_real_normal(np.random.default_rng(21), (4, 2, 2)), np.ones(4)),
        (_real_normal(np.random.default_rng(22), (5, 2)), np.ones(4)),
        (_real_normal(np.random.default_rng(23), (5, 2)), np.ones((5, 1))),
    ],
    ids=["one-dimensional-x", "three-dimensional-x", "length-mismatch", "two-dimensional-y"],
)
def test_invalid_shapes_raise_value_error(X, y):
    _expect_value_error(
        lambda: fit_linear_regression(X, y),
        "an invalid shape was accepted: X must be 2-D and y must be 1-D with one entry per row",
    )


def test_full_rank_ill_conditioned_design_is_not_rejected():
    # The rank rule applies to the design [1 | X]. Here X^T X is numerically singular (rank 2 by
    # matrix_rank), but the design itself has rank 3, so the fit must not be rejected.
    t = np.linspace(0.0, 1.0, 20)
    X = np.column_stack([t, t + 1e-8 * np.sin(7 * t)])
    y = 1.0 + t
    try:
        fit_linear_regression(X, y)
    except ValueError:
        pytest.fail(
            "a full-rank design with a large condition number was rejected: check the rank of "
            "[1 | X], not of X^T X, which squares the condition number"
        )


def test_non_finite_values_raise_value_error():
    X, y = _data(15, 10, 2)
    X_nan = X.copy()
    X_nan[3, 1] = np.nan
    y_inf = y.copy()
    y_inf[0] = np.inf
    _expect_value_error(
        lambda: fit_linear_regression(X_nan, y),
        "a NaN in X was accepted: reject non-finite values before solving",
    )
    _expect_value_error(
        lambda: fit_linear_regression(X, y_inf),
        "an infinite target was accepted: reject non-finite values before solving",
    )


def test_inputs_are_not_mutated():
    X, y = _data(16, 15, 3)
    X_copy, y_copy = X.copy(), y.copy()
    fit_linear_regression(X, y)
    np.testing.assert_array_equal(X, X_copy, err_msg="X was modified in place")
    np.testing.assert_array_equal(y, y_copy, err_msg="y was modified in place")


def test_integer_inputs_give_float_results():
    X = np.array([[1, 0], [0, 1], [2, 1], [1, 3], [3, 2]])
    y = np.array([2, 1, 4, 5, 6])
    intercept, coef = _fit(X, y)
    assert isinstance(intercept, float), "intercept must be a Python float"
    assert coef.dtype == np.float64, "coefficients must be float64"
    f_intercept, f_coef = _fit(X.astype(float), y.astype(float))
    np.testing.assert_allclose(
        [intercept, *coef],
        [f_intercept, *f_coef],
        rtol=1e-12,
        err_msg="integer inputs gave different results from the same values as floats",
    )


@pytest.mark.parametrize(
    "c", [-2.0, 0.5, 4.0], ids=["negative-factor", "fractional-factor", "large-factor"]
)
def test_scaling_y_scales_intercept_and_coefficients(c):
    X, y = _data(17, 25, 2)
    b0, b = _fit(X, y)
    c0, cb = _fit(X, c * y)
    np.testing.assert_allclose(
        [c0, *cb],
        c * np.array([b0, *b]),
        rtol=1e-9,
        atol=1e-12,
        err_msg="scaling y must scale the intercept and every coefficient by the same factor",
    )


def test_rescaling_a_feature_rescales_only_its_coefficient():
    X, y = _data(18, 25, 3)
    b0, b = _fit(X, y)
    X2 = X.copy()
    X2[:, 1] *= 4.0
    c0, cb = _fit(X2, y)
    np.testing.assert_allclose(
        c0, b0, rtol=1e-9, err_msg="rescaling a feature must not change the intercept"
    )
    np.testing.assert_allclose(
        cb[1],
        b[1] / 4.0,
        rtol=1e-9,
        err_msg="rescaling a feature must divide its own coefficient by the factor",
    )
    np.testing.assert_allclose(
        cb[[0, 2]],
        b[[0, 2]],
        rtol=1e-9,
        err_msg="rescaling a feature must not change the other coefficients",
    )


def test_shifting_a_feature_moves_only_the_intercept():
    X, y = _data(19, 25, 2)
    b0, b = _fit(X, y)
    shift = 3.0
    X2 = X.copy()
    X2[:, 0] += shift
    c0, cb = _fit(X2, y)
    np.testing.assert_allclose(cb, b, rtol=1e-9, err_msg="a shifted feature keeps its slope")
    np.testing.assert_allclose(
        c0,
        b0 - shift * b[0],
        rtol=1e-9,
        err_msg="a shift of the feature must move the intercept by -shift * slope",
    )


@pytest.mark.parametrize("scale", [1e8, 1e-8], ids=["huge-scale", "tiny-scale"])
def test_very_large_and_small_feature_scales_are_handled(scale):
    X, y = _data(20, 30, 2)
    b0, b = _fit(X, y)
    c0, cb = _fit(X * scale, y)
    np.testing.assert_allclose(
        c0, b0, rtol=1e-6, atol=1e-9, err_msg="scaling every feature must not change the intercept"
    )
    np.testing.assert_allclose(
        cb,
        b / scale,
        rtol=1e-6,
        err_msg="scaling every feature by s should divide every coefficient by s",
    )


@settings(max_examples=30, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(0, 10**6), n=st.integers(5, 25), d=st.integers(1, 3))
def test_random_fits_agree_with_numpy_least_squares(seed, n, d):
    X, y = _data(seed, n, d)
    design = np.hstack([np.ones((n, 1)), X])
    expected = np.linalg.lstsq(design, y, rcond=None)[0]
    intercept, coef = _fit(X, y)
    np.testing.assert_allclose(
        [intercept, *coef],
        expected,
        rtol=1e-7,
        atol=1e-9,
        err_msg="random fit differs from the least-squares solution",
    )


def test_nearly_collinear_full_rank_design_keeps_the_least_squares_coefficients():
    # y = 1 + t is fitted exactly by intercept 1, slope 1 and weight 0 on the wiggle column.
    # The design has full rank, but its columns are nearly parallel, so an accurate solve
    # recovers the coefficients to well inside the tolerance.
    t = np.linspace(0.0, 1.0, 20)
    X = np.column_stack([t, t + 1e-3 * np.sin(7 * t)])
    y = 1.0 + t
    intercept, coef = _fit(X, y)
    np.testing.assert_allclose(
        [intercept, *coef],
        [1.0, 1.0, 0.0],
        atol=1e-7,
        err_msg="nearly collinear full-rank design: coefficients are off the exact least-squares "
        "fit; a ridge term, jitter or truncated eigen-solve biases the nearly flat direction",
    )


def test_one_dimensional_x_without_intercept_raises_value_error():
    _expect_value_error(
        lambda: fit_linear_regression(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), fit_intercept=False
        ),
        "a one-dimensional X was accepted when no intercept is fitted: X must be 2-D (n, d)",
    )
