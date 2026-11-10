import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize
from scipy.special import expit
from submission import fit_logistic_regression

from dogma_data import load, rng


def _call(X, y, **kwargs):
    """Fit with NumPy warnings turned into errors. Hidden messages never show inputs or values."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = fit_logistic_regression(X, y, **kwargs)
            if not isinstance(result, tuple) or len(result) != 2:
                pytest.fail("return a two-item tuple containing the intercept and coefficients")
            return result
    except RuntimeWarning:
        pytest.fail(
            "the fit emitted a NumPy warning: avoid exp of a large positive number and log of zero"
        )
    except RuntimeError:
        pytest.fail(
            "the fit did not converge on a well-posed problem: check the gradient "
            "scale (the loss is a mean, so divide by n), the step direction and the stopping rule"
        )
    except Exception as exc:
        pytest.fail(f"the fit must accept valid inputs; the call raised {type(exc).__name__}")


def _expect_value_error(X, y, **kwargs):
    try:
        fit_logistic_regression(X, y, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"invalid input must raise ValueError; the function raised {type(exc).__name__}"
        )
    pytest.fail("invalid input must raise ValueError, not return a fit")


def _random_problem(seed, n_min=20, n_max=120, d_max=4):
    """A random problem with both classes present. The labels follow a noisy linear rule."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(n_min, n_max))
    d = int(rng.integers(1, d_max + 1))
    X = rng.normal(size=(n, d))
    beta = rng.normal(size=d)
    y = (rng.uniform(size=n) < expit(0.5 + X @ beta)).astype(float)
    y[0], y[1] = 0.0, 1.0
    return X, y


def _objective_and_gradient(theta, X, y, l2):
    """Independent copy of the objective, for the L-BFGS-B oracle. It uses np.logaddexp."""
    b, w = theta[0], theta[1:]
    eta = b + X @ w
    loss = np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.5 * l2 * np.dot(w, w)
    p = expit(eta)
    grad = np.empty_like(theta)
    grad[0] = np.mean(p - y)
    grad[1:] = X.T @ (p - y) / len(y) + l2 * w
    return loss, grad


def _oracle(X, y, l2):
    """L-BFGS-B to a tight tolerance on the same objective: an independent optimiser."""
    x0 = np.zeros(X.shape[1] + 1)
    res = minimize(
        _objective_and_gradient,
        x0,
        args=(X, y, l2),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-12},
    )
    return res.x


def _gradient_max(intercept, coef, X, y, l2):
    _, grad = _objective_and_gradient(np.concatenate([[intercept], coef]), X, y, l2)
    return float(np.max(np.abs(grad)))


@pytest.fixture(scope="module")
def raw_breast_cancer():
    """Real features on raw scales (means from tiny fractions to hundreds), nearly separable."""
    data = load("breast_cancer")
    return np.array(data.X, dtype=np.float64), np.array(data.y, dtype=np.float64)


def test_matches_an_independent_optimiser_on_raw_breast_cancer_scales(raw_breast_cancer):
    X, y = raw_breast_cancer
    want = _oracle(X, y, 1.0)
    b, w = _call(X, y, l2=1.0)
    np.testing.assert_allclose(
        np.concatenate([[b], w]),
        want,
        rtol=1e-4,
        atol=1e-4,
        err_msg="coefficients differ from the minimiser on real features with very different "
        "scales: the penalty and the gradient act on the raw coefficients, so do not standardise "
        "the objective; the oracle itself is only accurate to about 1e-4 on these features",
    )
    assert _gradient_max(b, w, X, y, 1.0) <= 1e-7, (
        "the returned point is not stationary on real features with very different scales: "
        "the stopping rule must test the gradient of the raw problem"
    )


@pytest.mark.parametrize(
    ("draw", "quantile"),
    [(0, 0.5), (2, 0.99)],
    ids=["balanced-separable", "rare-separable"],
)
def test_separable_real_features_with_a_weak_penalty_fit_without_overflow(
    raw_breast_cancer, draw, quantile
):
    # Labels are a linear rule on the real features: perfectly separable, with linear scores in
    # the hundreds along the way. The rare variant has about one positive in a hundred.
    X, _ = raw_breast_cancer
    key = f"cml/logistic-regression/p01-l2-logistic-fit:separable:{draw}"
    beta = rng(key).normal(size=X.shape[1])
    eta = X @ beta
    y = (eta > np.quantile(eta, quantile)).astype(float)
    b, w = _call(X, y, l2=1e-3)
    assert np.isfinite(b) and np.all(np.isfinite(w)), (
        "coefficients must stay finite on separable data under a positive penalty"
    )
    assert _gradient_max(b, w, X, y, 1e-3) <= 1e-7, (
        "the returned point is not stationary on separable real data: Newton steps can overshoot "
        "when the linear scores are large, so damp them and evaluate the loss without overflow"
    )


def test_nearly_separable_real_data_with_a_weak_penalty_is_finite_and_stationary(
    raw_breast_cancer,
):
    X, y = raw_breast_cancer
    b, w = _call(X, y, l2=0.01)
    assert np.isfinite(b) and np.all(np.isfinite(w)), (
        "coefficients must stay finite on nearly separable real data under a positive penalty"
    )
    assert np.max(np.abs(w)) <= 10.0, (
        "coefficients blow up on nearly separable real data: the penalty must bound them"
    )
    assert _gradient_max(b, w, X, y, 0.01) <= 1e-7, (
        "the returned point is not stationary on nearly separable real data"
    )


def test_matches_an_independent_optimiser_on_random_problems():
    for seed in range(8):
        X, y = _random_problem(seed)
        want = _oracle(X, y, 1.0)
        b, w = _call(X, y, l2=1.0)
        np.testing.assert_allclose(
            np.concatenate([[b], w]),
            want,
            rtol=1e-6,
            atol=1e-6,
            err_msg="coefficients differ from the minimiser of mean log-loss + (l2 / 2) * ||w||^2: "
            "check that the loss is averaged over samples and that the penalty is (l2 / 2) ||w||^2",
        )


def test_gradient_vanishes_at_the_returned_point_on_random_problems():
    for seed in range(8):
        X, y = _random_problem(100 + seed)
        b, w = _call(X, y, l2=0.5)
        assert _gradient_max(b, w, X, y, 0.5) <= 1e-7, (
            "the returned point is not stationary: the gradient of the penalised loss is not zero "
            "at the solution, so the iteration stopped early or the gradient is wrong"
        )


def test_intercept_is_not_penalised_when_features_are_all_zero():
    X = np.zeros((10, 2))
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    rate = y.mean()
    for l2 in (1.0, 10.0):
        b, w = _call(X, y, l2=l2)
        np.testing.assert_allclose(
            b,
            np.log(rate / (1.0 - rate)),
            atol=1e-8,
            err_msg="the intercept is shrunk towards zero: leave the intercept out of the penalty",
        )
        np.testing.assert_allclose(
            w, 0.0, atol=1e-10, err_msg="all-zero features must give all-zero coefficients"
        )


def test_mean_predicted_probability_equals_the_positive_rate():
    for seed in range(6):
        X, y = _random_problem(200 + seed)
        b, w = _call(X, y, l2=2.0)
        p = expit(b + X @ w)
        assert abs(np.mean(p) - np.mean(y)) <= 1e-7, (
            "mean predicted probability differs from the positive rate: the intercept gradient "
            "vanishes only when the intercept is unpenalised"
        )


def test_nearly_separable_data_keeps_the_penalised_coefficients_finite_and_moderate():
    X = np.array([[-3.0], [-2.0], [-1.0], [-0.4], [0.2], [0.1], [0.4], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    want = _oracle(X, y, 1.0)
    b, w = _call(X, y, l2=1.0)
    assert np.isfinite(b) and np.all(np.isfinite(w)), (
        "coefficients must stay finite under a penalty"
    )
    assert np.max(np.abs(np.concatenate([[b], w]))) <= 10.0, (
        "coefficients blow up on nearly separable data: the penalty must bound them"
    )
    np.testing.assert_allclose(
        np.concatenate([[b], w]),
        want,
        rtol=1e-6,
        atol=1e-6,
        err_msg="nearly separable fit disagrees with the penalised optimum",
    )


def test_perfectly_separated_data_with_a_penalty_stays_finite():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1], dtype=float)
    b, w = _call(X, y, l2=1.0)
    assert np.isfinite(b) and np.all(np.isfinite(w)), (
        "perfectly separated data must give finite coefficients when the penalty is positive"
    )
    assert _gradient_max(b, w, X, y, 1.0) <= 1e-7, "the returned point is not stationary"


def test_zero_penalty_on_non_separable_data_matches_the_maximum_likelihood_fit():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 2))
    y = (rng.uniform(size=80) < expit(0.3 + X @ np.array([1.0, -0.5]))).astype(float)
    y[0], y[1] = 0.0, 1.0
    want = _oracle(X, y, 0.0)
    b, w = _call(X, y, l2=0.0)
    np.testing.assert_allclose(
        np.concatenate([[b], w]),
        want,
        rtol=1e-6,
        atol=1e-6,
        err_msg="with l2 = 0 the fit must equal the maximum-likelihood estimate",
    )


def test_identical_columns_share_their_coefficient_equally():
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    X = np.column_stack([x, x])
    y = (rng.uniform(size=40) < expit(x)).astype(float)
    y[0], y[1] = 0.0, 1.0
    _, w = _call(X, y, l2=1.0)
    assert abs(w[0] - w[1]) <= 1e-8, (
        "identical columns must share the coefficient equally under an L2 penalty: the penalised "
        "optimum is unique, so a solver that splits them unevenly is wrong"
    )


def test_too_small_iteration_budget_raises_runtime_error():
    X, y = _random_problem(5)
    try:
        fit_logistic_regression(X, y, l2=1.0, max_iter=1)
    except RuntimeError:
        return
    except Exception as exc:
        pytest.fail(f"an exhausted budget must raise RuntimeError; raised {type(exc).__name__}")
    pytest.fail("an iteration budget that cannot reach the tolerance must raise RuntimeError")


INVALID_CASES = [
    ("negative-l2", np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), {"l2": -1.0}),
    ("nan-l2", np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), {"l2": float("nan")}),
    ("infinite-l2", np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), {"l2": float("inf")}),
    ("label-two", np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]), {}),
    ("signed-labels", np.array([[0.0], [1.0]]), np.array([-1.0, 1.0]), {}),
    ("one-class", np.array([[0.0], [1.0]]), np.array([1.0, 1.0]), {}),
    ("one-dimensional-x", np.array([0.0, 1.0]), np.array([0.0, 1.0]), {}),
    ("length-mismatch", np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0]), {}),
    ("nan-in-x", np.array([[0.0], [np.nan]]), np.array([0.0, 1.0]), {}),
    ("no-rows", np.zeros((0, 1)), np.zeros(0), {}),
    ("zero-iterations", np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), {"max_iter": 0}),
    ("zero-tolerance", np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), {"tol": 0.0}),
]


@pytest.mark.parametrize(
    ("X", "y", "kwargs"),
    [case[1:] for case in INVALID_CASES],
    ids=[case[0] for case in INVALID_CASES],
)
def test_invalid_inputs_raise_value_error(X, y, kwargs):
    _expect_value_error(X, y, **kwargs)


def test_integer_and_boolean_inputs_are_accepted():
    X = np.array([[0, 1], [1, 0], [2, 2], [3, 1], [0, 3]])
    y = np.array([False, False, True, True, True])
    b, w = _call(X, y, l2=1.0)
    assert isinstance(b, float), "the intercept must be a Python float"
    assert w.dtype == np.float64, "coef must be a float64 array"
    assert w.shape == (2,), "shape mismatch: coef needs one entry per feature"


def test_inputs_are_not_mutated():
    X, y = _random_problem(9)
    X_before, y_before = X.copy(), y.copy()
    _call(X, y, l2=1.0)
    np.testing.assert_array_equal(X, X_before, err_msg="X was modified in place")
    np.testing.assert_array_equal(y, y_before, err_msg="y was modified in place")


@settings(max_examples=30, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000))
def test_gradient_condition_holds_on_small_random_problems(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 15))
    X = rng.normal(size=(n, 2))
    y = (rng.uniform(size=n) < 0.5).astype(float)
    y[0], y[1] = 0.0, 1.0
    b, w = _call(X, y, l2=1.0)
    assert _gradient_max(b, w, X, y, 1.0) <= 1e-7, (
        "the returned point is not stationary on a small problem: check the gradient "
        "(1/n) X^T (p - y) + l2 * w and the intercept gradient mean(p - y)"
    )


@pytest.mark.parametrize("tol", [1e-8, 1e-11], ids=["ordinary", "strict"])
def test_returned_gradient_meets_the_requested_tolerance(tol):
    X, y = _random_problem(17)
    b, w = _call(X, y, l2=0.3, tol=tol)
    assert _gradient_max(b, w, X, y, 0.3) <= tol, (
        "the returned gradient must meet the caller's tolerance rather than a fixed looser cutoff"
    )


def test_stationarity_is_confirmed_by_finite_differences_of_the_objective():
    X, y = _random_problem(19, n_min=20, n_max=30, d_max=2)
    b, w = _call(X, y, l2=0.4)
    theta = np.r_[b, w]
    h = 1e-5
    numeric = np.empty_like(theta)
    for j in range(theta.size):
        step = np.zeros_like(theta)
        step[j] = h
        up = _objective_and_gradient(theta + step, X, y, 0.4)[0]
        down = _objective_and_gradient(theta - step, X, y, 0.4)[0]
        numeric[j] = (up - down) / (2 * h)
    np.testing.assert_allclose(
        numeric,
        0.0,
        atol=1e-8,
        rtol=0.0,
        err_msg="the returned fit is not stationary under independent objective finite differences",
    )


def test_column_vector_labels_raise_value_error():
    # The labels must be one-dimensional, one per row; a column of labels is a shape mistake.
    X = np.array([[0.5, 1.0], [1.5, 0.2], [0.3, 0.9]])
    y = np.array([[1.0], [0.0], [1.0]])
    try:
        fit_logistic_regression(X, y, l2=1.0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"two-dimensional labels must raise ValueError; the call raised {type(exc).__name__}"
        )
    pytest.fail("two-dimensional labels were accepted: y must be a one-dimensional array of labels")


def test_design_matrix_without_feature_columns_raises_value_error():
    # The statement requires at least one feature column, so a design with none is invalid input.
    X = np.zeros((3, 0))
    y = np.array([0.0, 1.0, 0.0])
    try:
        fit_logistic_regression(X, y, l2=1.0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"a design matrix with no feature columns must raise ValueError; raised "
            f"{type(exc).__name__}"
        )
    pytest.fail(
        "a design matrix with no feature columns was accepted: it needs at least one feature"
    )


def test_exhausted_budget_raises_even_when_one_coordinate_has_zero_gradient():
    # The second column is all zeros, so its gradient is exactly zero throughout. The fit has
    # not converged on the first coordinates, so the budget running out must still raise.
    X = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0], [-0.5, 0.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    try:
        fit_logistic_regression(X, y, l2=1.0, max_iter=1)
    except RuntimeError:
        return
    except Exception as exc:
        pytest.fail(f"an exhausted budget must raise RuntimeError; raised {type(exc).__name__}")
    pytest.fail(
        "a fit that uses up max_iter before the gradient tolerance is met must raise RuntimeError,"
        " not return a point whose gradient is still large in some coordinate"
    )


def test_exhausted_budget_raises_when_all_gradient_entries_are_negative():
    # Mostly positive labels and positive features: the gradient is negative and far from zero
    # after one iteration, so the budget must run out with RuntimeError.
    X = np.array([[0.5], [1.0], [1.5], [2.0], [0.2], [0.8], [1.2], [0.3]])
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    try:
        fit_logistic_regression(X, y, l2=1.0, max_iter=1)
    except RuntimeError:
        return
    except Exception as exc:
        pytest.fail(f"an exhausted budget must raise RuntimeError; raised {type(exc).__name__}")
    pytest.fail(
        "a fit that uses up max_iter before the gradient tolerance is met must raise RuntimeError,"
        " not return a point whose gradient is large in absolute value"
    )
