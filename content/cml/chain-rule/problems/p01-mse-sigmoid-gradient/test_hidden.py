import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.special import expit
from submission import mse_sigmoid_grad

ELEMENTS = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def _loss(w, X, y):
    return float(np.mean((expit(X @ w) - y) ** 2))


def _finite_difference_gradient(w, X, y, h=1e-6):
    """Independent reference: central differences of the loss, one weight at a time."""
    grad = np.zeros_like(w)
    for i in range(w.size):
        step = np.zeros_like(w)
        step[i] = h
        grad[i] = (_loss(w + step, X, y) - _loss(w - step, X, y)) / (2.0 * h)
    return grad


def _gradient_or_nan(w, X, y):
    """Return the gradient of shape (d,), or NaNs if the call raised or warned about overflow.

    Overflow warnings are errors here: a stable sigmoid never triggers them. A crash, a warning
    or a wrong shape then fails the assertion written for that test.
    """
    n_weights = np.shape(w)[0] if np.ndim(w) == 1 else 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = np.asarray(mse_sigmoid_grad(w, X, y), dtype=np.float64)
    except Exception:
        return np.full(n_weights, np.nan)
    return got if got.shape == (n_weights,) else np.full(n_weights, np.nan)


def _sigmoid_and_derivative_decimal(z):
    """High-precision reference: sigma(z) and sigma'(z) with 60 significant digits."""
    with localcontext() as ctx:
        ctx.prec = 60
        e = (-Decimal(z)).exp()
        return float(1 / (1 + e)), float(e / (1 + e) ** 2)


def test_gradient_matches_central_differences_of_the_loss():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(1, 9))
        d = int(rng.integers(1, 5))
        X = rng.normal(size=(n, d))
        y = rng.uniform(0.0, 1.0, size=n)
        w = rng.normal(size=d)
        want = _finite_difference_gradient(w, X, y)
        got = _gradient_or_nan(w, X, y)
        assert np.all(np.abs(got - want) <= 1e-7 * max(1.0, np.max(np.abs(want)))), (
            "gradient disagrees with finite differences of the loss: check the chain factors, "
            "the sigmoid derivative s(1 - s), the 2/n mean, and the transpose X.T"
        )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    arrays(np.float64, (4, 2), elements=ELEMENTS),
    arrays(np.float64, 4, elements=st.floats(0.0, 1.0)),
    arrays(np.float64, 2, elements=ELEMENTS),
)
def test_random_small_problems_match_finite_differences(X, y, w):
    want = _finite_difference_gradient(w, X, y)
    got = _gradient_or_nan(w, X, y)
    assert np.all(np.abs(got - want) <= 1e-7 * max(1.0, np.max(np.abs(want)))), (
        "gradient disagrees with finite differences on a random 4 x 2 problem: check the "
        "chain factors and the transpose"
    )


def test_single_sample_gradient_includes_the_sigmoid_derivative_factor():
    got = _gradient_or_nan(np.array([0.0]), np.array([[1.5]]), np.array([0.0]))
    assert abs(got[0] - 0.375) <= 1e-12, (
        "the sigmoid's local derivative sigma'(z) = s(1 - s) is part of the chain; leaving it "
        "out makes the gradient four times too large when z = 0"
    )


def test_three_sample_gradient_uses_the_mean_not_the_sum():
    got = _gradient_or_nan(
        np.array([0.0]), np.array([[1.0], [3.0], [-1.0]]), np.array([1.0, 0.0, 1.0])
    )
    assert abs(got[0] - 0.25) <= 1e-12, (
        "the loss is a mean over the n rows, so the outer factor is 2/n; using 2 or 2n instead "
        "makes the gradient off by a factor of n"
    )


@pytest.mark.parametrize(
    "logit", [800.0, -800.0], ids=["positive-saturation", "negative-saturation"]
)
@pytest.mark.parametrize("target", [0.0, 1.0], ids=["target-zero", "target-one"])
def test_saturated_logits_give_finite_zero_gradient(logit, target):
    got = _gradient_or_nan(np.array([logit]), np.array([[1.0]]), np.array([target]))
    assert np.all(np.isfinite(got)), (
        "saturated logits produced NaN or an overflow warning: build the sigmoid from exp(-|z|) "
        "on both sides, because exp(z) overflows for large positive z and exp(-z) for large "
        "negative z"
    )
    assert np.all(got == 0.0), (
        "a saturated sigmoid passes almost no gradient: the gradient should vanish, "
        "so check the local derivative factor"
    )


@pytest.mark.parametrize("logit", [30.0, 36.0], ids=["moderate-saturation", "deep-saturation"])
def test_gradient_keeps_relative_precision_deep_in_saturation(logit):
    got = _gradient_or_nan(np.array([logit]), np.array([[1.0]]), np.array([0.0]))
    s_ref, sp_ref = _sigmoid_and_derivative_decimal(logit)
    want = 2.0 * s_ref * sp_ref
    assert abs(got[0] - want) <= 1e-9 * abs(want), (
        "lost relative precision in the sigmoid derivative: s(1 - s) is wrong once s rounds "
        "to 1; compute sigma'(z) as e^{-|z|} / (1 + e^{-|z|})^2"
    )


BAD_SHAPES = [
    ("row mismatch", np.ones((3, 2)), np.ones(2), np.ones(2)),
    ("column mismatch", np.ones((3, 2)), np.ones(3), np.ones(3)),
    ("column-vector targets", np.ones((2, 1)), np.ones((2, 1)), np.ones(1)),
    ("one-dimensional X", np.ones(3), np.ones(3), np.ones(1)),
    ("empty rows", np.ones((0, 2)), np.ones(0), np.ones(2)),
    ("empty columns", np.ones((2, 0)), np.ones(2), np.ones(0)),
    ("two-dimensional w", np.ones((2, 2)), np.ones(2), np.ones((2, 1))),
    # A single target or a single row broadcasts silently in numpy: reject it explicitly.
    ("one target for several rows", np.ones((3, 2)), np.ones(1), np.ones(2)),
    ("several targets for one row", np.ones((1, 2)), np.ones(3), np.ones(2)),
]


@pytest.mark.parametrize("label, X, y, w", BAD_SHAPES, ids=[case[0] for case in BAD_SHAPES])
def test_shape_mistakes_raise_value_error(label, X, y, w):
    _raised_value_error(
        lambda: mse_sigmoid_grad(w, X, y),
        f"shape mistake ({label}): X must be 2-D, y and w 1-D, with one target per row of X "
        "and one weight per column of X",
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("where", ["X", "y", "w"], ids=["in-x", "in-y", "in-w"])
def test_non_finite_entries_raise_value_error(bad, where):
    X = np.array([[1.0, 2.0], [0.5, -1.0]])
    y = np.array([1.0, 0.0])
    w = np.array([0.1, -0.2])
    if where == "X":
        X[0, 0] = bad
    elif where == "y":
        y[1] = bad
    else:
        w[0] = bad
    _raised_value_error(
        lambda: mse_sigmoid_grad(w, X, y),
        f"a non-finite entry in {where} must raise ValueError, not return NaN",
    )


def test_inputs_are_not_mutated():
    X = np.array([[1.0, -2.0], [0.5, 3.0]])
    y = np.array([1.0, 0.0])
    w = np.array([0.2, -0.4])
    before = (X.copy(), y.copy(), w.copy())
    mse_sigmoid_grad(w, X, y)
    np.testing.assert_array_equal(X, before[0], err_msg="X was modified in place")
    np.testing.assert_array_equal(y, before[1], err_msg="y was modified in place")
    np.testing.assert_array_equal(w, before[2], err_msg="w was modified in place")


def test_integer_inputs_return_a_float64_vector_of_length_d():
    got = mse_sigmoid_grad([1, 0], [[1, 2], [0, 1]], [1, 0])
    assert isinstance(got, np.ndarray), "return a NumPy array, not a list or a scalar"
    assert got.shape == (2,), f"shape mismatch: expected one entry per weight, got {got.shape}"
    assert got.dtype == np.float64, "dtype should be float64 even for integer input"
    want = _finite_difference_gradient(
        np.array([1.0, 0.0]), np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([1.0, 0.0])
    )
    np.testing.assert_allclose(got, want, atol=1e-7, err_msg="integer input gives wrong values")


def test_targets_above_one_are_not_mutated():
    X = np.array([[1.0, -2.0], [0.5, 3.0]])
    y = np.array([2.0, 0.0])  # a target above 1 is a finite real number, so it is valid input
    w = np.array([0.2, -0.4])
    before = y.copy()
    mse_sigmoid_grad(w, X, y)
    np.testing.assert_array_equal(
        y,
        before,
        err_msg="must not mutate its inputs: a target above 1 was changed in place",
    )


def test_targets_are_used_at_full_double_precision():
    # one sample, one weight: the target 0.6 has no exact single-precision form
    X = np.array([[1.0]])
    w = np.array([0.5])
    y = np.array([0.6])
    s_ref, sp_ref = _sigmoid_and_derivative_decimal(0.5)
    want = 2.0 * (s_ref - 0.6) * sp_ref * 1.0
    got = _gradient_or_nan(w, X, y)
    assert abs(got[0] - want) <= 1e-10 * abs(want), (
        "the targets were rounded to single precision before the gradient was computed: "
        "each target must enter the gradient as given, at full double precision"
    )


def test_weights_are_used_at_full_double_precision():
    # one sample, one weight: the weight 0.1 has no exact single-precision form
    X = np.array([[1.0]])
    w = np.array([0.1])
    y = np.array([0.4])
    s_ref, sp_ref = _sigmoid_and_derivative_decimal(0.1)
    want = 2.0 * (s_ref - 0.4) * sp_ref * 1.0
    got = _gradient_or_nan(w, X, y)
    assert abs(got[0] - want) <= 1e-10 * abs(want), (
        "the weights were rounded to single precision before the gradient was computed: "
        "each weight must be used as given, at full double precision"
    )


def test_input_features_are_used_at_full_double_precision():
    # one sample, one weight: the feature 1.2345678901 has no exact single-precision form
    X = np.array([[1.2345678901]])
    w = np.array([0.5])
    y = np.array([0.25])
    s_ref, sp_ref = _sigmoid_and_derivative_decimal(1.2345678901 * 0.5)
    want = 2.0 * (s_ref - 0.25) * sp_ref * 1.2345678901
    got = _gradient_or_nan(w, X, y)
    assert abs(got[0] - want) <= 1e-10 * abs(want), (
        "the input features were rounded to single precision before the gradient was computed: "
        "each entry of X must be used as given, at full double precision"
    )
