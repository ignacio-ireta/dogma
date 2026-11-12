import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import expit, ndtr
from submission import logloss_grad_hess

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


def _tumour_margins(key, n, scale):
    """Real labels and real margins: n breast_cancer tumours scored by a linear model.

    The margin is the projection of six standardised features on the direction from the
    malignant mean to the benign mean of the drawn rows, times ``scale``. Most margins agree
    with the label and a few do not.
    """
    X, y = sample_rows("breast_cancer", data_rng(key), n, 6, standardize=True)
    direction = X[y == 1.0].mean(axis=0) - X[y == 0.0].mean(axis=0)
    return y, scale * (X @ direction)


def _call(y, margin):
    """Run with NumPy warnings as errors: an overflow warning is a stability failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = logloss_grad_hess(y, margin)
    except RuntimeWarning:
        pytest.fail(
            "the gradient or Hessian emitted a NumPy warning: avoid exp of a large positive number"
        )
    except Exception as exc:
        pytest.fail(f"valid input must be accepted; the call raised {type(exc).__name__}")
    if not (isinstance(result, tuple) and len(result) == 2):
        pytest.fail("return a two-item tuple (grad, hess)")
    return result


def _expect_value_error(mistake, y, margin):
    try:
        logloss_grad_hess(y, margin)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the call raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{mistake}; the call returned a result instead of raising ValueError")


def _per_example_loss(margin, y):
    """Bernoulli log-loss in its stable form: log(1 + e^m) - y*m, via logaddexp."""
    return np.logaddexp(0.0, margin) - y * margin


def test_gradient_and_hessian_match_central_differences_of_the_loss():
    margins = np.linspace(-20.0, 20.0, 81)
    for y_value in (0.0, 1.0):
        y = np.full(margins.shape, y_value)
        grad, hess = _call(y, margins)
        step1, step2 = 1e-5, 1e-4
        fd_grad = (
            _per_example_loss(margins + step1, y) - _per_example_loss(margins - step1, y)
        ) / (2 * step1)
        fd_hess = (
            _per_example_loss(margins + step2, y)
            - 2 * _per_example_loss(margins, y)
            + _per_example_loss(margins - step2, y)
        ) / step2**2
        np.testing.assert_allclose(
            grad,
            fd_grad,
            atol=1e-7,
            err_msg="gradient disagrees with the finite difference of the log-loss: "
            "it must be sigmoid(margin) - y",
        )
        np.testing.assert_allclose(
            hess,
            fd_hess,
            atol=1e-5,
            err_msg="hessian disagrees with the second finite difference of the log-loss: "
            "it must be sigmoid * (1 - sigmoid)",
        )


def test_matches_closed_form_on_random_moderate_margins():
    y, margin = _tumour_margins("logloss-grad-hess-p02:closed-form", 200, 1.0)
    grad, hess = _call(y, margin)
    p = expit(margin)
    np.testing.assert_allclose(
        grad, p - y, rtol=1e-12, atol=1e-12, err_msg="gradient differs from sigmoid(margin) - y"
    )
    np.testing.assert_allclose(
        hess,
        p * (1.0 - p),
        rtol=1e-10,
        atol=1e-14,
        err_msg="hessian differs from sigmoid * (1 - sigmoid) on moderate margins",
    )


def test_saturated_margins_are_finite_warning_free_and_exact():
    margin = np.array([1e4, -1e4, 2000.0, -2000.0, 750.0, -750.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    grad, hess = _call(y, margin)
    assert np.all(np.isfinite(grad)) and np.all(np.isfinite(hess)), (
        "non-finite output at |margin| up to 1e4: the sigmoid must not overflow"
    )
    np.testing.assert_allclose(
        grad,
        expit(margin) - y,
        atol=1e-12,
        err_msg="gradient is wrong at saturated margins: check the sign branch of the sigmoid",
    )
    np.testing.assert_allclose(
        hess, 0.0, atol=1e-12, err_msg="hessian must vanish once the margin saturates"
    )


def test_zero_margin_has_gradient_half_minus_label_and_hessian_quarter():
    grad, hess = _call(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        grad, [0.5, -0.5], atol=1e-14, err_msg="gradient at zero margin is wrong"
    )
    np.testing.assert_allclose(
        hess, [0.25, 0.25], atol=1e-14, err_msg="hessian at zero margin is wrong"
    )


def test_bounds_hold_for_any_finite_margin():
    rng = np.random.default_rng(12)
    margin = -1e4 + (1e4 - -1e4) * _real_unit(rng, 500)
    y = (_real_unit(rng, 500) < 0.5).astype(float)
    grad, hess = _call(y, margin)
    assert np.all(np.abs(grad) <= 1.0 + 1e-12), (
        "gradient magnitude exceeds one: sigmoid minus label is in [-1, 1]"
    )
    assert np.all(hess >= -1e-15) and np.all(hess <= 0.25 + 1e-12), (
        "hessian outside [0, 1/4]: it must be sigmoid * (1 - sigmoid)"
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.0, 1.0]),
)
def test_single_example_bounds_and_agreement_with_formula(margin, label):
    grad, hess = _call(np.array([label]), np.array([margin]))
    assert np.isfinite(grad[0]) and np.isfinite(hess[0]), "non-finite output for a finite margin"
    p = expit(margin)
    assert abs(grad[0] - (p - label)) <= 1e-12, "gradient disagrees with sigmoid(margin) - y"
    assert abs(hess[0] - p * (1.0 - p)) <= 1e-12, "hessian disagrees with sigmoid * (1 - sigmoid)"


def test_empty_input_returns_empty_float_arrays():
    grad, hess = _call(np.zeros(0), np.zeros(0))
    assert grad.shape == (0,) and hess.shape == (0,), (
        "empty input must return empty arrays of shape (0,)"
    )


def test_boolean_labels_are_accepted_and_return_float64():
    grad, hess = _call(np.array([True, False, True]), np.array([0.5, -0.5, 2.0]))
    assert grad.dtype == np.float64 and hess.dtype == np.float64, "outputs must be float64 arrays"
    np.testing.assert_allclose(
        grad,
        expit(np.array([0.5, -0.5, 2.0])) - np.array([1.0, 0.0, 1.0]),
        atol=1e-12,
        err_msg="boolean labels must behave as their binary numeric equivalents",
    )


INVALID = [
    ("label-minus-one", [-1.0, 1.0], [0.0, 0.0]),
    ("label-two", [0.0, 2.0], [0.0, 0.0]),
    ("length-mismatch", [0.0, 1.0, 1.0], [0.0, 0.0]),
    ("two-dimensional-margin", [0.0, 1.0], [[0.0], [0.0]]),
    ("nan-margin", [0.0, 1.0], [0.0, float("nan")]),
    ("infinite-margin", [0.0, 1.0], [float("inf"), 0.0]),
]


@pytest.mark.parametrize(
    ("y", "margin"), [case[1:] for case in INVALID], ids=[c[0] for c in INVALID]
)
def test_invalid_inputs_raise_value_error(y, margin):
    _expect_value_error(
        "labels must be 0 or 1, margins finite, and both arrays 1-D of equal length", y, margin
    )


def test_inputs_are_not_mutated():
    y, margin = _tumour_margins("logloss-grad-hess-p02:not-mutated", 20, 0.5)
    margin_before, y_before = margin.copy(), y.copy()
    _call(y, margin)
    np.testing.assert_array_equal(margin, margin_before, err_msg="margin was modified in place")
    np.testing.assert_array_equal(y, y_before, err_msg="y was modified in place")
