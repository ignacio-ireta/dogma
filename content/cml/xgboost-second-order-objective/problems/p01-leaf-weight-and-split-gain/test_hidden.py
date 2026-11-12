import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from submission import leaf_weight_and_split_gain

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


def _call(g_left, h_left, g_right, h_right, **kwargs):
    """Call the split with NumPy warnings as errors."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = leaf_weight_and_split_gain(g_left, h_left, g_right, h_right, **kwargs)
    except RuntimeWarning:
        pytest.fail(
            "the split arithmetic emitted a NumPy warning: guard the division by H + reg_lambda"
        )
    except Exception as exc:
        pytest.fail(f"valid input must be accepted; the call raised {type(exc).__name__}")
    if not (isinstance(result, tuple) and len(result) == 3):
        pytest.fail("return a three-item tuple (w_left, w_right, gain)")
    return result


def _expect_value_error(mistake, g_left, h_left, g_right, h_right, **kwargs):
    try:
        leaf_weight_and_split_gain(g_left, h_left, g_right, h_right, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the call raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{mistake}; the call returned a result instead of raising ValueError")


def _reference(g_l, h_l, g_r, h_r, *, lam, gam):
    """Closed forms evaluated with exactly rounded sums, as an independent path."""
    G_L, H_L = math.fsum(g_l), math.fsum(h_l)
    G_R, H_R = math.fsum(g_r), math.fsum(h_r)
    w_l = -G_L / (H_L + lam)
    w_r = -G_R / (H_R + lam)
    gain = (
        0.5 * (G_L**2 / (H_L + lam) + G_R**2 / (H_R + lam) - (G_L + G_R) ** 2 / (H_L + H_R + lam))
        - gam
    )
    return w_l, w_r, gain


def _random_split(seed):
    """A real candidate split: log-loss gradients and hessians of breast_cancer tumours.

    The margin is a weak linear score of three standardised features, as early in boosting, so
    g = sigmoid(margin) - y and h = sigmoid(margin) * (1 - sigmoid(margin)). The tumours go left
    or right by a fourth feature at a randomly chosen quantile; a split that leaves one side
    empty is skipped. The regularisation settings are hyperparameters and stay random.
    """
    rng = data_rng(f"split-gain-p01:split:{seed}")
    while True:
        n = int(rng.integers(2, 79))
        X, y = sample_rows("breast_cancer", rng, n, 4, standardize=True)
        left = X[:, 3] <= np.quantile(X[:, 3], float(rng.uniform(0.1, 0.9)))
        if 0 < int(left.sum()) < n:
            break
    p = 1.0 / (1.0 + np.exp(-0.5 * (X[:, :3] @ _real_normal(rng, 3))))
    g, h = p - y, p * (1.0 - p)
    g_l, g_r, h_l, h_r = g[left], g[~left], h[left], h[~left]
    lam = float(rng.uniform(0.0, 3.0))
    gam = float(rng.uniform(0.0, 1.0))
    return g_l, h_l, g_r, h_r, lam, gam


def _leaf_minimum(G, H, lam):
    """Minimum over w of G*w + (H + lam) * w**2 / 2, found by a bounded numeric search."""
    curvature = H + lam
    bound = 10.0 * (1.0 + abs(G)) / curvature
    res = minimize_scalar(
        lambda w: G * w + 0.5 * curvature * w * w,
        bounds=(-bound, bound),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(res.fun)


def test_matches_closed_form_on_random_splits():
    for seed in range(30):
        g_l, h_l, g_r, h_r, lam, gam = _random_split(seed)
        got = _call(g_l, h_l, g_r, h_r, reg_lambda=lam, gamma=gam)
        want = _reference(g_l, h_l, g_r, h_r, lam=lam, gam=gam)
        np.testing.assert_allclose(
            got,
            want,
            rtol=1e-9,
            atol=1e-12,
            err_msg="leaf weights or gain differ from the closed forms: check the sums G and H "
            "per side, the lambda in each denominator, and the parent term",
        )


def test_leaf_weights_are_stationary_points_of_the_leaf_objective():
    for seed in range(6):
        g_l, h_l, g_r, h_r, lam, gam = _random_split(200 + seed)
        w_l, w_r, _ = _call(g_l, h_l, g_r, h_r, reg_lambda=lam, gamma=gam)
        for w, g, h in ((w_l, g_l, h_l), (w_r, g_r, h_r)):
            G, H = float(np.sum(g)), float(np.sum(h))

            def quad(v, G=G, H=H, lam=lam):
                return G * v + 0.5 * (H + lam) * v * v

            step = 1e-6
            slope = (quad(w + step) - quad(w - step)) / (2 * step)
            assert abs(slope) <= 1e-6 * (1.0 + H + lam), (
                "a leaf weight is not a stationary point of the leaf objective: the weight must "
                "be -G / (H + reg_lambda) on that side"
            )


def test_gain_is_the_drop_in_leaf_objective_computed_numerically():
    for seed in range(8):
        g_l, h_l, g_r, h_r, lam, gam = _random_split(300 + seed)
        _, _, gain = _call(g_l, h_l, g_r, h_r, reg_lambda=lam, gamma=gam)
        G_L, H_L = float(np.sum(g_l)), float(np.sum(h_l))
        G_R, H_R = float(np.sum(g_r)), float(np.sum(h_r))
        parent = _leaf_minimum(G_L + G_R, H_L + H_R, lam)
        left = _leaf_minimum(G_L, H_L, lam)
        right = _leaf_minimum(G_R, H_R, lam)
        # Each node's value is its minimum plus gamma. Gain is parent value minus children.
        expected = (parent - left - right) - gam
        assert gain == pytest.approx(expected, abs=1e-7), (
            "split gain is not the drop in the regularised objective: use half the "
            "children-minus-parent score difference, minus gamma"
        )


def test_gamma_lowers_gain_one_for_one():
    g_l, h_l, g_r, h_r, lam, _ = _random_split(400)
    base = _call(g_l, h_l, g_r, h_r, reg_lambda=lam, gamma=0.0)[2]
    shifted = _call(g_l, h_l, g_r, h_r, reg_lambda=lam, gamma=0.3)[2]
    assert base - shifted == pytest.approx(0.3, abs=1e-12), (
        "gamma must be subtracted from the gain exactly once and must not change the leaf weights"
    )


def test_larger_reg_lambda_shrinks_both_leaf_weights():
    g_l, h_l, g_r, h_r, _, _ = _random_split(500)
    previous = None
    for lam in (0.1, 1.0, 10.0, 100.0):
        w_l, w_r, _ = _call(g_l, h_l, g_r, h_r, reg_lambda=lam, gamma=0.0)
        if previous is not None:
            assert abs(w_l) < abs(previous[0]) and abs(w_r) < abs(previous[1]), (
                "leaf weights must shrink towards zero as reg_lambda grows: the penalty belongs in "
                "the denominator H + reg_lambda"
            )
        previous = (w_l, w_r)


def test_empty_left_side_with_positive_lambda_gives_zero_weight():
    g_r, h_r = np.array([2.0, -1.0]), np.array([0.5, 0.5])
    w_l, w_r, gain = _call([], [], g_r, h_r, reg_lambda=2.0, gamma=0.1)
    assert w_l == pytest.approx(0.0, abs=1e-12), "an empty left side must get weight zero"
    assert w_r == pytest.approx(-1.0 / 3.0, abs=1e-12), (
        "right leaf weight is wrong for a one-sided split"
    )
    assert gain == pytest.approx(-0.1, abs=1e-12), (
        "a split with an empty side gains nothing: the gain must be exactly minus gamma"
    )


def test_empty_side_with_zero_lambda_raises_value_error():
    _expect_value_error(
        "an empty side with no positive reg_lambda has H + reg_lambda = 0 and must be rejected",
        [1.0],
        [1.0],
        [],
        [],
        reg_lambda=0.0,
        gamma=0.0,
    )


def test_zero_hessian_side_with_zero_lambda_raises_value_error():
    _expect_value_error(
        "a side whose hessians sum to zero, with no positive reg_lambda, must be rejected",
        [1.0, -1.0],
        [0.0, 0.0],
        [1.0],
        [1.0],
        reg_lambda=0.0,
        gamma=0.0,
    )


def test_zero_hessian_side_with_positive_lambda_is_accepted():
    w_l, _, _ = _call([1.0, 2.0], [0.0, 0.0], [0.5], [1.0], reg_lambda=2.0, gamma=0.0)
    assert w_l == pytest.approx(-1.5, abs=1e-12), (
        "a zero-hessian side with positive reg_lambda has weight -G / reg_lambda"
    )


INVALID_SCALARS = [
    ("negative-lambda", {"reg_lambda": -1.0, "gamma": 0.0}),
    ("nan-lambda", {"reg_lambda": float("nan"), "gamma": 0.0}),
    ("negative-gamma", {"reg_lambda": 1.0, "gamma": -0.1}),
    ("infinite-gamma", {"reg_lambda": 1.0, "gamma": float("inf")}),
]


@pytest.mark.parametrize(
    "kwargs", [case[1] for case in INVALID_SCALARS], ids=[case[0] for case in INVALID_SCALARS]
)
def test_invalid_regularisers_raise_value_error(kwargs):
    # Hessians sum to 2 on each side, so H + reg_lambda > 0 even for reg_lambda = -1.
    _expect_value_error(
        "reg_lambda and gamma must be finite and zero or positive",
        [1.0],
        [2.0],
        [1.0],
        [2.0],
        **kwargs,
    )


INVALID_SIDES = [
    ("negative-hessian", [1.0], [-0.5], [1.0], [1.0]),
    ("nan-gradient", [float("nan")], [1.0], [1.0], [1.0]),
    ("infinite-hessian", [1.0], [float("inf")], [1.0], [1.0]),
    ("length-mismatch", [1.0, 2.0], [1.0], [1.0], [1.0]),
    ("two-dimensional-gradient", [[1.0]], [1.0], [1.0], [1.0]),
]


@pytest.mark.parametrize(
    ("g_l", "h_l", "g_r", "h_r"),
    [case[1:] for case in INVALID_SIDES],
    ids=[case[0] for case in INVALID_SIDES],
)
def test_invalid_arrays_raise_value_error(g_l, h_l, g_r, h_r):
    _expect_value_error(
        "gradients and hessians must be 1-D, equal in length, finite, and the hessians >= 0",
        g_l,
        h_l,
        g_r,
        h_r,
        reg_lambda=1.0,
        gamma=0.0,
    )


def test_large_magnitudes_stay_finite_and_accurate():
    rng = np.random.default_rng(600)
    g_l = _real_normal(rng, 20) * 1e6
    h_l = 1e3 + (1e4 - 1e3) * _real_unit(rng, 20)
    g_r = _real_normal(rng, 15) * 1e6
    h_r = 1e3 + (1e4 - 1e3) * _real_unit(rng, 15)
    got = _call(g_l, h_l, g_r, h_r, reg_lambda=1.0, gamma=0.0)
    assert all(math.isfinite(v) for v in got), (
        "large gradients must not overflow: compute G^2 / (H + reg_lambda) without extra scaling"
    )
    np.testing.assert_allclose(
        got,
        _reference(g_l, h_l, g_r, h_r, lam=1.0, gam=0.0),
        rtol=1e-9,
        err_msg="large-magnitude split disagrees with the closed forms",
    )


def test_small_magnitudes_with_small_lambda_stay_accurate():
    rng = np.random.default_rng(601)
    g_l = _real_normal(rng, 10) * 1e-9
    h_l = 1e-10 + (1e-9 - 1e-10) * _real_unit(rng, 10)
    g_r = _real_normal(rng, 10) * 1e-9
    h_r = 1e-10 + (1e-9 - 1e-10) * _real_unit(rng, 10)
    got = _call(g_l, h_l, g_r, h_r, reg_lambda=1e-9, gamma=0.0)
    np.testing.assert_allclose(
        got,
        _reference(g_l, h_l, g_r, h_r, lam=1e-9, gam=0.0),
        rtol=1e-9,
        atol=1e-18,
        err_msg="small-magnitude split disagrees with the closed forms",
    )


def test_zero_gradients_give_zero_weights_and_gain_minus_gamma():
    w_l, w_r, gain = _call([0.0, 0.0], [1.0, 1.0], [0.0], [2.0], reg_lambda=1.0, gamma=0.2)
    assert (w_l, w_r) == pytest.approx((0.0, 0.0), abs=1e-12), (
        "zero gradients must give zero leaf weights"
    )
    assert gain == pytest.approx(-0.2, abs=1e-12), "zero gradients must give a gain of minus gamma"


@settings(max_examples=50, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000))
def test_gain_does_not_depend_on_sample_order(seed):
    g_l, h_l, g_r, h_r, lam, gam = _random_split(seed)
    rng = np.random.default_rng(seed + 1)
    perm_l = rng.permutation(len(g_l))
    perm_r = rng.permutation(len(g_r))
    original = _call(g_l, h_l, g_r, h_r, reg_lambda=lam, gamma=gam)
    permuted = _call(g_l[perm_l], h_l[perm_l], g_r[perm_r], h_r[perm_r], reg_lambda=lam, gamma=gam)
    np.testing.assert_allclose(
        permuted,
        original,
        rtol=1e-10,
        atol=1e-12,
        err_msg="the result depends on the order of the samples: the sums must be order-free",
    )


def test_inputs_are_not_mutated():
    g_l, h_l, g_r, h_r, lam, gam = _random_split(700)
    before = [a.copy() for a in (g_l, h_l, g_r, h_r)]
    _call(g_l, h_l, g_r, h_r, reg_lambda=lam, gamma=gam)
    for name, a, b in zip(
        ("g_left", "h_left", "g_right", "h_right"), (g_l, h_l, g_r, h_r), before, strict=True
    ):
        np.testing.assert_array_equal(a, b, err_msg=f"{name} was modified in place")
