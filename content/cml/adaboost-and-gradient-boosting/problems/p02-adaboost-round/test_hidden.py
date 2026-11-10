from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import adaboost_round


def _call(y, pred, weights):
    """Call the round and check the answer's types. Messages never show the inputs."""
    try:
        alpha, new_weights = adaboost_round(y, pred, weights)
    except ValueError:
        pytest.fail("valid round was rejected: check the input rules, and whether a sum overflowed")
    except Exception as exc:
        pytest.fail(f"valid round crashed the call ({type(exc).__name__}): check the contract")
    if not isinstance(alpha, float):
        pytest.fail("alpha must be returned as a Python float")
    new_weights = np.asarray(new_weights, dtype=np.float64)
    if new_weights.shape != (len(y),):
        pytest.fail("shape mismatch: the new weights need one entry per sample")
    return alpha, new_weights


def _expect_value_error(mistake, y, pred, weights):
    try:
        adaboost_round(y, pred, weights)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__} instead")
    pytest.fail(f"{mistake}; the function returned a round instead")


def _oracle(y, pred, weights):
    """Explicit reference: normalise, sum the mistake weights, scale each sample by its factor."""
    total = sum(weights)
    w = [v / total for v in weights]
    error = sum(w[i] for i in range(len(y)) if pred[i] != y[i])
    if error >= 0.5:
        return 0.0, w
    alpha = 0.5 * math.log((1.0 - error) / error)
    factors = [math.exp(alpha) if pred[i] != y[i] else math.exp(-alpha) for i in range(len(y))]
    raw = [w[i] * factors[i] for i in range(len(y))]
    z = sum(raw)
    return alpha, [v / z for v in raw]


def _random_round(seed, n=12):
    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 1], size=n)
    flips = rng.uniform(size=n) < 0.3
    pred = np.where(flips, -y, y)
    weights = rng.uniform(0.1, 2.0, size=n)
    return y, pred, weights


def test_matches_an_explicit_reference_on_random_rounds():
    checked = 0
    for seed in range(25):
        y, pred, weights = _random_round(seed)
        if np.all(pred == y):
            continue
        want_alpha, want_weights = _oracle(y.tolist(), pred.tolist(), weights.tolist())
        got_alpha, got_weights = _call(y, pred, weights)
        assert got_alpha == pytest.approx(want_alpha, rel=1e-9, abs=1e-12), (
            "alpha disagrees with the reference: use 0.5 * ln((1 - err) / err) with the normalised "
            "weights, and the weighted error, not the unweighted error rate"
        )
        np.testing.assert_allclose(
            got_weights,
            want_weights,
            rtol=1e-9,
            atol=1e-12,
            err_msg="new weights disagree with the reference: mistakes are scaled by e^alpha, "
            "correct samples by e^-alpha, then the weights are normalised",
        )
        checked += 1
    assert checked >= 15, "the random check must exercise many rounds"


def test_error_exactly_one_half_returns_zero_alpha_and_unchanged_weights():
    y = [1, -1, 1, -1]
    pred = [-1, 1, 1, -1]  # two of four uniform weights are wrong: error 0.5
    alpha, weights = _call(y, pred, [1.0, 1.0, 1.0, 1.0])
    assert alpha == 0.0, "an error of exactly one half is chance level: alpha must be 0"
    np.testing.assert_allclose(
        weights,
        [0.25, 0.25, 0.25, 0.25],
        rtol=0,
        atol=1e-12,
        err_msg="a rejected round must return the normalised weights, unchanged",
    )


def test_error_above_one_half_is_also_rejected():
    y = [1, 1, 1, 1]
    pred = [-1, -1, -1, 1]  # weighted error 0.75
    alpha, weights = _call(y, pred, [1.0, 1.0, 1.0, 1.0])
    assert alpha == 0.0, "an error above one half must give alpha 0, not a negative vote"
    np.testing.assert_allclose(
        weights,
        [0.25] * 4,
        rtol=0,
        atol=1e-12,
        err_msg="a rejected weak learner must preserve the normalised input weights",
    )


def test_after_the_update_the_same_predictions_have_error_one_half():
    for seed in range(15):
        y, pred, weights = _random_round(seed + 50)
        if np.all(pred == y) or np.all(pred != y):
            continue
        alpha, new_weights = _call(y, pred, weights)
        if alpha == 0.0:
            continue
        error_after = float(np.sum(new_weights[pred != y]))
        assert error_after == pytest.approx(0.5, abs=1e-12), (
            "after the update the same classifier must be at chance under the new weights: "
            "check the factors e^alpha and e^-alpha"
        )


def test_result_is_invariant_to_rescaling_the_weights():
    y, pred, weights = _random_round(77)
    a1, w1 = _call(y, pred, weights)
    a2, w2 = _call(y, pred, 7.3 * weights)
    assert a2 == pytest.approx(a1, rel=1e-12, abs=1e-12), (
        "alpha must depend on the normalised weights only: normalise before computing the error"
    )
    np.testing.assert_allclose(
        w2,
        w1,
        rtol=1e-12,
        atol=1e-12,
        err_msg="new weights must not depend on the scale of the input weights",
    )


def test_output_weights_are_positive_and_sum_to_one():
    for seed in range(10):
        y, pred, weights = _random_round(seed + 200)
        _, new_weights = _call(y, pred, weights)
        assert (new_weights >= 0.0).all(), "weights must be non-negative"
        assert abs(float(new_weights.sum()) - 1.0) <= 1e-12, (
            "the new weights must be normalised to sum to 1"
        )


def test_a_zero_weight_sample_stays_at_zero():
    y = [1, -1, 1, -1]
    pred = [1, -1, -1, -1]
    _, weights = _call(y, pred, [0.0, 1.0, 1.0, 1.0])
    assert weights[0] == 0.0, "a sample with zero weight must stay at zero after the update"
    assert abs(float(weights.sum()) - 1.0) <= 1e-12, "the updated weights must sum to one"


def test_inputs_are_not_mutated():
    y = np.array([1, -1, 1, 1, -1])
    pred = np.array([1, 1, 1, -1, -1])
    weights = np.array([0.5, 2.0, 1.0, 0.25, 3.0])
    y0, p0, w0 = y.copy(), pred.copy(), weights.copy()
    _call(y, pred, weights)
    np.testing.assert_array_equal(y, y0, err_msg="y was modified in place")
    np.testing.assert_array_equal(pred, p0, err_msg="pred was modified in place")
    np.testing.assert_array_equal(
        weights, w0, err_msg="weights were modified in place: normalise a copy"
    )


def test_perfect_classifier_raises_value_error():
    _expect_value_error(
        "a perfect classifier must raise ValueError: alpha would be infinite",
        [1, -1, 1],
        [1, -1, 1],
        [1.0, 2.0, 3.0],
    )


@pytest.mark.parametrize(
    ("y", "pred", "weights", "mistake"),
    [
        ([], [], [], "an empty round must raise ValueError"),
        ([1, -1], [1], [1.0, 1.0], "mismatched lengths must raise ValueError"),
        ([1, 0], [1, 1], [1.0, 1.0], "a label of 0 must raise ValueError: labels are -1 and +1"),
        ([1, -1], [1, 2], [1.0, 1.0], "a prediction outside -1 and +1 must raise ValueError"),
        (
            [1, 1, -1, 1],
            [-1, 1, -1, 1],
            [1.0, 1.0, 1.0, -0.5],
            "a negative weight must raise ValueError, even with a positive sum",
        ),
        ([1, -1], [1, -1], [0.0, 0.0], "weights with a zero sum must raise ValueError"),
        ([1, -1], [1, -1], [1.0, float("nan")], "a NaN weight must raise ValueError"),
        ([[1, -1]], [[1, -1]], [1.0, 1.0], "two-dimensional input must raise ValueError"),
    ],
    ids=[
        "empty",
        "length",
        "zero-label",
        "bad-pred",
        "negative-weight",
        "zero-sum",
        "nan-weight",
        "two-dimensional",
    ],
)
def test_invalid_round_raises_value_error(y, pred, weights, mistake):
    _expect_value_error(mistake, y, pred, weights)


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000), st.integers(min_value=2, max_value=15))
def test_non_chance_rounds_give_a_positive_vote_and_a_distribution(seed, n):
    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 1], size=n)
    pred = np.where(rng.uniform(size=n) < 0.2, -y, y)
    weights = rng.uniform(0.01, 5.0, size=n)
    if np.all(pred == y):
        return  # a perfect classifier is rejected; covered elsewhere
    alpha, new_weights = _call(y, pred, weights)
    if alpha == 0.0:
        return
    assert math.isfinite(alpha) and alpha > 0.0, (
        "a classifier better than chance must get a finite, positive vote"
    )
    assert abs(float(new_weights.sum()) - 1.0) <= 1e-12, "the new weights must sum to 1"


def test_large_finite_weight_scale_does_not_change_the_round():
    alpha, weights = _call([1, 1, 1, 1], [1, 1, 1, -1], [1e308] * 4)
    assert math.isclose(alpha, 0.5 * math.log(3.0), rel_tol=1e-12), (
        "normalising large finite weights must preserve the weighted error and vote"
    )
    np.testing.assert_allclose(
        weights,
        [1 / 6, 1 / 6, 1 / 6, 1 / 2],
        rtol=1e-12,
        atol=1e-12,
        err_msg="weight normalisation must remain a distribution when the raw sum overflows",
    )


def test_tiny_positive_weighted_error_has_a_finite_vote():
    alpha, weights = _call([1, 1], [1, -1], [1.0, 1e-320])
    assert math.isfinite(alpha) and alpha > 0.0, (
        "a positive representable error must yield a finite vote: subtract logarithms"
    )
    np.testing.assert_allclose(
        weights,
        [0.5, 0.5],
        rtol=1e-9,
        atol=1e-12,
        err_msg="a very small positive error must still balance correct and mistaken weight",
    )


def test_single_sample_round_is_accepted_and_rejected_at_chance():
    # One sample, wrongly predicted: the weighted error is 1, which is at or above one half.
    # The contract allows n >= 1, so the round must return alpha 0 and the weight of 1.
    alpha, weights = _call([1], [-1], [1.0])
    assert alpha == 0.0, "a single-sample round with error 1 is rejected: alpha must be 0"
    np.testing.assert_allclose(
        weights,
        [1.0],
        rtol=0,
        atol=1e-12,
        err_msg="a one-sample input is valid (n >= 1): do not reject it",
    )


def test_nan_weight_is_rejected_even_when_the_classifier_is_not_perfect():
    _expect_value_error(
        "a NaN weight must raise ValueError, even when the classifier makes a mistake",
        [1, -1],
        [1, 1],
        [1.0, float("nan")],
    )


def test_zero_weight_sum_is_rejected_even_when_the_classifier_is_not_perfect():
    _expect_value_error(
        "weights with a zero sum must raise ValueError, even when the classifier makes a mistake",
        [1, -1],
        [1, 1],
        [0.0, 0.0],
    )


def test_tiny_negative_weight_with_positive_sum_raises_value_error():
    y = np.array([1.0, -1.0, 1.0, -1.0])
    pred = np.array([1.0, 1.0, 1.0, -1.0])
    weights = np.array([1.0, 1.0, 1.0, -1e-300])  # negative, though the sum is positive
    try:
        adaboost_round(y, pred, weights)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a negative weight must raise ValueError; raised {type(exc).__name__}")
    pytest.fail("a negative weight must raise ValueError, not return a round")
