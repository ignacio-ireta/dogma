import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import logsumexp
from submission import cross_entropy_from_logits


def _call(logits, targets):
    """Call the function with NumPy warnings turned into errors.

    Hidden failures show only the first line of the message, so each line names the mistake and
    never includes the input, its shape or the expected value.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            loss, grad = cross_entropy_from_logits(logits, targets)
    except RuntimeWarning:
        pytest.fail(
            "cross-entropy overflowed or took the log of a zero probability: compute the "
            "log-sum-exp from the row-maximum-subtracted logits"
        )
    except Exception as exc:
        pytest.fail(
            "cross-entropy must accept valid logits and targets; the call raised "
            f"{type(exc).__name__}"
        )
    if type(loss) is not float:
        pytest.fail("the mean loss must be a Python float")
    if not isinstance(grad, np.ndarray) or grad.dtype != np.float64:
        pytest.fail("the gradient must be a float64 NumPy array")
    if grad.shape != np.shape(logits):
        pytest.fail("the gradient must have the same shape as the logits")
    return loss, grad


def _expect_value_error(logits, targets):
    try:
        cross_entropy_from_logits(logits, targets)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"invalid input must raise ValueError; the function raised {type(exc).__name__}"
        )
    pytest.fail("invalid input must raise ValueError, not return a loss")


def _reference_loss(logits, targets):
    """Independent reference: mean of logsumexp(z_i) - z_i[t_i], with scipy's log-sum-exp."""
    z = np.asarray(logits, dtype=np.float64)
    rows = np.arange(z.shape[0])
    return float(np.mean(logsumexp(z, axis=1) - z[rows, targets]))


def _finite_difference_gradient(logits, targets, h=1e-6):
    """Central differences of the reference loss with respect to every logit."""
    grad = np.zeros_like(logits, dtype=np.float64)
    for idx in np.ndindex(logits.shape):
        up = logits.astype(np.float64).copy()
        down = logits.astype(np.float64).copy()
        up[idx] += h
        down[idx] -= h
        grad[idx] = (_reference_loss(up, targets) - _reference_loss(down, targets)) / (2.0 * h)
    return grad


def test_loss_matches_a_log_sum_exp_reference_on_random_data():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=2.0, size=(20, 5))
    targets = rng.integers(0, 5, size=20)
    ref = _reference_loss(logits, targets)
    loss, _ = _call(logits, targets)
    assert abs(loss - ref) <= 1e-12, (
        "loss disagrees with the log-sum-exp reference: the loss is the mean of "
        "logsumexp(z_i) - z_i[t_i]"
    )


def test_gradient_matches_central_differences_of_the_loss():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(4, 3))
    targets = rng.integers(0, 3, size=4)
    fd = _finite_difference_gradient(logits, targets)
    _, grad = _call(logits, targets)
    np.testing.assert_allclose(
        grad,
        fd,
        atol=1e-7,
        err_msg="gradient disagrees with finite differences of the loss: the gradient of row i is "
        "(softmax_i - onehot_i) / n, with the 1/n from the mean and the sign from the log",
    )


@pytest.mark.parametrize(
    "logits, targets",
    [
        (np.array([[1e4, -1e4, 0.0]]), np.array([0])),
        (np.array([[1e4, -1e4, 0.0]]), np.array([1])),
        (np.array([[-1e4, -1e4 + 3.0], [1e4, 1e4]]), np.array([1, 0])),
    ],
    ids=["confident-correct", "confident-wrong", "mixed-rows"],
)
def test_extreme_logits_match_reference_without_warnings(logits, targets):
    ref = _reference_loss(logits, targets)
    loss, grad = _call(logits, targets)
    assert np.isfinite(loss) and np.all(np.isfinite(grad)), "extreme logits gave NaN or inf"
    assert abs(loss - ref) <= 1e-9 + 1e-12 * abs(ref), (
        "extreme logits gave the wrong loss: take log-probabilities from the logits, not from "
        "rounded probabilities, and subtract the maximum"
    )


def test_each_gradient_row_sums_to_zero():
    rng = np.random.default_rng(2)
    logits = rng.normal(scale=5.0, size=(7, 4))
    targets = rng.integers(0, 4, size=7)
    _, grad = _call(logits, targets)
    np.testing.assert_allclose(
        grad.sum(axis=1), 0.0, atol=1e-12, err_msg="each row of the gradient must sum to zero"
    )


def test_repeating_a_row_leaves_the_mean_loss_unchanged():
    row = np.array([[0.5, -1.0, 2.0]])
    target = np.array([2])
    single, _ = _call(row, target)
    repeated, _ = _call(np.repeat(row, 6, axis=0), np.repeat(target, 6))
    assert abs(repeated - single) <= 1e-12, (
        "the loss must be a mean over rows: repeating a row must not change it, so a sum is wrong"
    )


def test_two_class_problem_matches_the_reference():
    logits = np.array([[0.3, -0.7], [2.0, 1.0], [-3.0, 3.0]])
    targets = np.array([1, 0, 0])
    ref = _reference_loss(logits, targets)
    loss, grad = _call(logits, targets)
    assert abs(loss - ref) <= 1e-12, "two-class loss disagrees with the reference"
    assert grad.shape == (3, 2), "shape mismatch: the gradient has the shape of the logits"


def test_integer_logits_are_accepted_and_return_float_results():
    loss, grad = _call(np.array([[1, 2, 3], [0, 0, 0]]), np.array([2, 1]))
    assert isinstance(loss, float), "the loss must be a Python float"
    assert grad.dtype == np.float64, "the gradient must be float64 even for integer logits"
    want = _reference_loss(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]), np.array([2, 1]))
    assert abs(loss - want) <= 1e-12, (
        "integer logits give the wrong loss: cast them to float64 first"
    )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000))
def test_gradient_agrees_with_finite_differences_on_random_small_problems(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(1, 4)), int(rng.integers(2, 5))
    logits = rng.normal(size=(n, k))
    targets = rng.integers(0, k, size=n)
    fd = _finite_difference_gradient(logits, targets)
    _, grad = _call(logits, targets)
    np.testing.assert_allclose(
        grad,
        fd,
        atol=1e-7,
        err_msg="gradient disagrees with finite differences on a small problem: check the 1/n "
        "scaling and the sign of the one-hot term",
    )


@pytest.mark.parametrize(
    ("logits", "targets"),
    [
        (np.array([[0.0, 1.0]]), np.array([-1])),
        (np.array([[0.0, 1.0]]), np.array([2])),
        (np.array([[0.0, 1.0]]), np.array([0.0])),
        (np.array([[0.0, 1.0]]), np.array([True])),
        (np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0])),
        (np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[0], [1]])),
        (np.array([0.0, 1.0]), np.array([0])),
        (np.array([[0.0], [1.0]]), np.array([0, 0])),
        (np.zeros((0, 3)), np.zeros(0, dtype=int)),
        (np.array([[np.nan, 1.0]]), np.array([0])),
        (np.array([[np.inf, 1.0]]), np.array([1])),
    ],
    ids=[
        "negative-target",
        "target-too-large",
        "float-target",
        "boolean-target",
        "wrong-target-length",
        "two-dimensional-targets",
        "one-dimensional-logits",
        "one-class",
        "empty-batch",
        "nan-logit",
        "infinite-logit",
    ],
)
def test_invalid_inputs_raise_value_error(logits, targets):
    _expect_value_error(logits, targets)


def test_inputs_are_not_mutated():
    logits = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
    targets = np.array([2, 0])
    logits_before, targets_before = logits.copy(), targets.copy()
    _call(logits, targets)
    np.testing.assert_array_equal(logits, logits_before, err_msg="logits were modified in place")
    np.testing.assert_array_equal(targets, targets_before, err_msg="targets were modified in place")


def test_mean_of_large_finite_losses_does_not_overflow_the_reduction():
    logits = np.tile([0.0, -1e308], (3, 1))
    loss, grad = _call(logits, np.ones(3, dtype=int))
    np.testing.assert_allclose(
        loss,
        1e308,
        rtol=1e-12,
        err_msg="a representable mean loss must not overflow when reducing large row losses",
    )
    np.testing.assert_allclose(
        grad,
        np.tile([1.0 / 3, -1.0 / 3], (3, 1)),
        atol=1e-12,
        err_msg="extreme finite logits must keep the prediction-minus-target gradient",
    )


def test_opposite_float64_extremes_keep_correct_class_loss_finite_without_warnings():
    largest = np.finfo(np.float64).max
    loss, grad = _call(np.array([[largest, -largest]]), np.array([0]))
    assert loss == 0.0, "a saturated correct prediction must retain its representable zero loss"
    np.testing.assert_array_equal(
        grad,
        [[0.0, 0.0]],
        err_msg="a saturated correct prediction must have zero gradient without NaN",
    )


def test_batch_mean_can_be_finite_even_when_one_true_class_gap_is_not():
    largest = np.finfo(np.float64).max
    logits = np.array([[largest, -largest], [0.0, -largest], [0.0, -largest]])
    loss, _ = _call(logits, np.array([1, 0, 0]))
    np.testing.assert_allclose(
        loss,
        (largest / 3) * 2,
        rtol=1e-12,
        err_msg="scale an unrepresentable true-class gap before reduction if its mean is finite",
    )


def test_unrepresentable_mean_loss_returns_infinity_without_corrupting_gradient():
    largest = np.finfo(np.float64).max
    loss, grad = _call(np.array([[largest, -largest]]), np.array([1]))
    assert np.isposinf(loss), "a mean loss beyond float64 range must return positive infinity"
    np.testing.assert_array_equal(
        grad,
        [[1.0, -1.0]],
        err_msg="an out-of-range loss must still have a finite prediction-minus-target gradient",
    )


def test_uniform_huge_common_offset_retains_log_class_count_loss():
    loss, _ = _call(np.full((2, 3), 1e20), np.array([0, 2]))
    np.testing.assert_allclose(
        loss,
        np.log(3.0),
        rtol=1e-12,
        err_msg="a huge common logit offset must not erase the categorical log normalizer",
    )


def test_a_negative_target_anywhere_in_the_batch_is_rejected():
    try:
        cross_entropy_from_logits(np.zeros((2, 2)), np.array([0, -1]))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            "a batch with a negative class index must raise ValueError; "
            f"the function raised {type(exc).__name__}"
        )
    pytest.fail("a target below zero in any row must raise ValueError, not return a loss")


def test_a_too_large_target_anywhere_in_the_batch_is_rejected():
    try:
        cross_entropy_from_logits(np.zeros((2, 2)), np.array([0, 2]))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            "a batch with a class index past the last class must raise ValueError; "
            f"the function raised {type(exc).__name__}"
        )
    pytest.fail("a target at or above the class count in any row must raise ValueError")


def test_scalar_logits_raise_value_error():
    try:
        cross_entropy_from_logits(np.array(1.0), np.array([0]))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            "logits that are not a 2-D array must raise ValueError; "
            f"the function raised {type(exc).__name__}"
        )
    pytest.fail("logits that are not a 2-D array must raise ValueError, not return a loss")
