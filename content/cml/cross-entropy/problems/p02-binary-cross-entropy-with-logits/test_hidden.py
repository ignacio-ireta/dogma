import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import log_expit
from submission import binary_cross_entropy_with_logits

from dogma_data import rng, sample_rows


def _call(z, y):
    """Call the loss with NumPy warnings turned into errors.

    Hidden failures show only the first line of the message, so each line names the mistake and
    never includes the input, its shape or the expected value.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = binary_cross_entropy_with_logits(z, y)
            if type(got) is not float:
                pytest.fail("the mean loss must be a Python float")
            return got
    except RuntimeWarning:
        pytest.fail(
            "binary cross-entropy overflowed or took the log of a zero probability: use the form "
            "max(z, 0) - z y + log1p(exp(-|z|)), which never forms a probability"
        )
    except Exception as exc:
        pytest.fail(
            "binary cross-entropy must accept valid logits and targets; the call raised "
            f"{type(exc).__name__}"
        )


def _expect_value_error(z, y):
    try:
        binary_cross_entropy_with_logits(z, y)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"invalid input must raise ValueError; the function raised {type(exc).__name__}"
        )
    pytest.fail("invalid input must raise ValueError, not return a loss")


def _decimal_loss(z, y):
    """Independent reference: -y ln sigma(z) - (1 - y) ln sigma(-z), with 60 significant digits."""
    with localcontext() as ctx:
        ctx.prec = 60
        zz, yy = Decimal(z), Decimal(y)
        sigma_pos = 1 / (1 + (-zz).exp())
        sigma_neg = 1 / (1 + zz.exp())
        return float(-yy * sigma_pos.ln() - (1 - yy) * sigma_neg.ln())


def test_moderate_logits_match_the_scipy_log_sigmoid_reference():
    # A real linear score: standardised breast_cancer features projected on the direction from
    # the malignant mean to the benign mean of the drawn rows. Forty labels are smoothed (soft
    # targets), ten stay hard.
    X, label = sample_rows("breast_cancer", rng("bce-p02:reference"), 50, 6, standardize=True)
    z = X @ (X[label == 1].mean(axis=0) - X[label == 0].mean(axis=0))
    y = 0.9 * label + 0.05
    y[:10] = label[:10]
    want = float(np.mean(-y * log_expit(z) - (1.0 - y) * log_expit(-z)))
    got = _call(z, y)
    assert abs(got - want) <= 1e-12, (
        "mean binary cross-entropy disagrees with the reference: check the sign of the logit "
        "and the (1 - y) term"
    )


@pytest.mark.parametrize(
    ("z", "y"),
    [(40.0, 1.0), (-40.0, 0.0), (800.0, 0.0), (-800.0, 1.0), (1000.0, 0.3), (-1000.0, 0.7)],
    ids=["extreme-1", "extreme-2", "extreme-3", "extreme-4", "extreme-5", "extreme-6"],
)
def test_extreme_logits_match_the_high_precision_reference(z, y):
    got = _call(np.array([z]), np.array([y]))
    want = _decimal_loss(z, y)
    assert np.isfinite(got), (
        "an extreme logit gave inf or NaN: never form a probability that rounds to 0 or 1"
    )
    assert abs(got - want) <= 1e-12 + 1e-12 * abs(want), (
        "the loss is wrong at an extreme logit: the probability-based formula loses the tail, so "
        "compute from the logit with max(z, 0) and log1p(exp(-|z|))"
    )


def test_soft_targets_match_the_reference():
    z = np.array([0.0, 1.5, -2.5, 4.0])
    y = np.array([0.5, 0.1, 0.95, 0.25])
    want = float(np.mean([_decimal_loss(zi, yi) for zi, yi in zip(z, y, strict=True)]))
    got = _call(z, y)
    assert abs(got - want) <= 1e-12, "soft targets in [0, 1] must be accepted and weighted linearly"


def test_zero_logit_with_half_target_is_log_two():
    got = _call(np.array([0.0]), np.array([0.5]))
    assert abs(got - np.log(2.0)) <= 1e-12, "a logit of 0 with target 1/2 gives loss log 2"


def test_the_loss_is_a_mean_over_examples():
    z = np.array([1.0, -2.0, 0.5])
    y = np.array([1.0, 0.0, 1.0])
    single = _call(z[:1], y[:1])
    repeated = _call(np.repeat(z[:1], 5), np.repeat(y[:1], 5))
    assert abs(repeated - single) <= 1e-12, (
        "repeating an example must not change the mean loss: the loss is a mean"
    )
    assert (
        abs(_call(z, y) - np.mean([_call(z[i : i + 1], y[i : i + 1]) for i in range(3)])) <= 1e-12
    ), "the loss over a batch must equal the mean of the per-example losses"


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.floats(min_value=-20.0, max_value=20.0, allow_nan=False), min_size=1, max_size=8),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_random_moderate_inputs_match_the_reference_and_are_non_negative(values, target):
    z = np.array(values)
    y = np.full(z.shape, target)
    want = float(np.mean(-y * log_expit(z) - (1.0 - y) * log_expit(-z)))
    got = _call(z, y)
    assert got >= 0.0, "the mean binary cross-entropy is never negative"
    assert abs(got - want) <= 1e-12, "random moderate inputs disagree with the reference"


@pytest.mark.parametrize(
    ("z", "y"),
    [
        (np.array([0.0, 1.0]), np.array([1.0])),
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.zeros(0), np.zeros(0)),
        (np.array([[0.0], [1.0]]), np.array([1.0, 0.0])),
        (np.array([np.nan, 1.0]), np.array([1.0, 0.0])),
        (np.array([np.inf, 1.0]), np.array([1.0, 0.0])),
        (np.array([0.0, 1.0]), np.array([np.nan, 0.0])),
        (np.array([0.0, 1.0]), np.array([1.5, 0.0])),
        (np.array([0.0, 1.0]), np.array([-0.1, 0.0])),
        (np.array([0.0, 1.0]), np.array([0.0, np.inf])),
    ],
    ids=[
        "length-mismatch",
        "two-dimensional",
        "empty",
        "two-dimensional-logits",
        "nan-logit",
        "infinite-logit",
        "nan-target",
        "target-above-one",
        "target-below-zero",
        "infinite-target",
    ],
)
def test_invalid_inputs_raise_value_error(z, y):
    _expect_value_error(z, y)


def test_integer_inputs_return_a_float():
    got = binary_cross_entropy_with_logits(np.array([0, 2]), np.array([1, 0]))
    assert isinstance(got, float), "the loss must be a Python float"
    assert abs(got - _call(np.array([0.0, 2.0]), np.array([1.0, 0.0]))) <= 1e-12, (
        "integer inputs must give the same loss as their float equivalents"
    )


def test_inputs_are_not_mutated():
    z = np.array([800.0, -3.0, 0.5])
    y = np.array([0.0, 1.0, 0.25])
    z_before, y_before = z.copy(), y.copy()
    _call(z, y)
    np.testing.assert_array_equal(z, z_before, err_msg="the logits were modified in place")
    np.testing.assert_array_equal(y, y_before, err_msg="the targets were modified in place")


def test_mean_of_large_finite_losses_does_not_overflow_the_reduction():
    got = _call(np.full(3, 1e308), np.zeros(3))
    np.testing.assert_allclose(
        got,
        1e308,
        rtol=1e-12,
        err_msg="a representable mean loss must not overflow when summing large finite losses",
    )


def test_soft_target_near_one_avoids_cancelling_large_positive_logit_terms():
    target = np.nextafter(1.0, 0.0)
    got = _call(np.array([1e300]), np.array([target]))
    np.testing.assert_allclose(
        got,
        1e300 * (1.0 - target),
        rtol=1e-12,
        err_msg="a nearly-one soft target must retain the small negative-class weighted loss",
    )


def test_confident_correct_predictions_preserve_the_small_positive_loss():
    got = _call(np.array([40.0]), np.array([1.0]))
    np.testing.assert_allclose(
        got,
        _decimal_loss(40.0, 1.0),
        rtol=1e-12,
        atol=0.0,
        err_msg="a correct confident prediction must retain its positive tail loss",
    )


def test_mean_of_losses_of_very_different_sizes_does_not_overflow():
    got = _call(np.array([1e308, -30.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        got,
        5e307,
        rtol=1e-12,
        err_msg="the mean of a huge loss and a tiny loss is representable and must not overflow",
    )


def test_list_inputs_match_their_array_equivalents():
    # z and y are 1-D array-likes, so plain Python lists are valid inputs.
    got = _call([0.0, 2.0, -1.0], [1.0, 0.0, 0.5])
    want = _call(np.array([0.0, 2.0, -1.0]), np.array([1.0, 0.0, 0.5]))
    assert abs(got - want) <= 1e-12, (
        "list inputs must give the same loss as their array equivalents"
    )


def test_read_only_input_arrays_are_accepted_and_left_alone():
    z = np.array([800.0, -3.0, 0.5])
    y = np.array([0.0, 1.0, 0.25])
    for array in (z, y):
        array.setflags(write=False)
    try:
        binary_cross_entropy_with_logits(z, y)
    except Exception as exc:
        if "read-only" in str(exc):
            pytest.fail(
                "an input array was written to in place: inputs must not be mutated, so work "
                "on a copy",
                pytrace=False,
            )
        pytest.fail(
            "a valid call failed when its input arrays were read-only: a correct solution "
            "only reads its inputs",
            pytrace=False,
        )
