import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import expit
from submission import sigmoid

FINITE = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


def _evaluate(z):
    """Call sigmoid(z) with overflow and invalid-value warnings turned into errors.

    Hidden failures show only the first line of the message, so each line names the mistake and
    never includes the input, its shape or the expected value.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = sigmoid(z)
            if not isinstance(got, np.ndarray) or got.dtype != np.float64:
                pytest.fail("return a float64 NumPy array without caller-side conversion")
            return got
    except RuntimeWarning:
        pytest.fail(
            "sigmoid overflowed or raised an invalid-value warning on a finite input: "
            "never exponentiate a large positive number"
        )
    except Exception as exc:
        pytest.fail(
            f"sigmoid must accept an array of real numbers; the call raised {type(exc).__name__}"
        )


def _sigmoid_decimal(z):
    """Independent reference: sigma(z) with 60 significant digits, for a finite float z."""
    with localcontext() as ctx:
        ctx.prec = 60
        return float(1 / (1 + (-Decimal(z)).exp()))


def test_random_inputs_match_an_independent_stable_reference():
    rng = np.random.default_rng(0)
    z = rng.uniform(-50.0, 50.0, size=200)
    expected = expit(z)
    np.testing.assert_allclose(
        _evaluate(z),
        expected,
        rtol=1e-12,
        atol=1e-15,
        err_msg="sigmoid disagrees with the reference on random logits: check the sign "
        "convention and the branch that handles negative z",
    )


@pytest.mark.parametrize(
    "z",
    [-1e4, -1e3, -745.0, -710.0, 710.0, 745.0, 1e3, 1e4],
    ids=[
        "extreme-1",
        "extreme-2",
        "extreme-3",
        "extreme-4",
        "extreme-5",
        "extreme-6",
        "extreme-7",
        "extreme-8",
    ],
)
def test_extreme_logits_stay_finite_and_warning_free(z):
    got = _evaluate(np.array([z]))
    assert np.all(np.isfinite(got)), "a finite logit produced inf or NaN instead of a probability"
    assert np.all((got >= 0.0) & (got <= 1.0)), "sigmoid values must lie in [0, 1]"


@pytest.mark.parametrize(
    "z",
    [-20.0, -36.0, -40.0, -700.0],
    ids=["deep-tail-1", "deep-tail-2", "deep-tail-3", "deep-tail-4"],
)
def test_negative_tail_keeps_relative_precision(z):
    got = _evaluate(np.array([z]))
    want = _sigmoid_decimal(z)
    assert abs(got[0] - want) <= 1e-12 * abs(want), (
        "the negative tail lost relative precision: compute e / (1 + e) with e = exp(-|z|), "
        "not 1 - 1 / (1 + e), which rounds the tail to a wrong value"
    )


@pytest.mark.parametrize("z", [20.0, 36.0], ids=["small-side-1", "small-side-2"])
def test_small_side_of_the_symmetry_keeps_relative_precision(z):
    got = _evaluate(np.array([-z]))
    want = _sigmoid_decimal(-z)
    assert abs(got[0] - want) <= 1e-12 * abs(want), (
        "sigma(-z) for large z must keep its relative precision: it is the small value, so "
        "1 - sigma(z) is the wrong way to compute it"
    )


def test_sigmoid_of_zero_is_exactly_one_half_for_both_signed_zeros():
    got = _evaluate(np.array([0.0, -0.0]))
    assert np.all(got == 0.5), "sigma(0) must be exactly 0.5 for both signed zeros"


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.lists(FINITE, min_size=1, max_size=12))
def test_symmetry_holds_in_the_bulk_of_the_input_range(values):
    z = np.array(values)
    gap = np.abs(_evaluate(-z) + _evaluate(z) - 1.0)
    assert np.all(gap <= 1e-15), (
        "sigma(-z) and 1 - sigma(z) disagree: check both branches of the formula"
    )


def test_infinite_inputs_give_the_exact_saturated_limits():
    got = _evaluate(np.array([np.inf, -np.inf]))
    np.testing.assert_array_equal(
        got,
        [1.0, 0.0],
        err_msg="infinite logits must give exact limits: a clip or a floor on the logit or the "
        "output moves sigma(-inf) away from 0",
    )


def test_nan_input_raises_value_error():
    try:
        sigmoid(np.array([0.5, np.nan]))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"NaN input must raise ValueError; the function raised {type(exc).__name__}")
    pytest.fail("NaN input must raise ValueError, not return a value")


def test_empty_scalar_and_matrix_shapes_are_preserved():
    assert _evaluate(np.zeros((0, 3))).shape == (0, 3), "shape mismatch: empty input"
    assert _evaluate(np.zeros((2, 3))).shape == (2, 3), "shape mismatch: matrix input"
    assert _evaluate(1.5).shape == (), "shape mismatch: a scalar must give a 0-d array"


def test_integer_input_returns_float64():
    got = _evaluate(np.array([-3, 0, 3]))
    assert got.dtype == np.float64, "dtype should be float64 even for integer logits"
    np.testing.assert_allclose(
        got,
        expit(np.array([-3.0, 0.0, 3.0])),
        rtol=1e-12,
        err_msg="integer logits gave wrong values",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.lists(FINITE, min_size=1, max_size=12))
def test_output_is_a_probability_for_any_finite_logits(values):
    got = _evaluate(np.array(values))
    assert np.all(np.isfinite(got)), "output contains inf or NaN"
    assert np.all((got >= 0.0) & (got <= 1.0)), "output lies outside [0, 1]"


def test_input_array_is_not_mutated():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    before = z.copy()
    _evaluate(z)
    np.testing.assert_array_equal(z, before, err_msg="the input array was modified in place")
