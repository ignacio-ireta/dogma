import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import expit
from scipy.special import logit as scipy_logit
from submission import logit


def _call(p):
    """Call logit(p) with NumPy warnings turned into errors.

    Hidden failures show only the first line of the message, so each line names the mistake and
    never includes the input, its shape or the expected value.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = logit(p)
            if not isinstance(got, np.ndarray) or got.dtype != np.float64:
                pytest.fail("return a float64 NumPy array without caller-side conversion")
            return got
    except RuntimeWarning:
        pytest.fail(
            "logit emitted a NumPy warning on a valid probability: avoid dividing by, or taking "
            "the logarithm of, a value that underflows or is zero"
        )
    except Exception as exc:
        pytest.fail(
            f"logit must accept an array of probabilities; the call raised {type(exc).__name__}"
        )


def _expect_value_error(value):
    try:
        logit(value)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"invalid probability must raise ValueError; the function raised {type(exc).__name__}"
        )
    pytest.fail(
        "an input outside the open interval (0, 1) must raise ValueError, not return a value"
    )


def _logit_decimal(p):
    """Independent reference: log(p / (1 - p)) with 60 significant digits, for float p."""
    with localcontext() as ctx:
        ctx.prec = 60
        x = Decimal(p)
        return float((x / (1 - x)).ln())


def test_random_probabilities_match_an_independent_reference():
    rng = np.random.default_rng(0)
    p = rng.uniform(1e-6, 1.0 - 1e-6, size=300)
    expected = scipy_logit(p)
    np.testing.assert_allclose(
        _call(p),
        expected,
        rtol=1e-9,
        atol=1e-12,
        err_msg="logit disagrees with the reference on random probabilities: check the "
        "formula log(p / (1 - p))",
    )


def test_one_half_is_exactly_zero():
    got = _call(np.array([0.5]))
    assert got[0] == 0.0, "logit(1/2) must be exactly 0, not a rounding residue"


@pytest.mark.parametrize(
    "p",
    [0.5 + 2.9e-9, 0.5 - 1.3e-11, 0.5 - 4.1e-10],
    ids=["near-half-1", "near-half-2", "near-half-3"],
)
def test_near_one_half_keeps_relative_precision(p):
    got = _call(np.array([p]))[0]
    want = _logit_decimal(p)
    assert abs(got - want) <= 1e-9 * abs(want), (
        "logit lost relative precision near 1/2: the ratio p / (1 - p) is close to 1, so write "
        "the result with log1p((2p - 1) / (1 - p)) and never as log of that ratio"
    )


@pytest.mark.parametrize(
    "p",
    [1e-12, 1e-300, 2.0**-1074],
    ids=["near-zero-1", "near-zero-2", "near-zero-3"],
)
def test_tiny_probabilities_keep_relative_precision(p):
    got = _call(np.array([p]))[0]
    want = _logit_decimal(p)
    assert abs(got - want) <= 1e-9 * abs(want), (
        "logit lost relative precision for a tiny probability: a formula built from 2p - 1 or "
        "from a log1p of a value near -1 cancels here"
    )


@pytest.mark.parametrize(
    "p",
    [1.0 - 2**-53, 1.0 - 2**-40, 1.0 - 1e-12],
    ids=["near-one-1", "near-one-2", "near-one-3"],
)
def test_probabilities_near_one_keep_relative_precision(p):
    got = _call(np.array([p]))[0]
    want = _logit_decimal(p)
    assert abs(got - want) <= 1e-9 * abs(want), (
        "logit lost relative precision for a probability near 1: the value is large and positive, "
        "so its sign and magnitude must come from 1 - p, which is exact for p >= 1/2"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.floats(min_value=0.5, max_value=1.0, exclude_max=True))
def test_logit_is_odd_about_one_half(p):
    np.testing.assert_allclose(
        _call(np.array([1.0 - p])),
        -_call(np.array([p])),
        rtol=1e-12,
        atol=0.0,
        err_msg="logit must be odd about 1/2: logit(1 - p) = -logit(p)",
    )


def test_round_trip_through_the_logistic_function_recovers_the_logit():
    z = np.linspace(-10.0, 10.0, 81)
    np.testing.assert_allclose(
        _call(expit(z)),
        z,
        rtol=0.0,
        atol=1e-9,
        err_msg="logit does not invert the logistic function: check the sign and the "
        "scaling of log(p / (1 - p))",
    )


@pytest.mark.parametrize(
    "value",
    [0.0, 1.0, -0.25, 1.5, float("nan"), float("inf")],
    ids=["zero", "one", "negative", "above-one", "not-a-number", "infinite"],
)
def test_out_of_range_probabilities_raise_value_error(value):
    _expect_value_error(value)


def test_one_bad_entry_in_an_array_raises_value_error():
    _expect_value_error(np.array([0.3, 0.6, 1.0]))


def test_empty_input_gives_an_empty_array():
    got = _call(np.zeros((0, 2)) + 0.5)
    assert got.shape == (0, 2), "shape mismatch: empty input must give an empty array"


def test_float32_input_returns_float64_with_correct_values():
    got = _call(np.array([0.25, 0.75], dtype=np.float32))
    assert got.dtype == np.float64, "dtype should be float64 even for float32 probabilities"
    np.testing.assert_allclose(
        got, [-np.log(3.0), np.log(3.0)], rtol=1e-6, err_msg="float32 input gave wrong values"
    )


def test_input_array_is_not_mutated():
    p = np.array([0.1, 0.5, 0.9])
    before = p.copy()
    _call(p)
    np.testing.assert_array_equal(p, before, err_msg="the input array was modified in place")


def test_scalar_probability_gives_a_zero_dimensional_array():
    got = _call(0.25)
    assert got.ndim == 0, "shape mismatch: a scalar input must give a 0-d array, not a 1-d one"
    np.testing.assert_allclose(
        got,
        -np.log(3.0),
        rtol=1e-12,
        err_msg="wrong value for a scalar probability",
    )
