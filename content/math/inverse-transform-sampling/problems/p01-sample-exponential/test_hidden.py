"""Hidden tests: exact agreement with the protocol, the KS check, and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or an expected
answer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from submission import sample_exponential


class _Uniforms:
    """A stand-in generator that hands out the given uniforms in order.

    ``random(n)`` returns the next ``n`` values, and ``random()`` returns the next value as a
    scalar. Every call is recorded, so a test can check the call pattern.
    """

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)
        self.pos = 0
        self.calls = []

    def random(self, n=None):
        self.calls.append(n)
        if n is None:
            value = self.values[self.pos]
            self.pos += 1
            return float(value)
        out = self.values[self.pos : self.pos + n].copy()
        self.pos += n
        return out


def _run(call, mistake):
    """Call the learner on valid inputs. An exception becomes a failure that names the mistake."""
    try:
        return call()
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")


def _raises_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@settings(max_examples=120, deadline=None, report_multiple_bugs=False)
@given(
    seed=st.integers(min_value=0, max_value=10**6),
    n=st.integers(min_value=0, max_value=200),
    rate=st.floats(min_value=0.01, max_value=100.0),
)
def test_draws_reproduce_the_inverse_cdf_of_the_seeded_uniforms(seed, n, rate):
    u = np.random.default_rng(seed).random(n)
    expected = -np.log(1.0 - u) / rate
    got = _run(
        lambda: sample_exponential(rate, n, np.random.default_rng(seed)),
        "the draws must be the inverse-CDF map of rng.random(n): sample_exponential raised",
    )
    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-12,
        atol=0.0,
        err_msg="draws differ from the protocol: use u = rng.random(n) once, "
        "then x = -log1p(-u) / rate",
    )


def test_formula_holds_for_supplied_uniforms_including_the_ends_of_the_interval():
    u = np.array([0.0, 0.5, 0.9, 0.999999, 1.0 - 2.0**-53])
    got = _run(
        lambda: sample_exponential(3.0, len(u), _Uniforms(u)),
        "each uniform must be mapped by -log(1 - u) / rate: sample_exponential raised",
    )
    expected = np.array([-math.log(1.0 - float(v)) / 3.0 for v in u])
    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-12,
        atol=1e-15,
        err_msg="the map is not -log(1 - u) / rate: check the transform and the rate division",
    )


def test_tiny_uniforms_keep_their_magnitude_in_the_draw():
    u = np.array([1e-18, 1e-12, 3e-16])
    got = _run(
        lambda: sample_exponential(2.0, len(u), _Uniforms(u)),
        "each uniform must be mapped by -log(1 - u) / rate: sample_exponential raised",
    )
    np.testing.assert_allclose(
        got,
        u / 2.0,
        rtol=1e-9,
        err_msg="draws for tiny uniforms collapsed toward zero: use log1p(-u), not log(1 - u)",
    )


def test_draws_come_from_one_call_to_random_with_n():
    generator = _Uniforms([0.1, 0.2, 0.3, 0.4])
    _run(
        lambda: sample_exponential(1.0, 4, generator),
        "the draws must come from rng.random(n) alone: sample_exponential raised",
    )
    assert generator.calls == [4], (
        "the draws must come from one call rng.random(n), not from several calls or another method"
    )


def test_ks_goodness_of_fit_against_the_exponential_at_a_fixed_seed():
    rate = 2.5
    draws = _run(
        lambda: sample_exponential(rate, 20_000, np.random.default_rng(7)),
        "valid inputs must not raise: sample_exponential raised",
    )
    p_value = stats.kstest(draws, "expon", args=(0.0, 1.0 / rate)).pvalue
    assert p_value > 0.01, (
        "the draws do not follow the exponential distribution: check the rate and the transform"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    seed=st.integers(min_value=0, max_value=10**6),
    n=st.integers(min_value=1, max_value=100),
    rate=st.floats(min_value=0.05, max_value=20.0),
)
def test_doubling_the_rate_halves_every_draw_under_the_same_seed(seed, n, rate):
    slow = np.asarray(
        _run(
            lambda: sample_exponential(rate, n, np.random.default_rng(seed)),
            "valid inputs must not raise: sample_exponential raised",
        )
    )
    fast = np.asarray(
        _run(
            lambda: sample_exponential(2.0 * rate, n, np.random.default_rng(seed)),
            "valid inputs must not raise: sample_exponential raised",
        )
    )
    np.testing.assert_allclose(
        fast,
        slow / 2.0,
        rtol=1e-12,
        err_msg="scaling the rate must scale every draw by its inverse: check the division by rate",
    )


def test_output_is_a_float64_array_of_length_n():
    draws = _run(
        lambda: sample_exponential(1.5, 7, np.random.default_rng(3)),
        "valid inputs must not raise: sample_exponential raised",
    )
    assert isinstance(draws, np.ndarray), "return a NumPy array, not a list"
    assert draws.shape == (7,), "the output must have one draw per requested sample"
    assert draws.dtype == np.float64, "draws must be float64"


@pytest.mark.parametrize(
    "bad_rate",
    [0.0, -1.0, float("nan"), float("inf")],
    ids=["zero", "negative", "not-a-number", "infinite"],
)
def test_rate_that_is_not_finite_and_positive_raises_value_error(bad_rate):
    _raises_value_error(
        lambda: sample_exponential(bad_rate, 5, np.random.default_rng(0)),
        "rate must be finite and positive: raise ValueError for zero, negative, NaN or inf rates",
    )


def test_negative_count_raises_value_error():
    # The stand-in does not reject a negative count itself, so the check must be the learner's.
    _raises_value_error(
        lambda: sample_exponential(1.0, -1, _Uniforms([0.1, 0.2])),
        "a negative number of draws is invalid: raise ValueError",
    )
