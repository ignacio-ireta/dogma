from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import discounted_returns

from dogma_data import rng as data_rng
from dogma_data import sample_window


def _close(
    actual, desired, *, err_msg="values differ from the expected result", atol=0.0, rtol=1e-7
):
    """numpy.testing.assert_allclose, but the first failure line is the named message."""
    if np.shape(actual) != np.shape(desired):
        pytest.fail(f"{err_msg} (shape differs from the contract)", pytrace=False)
    if not np.allclose(actual, desired, atol=atol, rtol=rtol, equal_nan=False):
        pytest.fail(err_msg, pytrace=False)


def _equal(actual, desired, *, err_msg="values differ from the expected result"):
    """numpy.testing.assert_array_equal, but the first failure line is the named message."""
    if np.shape(actual) != np.shape(desired) or not np.array_equal(actual, desired):
        pytest.fail(err_msg, pytrace=False)


def _expect(condition, message):
    if not condition:
        pytest.fail(message, pytrace=False)


def _call(rewards, gamma):
    """Run the learner's function and report any failure as the kind of mistake it is."""
    try:
        return discounted_returns(rewards, gamma)
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(f"valid input raised {type(exc).__name__}: check indexing and dtypes")


def _rejects(rewards, gamma):
    try:
        discounted_returns(rewards, gamma)
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception type")
    pytest.fail("invalid input was accepted: check the rewards and gamma validation")


def _forward_reference(rewards, gamma):
    """Independent oracle: the defining double sum, no recurrence."""
    total = len(rewards)
    return np.array(
        [sum((gamma**k) * float(rewards[t + k]) for k in range(total - t)) for t in range(total)]
    )


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    st.integers(0, 30),
    st.floats(0.0, 1.0, allow_nan=False),
    st.integers(0, 10_000),
)
def test_random_episodes_match_forward_definition(length, gamma, seed):
    # Real rewards: month-to-month changes of El Niño sea-surface temperature, in degrees.
    rewards = np.diff(sample_window("elnino", np.random.default_rng(seed), length + 1))
    got = _call(rewards, gamma)
    want = _forward_reference(rewards, gamma)
    _expect(
        got.shape == want.shape,
        "output shape must be (T,), one return per step",
    )
    _close(
        got,
        want,
        rtol=1e-9,
        atol=1e-9,
        err_msg="returns disagree with the defining discounted sum: check the recurrence order",
    )


def test_dyadic_discount_matches_exact_rational_arithmetic():
    # gamma = 1/2 and integer rewards keep every intermediate value exactly representable in
    # float64 (at most 40 fractional bits for 40 steps, and under 11 integer bits), so an
    # exact result is required. The rewards are real whole numbers: year-to-year changes of
    # the annual flow of the Nile.
    rng = data_rng("p01-discounted-returns:dyadic")
    for _ in range(20):
        length = int(rng.integers(0, 41))
        rewards = np.diff(sample_window("nile", rng, length + 1)).astype(np.int64)
        exact = []
        for t in range(length):
            total = Fraction(0)
            for k in range(length - t):
                total += Fraction(int(rewards[t + k])) / (2**k)
            exact.append(float(total))
        got = _call(rewards, 0.5)
        if not np.array_equal(got, np.array(exact, dtype=np.float64)):
            pytest.fail("dyadic-discount returns are not exact: check the reverse recurrence")


def test_gamma_zero_returns_the_rewards_themselves():
    rewards = np.array([3.0, -1.5, 0.25, 8.0])
    got = _call(rewards, 0.0)
    _equal(got, rewards, err_msg="gamma=0 must return the immediate rewards")


def test_gamma_one_matches_reversed_cumulative_sum():
    rewards = np.diff(sample_window("nile", data_rng("p01-discounted-returns:suffix"), 26))
    got = _call(rewards, 1.0)
    _equal(
        got,
        np.cumsum(rewards[::-1])[::-1],
        err_msg="gamma=1 must give the suffix sum of rewards from each step",
    )


def test_single_step_episode_returns_its_reward():
    _equal(_call(np.array([-2.5]), 0.9), [-2.5])


def test_integer_rewards_return_float64():
    out = _call(np.array([1, 2, 3]), 0.5)
    _expect(
        out.dtype == np.float64,
        "return float64 returns even for integer rewards",
    )
    _close(out, [2.75, 3.5, 3.0])


def test_large_magnitudes_stay_finite_and_accurate():
    rewards = np.full(100, 1e300)
    gamma = 0.9
    got = _call(rewards, gamma)
    _expect(
        np.all(np.isfinite(got)),
        "large rewards must not overflow the return",
    )
    expected = 1e300 * (1 - gamma**100) / (1 - gamma)
    _expect(
        math.isclose(got[0], expected, rel_tol=1e-9),
        "large-magnitude return is inaccurate",
    )


def test_tiny_magnitudes_keep_relative_accuracy():
    rewards = np.full(50, 1e-200)
    gamma = 0.5
    got = _call(rewards, gamma)
    expected = 1e-200 * (1 - gamma**50) / (1 - gamma)
    _expect(
        math.isclose(got[0], expected, rel_tol=1e-9),
        "tiny rewards lost relative accuracy",
    )


def test_inputs_are_not_mutated():
    rewards = np.array([1.0, 2.0, 3.0])
    original = rewards.copy()
    _call(rewards, 0.7)
    _call(rewards, 0.0)
    _equal(rewards, original, err_msg="the rewards array was modified")


def test_invalid_gamma_rejected():
    for gamma in (-0.1, 1.1, float("nan"), float("inf"), True):
        _rejects(np.array([1.0]), gamma)


def test_invalid_rewards_rejected():
    _rejects(np.ones((2, 2)), 0.9)
    _rejects(np.array([1 + 2j]), 0.9)
    _rejects(np.array([True, False]), 0.9)
    _rejects(np.array([1.0, float("nan")]), 0.9)
    _rejects(np.array([1.0, float("inf")]), 0.9)
    _rejects(np.array(["a", "b"]), 0.9)


def test_read_only_input_arrays_are_accepted_and_left_alone():
    rewards = np.array([1.0, 2.0, 3.0])
    for array in (rewards,):
        array.setflags(write=False)
    try:
        discounted_returns(rewards, 0.7)
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
