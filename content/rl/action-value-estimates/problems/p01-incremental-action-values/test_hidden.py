from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import action_value_estimates

from dogma_data import retail_events


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


ATOL = 1e-9
RTOL = 1e-9


def _call(actions, rewards, n_actions, step_size=None, initial_value=0.0):
    try:
        return action_value_estimates(actions, rewards, n_actions, step_size, initial_value)
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(f"valid input raised {type(exc).__name__}: check shapes and indices")


def _rejects(*args, **kwargs):
    try:
        action_value_estimates(*args, **kwargs)
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception type")
    pytest.fail("invalid input was accepted: check the argument and dtype validation")


def _batch_means(actions, rewards, n_actions, initial_value):
    """Independent oracle: the mean of each action's rewards, or the initial value if unpulled."""
    out = np.full(n_actions, float(initial_value))
    for a in range(n_actions):
        mask = actions == a
        if mask.any():
            out[a] = float(np.mean(rewards[mask]))
    return out


def _closed_form_constant(actions, rewards, n_actions, alpha, initial_value):
    """Independent oracle: Q_n = (1-alpha)^n Q_0 + sum alpha (1-alpha)^(n-i) r_i per action."""
    out = np.empty(n_actions)
    for a in range(n_actions):
        seq = rewards[actions == a]
        n = len(seq)
        weights = alpha * (1 - alpha) ** (n - 1 - np.arange(n))
        out[a] = (1 - alpha) ** n * initial_value + float(np.sum(weights * seq))
    return out


@functools.cache
def _pulls():
    """Real pulls: the weekday and the amount of every invoice line of Online Retail II.

    A pull is one real order line. The arm is the weekday the order was placed, counted
    modulo the number of arms, and the reward is what the line paid. Amounts are skewed,
    repeat often, and are negative for refunds; busy weekdays are pulled more often.
    """
    lines = retail_events()
    weekdays = lines["ts"].dt.day_of_week.to_numpy(dtype=np.int64)
    amounts = lines["amount"].to_numpy(dtype=np.float64)
    weekdays.setflags(write=False)
    amounts.setflags(write=False)
    return weekdays, amounts


def _episode(rng, n_actions, length):
    """``length`` real pulls picked by ``rng``: (actions, rewards)."""
    weekdays, amounts = _pulls()
    picked = rng.integers(0, amounts.shape[0], size=length)
    return weekdays[picked] % n_actions, amounts[picked].copy()


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    st.integers(1, 5),
    st.integers(0, 60),
    st.integers(0, 10_000),
    st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
)
def test_sample_average_equals_batch_mean(n_actions, length, seed, initial_value):
    rng = np.random.default_rng(seed)
    actions, rewards = _episode(rng, n_actions, length)
    got = _call(actions, rewards, n_actions, None, initial_value)
    want = _batch_means(actions, rewards, n_actions, initial_value)
    _close(
        got,
        want,
        atol=ATOL,
        rtol=RTOL,
        err_msg="sample-average estimate differs from the batch mean: "
        "check that n counts this pull",
    )


def test_constant_step_size_matches_exponential_weights():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n_actions = int(rng.integers(1, 4))
        length = int(rng.integers(1, 40))
        actions, rewards = _episode(rng, n_actions, length)
        rewards = rewards / 100.0
        alpha = float(rng.uniform(0.05, 1.0))
        got = _call(actions, rewards, n_actions, alpha, 0.7)
        want = _closed_form_constant(actions, rewards, n_actions, alpha, 0.7)
        _close(
            got,
            want,
            atol=ATOL,
            rtol=RTOL,
            err_msg="constant step size disagrees with the exponential-weight form: "
            "check the (1 - alpha) decay of past rewards",
        )


def test_first_sample_average_pull_overwrites_initial_value():
    got_a = _call(np.array([0]), np.array([2.5]), 1, None, 100.0)
    got_b = _call(np.array([0]), np.array([2.5]), 1, None, -100.0)
    _equal(got_a, [2.5], err_msg="first pull must replace the initial value")
    _equal(got_b, [2.5], err_msg="first pull must replace the initial value")


def test_sample_average_forgets_initial_value_after_every_action_pulled():
    actions = np.array([0, 1, 0, 1, 1])
    rewards = np.array([1.0, -2.0, 3.0, 4.0, 0.5])
    a = _call(actions, rewards, 2, None, 0.0)
    b = _call(actions, rewards, 2, None, 50.0)
    _close(a, b, atol=ATOL, rtol=RTOL, err_msg="initial value must not affect pulled actions")


def test_optimistic_start_decays_toward_reward_with_constant_step():
    got = _call(np.array([0]), np.array([0.0]), 2, 0.1, 10.0)
    _close(
        got,
        [9.0, 10.0],
        atol=ATOL,
        rtol=RTOL,
        err_msg="optimistic start must move by alpha * (reward - estimate) on the first pull",
    )


def test_step_size_one_keeps_only_last_reward_per_action():
    actions = np.array([0, 1, 0, 1, 0])
    rewards = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    got = _call(actions, rewards, 2, 1.0, 0.0)
    _equal(got, [9.0, 8.0], err_msg="alpha=1 keeps only the latest reward")


def test_empty_stream_returns_initial_values():
    got = _call(np.array([], dtype=np.int64), np.array([]), 3, None, 2.5)
    _equal(got, [2.5, 2.5, 2.5], err_msg="empty stream must do nothing")
    _expect(
        got.dtype == np.float64,
        "estimates must be float64",
    )


def test_single_action_problem_is_the_running_mean():
    rewards = np.array([1.0, 2.0, 6.0])
    got = _call(np.zeros(3, dtype=np.int64), rewards, 1)
    _close(got, [3.0], atol=ATOL, rtol=RTOL)


def test_large_magnitudes_stay_finite_and_accurate():
    rewards = np.array([1e150, 2e150, 3e150, 4e150])
    got = _call(np.zeros(4, dtype=np.int64), rewards, 1)
    _expect(
        np.all(np.isfinite(got)),
        "large rewards must not overflow the estimate",
    )
    _expect(
        math.isclose(got[0], float(np.mean(rewards)), rel_tol=1e-9),
        "large-magnitude estimate lost relative accuracy",
    )


def test_tiny_magnitudes_keep_relative_accuracy():
    rewards = np.array([1e-200, 3e-200, 5e-200])
    got = _call(np.zeros(3, dtype=np.int64), rewards, 1)
    _expect(
        math.isclose(got[0], 3e-200, rel_tol=1e-9),
        "tiny rewards lost relative accuracy",
    )


def test_integer_rewards_return_float64():
    got = _call(np.array([0, 0]), np.array([1, 2]), 1)
    _expect(
        got.dtype == np.float64,
        "estimates must be float64 even for integer rewards",
    )
    _close(got, [1.5], atol=ATOL)


def test_inputs_are_not_mutated():
    actions = np.array([0, 1, 1])
    # Negative rewards too, so an in-place sign change or clip is visible in the input.
    rewards = np.array([1.0, -2.0, 3.0])
    a0, r0 = actions.copy(), rewards.copy()
    _call(actions, rewards, 2, 0.3, 1.0)
    _call(actions, rewards, 2, None, 0.0)
    _equal(actions, a0, err_msg="the actions array was modified")
    _equal(rewards, r0, err_msg="the rewards array was modified")


def test_invalid_step_size_rejected():
    actions = np.array([0])
    rewards = np.array([1.0])
    for step_size in (0.0, -0.1, 1.5, float("nan"), True):
        _rejects(actions, rewards, 1, step_size=step_size)


def test_invalid_arguments_rejected():
    _rejects(np.array([0]), np.array([1.0]), 0)
    _rejects(np.array([0]), np.array([1.0]), True)
    _rejects(np.array([1]), np.array([1.0]), 1)
    _rejects(np.array([-1]), np.array([1.0]), 1)
    _rejects(np.array([0]), np.array([1.0]), 1, initial_value=float("inf"))


def test_invalid_pulls_rejected():
    _rejects(np.array([0.0]), np.array([1.0]), 1)
    _rejects(np.array([0, 0]), np.array([1.0]), 1)
    _rejects(np.array([[0]]), np.array([[1.0]]), 1)
    _rejects(np.array([0]), np.array([np.nan]), 1)
    _rejects(np.array([0]), np.array([1 + 1j]), 1)
    _rejects(np.array([0]), np.array([True]), 1)


def test_negative_action_index_is_rejected_after_a_valid_index():
    _rejects(np.array([0, -1]), np.array([1.0, 2.0]), 2)


def test_action_index_past_the_end_is_rejected_before_a_valid_index():
    _rejects(np.array([0, 2]), np.array([1.0, 2.0]), 2)


def test_zero_actions_is_rejected_even_for_an_empty_stream():
    _rejects(np.array([], dtype=np.int64), np.array([], dtype=np.float64), 0)


def test_unsigned_integer_actions_are_valid():
    # uint8 is an integer array, which the Contract accepts. Pulls: action 0 gets 2.0,
    # action 1 gets 6.0, each mean is the single reward.
    got = _call(np.array([0, 1], dtype=np.uint8), np.array([2.0, 6.0]), 2)
    _close(got, [2.0, 6.0], atol=ATOL, err_msg="unsigned integer action indices must be accepted")
