from __future__ import annotations

import signal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import iterative_policy_evaluation


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


ATOL = 1e-8
RTOL = 1e-8


def _call(P, R, pi, gamma, tol=1e-10):
    try:
        return iterative_policy_evaluation(P, R, pi, gamma, tol)
    except Exception as exc:  # a valid call must not fail with a generic error
        pytest.fail(f"valid input raised {type(exc).__name__}: check shapes and the update")


class _Hang(BaseException):
    """Raised by the alarm when a call hangs. A BaseException, so learner code cannot swallow it."""


def _on_alarm(signum, frame):
    raise _Hang


def _rejects(*args):
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(5)
    try:
        iterative_policy_evaluation(*args)
    except ValueError:
        return
    except _Hang:
        pytest.fail(
            "invalid input made the iteration run forever: check finite values, "
            "gamma < 1 and tol > 0"
        )
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception type")
    finally:
        signal.alarm(0)
    pytest.fail("invalid input was accepted: check the shape, probability and discount rules")


def _policy_model(P, R, pi):
    """Independent oracle pieces: the policy-averaged transition matrix and reward vector."""
    p_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.einsum("sa,sa->s", pi, R)
    return p_pi, r_pi


def _linear_system_oracle(P, R, pi, gamma):
    """Solve (I - gamma P_pi) V = R_pi directly; the learner must not do this."""
    p_pi, r_pi = _policy_model(P, R, pi)
    return np.linalg.solve(np.eye(p_pi.shape[0]) - gamma * p_pi, r_pi)


def _random_mdp(rng, n_states, n_actions):
    P = rng.random((n_states, n_actions, n_states))
    P /= P.sum(axis=2, keepdims=True)
    R = rng.normal(size=(n_states, n_actions))
    pi = rng.random((n_states, n_actions))
    pi /= pi.sum(axis=1, keepdims=True)
    return P, R, pi


def test_random_mdps_match_linear_system_oracle():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        n_states, n_actions = int(rng.integers(1, 7)), int(rng.integers(1, 5))
        gamma = (0.0, 0.5, 0.9)[seed % 3]
        P, R, pi = _random_mdp(rng, n_states, n_actions)
        got = _call(P, R, pi, gamma, tol=1e-10)
        want = _linear_system_oracle(P, R, pi, gamma)
        _expect(
            got.shape == (n_states,),
            "output must have one value per state, shape (S,)",
        )
        _close(
            got,
            want,
            atol=ATOL,
            rtol=RTOL,
            err_msg="values differ from the Bellman solution: check the policy average "
            "over actions, the next-state sum and the stopping rule",
        )


def test_absorbing_state_reward_matches_geometric_series():
    P = np.eye(2).reshape(2, 1, 2)  # each state is absorbing under its single action
    R = np.array([[2.0], [-1.0]])
    pi = np.ones((2, 1))
    got = _call(P, R, pi, 0.99, tol=1e-12)
    _close(
        got,
        [200.0, -100.0],
        atol=ATOL,
        rtol=RTOL,
        err_msg="absorbing states must give r / (1 - gamma); the discount is misapplied",
    )


def test_zero_discount_returns_policy_expected_reward():
    rng = np.random.default_rng(3)
    P, R, pi = _random_mdp(rng, 4, 3)
    got = _call(P, R, pi, 0.0)
    _, r_pi = _policy_model(P, R, pi)
    _close(
        got, r_pi, atol=ATOL, rtol=RTOL, err_msg="gamma=0 must return the policy-averaged reward"
    )


def test_duplicated_action_with_split_probability_leaves_values_unchanged():
    rng = np.random.default_rng(17)
    P, R, pi = _random_mdp(rng, 3, 2)
    P3 = np.concatenate([P[:, [0]], P[:, [0]], P[:, [1]]], axis=1)
    R3 = np.concatenate([R[:, [0]], R[:, [0]], R[:, [1]]], axis=1)
    pi3 = np.stack([pi[:, 0] / 2, pi[:, 0] / 2, pi[:, 1]], axis=1)
    base = _call(P, R, pi, 0.8)
    expanded = _call(P3, R3, pi3, 0.8)
    _close(
        expanded,
        base,
        atol=ATOL,
        rtol=RTOL,
        err_msg="splitting one action's probability across copies changed the values",
    )


def test_single_state_with_several_actions():
    P = np.ones((1, 3, 1))
    R = np.array([[1.0, 2.0, 3.0]])
    pi = np.array([[0.2, 0.3, 0.5]])
    got = _call(P, R, pi, 0.5, tol=1e-12)
    _close(
        got,
        [2.3 / 0.5],
        atol=ATOL,
        rtol=RTOL,
        err_msg="single state must average R over pi, then divide by 1 - gamma",
    )


def test_large_reward_magnitudes_keep_relative_accuracy():
    rng = np.random.default_rng(29)
    P, R, pi = _random_mdp(rng, 3, 2)
    R = R * 1e6
    got = _call(P, R, pi, 0.9)
    want = _linear_system_oracle(P, R, pi, 0.9)
    _close(
        got,
        want,
        atol=ATOL,
        rtol=RTOL,
        err_msg="large rewards lost relative accuracy: check the stopping threshold",
    )


def test_integer_model_returns_float64_values():
    P = np.array([[[0, 1]], [[1, 0]]])
    R = np.array([[1], [0]])
    pi = np.array([[1], [1]])
    got = _call(P, R, pi, 0.5)
    _expect(
        got.dtype == np.float64,
        "return float64 values even for integer-valued inputs",
    )
    _close(got, [4 / 3, 2 / 3], atol=ATOL)


@settings(max_examples=25, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(1, 5), st.integers(1, 4), st.integers(0, 10_000))
def test_random_policies_satisfy_bellman_equation(n_states, n_actions, seed):
    rng = np.random.default_rng(seed)
    P, R, pi = _random_mdp(rng, n_states, n_actions)
    got = _call(P, R, pi, 0.7)
    p_pi, r_pi = _policy_model(P, R, pi)
    residual = got - (r_pi + 0.7 * p_pi @ got)
    _expect(
        float(np.max(np.abs(residual))) < 1e-8,
        "result is not a fixed point of the Bellman expectation update",
    )


def test_inputs_are_not_mutated():
    rng = np.random.default_rng(41)
    P, R, pi = _random_mdp(rng, 3, 2)
    # Rows that are valid but off by 5e-10, so renormalising in place would change the bits.
    pi[:, 0] *= 1.0 + 5e-10
    originals = [P.copy(), R.copy(), pi.copy()]
    _call(P, R, pi, 0.6)
    for array, original in zip((P, R, pi), originals, strict=True):
        _equal(array, original, err_msg="an input array was modified")


def test_invalid_discount_and_tolerance_rejected():
    P = np.array([[[1.0]]])
    R = np.array([[1.0]])
    pi = np.array([[1.0]])
    for gamma in (1.0, 1.5, -0.1, float("nan"), True):
        _rejects(P, R, pi, gamma)
    for tol in (0.0, -1e-3, float("inf"), True):
        _rejects(P, R, pi, 0.5, tol)


def test_invalid_probabilities_rejected():
    P = np.array([[[0.5, 0.4]], [[0.0, 1.0]]])
    R = np.zeros((2, 1))
    pi = np.ones((2, 1))
    _rejects(P, R, pi, 0.5)
    P_ok = np.array([[[0.5, 0.5]], [[0.0, 1.0]]])
    _rejects(P_ok, R, np.array([[0.7], [1.0]]), 0.5)
    _rejects(P_ok, R, np.array([[-0.2], [1.0]]), 0.5)


def test_shape_mismatch_and_nonfinite_rejected():
    P = np.array([[[1.0]]])
    _rejects(P, np.zeros((2, 1)), np.ones((1, 1)), 0.5)
    _rejects(P, np.array([[np.nan]]), np.ones((1, 1)), 0.5)
    _rejects(P, np.array([[1 + 1j]]), np.ones((1, 1)), 0.5)
    _rejects(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)), 0.5)


def test_negative_transition_entry_is_rejected_even_when_the_row_sums_to_one():
    P = np.array([[[1.5, -0.5]], [[0.0, 1.0]]])
    _rejects(P, np.zeros((2, 1)), np.ones((2, 1)), 0.5)


def test_negative_policy_entry_is_rejected_even_when_the_row_sums_to_one():
    P = np.ones((1, 2, 1))
    _rejects(P, np.array([[1.0, 1.0]]), np.array([[1.5, -0.5]]), 0.5)


def test_zero_policy_probability_is_accepted():
    P = np.ones((1, 2, 1))
    got = _call(P, np.array([[2.0, 5.0]]), np.array([[1.0, 0.0]]), 0.5)
    _close(
        got,
        [4.0],
        atol=ATOL,
        rtol=RTOL,
        err_msg="a policy with zero probability on some action is valid: zero entries of pi "
        "must be accepted",
    )


def test_nan_reward_is_rejected_when_another_reward_is_finite():
    P = np.array([[[0.0, 1.0]], [[1.0, 0.0]]])
    _rejects(P, np.array([[np.nan], [1.0]]), np.ones((2, 1)), 0.5)


def test_stopping_rule_is_strict_when_the_change_equals_the_tolerance():
    # The first sweep changes the value by exactly tol (0.5), so the strict rule keeps going
    # and the second sweep, which changes it by 0.25, is the one returned.
    got = _call(np.ones((1, 1, 1)), np.array([[0.5]]), np.ones((1, 1)), 0.5, tol=0.5)
    _close(
        got,
        [0.75],
        atol=ATOL,
        rtol=RTOL,
        err_msg="the loop must stop only when the change is strictly below tol: check the "
        "comparison",
    )


def test_bool_false_discount_is_rejected():
    _rejects(np.array([[[1.0]]]), np.array([[1.0]]), np.array([[1.0]]), False)
