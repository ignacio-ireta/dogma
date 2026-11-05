from __future__ import annotations

import numpy as np
from submission import iterative_policy_evaluation


def test_single_state_geometric_series():
    P = np.array([[[1.0]]])
    R = np.array([[2.0]])
    pi = np.array([[1.0]])
    out = iterative_policy_evaluation(P, R, pi, 0.5)
    np.testing.assert_allclose(out, [4.0], err_msg="single state must give R / (1 - gamma)")


def test_two_state_cycle_matches_hand_solution():
    P = np.array([[[0.0, 1.0]], [[1.0, 0.0]]])
    R = np.array([[1.0], [0.0]])
    pi = np.ones((2, 1))
    out = iterative_policy_evaluation(P, R, pi, 0.5)
    np.testing.assert_allclose(
        out, [4 / 3, 2 / 3], atol=1e-8, err_msg="two-state cycle values are wrong"
    )


def test_zero_discount_returns_expected_reward_for_one_state():
    P = np.array([[[1.0]]])
    R = np.array([[3.0]])
    pi = np.array([[1.0]])
    out = iterative_policy_evaluation(P, R, pi, 0.0)
    np.testing.assert_allclose(out, [3.0], err_msg="gamma=0 must return the immediate reward")
