from __future__ import annotations

import numpy as np
from submission import action_value_estimates


def test_sample_average_of_two_actions():
    out = action_value_estimates(np.array([0, 1, 0]), np.array([1.0, 4.0, 3.0]), 2)
    np.testing.assert_allclose(out, [2.0, 4.0], err_msg="sample average must be the batch mean")


def test_constant_step_size_matches_hand_computation():
    out = action_value_estimates(np.array([0, 0]), np.array([1.0, 0.0]), 1, step_size=0.5)
    np.testing.assert_allclose(out, [0.25], err_msg="constant step size update is wrong")


def test_untried_actions_keep_initial_value():
    out = action_value_estimates(np.array([0]), np.array([2.0]), 3, initial_value=5.0)
    np.testing.assert_allclose(
        out, [2.0, 5.0, 5.0], err_msg="actions never pulled must keep the initial value"
    )
