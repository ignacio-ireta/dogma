from __future__ import annotations

import numpy as np
from submission import q_learning_update, train_q_learning


def test_non_terminal_update_bootstraps_from_the_best_next_action():
    Q = np.zeros((9, 4))
    Q[1] = [0.2, 1.0, 0.4, 0.0]
    out = q_learning_update(Q, 0, 2, 0.0, 1, False, 0.5, 0.9)
    np.testing.assert_allclose(
        out[0, 2], 0.45, err_msg="target must use the max over next actions: 0.9 * 1.0 * 0.5"
    )


def test_terminal_update_ignores_the_next_states_values():
    Q = np.zeros((9, 4))
    Q[8, :] = 1e6
    out = q_learning_update(Q, 7, 1, 1.0, 8, True, 0.5, 0.9)
    np.testing.assert_allclose(
        out[7, 1], 0.5, err_msg="a terminal target is the reward alone; do not bootstrap"
    )


def test_zero_episodes_returns_the_zero_table():
    out = train_q_learning(0, seed=0)
    assert out.shape == (9, 4), "the Q table must have one row per state and one column per action"
    np.testing.assert_array_equal(out, np.zeros((9, 4)), err_msg="no episodes must mean no updates")
