# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
import numpy as np
from submission import expected_sarsa_update, sarsa_update

TABLE = [[0.0, 0.0], [1.0, 3.0]]


def test_sarsa_example_bootstraps_from_the_action_taken_next():
    np.testing.assert_allclose(
        sarsa_update(TABLE, s=0, a=1, r=1.0, s_next=1, a_next=0, alpha=0.5, gamma=0.5),
        [[0.0, 0.75], [1.0, 3.0]],
        atol=1e-9,
        err_msg="Sarsa target must be r + gamma * Q[s_next, a_next] with the action taken next",
    )


def test_expected_sarsa_example_averages_under_the_epsilon_greedy_policy():
    np.testing.assert_allclose(
        expected_sarsa_update(TABLE, s=0, a=1, r=1.0, s_next=1, alpha=0.5, gamma=0.5, epsilon=0.5),
        [[0.0, 1.125], [1.0, 3.0]],
        atol=1e-9,
        err_msg="expected Sarsa must average Q[s_next, :] under the epsilon-greedy weights",
    )


def test_terminal_next_state_targets_the_reward_alone():
    np.testing.assert_allclose(
        sarsa_update(TABLE, s=0, a=1, r=1.0, s_next=None, a_next=None, alpha=0.5, gamma=0.5),
        [[0.0, 0.5], [1.0, 3.0]],
        atol=1e-9,
        err_msg="a terminal next state has no bootstrap: the target is the reward alone",
    )
