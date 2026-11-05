from __future__ import annotations

import numpy as np
from submission import epsilon_greedy_action


def test_zero_epsilon_picks_the_greedy_action():
    rng = np.random.default_rng(0)
    choice = epsilon_greedy_action(np.array([0.1, 0.9, 0.3]), 0.0, rng)
    assert choice == 1, "zero epsilon must return the greedy action"


def test_tie_at_zero_epsilon_picks_lowest_index():
    rng = np.random.default_rng(0)
    choice = epsilon_greedy_action(np.array([2.0, 2.0, 1.0]), 0.0, rng)
    assert choice == 0, "tie must return the lowest index"


def test_single_action_is_always_returned():
    rng = np.random.default_rng(5)
    for epsilon in (0.0, 0.5, 1.0):
        choice = epsilon_greedy_action(np.array([-3.0]), epsilon, rng)
        assert choice == 0, "a single action must always be the one returned"
