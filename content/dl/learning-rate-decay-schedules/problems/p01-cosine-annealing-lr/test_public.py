# Public tests: the examples from problem.md plus one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import pytest
from submission import cosine_annealing_lr


def test_first_step_returns_the_base_rate():
    assert cosine_annealing_lr(0, 0.1, 4) == pytest.approx(0.1, rel=1e-12), (
        "step 0 must return base_lr: the half-cosine has not moved yet"
    )


def test_midpoint_is_the_average_of_base_rate_and_floor():
    assert cosine_annealing_lr(2, 0.1, 4) == pytest.approx(0.05, rel=1e-12), (
        "midpoint should be halfway between base_lr and eta_min"
    )


def test_quarter_step_uses_the_cosine_not_a_straight_line():
    expected = 0.1 * (1 + math.cos(math.pi / 4)) / 2
    assert cosine_annealing_lr(1, 0.1, 4) == pytest.approx(expected, rel=1e-12), (
        "rate at step 1 must follow the cosine phase pi / 4, not linear interpolation"
    )


def test_rate_after_t_max_holds_the_floor_with_nonzero_eta_min():
    assert cosine_annealing_lr(4, 0.1, 4, eta_min=0.01) == pytest.approx(0.01, rel=1e-12), (
        "at t_max the rate must be the floor"
    )
