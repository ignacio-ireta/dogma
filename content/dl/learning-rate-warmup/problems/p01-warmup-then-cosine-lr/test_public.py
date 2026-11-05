# Public tests: the examples from problem.md plus one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import warmup_cosine_lr


def test_ramp_starts_at_zero():
    assert warmup_cosine_lr(0, 1.0, 4, 12, 0.1) == pytest.approx(0.0, abs=1e-12), (
        "step 0 must return 0.0 when warmup is positive: the ramp starts from zero"
    )


def test_warmup_boundary_is_exactly_the_base_rate():
    assert warmup_cosine_lr(4, 1.0, 4, 12, 0.1) == pytest.approx(1.0, rel=1e-12), (
        "at step == warmup_steps the rate must be base_lr: check the boundary branch"
    )


def test_midpoint_of_cosine_is_halfway_between_base_and_floor():
    assert warmup_cosine_lr(8, 1.0, 4, 12, 0.1) == pytest.approx(0.55, rel=1e-12), (
        "halfway through the cosine the rate must be the average of base_lr and min_lr"
    )


def test_step_at_total_steps_returns_the_floor():
    assert warmup_cosine_lr(12, 1.0, 4, 12, 0.1) == pytest.approx(0.1, rel=1e-12), (
        "at total_steps the rate must be min_lr"
    )
