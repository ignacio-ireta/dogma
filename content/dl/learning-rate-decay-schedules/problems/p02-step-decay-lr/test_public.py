# Public tests: the examples from problem.md plus one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import step_decay_lr


def test_rate_is_unchanged_inside_the_first_block():
    assert step_decay_lr(1, 0.1, 2, 0.5) == pytest.approx(0.1, rel=1e-12), (
        "inside the first block of updates the rate must still equal base_lr"
    )


def test_first_drop_happens_at_step_equal_to_step_size():
    assert step_decay_lr(2, 0.1, 2, 0.5) == pytest.approx(0.05, rel=1e-12), (
        "the first drop must take effect at step == step_size, not one step later"
    )


def test_two_completed_blocks_apply_two_drops():
    assert step_decay_lr(5, 0.1, 2, 0.5) == pytest.approx(0.025, rel=1e-12), (
        "floor(5 / 2) = 2 completed blocks must give two multiplications by gamma"
    )


def test_gamma_one_gives_a_constant_rate():
    assert step_decay_lr(9, 0.3, 3, 1.0) == pytest.approx(0.3, rel=1e-12), (
        "gamma equal to one must never change the rate"
    )
