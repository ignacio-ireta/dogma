# Public tests: the examples from problem.md plus one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import inverse_sqrt_lr


def test_first_update_is_on_the_ramp():
    assert inverse_sqrt_lr(1, 4, 4) == pytest.approx(0.0625, rel=1e-12), (
        "step 1 is the first update; on the ramp the step * warmup**-1.5 term is smaller"
    )


def test_peak_is_at_warmup_steps():
    assert inverse_sqrt_lr(4, 4, 4) == pytest.approx(0.25, rel=1e-12), (
        "both terms are equal at step == warmup_steps, which is the peak"
    )


def test_decay_after_the_peak_follows_inverse_square_root():
    assert inverse_sqrt_lr(16, 4, 4) == pytest.approx(0.125, rel=1e-12), (
        "after warmup the rate must fall as step ** -0.5"
    )


def test_step_zero_is_rejected():
    with pytest.raises(ValueError):
        inverse_sqrt_lr(0, 4, 4)
