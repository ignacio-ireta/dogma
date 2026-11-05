# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import DynamicLossScaler


def test_fresh_scaler_starts_at_init_scale_with_no_good_steps():
    scaler = DynamicLossScaler(init_scale=1024.0, growth_interval=3)
    assert scaler.scale == 1024.0
    assert scaler.good_steps == 0


def test_overflow_skips_step_halves_scale_resets_count():
    scaler = DynamicLossScaler(init_scale=1024.0, growth_interval=3)
    assert scaler.update(found_inf=False) is True
    assert scaler.good_steps == 1
    assert scaler.update(found_inf=True) is False, "an overflowing step must be skipped"
    assert scaler.scale == 512.0, "overflow must halve the scale (backoff_factor)"
    assert scaler.good_steps == 0, "overflow must reset the good-step count"


def test_growth_happens_after_a_full_interval_of_good_steps():
    scaler = DynamicLossScaler(init_scale=1024.0, growth_interval=3)
    for _ in range(2):
        scaler.update(found_inf=False)
    assert scaler.scale == 1024.0, "the scale grew before a full interval of good steps"
    scaler.update(found_inf=False)
    assert scaler.scale == 2048.0, "the scale must double after growth_interval good steps"
    assert scaler.good_steps == 0


def test_non_positive_initial_scale_raises_value_error():
    with pytest.raises(ValueError):
        DynamicLossScaler(init_scale=0.0)
    with pytest.raises(ValueError):
        DynamicLossScaler(init_scale=-1.0)
