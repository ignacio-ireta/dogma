import numpy as np
import pytest
from submission import binary_cross_entropy_with_logits


def _call(z, y, mistake):
    """Call the loss; a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return float(binary_cross_entropy_with_logits(z, y))
    except ValueError as exc:
        pytest.fail(f"{mistake}; the function raised ValueError: {exc}")
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__}: {exc}")


def test_zero_logits_give_log_two():
    got = _call(np.array([0.0, 0.0]), np.array([1.0, 0.0]), "zero logits failed")
    assert abs(got - np.log(2.0)) <= 1e-12, (
        "a logit of 0 gives probability 1/2, so each loss is log 2"
    )


def test_mixed_logits_match_the_reference_mean():
    got = _call(np.array([2.0, -1.0]), np.array([1.0, 0.0]), "mixed logits failed")
    assert abs(got - 0.2200948492805977) <= 1e-12, (
        "mean of log(1 + exp(-2)) and log(1 + exp(-1)), using the correct side for each target"
    )


def test_confidently_wrong_logit_gives_its_magnitude_without_overflow():
    got = _call(np.array([800.0]), np.array([0.0]), "a confidently wrong logit must stay finite")
    assert abs(got - 800.0) <= 1e-9, "a logit of 800 with target 0 has loss 800"


def test_targets_outside_the_unit_interval_raise_value_error():
    try:
        binary_cross_entropy_with_logits(np.array([0.0]), np.array([2.0]))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"a target of 2 must raise ValueError; the function raised {type(exc).__name__}"
        )
    pytest.fail("a target above 1 must raise ValueError, not return a loss")
