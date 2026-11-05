import pytest
from submission import sample_size_two_proportions


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_example_for_baseline_ten_percent_and_two_point_effect():
    assert sample_size_two_proportions(0.10, 0.02) == 3841, (
        "10% baseline, 2-point effect: check the pooled H0 term, the unpooled H1 term, then ceil"
    )


def test_stricter_alpha_and_higher_power_need_more_users():
    assert sample_size_two_proportions(0.10, 0.02, alpha=0.01, power=0.9) == 7281, (
        "alpha = 0.01 and power = 0.9 change both quantiles: check which argument feeds which"
    )


def test_baseline_plus_effect_above_one_raises_value_error():
    _raised_value_error(
        lambda: sample_size_two_proportions(0.98, 0.05),
        "the treatment rate 1.03 is not a probability: raise ValueError",
    )
