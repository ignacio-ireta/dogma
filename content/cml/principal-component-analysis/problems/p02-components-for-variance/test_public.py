import pytest
from submission import components_for_variance

WORKED = [[-3, -1], [0, -2], [0, 3]]  # explained-variance ratios are 0.75 and 0.25


def _call(X, threshold):
    try:
        return components_for_variance(X, threshold)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(
            f"components_for_variance raised {type(exc).__name__} on a valid input: return the "
            "position of the first cumulative ratio that reaches the threshold, counted from one"
        )


def test_threshold_below_first_ratio_keeps_one_component():
    result = _call(WORKED, 0.7)
    assert result == 1, "0.75 >= 0.7 already, so one component is enough"


def test_threshold_above_first_ratio_needs_second_component():
    result = _call(WORKED, 0.8)
    assert result == 2, "the first ratio 0.75 is below 0.8, so the cumulative sum must reach two"


def test_threshold_equal_to_cumulative_ratio_counts_as_reached():
    result = _call(WORKED, 0.75)
    assert result == 1, "a cumulative ratio equal to the threshold must count as reaching it"


def test_threshold_outside_zero_to_one_raises_value_error():
    try:
        components_for_variance(WORKED, 0.0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a threshold of 0 raised {type(exc).__name__}; raise ValueError")
    pytest.fail("a threshold of 0 is outside (0, 1]: raise ValueError")
