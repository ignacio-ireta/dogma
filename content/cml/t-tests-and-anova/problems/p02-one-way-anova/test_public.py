import numpy as np
import pytest
from submission import one_way_anova


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_two_group_example_matches_the_textbook_f():
    np.testing.assert_allclose(
        one_way_anova([1, 2, 3], [4, 5, 6]),
        [13.5, 0.0213116411],
        atol=1e-8,
        err_msg="two-group example: F = MSB / MSW with df (1, 4), where MSB = 13.5 and MSW = 1",
    )


def test_three_group_example_matches_the_reference():
    np.testing.assert_allclose(
        one_way_anova([4.1, 5.0, 4.6, 4.8], [5.9, 6.2, 5.5, 6.4], [5.0, 4.9, 5.3, 5.1]),
        [17.7814070352, 0.0007476779],
        atol=1e-8,
        err_msg="three-group example differs: check the between and within sums of squares",
    )


def test_identical_constant_groups_raise_value_error():
    _raised_value_error(
        lambda: one_way_anova([1, 1], [2, 2]),
        "within-group variation is zero, so F is undefined: raise ValueError",
    )


def test_single_group_raises_value_error():
    _raised_value_error(
        lambda: one_way_anova([1, 2, 3]),
        "one group gives no between-group variation: raise ValueError when fewer than two groups",
    )
