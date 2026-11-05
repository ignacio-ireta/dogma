import pytest
from submission import permutation_test_mean_diff


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_identical_constant_groups_have_p_equal_to_one():
    p = permutation_test_mean_diff([5, 5, 5], [5, 5, 5], n_permutations=99, seed=0)
    assert p == 1.0, (
        "identical groups: every relabelling ties the observed difference, so p must be 1"
    )


def test_separated_groups_have_p_near_the_exact_value():
    p = permutation_test_mean_diff([1, 2, 3], [7, 8, 9], n_permutations=9999, seed=0)
    assert abs(p - 0.1) <= 0.02, (
        "p should be close to the exact two-sided value 2/20 = 0.1: check the statistic and tails"
    )


def test_empty_group_raises_value_error():
    _raised_value_error(
        lambda: permutation_test_mean_diff([], [1.0, 2.0], seed=0),
        "an empty group has no mean: raise ValueError",
    )
