import numpy as np
import pytest
from submission import welch_t_test


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_example_matches_the_welch_statistic_df_and_p_value():
    np.testing.assert_allclose(
        welch_t_test([1, 2, 3, 4, 5], [2, 4, 6, 8, 10, 12, 14]),
        [-2.8097574347, 8.0371057514, 0.0227472553],
        atol=1e-8,
        err_msg="example differs: use Welch's standard error and the Satterthwaite df",
    )


def test_identical_groups_give_zero_t_and_p_one():
    np.testing.assert_allclose(
        welch_t_test([1, 2, 3], [1, 2, 3]),
        [0.0, 4.0, 1.0],
        atol=1e-12,
        err_msg="identical groups should give t = 0, df = 2(n - 1) = 4 and p = 1",
    )


def test_both_samples_constant_raises_value_error():
    _raised_value_error(
        lambda: welch_t_test([1, 1, 1], [2, 2, 2, 2]),
        "both samples are constant, so the standard error is zero: raise ValueError",
    )


def test_sample_with_one_observation_raises_value_error():
    _raised_value_error(
        lambda: welch_t_test([1.0], [2.0, 3.0, 4.0]),
        "a group with fewer than two observations has no variance: raise ValueError",
    )
