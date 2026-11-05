import numpy as np
import pytest
from submission import two_proportion_z_test


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_example_control_converts_more_gives_negative_z():
    np.testing.assert_allclose(
        two_proportion_z_test(120, 1000, 90, 1000),
        [-2.1882658846, 0.0286502404],
        atol=1e-8,
        err_msg="z must be (p_B - p_A) / pooled SE: B converted less, so z is negative",
    )


def test_equal_rates_give_z_zero_and_p_one():
    np.testing.assert_allclose(
        two_proportion_z_test(50, 100, 50, 100),
        [0.0, 1.0],
        atol=1e-12,
        err_msg="equal rates should give z = 0 and a two-sided p of 1",
    )


def test_pooled_rate_of_zero_raises_value_error():
    _raised_value_error(
        lambda: two_proportion_z_test(0, 10, 0, 10),
        "no conversions in either arm leave the pooled SE at zero: raise ValueError",
    )
