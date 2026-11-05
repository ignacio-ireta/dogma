import numpy as np
import pytest
from submission import mean_confidence_interval


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_example_interval_for_one_to_five_uses_t_with_four_degrees_of_freedom():
    np.testing.assert_allclose(
        mean_confidence_interval([1, 2, 3, 4, 5]),
        [3.0, 1.0367568385, 4.9632431615],
        atol=1e-9,
        err_msg="interval for 1..5 is wrong: the margin is t(0.975, df=4) * s / sqrt(5)",
    )


def test_constant_sample_gives_a_point_interval():
    np.testing.assert_allclose(
        mean_confidence_interval([2, 2, 2]),
        [2.0, 2.0, 2.0],
        atol=1e-12,
        err_msg="a constant sample has zero spread, so lo and hi should equal the mean",
    )


def test_single_observation_raises_value_error():
    _raised_value_error(
        lambda: mean_confidence_interval([7.0]),
        "one observation has no standard error: raise ValueError when n < 2",
    )


def test_confidence_of_one_raises_value_error():
    _raised_value_error(
        lambda: mean_confidence_interval([1.0, 2.0, 3.0], confidence=1.0),
        "confidence must be strictly between 0 and 1: raise ValueError at 1.0",
    )
