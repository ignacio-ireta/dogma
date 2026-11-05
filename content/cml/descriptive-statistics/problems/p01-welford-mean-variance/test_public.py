import math

import numpy as np
import pytest
from submission import welford


def test_small_sample_matches_hand_values():
    n, mean, variance = welford([1.0, 3.0, 4.0, 8.0, 9.0])
    assert n == 5, "count must be the number of values"
    np.testing.assert_allclose(mean, 5.0, atol=1e-12, err_msg="mean of 1,3,4,8,9 is 5")
    np.testing.assert_allclose(
        variance,
        11.5,
        atol=1e-12,
        err_msg="sample variance divides M2 by n - 1: expected 46 / 4 = 11.5",
    )


def test_single_value_has_nan_variance():
    n, mean, variance = welford([4.0])
    assert n == 1 and mean == 4.0, "one value: count 1 and mean equal to that value"
    assert math.isnan(variance), "one value must give a nan variance, not 0.0 or another number"


def test_empty_input_raises_value_error():
    try:
        welford([])
    except ValueError:
        return
    pytest.fail("empty input was accepted: raise ValueError when there are no values to average")


def test_generator_input_gives_the_same_answer_as_a_list():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    n, mean, variance = welford(v for v in values)
    assert n == 8, "a generator must be counted as a stream of values"
    np.testing.assert_allclose(mean, 5.0, atol=1e-12)
    np.testing.assert_allclose(variance, 32.0 / 7.0, atol=1e-12)
