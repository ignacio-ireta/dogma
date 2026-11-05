import numpy as np
import pytest
from submission import numerical_gradient


def test_gradient_of_sum_of_squares_is_twice_x():
    np.testing.assert_allclose(
        numerical_gradient(lambda v: float(np.sum(v**2)), [1.0, 2.0]),
        [2.0, 4.0],
        rtol=1e-8,
        err_msg="the gradient of sum(x_i^2) is 2x, which is (2, 4) at (1, 2)",
    )


def test_gradient_of_linear_function_is_its_coefficient_vector():
    np.testing.assert_allclose(
        numerical_gradient(lambda v: 3 * v[0] - v[1], [0.7, -2.0]),
        [3.0, -1.0],
        atol=1e-9,
        err_msg="the partials of 3*x0 - x1 are 3 and -1, whatever the point",
    )


def test_empty_input_raises_value_error():
    try:
        numerical_gradient(lambda v: 0.0, [])
    except ValueError:
        return
    pytest.fail("an empty x must raise ValueError: check the size before looping over coordinates")
