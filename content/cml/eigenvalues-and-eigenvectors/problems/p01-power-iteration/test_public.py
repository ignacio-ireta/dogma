import numpy as np
import pytest
from submission import power_iteration


def test_symmetric_two_by_two_returns_three_and_the_positive_diagonal_direction():
    value, vector = power_iteration([[2, 1], [1, 2]])
    assert value == pytest.approx(3.0, abs=1e-9), "the dominant eigenvalue of [[2, 1], [1, 2]] is 3"
    np.testing.assert_allclose(
        vector,
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        atol=1e-9,
        err_msg="the eigenvector for 3 is (1, 1)/sqrt(2); its first entry must be positive",
    )


def test_diagonal_matrix_returns_its_largest_entry_and_axis_vector():
    value, vector = power_iteration(np.diag([5.0, 1.0]))
    assert value == pytest.approx(5.0, abs=1e-9), (
        "the largest diagonal entry is the dominant eigenvalue"
    )
    np.testing.assert_allclose(
        vector,
        [1.0, 0.0],
        atol=1e-9,
        err_msg="the eigenvector for the largest diagonal entry is the first axis",
    )


def test_non_symmetric_matrix_raises_value_error():
    with pytest.raises(ValueError, match="symmetric"):
        power_iteration([[1, 2], [0, 1]])


def test_zero_matrix_returns_zero_eigenvalue_and_a_unit_vector():
    value, vector = power_iteration(np.zeros((3, 3)))
    assert value == 0.0, "the zero matrix has eigenvalue 0"
    assert np.sqrt(vector @ vector) == pytest.approx(1.0, abs=1e-12), (
        "the vector must have unit length"
    )
