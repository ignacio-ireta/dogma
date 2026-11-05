import numpy as np
import pytest
from submission import determinant


def test_two_by_two_needs_a_row_swap_and_gives_negative_two():
    np.testing.assert_allclose(
        determinant([[1, 2], [3, 4]]),
        -2.0,
        atol=1e-9,
        err_msg="det([[1, 2], [3, 4]]) is -2: a row swap must flip the sign of the pivot product",
    )


def test_identity_has_determinant_one():
    np.testing.assert_allclose(
        determinant(np.eye(3)),
        1.0,
        atol=1e-12,
        err_msg="the identity matrix must have determinant 1",
    )


def test_singular_matrix_has_determinant_zero():
    assert determinant([[1, 2], [2, 4]]) == pytest.approx(0.0, abs=1e-12), (
        "a matrix with a repeated row is singular, so its determinant is 0"
    )


def test_non_square_matrix_raises_value_error():
    with pytest.raises(ValueError):
        determinant([[1, 2, 3], [4, 5, 6]])
