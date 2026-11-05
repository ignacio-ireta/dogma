import numpy as np
import pytest
from submission import gauss_jordan_inverse


def test_two_by_two_matrix_matches_closed_form_inverse():
    np.testing.assert_allclose(
        gauss_jordan_inverse([[4, 7], [2, 6]]),
        [[0.6, -0.7], [-0.2, 0.4]],
        atol=1e-9,
        err_msg="2x2 inverse is wrong: use (1/det) [[d, -b], [-c, a]] as a check",
    )


def test_zero_first_pivot_needs_a_row_swap():
    np.testing.assert_allclose(
        gauss_jordan_inverse([[0, 1], [1, 0]]),
        [[0.0, 1.0], [1.0, 0.0]],
        atol=1e-9,
        err_msg="a zero pivot must be avoided by swapping in a row with a nonzero entry",
    )


def test_singular_matrix_raises_value_error_mentioning_singular():
    with pytest.raises(ValueError, match="singular"):
        gauss_jordan_inverse([[1, 2], [2, 4]])


def test_non_square_matrix_raises_value_error():
    with pytest.raises(ValueError):
        gauss_jordan_inverse([[1, 2, 3], [4, 5, 6]])
