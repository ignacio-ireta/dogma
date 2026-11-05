import numpy as np
from submission import matrix_rank


def test_full_rank_two_by_two_has_rank_two():
    assert matrix_rank([[1, 2], [3, 4]]) == 2, "a nonsingular 2x2 matrix has rank 2"


def test_proportional_rows_have_rank_one():
    assert matrix_rank([[1, 2], [2, 4]]) == 1, (
        "rows that are multiples of each other leave one independent row: check elimination"
    )


def test_zero_matrix_has_rank_zero_and_wide_shape_is_handled():
    assert matrix_rank(np.zeros((2, 3))) == 0, "the zero matrix has rank 0"


def test_explicit_tolerance_treats_a_tiny_pivot_as_zero():
    A = [[1.0, 0.0], [0.0, 1e-10]]
    assert matrix_rank(A, tol=1e-8) == 1, "a pivot below an explicit tol must not count"
    assert matrix_rank(A) == 2, "with the default tolerance, a pivot of 1e-10 is real"
