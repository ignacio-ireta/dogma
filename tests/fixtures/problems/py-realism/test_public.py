import numpy as np
from submission import column_means


def test_means_of_a_small_matrix():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(column_means(matrix), [2.0, 3.0]), "each column mean should be 2 and 3"


def test_single_row_is_its_own_mean():
    matrix = np.array([[5.0, -1.0, 0.5]])
    assert np.allclose(column_means(matrix), [5.0, -1.0, 0.5]), "one row gives its own values"
