import numpy as np
from submission import knn_classify


def test_majority_vote_among_the_three_nearest_rows():
    X = [[0, 0], [1, 0], [0, 1], [5, 5]]
    out = knn_classify(X, [0, 1, 1, 0], [[0.2, 0.2]], k=3)
    np.testing.assert_array_equal(
        out, [1], err_msg="majority vote is wrong: count the labels of the k nearest rows only"
    )


def test_vote_tie_goes_to_the_class_of_the_nearest_tied_neighbour():
    out = knn_classify([[0, 0], [2, 0], [1, 0], [9, 9]], [7, 3, 5, 9], [[1.2, 0]], k=2)
    np.testing.assert_array_equal(
        out,
        [5],
        err_msg="vote tie resolved wrongly: the tie goes to the nearest tied neighbour's label, "
        "not to the smallest label",
    )


def test_equal_distances_are_ordered_by_lower_training_index():
    out = knn_classify([[0, 0], [1, 0]], [4, 2], [[0.5, 0]], k=1)
    np.testing.assert_array_equal(
        out,
        [4],
        err_msg="equal distances must be broken by the lower training index",
    )


def test_k_above_training_size_raises_value_error():
    X = [[0, 0], [1, 0], [0, 1], [5, 5]]
    try:
        knn_classify(X, [0, 1, 1, 0], [[0.2, 0.2]], k=5)
    except ValueError:
        return
    raise AssertionError("k larger than the number of training rows must raise ValueError")
