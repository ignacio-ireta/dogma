import numpy as np
from submission import knn_regress

X = [[0.0], [1.0], [2.0], [10.0]]
Y = [0.0, 1.0, 2.0, 10.0]


def test_uniform_weights_average_the_k_nearest_targets():
    out = knn_regress(X, Y, [[1.2]], k=2)
    np.testing.assert_allclose(
        out,
        [1.5],
        atol=1e-12,
        err_msg="uniform prediction is wrong: average the targets of the k nearest rows",
    )


def test_distance_weights_pull_the_prediction_towards_the_closer_neighbour():
    out = knn_regress(X, Y, [[1.2]], k=2, weights="distance")
    np.testing.assert_allclose(
        out,
        [1.2],
        atol=1e-12,
        err_msg="distance-weighted prediction is wrong: use weights 1/d and normalise them",
    )


def test_exact_matches_are_averaged_under_distance_weights():
    out = knn_regress([[0.0], [0.0], [3.0]], [4.0, 8.0, 100.0], [[0.0]], k=3, weights="distance")
    np.testing.assert_allclose(
        out,
        [6.0],
        atol=1e-12,
        err_msg="zero-distance neighbours must be averaged alone when weights are distance-based",
    )


def test_unknown_weights_value_raises_value_error():
    try:
        knn_regress(X, Y, [[1.2]], k=2, weights="median")
    except ValueError:
        return
    raise AssertionError('weights other than "uniform" or "distance" must raise ValueError')
