import numpy as np
from submission import kmeans


def test_two_tight_pairs_converge_to_their_means():
    X = [[0, 0], [0, 1], [10, 10], [10, 11]]
    centroids, labels, inertia, n_iter = kmeans(X, [[0, 0], [10, 10]])
    np.testing.assert_allclose(
        centroids,
        [[0.0, 0.5], [10.0, 10.5]],
        atol=1e-12,
        err_msg="centroids are wrong: each centroid must move to the mean of its samples",
    )
    np.testing.assert_array_equal(labels, [0, 0, 1, 1], err_msg="labels do not match the clusters")
    assert inertia == 1.0, "inertia is wrong: sum the squared distances to the returned centroids"
    assert n_iter == 2, (
        "iteration count is wrong: the second update changes nothing, so the run stops there"
    )


def test_empty_cluster_keeps_its_previous_centroid():
    X = [[0, 0], [0, 1], [10, 10]]
    centroids, labels, _, _ = kmeans(X, [[0, 0], [100, 100]])
    assert np.shape(centroids) == (2, 2), (
        "shape mismatch: an empty cluster must keep its row, so centroids stay (k, d)"
    )
    np.testing.assert_allclose(
        centroids[1],
        [100.0, 100.0],
        atol=1e-12,
        err_msg="an empty cluster must keep its previous centroid, not become NaN or zero",
    )
    assert np.all(np.isfinite(centroids)), "centroids must stay finite when a cluster is empty"
    np.testing.assert_array_equal(
        labels, [0, 0, 0], err_msg="labels do not match the nearest centroid"
    )


def test_init_with_wrong_number_of_columns_raises_value_error():
    try:
        kmeans([[0, 0], [0, 1]], [[0, 0, 0]])
    except ValueError:
        return
    raise AssertionError("init_centroids with a different column count must raise ValueError")
