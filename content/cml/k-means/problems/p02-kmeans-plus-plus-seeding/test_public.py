import numpy as np
from submission import kmeans_plus_plus


def test_seed_zero_on_four_points_gives_the_protocol_centres():
    out = kmeans_plus_plus([[0.0], [1.0], [3.0], [7.0]], 2, seed=0)
    np.testing.assert_array_equal(
        out,
        [[7.0], [0.0]],
        err_msg="centres differ from the protocol: rng.integers(n) first, then rng.choice(n, p=D2)",
    )


def test_k_equal_to_one_returns_the_first_protocol_row():
    X = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 1.0]])
    out = kmeans_plus_plus(X, 1, seed=3)
    rng = np.random.default_rng(3)
    np.testing.assert_array_equal(
        out,
        X[[int(rng.integers(3))]],
        err_msg="with k = 1 the single centre must be the row drawn by rng.integers(n)",
    )


def test_too_few_distinct_rows_raises_value_error():
    try:
        kmeans_plus_plus([[1.0], [1.0], [2.0]], 3, seed=0)
    except ValueError:
        return
    raise AssertionError("fewer distinct rows than k must raise ValueError")
