import numpy as np
import pytest
from submission import cosine_similarity_matrix


def test_axis_vectors_against_diagonal_direction_give_one_over_root_two():
    np.testing.assert_allclose(
        cosine_similarity_matrix([[1, 0], [0, 1]], [[1, 1]]),
        [[1 / np.sqrt(2)], [1 / np.sqrt(2)]],
        atol=1e-9,
        err_msg="each axis vs (1, 1) should be 1/sqrt(2): check the dot product and the norms",
    )


def test_same_direction_gives_one_and_orthogonal_gives_zero():
    np.testing.assert_allclose(
        cosine_similarity_matrix([[3, 4]], [[6, 8], [-4, 3], [0, 0]]),
        [[1.0, 0.0, 0.0]],
        atol=1e-9,
        err_msg="same direction should give 1, orthogonal 0, and a zero row 0 by convention",
    )


def test_zero_row_gives_zero_similarity_not_nan():
    sims = cosine_similarity_matrix([[0, 0], [1, 2]], [[1, 2]])
    assert np.all(np.isfinite(sims)), (
        "a zero-norm row gave NaN or inf: divide by nonzero norms only"
    )
    np.testing.assert_allclose(
        sims,
        [[0.0], [1.0]],
        atol=1e-12,
        err_msg="a zero-norm row must give similarity 0.0 to every vector",
    )


def test_feature_count_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        cosine_similarity_matrix([[1, 2]], [[1, 2, 3]])
