import numpy as np
import pytest
from submission import vector_projection


def test_example_projection_onto_x_axis_keeps_x_component():
    np.testing.assert_allclose(
        vector_projection([3, 4], [1, 0]),
        [3.0, 0.0],
        atol=1e-9,
        err_msg="projecting onto the x-axis should keep only the x-component",
    )


def test_example_projection_onto_ones_gives_the_mean():
    np.testing.assert_allclose(
        vector_projection([1, 2, 3], [1, 1, 1]),
        [2.0, 2.0, 2.0],
        atol=1e-9,
        err_msg="projection onto the all-ones vector should be the mean repeated",
    )


def test_orthogonal_vectors_project_to_zero():
    np.testing.assert_allclose(
        vector_projection([1, 0], [0, 5]),
        [0.0, 0.0],
        atol=1e-12,
        err_msg="orthogonal vectors have zero dot product, so the projection is the zero vector",
    )


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_zero_vector_b_raises_value_error():
    _raised_value_error(
        lambda: vector_projection([1, 2], [0, 0]),
        "b is the zero vector, so there is no line to project onto: raise ValueError",
    )
