import numpy as np
import pytest
from submission import correlation_matrix


def test_negative_correlation_from_a_small_example():
    np.testing.assert_allclose(
        correlation_matrix([[1, 3], [2, 1], [3, 2]]),
        [[1.0, -0.5], [-0.5, 1.0]],
        atol=1e-9,
        err_msg="expected r = -0.5: the cross-product is -1 and each variance is 2",
    )


def test_perfectly_linear_columns_have_correlation_one():
    np.testing.assert_allclose(
        correlation_matrix([[1, 2], [2, 4], [3, 6]]),
        [[1.0, 1.0], [1.0, 1.0]],
        atol=1e-9,
        err_msg="column 2 is 2 times column 1, so the correlation is exactly 1",
    )


def test_constant_column_gives_nan_row_and_column_including_diagonal():
    result = correlation_matrix([[1, 5], [2, 5], [3, 5]])
    assert result[0, 0] == pytest.approx(1.0), (
        "the non-constant column still has correlation 1 with itself"
    )
    assert np.isnan(result[1]).all() and np.isnan(result[:, 1]).all(), (
        "a constant column has no correlation: its whole row and column should be NaN"
    )


def test_one_observation_raises_value_error():
    try:
        correlation_matrix([[1.0, 2.0]])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"one observation has no correlation; raised {type(exc).__name__}: {exc}")
    pytest.fail("correlation needs at least two observations: raise ValueError")
