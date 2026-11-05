import numpy as np
import pytest
from submission import pseudo_inverse


def _call(A, **kwargs):
    try:
        return np.asarray(pseudo_inverse(A, **kwargs), dtype=np.float64)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(
            f"pseudo_inverse raised {type(exc).__name__} on a valid input: return the (n, m) "
            "matrix V diag(s_plus) U^T built from the SVD of A"
        )


def _check_close(actual, expected, mistake):
    if actual.shape != np.shape(expected) or not np.allclose(
        actual, expected, rtol=1e-9, atol=1e-9
    ):
        pytest.fail(mistake)


def test_invertible_diagonal_matrix_gives_reciprocal_diagonal():
    _check_close(
        _call([[2.0, 0.0], [0.0, 4.0]]),
        [[0.5, 0.0], [0.0, 0.25]],
        "for an invertible diagonal matrix the pseudo-inverse is the diagonal of reciprocals",
    )


def test_rank_one_matrix_drops_the_zero_singular_value():
    _check_close(
        _call([[1.0, 2.0], [2.0, 4.0]]),
        [[0.04, 0.08], [0.08, 0.16]],
        "for a rank-one matrix only the nonzero singular value is inverted, "
        "so the result is A / 25",
    )


def test_cutoff_is_relative_to_the_largest_singular_value():
    _check_close(
        _call([[1000.0, 0.0], [0.0, 0.5]], rcond=1e-3),
        [[0.001, 0.0], [0.0, 0.0]],
        "the cutoff is rcond times the largest singular value (1), so the singular value 0.5 is "
        "dropped",
    )


def test_negative_rcond_raises_value_error():
    try:
        pseudo_inverse([[1.0, 0.0]], rcond=-1.0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a negative rcond raised {type(exc).__name__}; raise ValueError")
    pytest.fail("a negative rcond must raise ValueError")
