import numpy as np
import pytest
from submission import best_rank_k


def _call(A, k):
    try:
        return best_rank_k(A, k)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(
            f"best_rank_k raised {type(exc).__name__} on a valid input: return (A_k, error), with "
            "A_k built from the first k singular terms and error from the discarded singular values"
        )


def _check_close(actual, expected, mistake):
    actual = np.asarray(actual, dtype=np.float64)
    if actual.shape != np.shape(expected) or not np.allclose(
        actual, expected, rtol=1e-9, atol=1e-9
    ):
        pytest.fail(mistake)


def test_diagonal_matrix_keeps_its_largest_entry_and_reports_the_other():
    A_k, error = _call([[3.0, 0.0], [0.0, 1.0]], 1)
    _check_close(
        A_k,
        [[3.0, 0.0], [0.0, 0.0]],
        "A_k is wrong: the rank-one approximation of diag(3, 1) keeps the 3 and drops the 1",
    )
    if abs(float(error) - 1.0) > 1e-9:
        pytest.fail(
            "the error is wrong: it is the discarded singular value 1, since sqrt(1**2) = 1"
        )


def test_k_zero_returns_zero_matrix_and_the_full_frobenius_norm():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    A_k, error = _call(A, 0)
    _check_close(
        A_k, np.zeros((2, 2)), "k = 0 keeps no singular terms: A_k must be the zero matrix"
    )
    if abs(float(error) - np.linalg.norm(A)) > 1e-9:
        pytest.fail("with k = 0 every singular value is discarded: the error is the norm of A")


def test_k_above_min_of_rows_and_columns_raises_value_error():
    try:
        best_rank_k([[1.0, 2.0], [3.0, 4.0]], 3)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"k above min(m, n) raised {type(exc).__name__}; raise ValueError")
    pytest.fail("k above min(m, n) must raise ValueError")


def test_full_rank_k_returns_the_input_with_zero_error():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    A_k, error = _call(A, 2)
    _check_close(A_k, A, "keeping every singular term must reproduce A")
    if abs(float(error)) > 1e-9:
        pytest.fail("nothing is discarded when k = min(m, n), so the error must be zero")
