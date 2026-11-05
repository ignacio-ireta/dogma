import warnings

import numpy as np
import pytest
from submission import softmax


def _call(logits, mistake, **kwargs):
    """Call softmax with NumPy warnings turned into errors; a crash is a named contract failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return np.asarray(softmax(logits, **kwargs), dtype=np.float64)
    except RuntimeWarning as exc:
        pytest.fail(f"{mistake}; NumPy warned: {exc}")
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__}: {exc}")


def test_plain_vector_matches_the_textbook_values():
    np.testing.assert_allclose(
        _call(np.array([1.0, 2.0, 3.0]), "softmax of a vector failed"),
        [0.09003057, 0.24472847, 0.66524096],
        rtol=1e-7,
        err_msg="wrong distribution: softmax(z)_j = exp(z_j) / sum_k exp(z_k)",
    )


def test_rows_of_a_matrix_each_sum_to_one():
    got = _call(np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]]), "softmax of a matrix failed")
    np.testing.assert_allclose(
        got.sum(axis=-1),
        [1.0, 1.0],
        rtol=1e-12,
        err_msg="each row must sum to 1 along the last axis",
    )


def test_extreme_logits_are_finite_and_warning_free():
    got = _call(np.array([1e4, 1e4 - 1.0]), "extreme logits must not overflow")
    np.testing.assert_allclose(
        got,
        [0.7310585786300049, 0.2689414213699951],
        rtol=1e-12,
        err_msg="softmax must subtract the maximum before exponentiating: a shift by a "
        "constant leaves the answer unchanged, and exp of 1e4 overflows",
    )


def test_axis_zero_normalises_each_column():
    got = _call(np.array([[1.0, 2.0], [3.0, 4.0]]), "softmax along axis 0 failed", axis=0)
    np.testing.assert_allclose(
        got.sum(axis=0), [1.0, 1.0], rtol=1e-12, err_msg="columns must sum to 1"
    )
