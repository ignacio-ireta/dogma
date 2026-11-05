from __future__ import annotations

import numpy as np
import pytest
from submission import dense_forward


def _call(X, W, b, mistake):
    """Call dense_forward on a valid input; any exception is a named contract failure."""
    try:
        return dense_forward(X, W, b)
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__}")


def test_worked_example_with_two_units_and_a_bias():
    X = np.array([[1.0, 2.0]])
    W = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    b = np.array([0.5, 0.0, -1.0])
    np.testing.assert_allclose(
        _call(X, W, b, "the one-row example failed"),
        [[1.5, 2.0, 7.0]],
        atol=1e-12,
        err_msg="Y = X @ W + b: the row [1, 2] times W is [1, 2, 8], then the bias is added",
    )


def test_identity_weights_and_zero_bias_return_the_batch_unchanged():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(
        _call(X, np.eye(2), np.zeros(2), "identity weights failed"),
        X,
        atol=1e-12,
        err_msg="X @ I + 0 must equal X: check the product order and the bias",
    )


def test_feature_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        dense_forward(np.zeros((2, 3)), np.zeros((4, 2)), np.zeros(2))


def test_empty_batch_returns_zero_rows_of_output_width():
    got = _call(np.zeros((0, 3)), np.zeros((3, 2)), np.zeros(2), "empty batch failed")
    assert got.shape == (0, 2), "shape mismatch: an empty batch gives (0, d_out) rows"
