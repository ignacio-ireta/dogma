from __future__ import annotations

import numpy as np
import pytest
from submission import dense_backward


def test_one_row_example():
    out = dense_backward(np.array([[2.0, 3.0]]), np.array([[4.0], [5.0]]), np.array([[2.0]]))
    for a, b in zip(
        out, (np.array([[8.0, 10.0]]), np.array([[4.0], [6.0]]), np.array([2.0])), strict=True
    ):
        np.testing.assert_allclose(a, b)


def test_mean_example():
    out = dense_backward(np.array([[1.0], [3.0]]), np.array([[2.0]]), np.ones((2, 1)), "mean")
    for a, b in zip(out, (np.ones((2, 1)), np.array([[2.0]]), np.array([1.0])), strict=True):
        np.testing.assert_allclose(a, b)


def test_empty_rejected():
    with pytest.raises(ValueError):
        dense_backward(np.empty((0, 1)), np.ones((1, 1)), np.empty((0, 1)))
