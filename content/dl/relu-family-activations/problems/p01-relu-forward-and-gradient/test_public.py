from __future__ import annotations

import numpy as np
import pytest
from submission import relu_family


def _close(got, want, message):
    assert isinstance(got, np.ndarray), "return type must be a NumPy array"
    assert got.dtype == np.float64, "output dtype must be float64"
    assert got.shape == np.shape(want), "output shape must match the contract"
    assert np.allclose(got, want, atol=1e-9, rtol=1e-9), message


def _invalid(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid input must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid input was accepted instead of raising ValueError")


def test_relu_example():
    y, d = relu_family([-2, 0, 3])
    _close(y, [0, 0, 3], "ReLU must gate negative values")
    _close(d, [0, 0, 1], "ReLU local derivative has the wrong sign mask")


def test_leaky_example():
    y, d = relu_family([-2, 0, 3], 0.1)
    _close(y, [-0.2, 0, 3], "negative branch must retain the configured slope")
    _close(d, [0.1, 0, 1], "zero derivative convention must be applied separately")


def test_empty_preserves_shape():
    y, d = relu_family(np.empty((2, 0)))
    _close(y, np.empty((2, 0)), "empty values must be preserved")
    _close(d, np.empty((2, 0)), "empty derivatives must be preserved")
