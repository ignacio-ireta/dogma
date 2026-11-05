from __future__ import annotations

import numpy as np
import pytest
from submission import smooth_derivatives


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


def test_zero_derivatives():
    for got, want in zip(smooth_derivatives([0]), ([1], [0.5], [0.5]), strict=True):
        _close(got, want, "derivative at zero is incorrect")


def test_positive_example():
    expected = ([0.41997434161402614], [1.0833154705876864], [0.9276705118714867])
    for got, want in zip(smooth_derivatives([1]), expected, strict=True):
        _close(got, want, "worked derivative example is incorrect")


def test_empty_preserves_shape():
    for got in smooth_derivatives(np.empty((0, 3))):
        _close(got, np.empty((0, 3)), "empty output must retain all axes")
