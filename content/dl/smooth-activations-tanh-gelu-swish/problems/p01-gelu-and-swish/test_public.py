from __future__ import annotations

import numpy as np
import pytest
from submission import smooth_activations


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


def test_zero_values():
    for got in smooth_activations([0]):
        _close(got, [0], "activation at zero is incorrect")


def test_signed_example():
    expected = (
        [-0.7615941559557649, 0.7615941559557649],
        [-0.15865525393145707, 0.8413447460685429],
        [-0.2689414213699951, 0.7310585786300049],
    )
    for got, want in zip(smooth_activations([-1, 1]), expected, strict=True):
        _close(got, want, "signed activation example is incorrect")


def test_empty_preserves_shape():
    for got in smooth_activations(np.empty((0, 3))):
        _close(got, np.empty((0, 3)), "empty output must retain all axes")
