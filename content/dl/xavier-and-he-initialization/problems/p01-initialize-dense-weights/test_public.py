from __future__ import annotations

import numpy as np
import pytest
from submission import initialize_dense


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


def test_xavier_shape_and_bounds():
    result = initialize_dense(2, 3, "xavier_uniform", 7)
    assert result.shape == (2, 3), "weights use input-output layout"
    assert np.all(np.abs(result) <= np.sqrt(6 / 5)), "Xavier support must match its variance target"


def test_he_example():
    _close(
        initialize_dense(1, 1, "he_normal", 0),
        [[0.1778093838704446]],
        "He seeded example is incorrect",
    )


def test_invalid_zero_fan():
    _invalid(initialize_dense, 0, 3, "he_normal", 0)
