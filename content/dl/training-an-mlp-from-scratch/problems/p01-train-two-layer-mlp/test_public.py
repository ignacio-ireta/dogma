from __future__ import annotations

import numpy as np
import pytest
from submission import train_mlp


def test_initial_checkpoint_example():
    params, losses = train_mlp(np.array([[0.0]]), np.array([[2.0]]), 1, 0, 0.1, 0)
    np.testing.assert_allclose(losses, [2.0])
    assert [a.shape for a in params] == [(1, 1), (1,), (1, 1), (1,)]


def test_one_step_zero_input_example():
    _, losses = train_mlp(np.array([[0.0]]), np.array([[2.0]]), 1, 1, 0.1, 0)
    assert losses.shape == (2,) and losses[1] < losses[0]


def test_invalid_learning_rate():
    with pytest.raises(ValueError):
        train_mlp(np.ones((1, 1)), np.ones((1, 1)), 1, 1, 0.0, 0)
