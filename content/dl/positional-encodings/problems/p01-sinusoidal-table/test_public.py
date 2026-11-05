# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import pytest
import torch
from submission import sinusoidal_encoding


def test_worked_example_three_positions_four_columns():
    got = sinusoidal_encoding(3, 4)
    want = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [math.sin(1.0), math.cos(1.0), math.sin(0.01), math.cos(0.01)],
            [math.sin(2.0), math.cos(2.0), math.sin(0.02), math.cos(0.02)],
        ],
        dtype=torch.float64,
    )
    assert got.shape == (3, 4), "shape must be (max_len, d_model)"
    torch.testing.assert_close(
        got,
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="pair i uses angle pos / 10000**(2i/d_model): check exponent and sin/cos columns",
    )


def test_zero_positions_returns_empty_table():
    got = sinusoidal_encoding(0, 6)
    assert tuple(got.shape) == (0, 6), "max_len = 0 must return shape (0, d_model)"


def test_odd_d_model_raises_value_error():
    with pytest.raises(ValueError):
        sinusoidal_encoding(2, 3)
