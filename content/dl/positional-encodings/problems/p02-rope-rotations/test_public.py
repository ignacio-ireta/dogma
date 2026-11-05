# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

import pytest
import torch
from submission import apply_rope


def test_first_row_unchanged_and_second_row_rotated_by_one_radian():
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    want = torch.tensor(
        [[1.0, 0.0], [math.cos(1.0), math.sin(1.0)]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        apply_rope(x),
        want,
        atol=1e-9,
        rtol=1e-9,
        msg="row t must rotate the adjacent pair (2i, 2i+1) by t * theta_i: check the pair layout",
    )


def test_single_row_is_position_zero_and_stays_unchanged():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    torch.testing.assert_close(
        apply_rope(x), x, atol=1e-12, rtol=0, msg="position 0 must be the identity rotation"
    )


def test_odd_feature_width_raises_value_error():
    with pytest.raises(ValueError):
        apply_rope(torch.zeros(2, 3, dtype=torch.float64))
