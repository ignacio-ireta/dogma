from __future__ import annotations

import numpy as np
from submission import discounted_returns


def test_worked_example_with_half_discount():
    out = discounted_returns(np.array([1.0, 0.0, 2.0]), 0.5)
    np.testing.assert_allclose(out, [1.5, 1.0, 2.0], err_msg="return recurrence is wrong")


def test_gamma_one_gives_suffix_sums():
    out = discounted_returns(np.array([1.0, 1.0, 1.0]), 1.0)
    np.testing.assert_allclose(out, [3.0, 2.0, 1.0], err_msg="gamma=1 must give suffix sums")


def test_empty_episode_returns_empty_float_array():
    out = discounted_returns(np.array([]), 0.9)
    assert out.shape == (0,), "an episode with no steps has no returns"
    assert out.dtype == np.float64, "returns must be float64 even for an empty episode"
