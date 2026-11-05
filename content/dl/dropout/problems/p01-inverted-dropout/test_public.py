from __future__ import annotations

import numpy as np
from submission import dropout


def test_seeded_example():
    out, dx, mask = dropout(np.array([[2.0, 4.0]]), np.ones((1, 2)), 0.5, True, 0)
    np.testing.assert_array_equal(out, [[4.0, 0.0]])
    np.testing.assert_array_equal(dx, [[2.0, 0.0]])
    np.testing.assert_array_equal(mask, [[2.0, 0.0]])


def test_eval_example():
    out, dx, mask = dropout(np.array([[2.0, 4.0]]), np.array([[3.0, 5.0]]), 0.5, False, 0)
    np.testing.assert_array_equal(out, [[2.0, 4.0]])
    np.testing.assert_array_equal(dx, [[3.0, 5.0]])
    np.testing.assert_array_equal(mask, [[1.0, 1.0]])


def test_zero_probability_identity():
    out, dx, _ = dropout(np.array([[1.0]]), np.array([[7.0]]), 0.0, True, 0)
    np.testing.assert_array_equal(out, [[1.0]])
    np.testing.assert_array_equal(dx, [[7.0]])
