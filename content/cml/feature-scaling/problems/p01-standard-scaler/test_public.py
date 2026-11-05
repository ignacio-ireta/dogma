from __future__ import annotations

import numpy as np
from submission import standardize


def test_training_statistics_transform_unseen_values():
    out, mean, scale = standardize([[1, 10], [3, 14]], [[5, 8]])
    np.testing.assert_allclose(out, [[3, -2]], err_msg="use training statistics")
    np.testing.assert_allclose(mean, [2, 12], err_msg="mean is per training column")
    np.testing.assert_allclose(scale, [1, 2], err_msg="use population standard deviation")


def test_constant_column_has_unit_scale():
    out, mean, scale = standardize([[4, 1], [4, 3]], [[7, 2]])
    np.testing.assert_allclose(out, [[3, 0]], err_msg="constant columns still subtract their mean")
    np.testing.assert_allclose(mean, [4, 2])
    np.testing.assert_allclose(scale, [1, 1])


def test_empty_transform_preserves_feature_axis():
    out, mean, scale = standardize([[2, 8]], np.empty((0, 2)))
    assert out.shape == (0, 2), "empty transform must keep the feature axis"
    np.testing.assert_array_equal(mean, [2, 8])
    np.testing.assert_array_equal(scale, [1, 1])
