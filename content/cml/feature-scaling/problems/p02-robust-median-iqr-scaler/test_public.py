import numpy as np
import pytest
from submission import robust_standardize


def test_median_and_iqr_come_from_the_training_rows():
    out, center, scale = robust_standardize([[1, 10], [3, 14], [5, 18], [9, 22]], [[5, 8]])
    np.testing.assert_allclose(center, [4, 16], err_msg="center must be the training median")
    np.testing.assert_allclose(
        scale, [3.5, 6], err_msg="scale must be the 75th minus 25th percentile"
    )
    np.testing.assert_allclose(
        out, [[0.2857142857, -4 / 3]], err_msg="transform uses the training center and scale"
    )


def test_constant_training_column_gets_unit_scale():
    out, _center, scale = robust_standardize([[4, 1], [4, 1], [4, 1]], [[7, 2]])
    np.testing.assert_allclose(scale, [1, 1], err_msg="a zero IQR must be replaced by one")
    np.testing.assert_allclose(
        out, [[3, 1]], err_msg="constant columns still subtract their median"
    )


def test_empty_values_keep_the_feature_axis():
    out, center, scale = robust_standardize([[2, 8], [4, 6]], np.empty((0, 2)))
    if out.shape != (0, 2):
        pytest.fail("an empty transform must keep the feature axis")
    if center.shape != (2,) or scale.shape != (2,):
        pytest.fail("statistics must be one-dimensional with one entry per feature")
