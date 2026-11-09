"""Hidden tests on real and semi-synthetic data, loaded once per module.

The oracles are NumPy's own reductions on the same arrays, so the expected values are computed
at run time and never copied from a draw.
"""

import numpy as np
import pytest
from submission import column_means

from dogma_data import load, load_series, rng, synth


@pytest.fixture(scope="module")
def breast_cancer():
    return load("breast_cancer").X


@pytest.fixture(scope="module")
def mixed_scales():
    return synth.correlated_mixed_scale(rng("py-realism:cms:0"), 200, 8)


@pytest.fixture(scope="module")
def co2_column():
    # co2 has 59 missing months. The mean must skip them, not return NaN.
    return load_series("co2").reshape(-1, 1)


def test_real_features_on_very_different_scales(breast_cancer):
    got = np.asarray(column_means(breast_cancer), dtype=float)
    expected = np.mean(breast_cancer, axis=0)
    assert got.shape == expected.shape, "one mean per feature column"
    assert np.allclose(got, expected, rtol=1e-9, atol=0.0), "column means must match numpy"


def test_correlated_mixed_scale_matrix(mixed_scales):
    got = np.asarray(column_means(mixed_scales), dtype=float)
    assert np.allclose(got, np.mean(mixed_scales, axis=0), rtol=1e-9, atol=0.0)


def test_missing_values_are_skipped(co2_column):
    got = np.asarray(column_means(co2_column), dtype=float)
    assert np.isfinite(got).all(), "NaN entries must be ignored, not propagated"
    assert np.allclose(got, np.nanmean(co2_column, axis=0), rtol=1e-9, atol=0.0)
