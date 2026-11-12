from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import StandardScaler
from submission import standardize

from dogma_data import rng, sample_rows


def _call(train, values):
    try:
        result = standardize(train, values)
    except Exception as exc:
        pytest.fail(f"valid scaler input was rejected: {type(exc).__name__}")
    assert isinstance(result, tuple) and len(result) == 3, "return values, mean and scale"
    for array in result:
        assert isinstance(array, np.ndarray), "return NumPy arrays"
        assert array.dtype == np.float64, "scaler outputs must be float64"
    return result


def _reject(train, values):
    try:
        standardize(train, values)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid scaler input must raise ValueError, not {type(exc).__name__}")
    pytest.fail("invalid scaler input must raise ValueError")


@settings(max_examples=35, deadline=None, report_multiple_bugs=False)
@given(st.integers(0, 10000), st.integers(2, 25), st.integers(1, 6))
def test_matches_sklearn_on_seeded_random_matrices(seed, rows, columns):
    # Raw wine: the columns sit on scales from hundredths to thousands. The scaler is fitted on
    # `rows` wines and applied to seven other wines.
    drawn, _ = sample_rows("wine", rng(f"standard-scaler-p01:sklearn:{seed}"), rows + 7, columns)
    train, values = drawn[:rows], drawn[rows:]
    expected = StandardScaler().fit(train)
    out, mean, scale = _call(train, values)
    checks = [
        (
            out,
            expected.transform(values),
            "transformed values differ: check (values - mean) / scale",
        ),
        (mean, expected.mean_, "mean differs: the mean must come from the training rows only"),
        (scale, expected.scale_, "scale differs: use the population std, ddof=0, of each column"),
    ]
    for actual, target, mistake in checks:
        np.testing.assert_allclose(actual, target, atol=1e-9, rtol=1e-9, err_msg=mistake)


def test_held_out_values_never_influence_fit_statistics():
    train = np.array([[0, 4], [4, 8], [8, 12]])
    first = _call(train, [[100, -100]])
    second = _call(train, [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(first[1], second[1], err_msg="held-out data changed the mean")
    np.testing.assert_array_equal(first[2], second[2], err_msg="held-out data changed the scale")
    np.testing.assert_allclose(first[1], [4, 8], err_msg="mean must use only training rows")


def test_population_variance_not_sample_variance():
    out, _, scale = _call([[1], [5]], [[3], [7]])
    np.testing.assert_allclose(scale, [2], err_msg="divide variance by n, not n minus one")
    np.testing.assert_allclose(out, [[0], [2]], err_msg="population scale was not used")


def test_constant_training_features_still_transform_new_values():
    out, mean, scale = _call([[3, 8], [3, 8]], [[5, 7], [3, 8]])
    np.testing.assert_array_equal(mean, [3, 8], err_msg="constant-column mean was lost")
    np.testing.assert_array_equal(scale, [1, 1], err_msg="zero scales need unit replacement")
    np.testing.assert_array_equal(
        out, [[2, -1], [0, 0]], err_msg="constant columns must still transform new rows"
    )


def test_single_training_row_and_empty_transform():
    out, mean, scale = _call([[2, 4, 6]], np.empty((0, 3)))
    assert out.shape == (0, 3), "empty transform has the wrong feature count"
    assert mean.shape == scale.shape == (3,), "statistics must be one-dimensional"
    np.testing.assert_array_equal(scale, [1, 1, 1], err_msg="one-row training is constant")


def test_magnitude_bound_is_inclusive_at_1e100():
    out, _mean, scale = _call([[-1e100], [1e100]], [[1e100]])
    np.testing.assert_allclose(out, [[1.0]], rtol=1e-12, err_msg="1e100 is within the bound")
    np.testing.assert_allclose(scale, [1e100], rtol=1e-12, err_msg="scale of +-1e100 is 1e100")


def test_magnitude_just_above_the_bound_is_rejected():
    _reject([[np.nextafter(1e100, np.inf)]], [[1.0]])


def test_nan_in_values_is_rejected():
    _reject([[1.0], [2.0]], [[np.nan]])


def test_constant_column_at_a_large_offset_is_centred_exactly():
    value = 1e8 + 0.1
    out, _mean, scale = _call([[value], [value], [value]], [[value]])
    np.testing.assert_allclose(
        out,
        [[0.0]],
        rtol=0.0,
        atol=1e-9,
        err_msg="a constant column's mean is its own value, so that value must map to 0",
    )
    np.testing.assert_array_equal(scale, [1.0], err_msg="an exactly constant column has scale 1")


@pytest.mark.parametrize("magnitude", [1e90, 1e-90], ids=["large", "small"])
def test_extreme_scales_preserve_standardized_values(magnitude):
    out, _, _ = _call(np.array([[1, 2], [3, 6]]) * magnitude, np.array([[5, 0]]) * magnitude)
    np.testing.assert_allclose(
        out,
        [[3, -2]],
        atol=1e-9,
        rtol=1e-9,
        err_msg="changing physical units changed standardized values",
    )


def test_smallest_normal_fitted_scale_is_supported():
    tiny = np.finfo(np.float64).tiny
    out, mean, scale = _call([[-tiny], [tiny]], [[tiny]])
    np.testing.assert_array_equal(scale, [tiny], err_msg="small normal scales remain supported")
    np.testing.assert_array_equal(mean, [0], err_msg="symmetric training data has zero mean")
    np.testing.assert_allclose(
        out, [[1]], rtol=1e-9, atol=1e-9, err_msg="small normal scales must not be replaced by one"
    )


def test_readonly_inputs_are_not_mutated_and_shapes_are_preserved():
    train = np.array([[1.0, 2], [5, 10], [9, 18]])
    values = np.array([[3.0, 7], [9, 12]])
    before_train, before_values = train.copy(), values.copy()
    train.flags.writeable = False
    values.flags.writeable = False
    out, mean, scale = _call(train, values)
    np.testing.assert_array_equal(train, before_train, err_msg="training input was mutated")
    np.testing.assert_array_equal(values, before_values, err_msg="transform input was mutated")
    assert out.shape == values.shape, "transform shape differs from input"
    assert mean.shape == scale.shape == (2,), "statistics lost their feature axis"


def test_writable_inputs_are_left_unchanged_by_the_call():
    train = np.array([[1.0, 2], [5, 10], [9, 18]])
    values = np.array([[3.0, 7], [9, 12]])
    before_train, before_values = train.copy(), values.copy()
    _call(train, values)
    np.testing.assert_array_equal(
        train, before_train, err_msg="training input was modified in place: write new arrays"
    )
    np.testing.assert_array_equal(
        values, before_values, err_msg="transform input was modified in place: write new arrays"
    )


@pytest.mark.parametrize(
    ("train", "values"),
    [
        ([1, 2], [[1, 2]]),
        ([[1, 2]], [1, 2]),
        (np.ones((1, 2, 1)), [[1, 2]]),
        (np.empty((0, 2)), [[1, 2]]),
        (np.empty((3, 0)), np.empty((1, 0))),
        ([[1, 2]], [[1]]),
        ([[np.nan, 2]], [[1, 2]]),
        ([[1, 2]], [[np.inf, 2]]),
        ([[-np.inf, 2]], [[1, 2]]),
        ([[1e101]], [[1]]),
        ([[1]], [[-1e101]]),
    ],
    ids=[
        "train-rank",
        "values-rank",
        "tensor",
        "no-rows",
        "no-features",
        "columns",
        "train-nan",
        "values-infinite",
        "train-infinite",
        "train-range",
        "values-range",
    ],
)
def test_invalid_inputs_raise_value_error(train, values):
    _reject(train, values)


def test_scale_of_a_tiny_spread_does_not_underflow_when_one_value_sits_at_the_mean():
    t = 1e-300
    _out, _mean, scale = _call([[0.0], [t], [2.0 * t]], [[t]])
    np.testing.assert_allclose(
        scale,
        [t * np.sqrt(2.0 / 3.0)],
        rtol=1e-9,
        err_msg="the population scale of a tiny nonzero spread must be computed without "
        "underflow, even when one training value equals the mean",
    )
