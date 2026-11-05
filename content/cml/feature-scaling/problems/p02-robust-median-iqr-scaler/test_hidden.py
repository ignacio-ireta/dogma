"""Hidden tests: scikit-learn's RobustScaler as the oracle, plus robustness, edge and validation
cases. Each failure names the likely mistake first."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import RobustScaler
from submission import robust_standardize

SETTINGS = settings(max_examples=40, deadline=None, report_multiple_bugs=False)


def _call(train, values):
    try:
        result = robust_standardize(train, values)
    except Exception as exc:
        pytest.fail(f"valid robust-scaler input was rejected: {type(exc).__name__}")
    if not (isinstance(result, tuple) and len(result) == 3):
        pytest.fail("return the transformed values, the center and the scale")
    for array in result:
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            pytest.fail("the three outputs must be float64 NumPy arrays")
    return result


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@SETTINGS
@given(st.integers(0, 10_000), st.integers(1, 30), st.integers(1, 5))
def test_matches_sklearn_robust_scaler_on_seeded_matrices(seed, rows, columns):
    rng = np.random.default_rng(seed)
    train = rng.normal(size=(rows, columns)) * np.arange(1, columns + 1) + 3
    values = rng.normal(size=(6, columns)) * 2 - 1
    expected = RobustScaler().fit(train)
    out, center, scale = _call(train, values)
    checks = [
        (center, expected.center_, "center differs from the training median"),
        (scale, expected.scale_, "scale differs from the 75th minus 25th percentile range"),
        (
            out,
            expected.transform(values),
            "transformed values differ: check (values - center) / scale",
        ),
    ]
    for actual, target, mistake in checks:
        np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-9, err_msg=mistake)


def test_single_extreme_value_leaves_center_and_scale_unchanged():
    base = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])
    moved = base.copy()
    moved[-1, 0] = 1e9
    _, center, scale = _call(base, [[0.0]])
    _, center_moved, scale_moved = _call(moved, [[0.0]])
    if not np.allclose(center, center_moved) or not np.allclose(scale, scale_moved):
        pytest.fail("the center or scale moved when only the maximum changed: use median and IQR")


def test_held_out_values_never_influence_the_fitted_statistics():
    train = np.array([[0.0, 4.0], [4.0, 8.0], [8.0, 12.0], [2.0, 6.0]])
    first = _call(train, [[100.0, -100.0]])
    second = _call(train, [[1.0, 2.0], [3.0, 4.0]])
    if not np.array_equal(first[1], second[1]) or not np.array_equal(first[2], second[2]):
        pytest.fail("held-out values changed the center or the scale: fit on training rows only")


def test_even_and_odd_training_counts_use_the_median_rule():
    _, center_even, _ = _call([[1.0], [2.0], [4.0], [10.0]], [[0.0]])
    _, center_odd, _ = _call([[1.0], [2.0], [4.0]], [[0.0]])
    if not np.isclose(center_even[0], 3.0):
        pytest.fail("an even count needs the mean of the two middle values as the median")
    if not np.isclose(center_odd[0], 2.0):
        pytest.fail("an odd count needs the middle value as the median")


def test_tiny_but_real_iqr_is_kept_as_the_scale():
    train = np.array([[0.0], [1e-13], [2e-13], [3e-13], [4e-13]])
    _, _, scale = _call(train, [[0.0]])
    if not np.isclose(scale[0], 2e-13, rtol=1e-9):
        pytest.fail(
            "a real IQR of ordinary float size was replaced by one: only near-zero IQRs are"
        )


def test_integer_inputs_give_float_outputs_equal_to_float_inputs():
    x_int = np.array([[1, 0], [0, 2], [1, 1], [2, 1], [5, 9]])
    out_int = _call(x_int, np.array([[3, 4]]))
    out_float = _call(x_int.astype(float), np.array([[3.0, 4.0]]))
    for a, b in zip(out_int, out_float, strict=True):
        if not np.allclose(a, b, rtol=1e-12, atol=1e-12):
            pytest.fail("integer inputs must give the same numbers as the same values as floats")


def test_zero_rows_in_values_return_empty_outputs_with_feature_axis():
    out, center, scale = _call([[1.0, 2.0], [3.0, 4.0]], np.empty((0, 2)))
    if out.shape != (0, 2) or center.shape != (2,) or scale.shape != (2,):
        pytest.fail("shape mismatch: an empty transform keeps the feature axis")


@pytest.mark.parametrize(
    ("train", "values"),
    [
        ([1.0, 2.0], [[1.0, 2.0]]),
        ([[1.0, 2.0]], [1.0, 2.0]),
        (np.ones((1, 2, 1)), [[1.0, 2.0]]),
        (np.empty((0, 2)), [[1.0, 2.0]]),
        ([[1.0, 2.0]], [[1.0]]),
        ([[np.nan, 2.0]], [[1.0, 2.0]]),
        ([[1.0, 2.0]], [[np.inf, 2.0]]),
    ],
    ids=[
        "train-rank",
        "values-rank",
        "tensor",
        "no-rows",
        "columns",
        "train-nan",
        "values-infinite",
    ],
)
def test_invalid_inputs_raise_value_error(train, values):
    _expect_value_error(
        lambda: robust_standardize(train, values),
        "invalid input was accepted: check ranks, feature counts and finiteness",
    )


def test_inputs_are_not_mutated():
    train = np.array([[9.0, 1.0], [3.0, 5.0], [1.0, 7.0], [5.0, 3.0]])
    values = np.array([[2.0, 6.0]])
    before_train, before_values = train.copy(), values.copy()
    _call(train, values)
    if not np.array_equal(train, before_train):
        pytest.fail("the training array was sorted or modified in place: sort a copy")
    if not np.array_equal(values, before_values):
        pytest.fail("the values array was modified in place: return a new array")


def test_opposite_extreme_values_have_representable_quartiles_and_center():
    train = np.array([[-1e308], [1e308]])
    out, center, scale = _call(train, train)
    if not np.allclose(center, [0.0], rtol=0.0, atol=0.0):
        pytest.fail("opposite extreme values need a finite median without subtractive overflow")
    if not np.allclose(scale, [1e308], rtol=1e-9, atol=0.0):
        pytest.fail("interpolated quartiles must stay finite when their IQR is representable")
    if not np.allclose(out, [[-1.0], [1.0]], rtol=1e-9, atol=0.0):
        pytest.fail("finite extreme training values must retain their standardized coordinates")


def test_iqr_below_ten_machine_epsilons_gets_unit_scale():
    _, _, scale = _call(np.arange(5).reshape(-1, 1) * 5e-16, [[0.0]])
    if not np.isclose(scale[0], 1.0, rtol=1e-9, atol=0.0):
        pytest.fail("an IQR of 1e-15 is below 10 machine epsilons: its scale must be 1.0")


def test_iqr_just_above_ten_machine_epsilons_is_kept():
    _, _, scale = _call(np.arange(5).reshape(-1, 1) * 1.5e-15, [[0.0]])
    if not np.isclose(scale[0], 3e-15, rtol=1e-9, atol=0.0):
        pytest.fail("an IQR of 3e-15 is above 10 machine epsilons: it must be kept as the scale")


def test_near_zero_iqr_uses_machine_epsilon_policy_but_larger_iqr_survives():
    for step, expected in [(1e-17, 1.0), (1e-13, 2e-13)]:
        _, _, scale = _call(np.arange(5).reshape(-1, 1) * step, [[0.0]])
        if not np.isclose(scale[0], expected, rtol=1e-9, atol=0.0):
            pytest.fail(
                "apply the stated machine-epsilon fallback without discarding larger nonzero IQRs"
            )


def test_iqr_just_below_ten_epsilons_gets_unit_scale():
    eps = float(np.finfo(np.float64).eps)
    _, _, scale = _call(np.arange(5).reshape(-1, 1) * (4.75 * eps), [[0.0]])
    np.testing.assert_allclose(
        scale,
        [1.0],
        rtol=0.0,
        atol=0.0,
        err_msg="a scale of 9.5 machine epsilons is below ten, so the column gets scale one",
    )


def test_iqr_just_above_ten_epsilons_is_kept():
    eps = float(np.finfo(np.float64).eps)
    _, _, scale = _call(np.arange(5).reshape(-1, 1) * (5.25 * eps), [[0.0]])
    np.testing.assert_allclose(
        scale,
        [10.5 * eps],
        rtol=1e-9,
        atol=0.0,
        err_msg="a scale of 10.5 machine epsilons is not below ten, so it must be kept",
    )
