import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import iqr_fences

from dogma_data import rng, sample_rows, sample_window


def _reference(values, k):
    """Fences from NumPy's linear quantiles, the reference for the formula."""
    arr = np.asarray(values, dtype=np.float64)
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 50, 201], ids=list("abcdefgh"))
@pytest.mark.parametrize("k", [0.0, 1.5, 3.0], ids=["zero-k", "fractional-k", "large-k"])
def test_matches_numpy_linear_quantiles(n, k):
    # Real household incomes from engel: right-skewed, with a few very large values.
    values = sample_rows("engel", rng(f"iqr-fences-p02:numpy:{n}"), n)[0][:, 0]
    lower, upper = iqr_fences(values, k=k)
    ref_lower, ref_upper = _reference(values, k)
    np.testing.assert_allclose(
        [lower, upper],
        [ref_lower, ref_upper],
        rtol=1e-12,
        atol=1e-12,
        err_msg="fences differ from linear-interpolated quartiles: check the position h = q(n - 1)",
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=30),
    st.floats(0.0, 4.0),
)
def test_property_agrees_with_numpy_on_any_finite_list(values, k):
    lower, upper = iqr_fences(values, k=k)
    ref_lower, ref_upper = _reference(values, k)
    tol = 1e-9 * (1.0 + np.max(np.abs(values)))
    np.testing.assert_allclose(
        [lower, upper],
        [ref_lower, ref_upper],
        rtol=1e-9,
        atol=tol,
        err_msg="fences differ from NumPy's linear quantiles on a random list",
    )


def test_extreme_maximum_leaves_the_fences_unchanged():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    base = iqr_fences(values)
    moved = values.copy()
    moved[np.argmax(moved)] = 1e12
    np.testing.assert_allclose(
        iqr_fences(moved),
        base,
        rtol=1e-12,
        err_msg="the fences moved when only the maximum changed: quartiles must be robust",
    )
    assert np.mean(moved) > 10 * np.mean(values), "sanity: the mean is not robust"


def test_single_value_has_zero_width_fences():
    assert iqr_fences([4.5]) == pytest.approx((4.5, 4.5), abs=1e-12), (
        "a single value must give zero-width fences at that value"
    )


def test_constant_data_has_zero_width_fences():
    assert iqr_fences([7.0] * 6) == pytest.approx((7.0, 7.0), abs=1e-12), (
        "constant data must give zero-width fences at the constant"
    )


def test_duplicates_are_handled_by_the_interpolation():
    lower, upper = iqr_fences([5, 5, 5, 5, 100])
    assert upper == pytest.approx(5.0, abs=1e-12), (
        "Q1 = Q3 = 5 here, so the upper fence is 5 and the outlier 100 is above it"
    )
    assert lower == pytest.approx(5.0, abs=1e-12), "the lower fence is also 5 here, since Q1 = Q3"


def test_wider_k_gives_wider_fences():
    values = sample_rows("engel", rng("iqr-fences-p02:widen"), 40)[0][:, 0]
    lo1, up1 = iqr_fences(values, k=1.0)
    lo2, up2 = iqr_fences(values, k=2.5)
    assert lo2 <= lo1 and up2 >= up1, "a larger k must widen both fences, never narrow them"


def test_shifting_and_scaling_the_data_moves_and_scales_the_fences():
    values = sample_window("nile", rng("iqr-fences-p02:shift"), 25)  # real annual river flow
    lo, up = iqr_fences(values)
    lo_shift, up_shift = iqr_fences(values + 10.0)
    assert lo_shift == pytest.approx(lo + 10.0, rel=1e-12), (
        "shifting the data must shift the lower fence"
    )
    assert up_shift == pytest.approx(up + 10.0, rel=1e-12), (
        "shifting the data must shift the upper fence"
    )
    lo_scale, up_scale = iqr_fences(values * 3.0)
    assert lo_scale == pytest.approx(lo * 3.0, rel=1e-12), (
        "scaling the data must scale the lower fence"
    )
    assert up_scale == pytest.approx(up * 3.0, rel=1e-12), (
        "scaling the data must scale the upper fence"
    )


def test_input_is_not_mutated_or_sorted_in_place():
    values = np.array([9.0, 1.0, 4.0, 8.0, 3.0])
    original = values.copy()
    iqr_fences(values)
    np.testing.assert_array_equal(
        values,
        original,
        err_msg="the caller's array was changed: sort a copy, not the input",
    )


def test_integer_input_gives_float_fences():
    lower, upper = iqr_fences([1, 3, 4, 8, 9])
    assert isinstance(lower, float) and isinstance(upper, float), (
        "fences must be Python floats even for integer input"
    )


@pytest.mark.parametrize(
    "values",
    [[[1.0, 2.0], [3.0, 4.0]], [], [1.0, float("nan"), 3.0], [1.0, float("inf")]],
    ids=["two-dimensional", "empty", "NaN", "infinite"],
)
def test_invalid_data_raises_value_error(values):
    _expect_value_error(
        lambda: iqr_fences(values),
        "invalid data was accepted: xs must be a non-empty 1-D array of finite numbers",
    )


def test_negative_k_raises_value_error():
    _expect_value_error(
        lambda: iqr_fences([1.0, 2.0, 3.0], k=-1.0),
        "a negative k was accepted: k must be non-negative",
    )
