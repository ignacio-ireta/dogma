from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import zero_point_quantize


def checked(call):
    """Run a learner call and fail with a named contract message if it raises."""
    try:
        result = call()
    except ValueError:
        pytest.fail(
            "valid input raised ValueError: check the accepted dtypes and shapes", pytrace=False
        )
    except Exception:
        pytest.fail("valid input failed: honor the documented interface", pytrace=False)
    if not isinstance(result, tuple) or len(result) != 3:
        pytest.fail("return the triple (q, scale, zero_point)", pytrace=False)
    q, scale, zero_point = result
    if not isinstance(q, np.ndarray) or q.dtype != np.int8:
        pytest.fail("codes must be an int8 array, not another integer dtype", pytrace=False)
    if isinstance(zero_point, bool) or not isinstance(zero_point, int):
        pytest.fail("zero point must be a Python int (not bool, float or array)", pytrace=False)
    if not isinstance(scale, float) or not scale > 0:
        pytest.fail("scale must be a positive Python float", pytrace=False)
    if not -128 <= zero_point <= 127:
        pytest.fail("zero point must lie in the int8 range", pytrace=False)
    return q, scale, zero_point


def rejects(call):
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception", pytrace=False)
    pytest.fail("invalid input must raise ValueError", pytrace=False)


def reference(values: list[float]):
    """Pure-Python model of the affine grid, with Python round() (ties to even)."""
    lo = min(*values, 0.0)
    hi = max(*values, 0.0)
    scale = 1.0 if hi == lo else (hi - lo) / 255
    zero_point = -128 - round(lo / scale)
    codes = [min(127, max(-128, round(v / scale) + zero_point)) for v in values]
    return codes, scale, zero_point


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(0, 10_000), st.sampled_from([-3.0, -0.5, 0.0, 0.5, 2.0, 7.0]))
def test_random_inputs_match_the_affine_model(seed, shift):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, 5)) + shift
    q, scale, zero_point = checked(lambda: zero_point_quantize(x))
    want_codes, want_scale, want_zp = reference(x.ravel().tolist())
    if zero_point != want_zp:
        pytest.fail("zero point mismatch: check lo/hi include 0 and the -128 anchor", pytrace=False)
    if abs(scale - want_scale) > 1e-9 * want_scale:
        pytest.fail("scale mismatch: use (hi - lo) / 255, not / 256 or / 254", pytrace=False)
    np.testing.assert_array_equal(
        q.ravel(),
        np.array(want_codes, dtype=np.int8),
        err_msg="code mismatch: check round-half-even and the zero-point addition",
    )


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(0, 10_000), st.integers(1, 4))
def test_error_is_within_half_a_step_for_every_element(seed, ndim):
    rng = np.random.default_rng(seed)
    shape = (3,) * ndim
    x = rng.uniform(-5.0, 9.0, size=shape)
    q, scale, zero_point = checked(lambda: zero_point_quantize(x))
    x_hat = scale * (q.astype(np.float64) - zero_point)
    if np.any(np.abs(x - x_hat) > scale / 2 * (1 + 1e-9)):
        pytest.fail(
            "error bound broken: each element must be within scale / 2 of its reconstruction",
            pytrace=False,
        )


@settings(max_examples=40, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(0, 10_000))
def test_zero_is_exact_inside_any_range(seed):
    rng = np.random.default_rng(seed)
    x = np.concatenate([[0.0], rng.uniform(0.1, 4.0, size=5)])
    q, scale, zero_point = checked(lambda: zero_point_quantize(x))
    if (int(q[0]) - zero_point) * scale != 0.0:
        pytest.fail(
            "zero is not exact: 0.0 must have its own code and dequantize to 0.0",
            pytrace=False,
        )


def test_degenerate_all_zero_tensor_is_exact_and_unit_scaled():
    q, scale, zero_point = checked(lambda: zero_point_quantize(np.zeros((2, 2))))
    if q.shape != (2, 2) or np.any(q != -128):
        pytest.fail("all-zero mistake: every code must be the zero code -128", pytrace=False)
    if scale != 1.0 or zero_point != -128:
        pytest.fail("all-zero mistake: scale must be 1.0 and zero point -128", pytrace=False)


def test_single_positive_value_uses_the_top_code():
    q, scale, zero_point = checked(lambda: zero_point_quantize(np.array([5.0])))
    if int(q[0]) != 127:
        pytest.fail("range mistake: the maximum of [0, hi] must use code 127", pytrace=False)
    if zero_point != -128 or abs(scale - 5.0 / 255) > 1e-15:
        pytest.fail(
            "range mistake: lo is 0 here, so scale is hi / 255 and zero point -128", pytrace=False
        )


@settings(max_examples=30, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(0, 10_000), st.sampled_from([1e-30, 1e-6, 1e6, 1e30]))
def test_small_and_large_magnitudes_are_stable(seed, magnitude):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=6) * magnitude
    q, scale, zero_point = checked(lambda: zero_point_quantize(x))
    if not np.isfinite(scale) or scale <= 0:
        pytest.fail(
            "extreme magnitudes mistake: the scale must stay finite and positive", pytrace=False
        )
    x_hat = scale * (q.astype(np.float64) - zero_point)
    if np.any(np.abs(x - x_hat) > scale / 2 * (1 + 1e-9)):
        pytest.fail(
            "extreme magnitudes mistake: the error bound must hold at every scale", pytrace=False
        )


def test_integer_input_is_accepted_and_divided_in_float():
    q, scale, zero_point = checked(lambda: zero_point_quantize(np.array([-100, 0, 155])))
    np.testing.assert_array_equal(q, np.array([-128, -28, 127], dtype=np.int8))
    assert scale == 1.0 and zero_point == -28, "integer input mistake: scale and zero point differ"


def test_invalid_dtypes_shapes_and_values_raise():
    rejects(lambda: zero_point_quantize(np.array([True, False])))
    rejects(lambda: zero_point_quantize(np.array([1 + 1j])))
    rejects(lambda: zero_point_quantize(np.array([], dtype=float)))
    rejects(lambda: zero_point_quantize(np.array([np.nan, 1.0])))
    rejects(lambda: zero_point_quantize(np.array([-np.inf, 1.0])))
    rejects(lambda: zero_point_quantize(np.float64(3.0)))


def test_inputs_are_not_mutated():
    x = np.array([[-1.5, 0.25], [3.0, 0.0]])
    original = x.copy()
    zero_point_quantize(x)
    np.testing.assert_array_equal(x, original, err_msg="input array was mutated")


def test_exact_ties_round_to_even_codes():
    q, scale, zero_point = checked(lambda: zero_point_quantize(np.array([-2.5, 0.5, 252.5])))
    if zero_point != -126 or scale != 1.0:
        pytest.fail("tie mistake: lo = -2.5 gives scale 1.0 and zero point -126", pytrace=False)
    np.testing.assert_array_equal(
        q,
        np.array([-128, -126, 126], dtype=np.int8),
        err_msg="tie mistake: 0.5 must round to the even code 0, not 1 (no floor(x + 0.5))",
    )


def test_top_value_that_rounds_past_127_is_clipped():
    q, scale, zero_point = checked(lambda: zero_point_quantize(np.array([-253.5, 1.5])))
    if zero_point != 126 or scale != 1.0:
        pytest.fail(
            "range mistake: lo = -253.5 and hi = 1.5 give scale 1.0 and zero point 126",
            pytrace=False,
        )
    np.testing.assert_array_equal(
        q,
        np.array([-128, 127], dtype=np.int8),
        err_msg="clip mistake: hi rounds to code 128 on the shifted grid; clip it to 127",
    )


def test_underflowing_scale_is_rejected():
    rejects(lambda: zero_point_quantize(np.array([5e-324])))
    rejects(lambda: zero_point_quantize(np.array([1e-320, 0.0])))


def test_positive_infinity_next_to_finite_values_is_rejected():
    rejects(lambda: zero_point_quantize(np.array([1.0, np.inf])))


def test_scale_equal_to_the_smallest_normal_float64_is_accepted():
    tiny = np.finfo(np.float64).tiny
    q, _scale, _zero_point = checked(lambda: zero_point_quantize(np.array([255.0 * tiny])))
    if int(q[0]) != 127:
        pytest.fail(
            "a scale exactly equal to the smallest normal float64 is not underflow: "
            "the top value must still map to code 127",
            pytrace=False,
        )


def test_tiny_but_normal_scale_is_accepted():
    q, _scale, zero_point = checked(lambda: zero_point_quantize(np.array([1e-40, 0.0])))
    if zero_point != -128 or int(q[0]) != 127:
        pytest.fail(
            "a normal but tiny scale is valid: lo is 0 here, so the top value maps to code 127",
            pytrace=False,
        )
