from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import absmax_dequantize, absmax_quantize


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
    if not isinstance(result, tuple) or len(result) != 2:
        pytest.fail("return the pair (q, scale)", pytrace=False)
    q, scale = result
    if not isinstance(q, np.ndarray) or q.dtype != np.int8:
        pytest.fail("codes must be an int8 array, not another integer dtype", pytrace=False)
    if not isinstance(scale, np.ndarray) or scale.dtype != np.float64:
        pytest.fail("scale must be a float64 array", pytrace=False)
    return q, scale


def rejects(call):
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception", pytrace=False)
    pytest.fail("invalid input must raise ValueError", pytrace=False)


def reference(x: np.ndarray, axis: int | None):
    """Pure-Python model: one group per channel, Python round() (ties to even), clamp to 127."""
    values = np.asarray(x, dtype=np.float64)
    if axis is None:
        moved = values.reshape(1, -1)
    else:
        moved = np.moveaxis(values, axis, 0).reshape(values.shape[axis], -1)
    scales, codes = [], []
    for row in moved.tolist():
        amax = max(abs(v) for v in row)
        s = amax / 127 if amax > 0 else 1.0
        scales.append(s)
        codes.append([min(127, max(-127, round(v / s))) for v in row])
    code_array = np.array(codes, dtype=np.int64)
    if axis is None:
        return code_array.reshape(values.shape), np.array(scales[0])
    shape = np.moveaxis(values, axis, 0).shape
    return np.moveaxis(code_array.reshape(shape), 0, axis), np.array(scales)


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    st.integers(0, 10_000),
    st.sampled_from([None, 0, 1, -1]),
    st.sampled_from([(4,), (3, 5), (2, 3, 4)]),
)
def test_random_inputs_match_scalar_model(seed, axis, shape):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=3.0, size=shape)
    if axis is not None and axis >= len(shape):
        return
    q, scale = checked(lambda: absmax_quantize(x, axis))
    want_q, want_scale = reference(x, axis)
    if q.shape != x.shape:
        pytest.fail("codes must keep the input shape", pytrace=False)
    np.testing.assert_array_equal(
        q,
        want_q,
        err_msg="code mismatch: check the absmax axes, the divisor 127, and round-half-even",
    )
    np.testing.assert_allclose(
        scale,
        want_scale,
        atol=1e-9,
        rtol=1e-9,
        err_msg="scale mismatch: check per-channel versus per-tensor absmax",
    )


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(0, 10_000), st.sampled_from([None, 0, 1]))
def test_round_trip_error_is_within_half_a_scale_step(seed, axis):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3, 4)) * 10.0 ** rng.integers(-3, 4)
    q, scale = checked(lambda: absmax_quantize(x, axis))
    x_hat = absmax_dequantize(q, scale, axis)
    if x_hat.shape != x.shape:
        pytest.fail("dequantized shape must match the input shape", pytrace=False)
    if axis is None:
        bound = np.full(x.shape, float(scale))
    else:
        shape = [1, 1]
        shape[axis] = -1
        bound = np.broadcast_to(scale.reshape(shape), x.shape)
    # Float rounding in x / scale can push a value a hair past the midpoint; allow 1e-9 relative.
    if np.any(np.abs(x - x_hat) > bound / 2 * (1 + 1e-9)):
        pytest.fail(
            "error bound broken: the reconstruction must lie within scale / 2 "
            "(check the scale or the rounding)",
            pytrace=False,
        )


def test_codes_stay_in_symmetric_range_and_absmax_reaches_edge():
    x = np.array([[-7.0, 3.0, 7.0], [2.0, -2.0, 0.0]])
    q, _scale = checked(lambda: absmax_quantize(x))
    if q.min() < -127 or q.max() > 127:
        pytest.fail("codes must stay in [-127, 127]; -128 is not used", pytrace=False)
    if int(q.max()) != 127 or int(q.min()) != -127:
        pytest.fail("the absmax entries must map to +127 and -127", pytrace=False)


def test_negative_zero_is_a_zero_code():
    q, _ = checked(lambda: absmax_quantize(np.array([-0.0, 0.0, 1.0])))
    if q[0] != 0 or q[1] != 0:
        pytest.fail("signed zero must quantize to code 0", pytrace=False)


def test_all_zero_channel_is_isolated_from_other_channels():
    x = np.array([[0.0, 0.0, 0.0], [1.0, -4.0, 2.0]])
    q, scale = checked(lambda: absmax_quantize(x, axis=0))
    if float(scale[0]) != 1.0 or np.any(q[0] != 0):
        pytest.fail("all-zero channel mistake: scale 1.0 and codes 0, without NaN", pytrace=False)
    np.testing.assert_array_equal(
        q[1],
        np.array([32, -127, 64], dtype=np.int8),
        err_msg="a zero channel leaked into its neighbour: scale is per channel",
    )


def test_integer_input_is_cast_before_division():
    q, scale = checked(lambda: absmax_quantize(np.array([254, -100, 3])))
    np.testing.assert_array_equal(
        q,
        np.array([127, -50, 2], dtype=np.int8),
        err_msg="integer input mistake: divide in float, and round 1.5 to 2 (even)",
    )
    assert float(scale) == 2.0, "scale mistake: 254 / 127 = 2.0"


def test_very_small_and_very_large_magnitudes_round_trip():
    for magnitude in (1e-30, 1e30):
        x = np.array([magnitude, -magnitude / 3.0, 0.0])
        q, scale = checked(lambda x=x: absmax_quantize(x))
        if float(scale) != magnitude / 127:
            pytest.fail("scale mistake at extreme magnitudes: use absmax / 127", pytrace=False)
        if int(q[0]) != 127 or int(q[2]) != 0:
            pytest.fail("extreme magnitudes must keep the absmax at code 127", pytrace=False)


def test_invalid_dtypes_shapes_axes_and_values_raise():
    rejects(lambda: absmax_quantize(np.array([True, False])))
    rejects(lambda: absmax_quantize(np.array([1 + 2j])))
    rejects(lambda: absmax_quantize(np.array([1.0, np.nan])))
    rejects(lambda: absmax_quantize(np.array([np.inf, 1.0])))
    rejects(lambda: absmax_quantize(np.array([], dtype=float)))
    rejects(lambda: absmax_quantize(np.float64(2.0)))
    rejects(lambda: absmax_quantize(np.ones((2, 3)), axis=2))
    rejects(lambda: absmax_quantize(np.ones((2, 3)), axis=True))


def test_inputs_are_not_mutated():
    x = np.array([[1.0, -8.0], [0.25, 3.0]])
    original = x.copy()
    absmax_quantize(x, axis=1)
    absmax_quantize(x)
    np.testing.assert_array_equal(x, original, err_msg="input array was mutated")


def test_dequantize_shapes_and_scale_rules():
    q = np.array([[127, -1], [2, 0]], dtype=np.int8)
    out = absmax_dequantize(q, np.array([2.0, 0.5]), axis=0)
    if out.dtype != np.float64 or out.shape != q.shape:
        pytest.fail("dequantize must return float64 with the shape of q", pytrace=False)
    np.testing.assert_allclose(
        out,
        [[254.0, -2.0], [1.0, 0.0]],
        err_msg="each code must be multiplied by its own group's scale",
    )
    rejects(lambda: absmax_dequantize(q, np.array([2.0, 0.5, 1.0]), axis=0))
    rejects(lambda: absmax_dequantize(q.astype(np.int32), np.array([2.0, 0.5]), axis=0))
    rejects(lambda: absmax_dequantize(q, np.array(0.0)))


def test_underflowing_scale_is_rejected():
    rejects(lambda: absmax_quantize(np.array([5e-324, -5e-324])))
    rejects(lambda: absmax_quantize(np.array([[1.0, 2.0], [1e-320, 0.0]]), axis=0))


def test_dequantize_per_channel_scale_with_wrong_length_is_rejected():
    q = np.array([[1, -2], [3, 4]], dtype=np.int8)
    rejects(lambda: absmax_dequantize(q, np.array(2.0), axis=0))
    rejects(lambda: absmax_dequantize(q, np.array([2.0]), axis=0))


def test_dequantize_per_tensor_scale_with_extra_entries_is_rejected():
    q = np.array([[1, -2], [3, 4]], dtype=np.int8)
    rejects(lambda: absmax_dequantize(q, np.array([2.0])))
    rejects(lambda: absmax_dequantize(q, np.array([2.0, 1.0])))


def test_empty_array_is_rejected_even_when_one_axis_is_nonempty():
    rejects(lambda: absmax_quantize(np.zeros((0, 3)), axis=0))


def test_nonfinite_entry_in_dequantize_scale_is_rejected():
    q = np.array([[1, -2], [3, 4]], dtype=np.int8)
    rejects(lambda: absmax_dequantize(q, np.array([2.0, np.inf]), axis=0))
    rejects(lambda: absmax_dequantize(q, np.array([2.0, np.nan]), axis=0))


def test_one_nonpositive_entry_in_dequantize_scale_is_rejected():
    q = np.array([[1, -2], [3, 4]], dtype=np.int8)
    rejects(lambda: absmax_dequantize(q, np.array([0.0, 0.5]), axis=0))
    rejects(lambda: absmax_dequantize(q, np.array([-1.0, 0.5]), axis=0))


def test_per_tensor_dequantize_returns_float64():
    out = absmax_dequantize(np.array([[1, -2]], dtype=np.int8), np.array(0.5))
    if out.dtype != np.float64:
        pytest.fail("per-tensor dequantize must return a float64 array, not a narrower float")
    np.testing.assert_allclose(
        out, [[0.5, -1.0]], err_msg="each code must be multiplied by the one scale"
    )


def test_tiny_but_normal_scale_is_accepted():
    x = np.array([1e-40, 0.0])
    q, scale = checked(lambda: absmax_quantize(x))
    if int(q[0]) != 127 or int(q[1]) != 0:
        pytest.fail("a normal but tiny scale must still map the absmax to code 127")
    np.testing.assert_allclose(
        float(scale), 1e-40 / 127, rtol=1e-12, err_msg="scale must be absmax divided by 127"
    )
