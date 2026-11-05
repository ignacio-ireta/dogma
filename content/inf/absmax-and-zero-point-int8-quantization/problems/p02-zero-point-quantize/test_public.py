from __future__ import annotations

import numpy as np
import pytest
from submission import zero_point_quantize


def test_negative_and_positive_range_uses_the_shifted_grid():
    q, scale, zero_point = zero_point_quantize(np.array([-100.0, 0.0, 155.0]))
    np.testing.assert_array_equal(
        q,
        np.array([-128, -28, 127], dtype=np.int8),
        err_msg="code mismatch: the zero point must send lo to -128 and 0.0 to its own code",
    )
    assert scale == 1.0, "scale mistake: (hi - lo) / 255 with hi - lo = 255"
    assert zero_point == -28, "zero point mistake: zero_point = -128 - round(lo / scale)"


def test_all_positive_input_still_includes_zero_in_the_range():
    q, _scale, zero_point = zero_point_quantize(np.array([1.0, 2.0, 255.0]))
    np.testing.assert_array_equal(
        q,
        np.array([-127, -126, 127], dtype=np.int8),
        err_msg="range mistake: lo must be min(min(x), 0), so lo = 0 here",
    )
    assert zero_point == -128, "zero point mistake: with lo = 0 the zero point is -128"


def test_all_zero_tensor_has_unit_scale_and_dequantizes_to_zero():
    try:
        q, scale, zero_point = zero_point_quantize(np.zeros(2))
    except Exception:
        pytest.fail(
            "all-zero tensor raised: lo == hi needs scale 1.0, not a division by zero",
            pytrace=False,
        )
    np.testing.assert_array_equal(q, np.array([-128, -128], dtype=np.int8))
    assert scale == 1.0, "degenerate range mistake: use scale 1.0 when lo == hi"
    assert (int(q[0]) - zero_point) * scale == 0.0, "zero must dequantize to exactly 0.0"
