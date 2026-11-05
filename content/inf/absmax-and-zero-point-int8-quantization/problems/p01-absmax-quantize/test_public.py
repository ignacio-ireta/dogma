from __future__ import annotations

import numpy as np
import pytest
from submission import absmax_quantize


def test_ties_round_to_even_codes_with_unit_scale():
    q, scale = absmax_quantize(np.array([127.0, 0.5, 1.5, 2.5, -2.5]))
    np.testing.assert_array_equal(
        q,
        np.array([127, 0, 2, 2, -2], dtype=np.int8),
        err_msg="rounding mistake: ties must go to the even code (0.5 -> 0, 2.5 -> 2)",
    )
    assert float(scale) == 1.0, "scale mistake: per-tensor scale is absmax / 127"


def test_per_channel_rows_get_their_own_scale():
    x = np.array([[127.0, 1.0, -2.0], [0.5, -0.3, 0.1]])
    try:
        q, scale = absmax_quantize(x, axis=0)
    except Exception:
        pytest.fail(
            "per-channel call with axis=0 raised: the scale must have shape (2,)", pytrace=False
        )
    np.testing.assert_array_equal(
        q,
        np.array([[127, 1, -2], [127, -76, 25]], dtype=np.int8),
        err_msg="per-channel mistake: the absmax was taken over the wrong axes or globally",
    )
    np.testing.assert_allclose(
        scale,
        [1.0, 0.5 / 127],
        err_msg="per-channel mistake: each channel needs absmax_c / 127",
    )


def test_all_zero_tensor_has_unit_scale_and_zero_codes():
    try:
        q, scale = absmax_quantize(np.zeros(3))
    except Exception:
        pytest.fail(
            "all-zero tensor raised: a zero absmax needs scale 1.0, not a division by zero",
            pytrace=False,
        )
    np.testing.assert_array_equal(
        q, np.zeros(3, dtype=np.int8), err_msg="zero input must give zero codes"
    )
    assert float(scale) == 1.0, "zero group mistake: use scale 1.0, not 0 or NaN"
