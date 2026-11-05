import numpy as np
from submission import ks_drift


def test_identical_samples_have_zero_statistic_unit_p_value_and_no_drift():
    statistic, p_value, drifted = ks_drift(np.arange(10.0), np.arange(10.0))
    assert statistic == 0.0, "identical samples have no gap between their CDFs"
    assert abs(p_value - 1.0) < 1e-12, "a zero statistic gives a p-value of exactly one"
    assert drifted is False, "identical samples must not be flagged as drift"


def test_shifted_samples_match_the_worked_statistic_and_p_value():
    statistic, p_value, drifted = ks_drift(np.arange(1.0, 11.0), np.arange(6.0, 16.0))
    np.testing.assert_allclose(
        statistic, 0.5, rtol=1e-12, err_msg="statistic must be the largest CDF gap, 0.5"
    )
    np.testing.assert_allclose(
        p_value,
        0.1640791977266521,
        rtol=1e-9,
        err_msg="p-value must be the limiting Kolmogorov distribution at sqrt(5) * 0.5",
    )
    assert drifted is False, "p = 0.164 is above alpha = 0.05, so there is no drift"


def test_left_shift_is_detected_as_strongly_as_right_shift():
    left = ks_drift(np.arange(10.0), np.arange(10.0) - 5.0)
    right = ks_drift(np.arange(10.0), np.arange(10.0) + 5.0)
    assert left[0] == right[0] == 0.5, (
        "a shift to the left must give the same statistic as one to the right"
    )


def test_alpha_outside_the_open_interval_raises_value_error():
    try:
        ks_drift([1.0, 2.0], [1.0, 2.0], alpha=1.0)
    except ValueError:
        return
    raise AssertionError("alpha = 1 was accepted: alpha must be strictly between 0 and 1")
