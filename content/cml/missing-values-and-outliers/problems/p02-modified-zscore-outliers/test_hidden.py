"""Hidden tests: an independent MAD from SciPy as the oracle, plus invariance,
zero-MAD, strictness and validation cases. Each failure names the likely mistake first."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import median_abs_deviation
from submission import robust_zscore_outliers

SETTINGS = settings(max_examples=60, deadline=None, report_multiple_bugs=False)


def _reference(x, threshold):
    """Independent route: SciPy's MAD, and the same formula written out with explicit branches."""
    values = np.asarray(x, dtype=np.float64)
    median = np.median(values)
    mad = median_abs_deviation(values)
    if mad == 0:
        return values != median
    score = 0.6745 * (values - median) / mad
    return np.abs(score) > threshold


def _run(x, threshold=3.5):
    try:
        out = robust_zscore_outliers(x, threshold)
    except Exception as exc:
        pytest.fail(f"valid outlier request crashed with {type(exc).__name__}: check the MAD rule")
    if not isinstance(out, np.ndarray) or out.dtype != np.bool_:
        pytest.fail("return a boolean NumPy array with one flag per value")
    return out


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def _sample(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    values = rng.normal(size=n) * 2 + 3
    k = int(rng.integers(0, 4))
    values[: min(k, n)] += rng.choice([-1, 1]) * rng.uniform(10, 60, size=min(k, n))
    return values


@SETTINGS
@given(st.integers(0, 10_000), st.sampled_from([1.0, 2.0, 3.5, 5.0]))
def test_flags_match_independent_scipy_mad_computation(seed, threshold):
    x = _sample(seed)
    want = _reference(x, threshold)
    if not np.array_equal(_run(x, threshold), want):
        pytest.fail(
            "flags differ from the modified z-score: check 0.6745, the MAD and the strict test"
        )


def test_single_large_value_is_flagged_where_the_classic_rule_would_hide_it():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    z_classic = (x - x.mean()) / x.std()
    if np.any(np.abs(z_classic) > 3):
        pytest.fail("sanity: the classic z-score should not flag this sample at threshold 3")
    if not _run(x)[4]:
        pytest.fail("the extreme value must be flagged by the median and MAD rule")


def test_zero_mad_flags_values_different_from_the_median_only():
    if _run([-1.0, 0.0, 0.0, 0.0]).tolist() != [True, False, False, False]:
        pytest.fail("zero MAD: values different from the median must be flagged")
    if _run([5.0, 5.0, 5.0, 5.0, 5.0]).any():
        pytest.fail("zero MAD with every value equal to the median must flag nothing")


@SETTINGS
@given(st.integers(0, 10_000), st.floats(0.1, 50), st.floats(-100, 100))
def test_flags_are_invariant_to_positive_scaling_and_shifting(seed, scale, shift):
    x = _sample(seed)
    if not np.array_equal(_run(x), _run(x * scale + shift)):
        pytest.fail(
            "flags changed under a positive scale and a shift: the score must be scale-free"
        )


def test_comparison_is_strict_at_the_threshold():
    # Median 0 and MAD 1, so M = 0.6745 * x exactly for x = +-2 (doubling is exact in float64).
    x = np.array([-1.0, 0.0, 1.0, 2.0, 0.0])
    boundary = 0.6745 * 2.0
    flags = robust_zscore_outliers(x, boundary)
    if flags[3] or flags[0]:
        pytest.fail(
            "a value with |M| exactly equal to the threshold must not be flagged: use a strict '>'"
        )
    if not robust_zscore_outliers(x, np.nextafter(boundary, 0.0))[3]:
        pytest.fail("a value with |M| just above the threshold must be flagged")


@SETTINGS
@given(st.integers(0, 10_000))
def test_flags_are_symmetric_under_negation(seed):
    x = _sample(seed)
    if not np.array_equal(_run(x), _run(-x)):
        pytest.fail("negating the sample changed the flags: the rule must be two-sided")


@SETTINGS
@given(st.integers(0, 10_000))
def test_a_larger_threshold_never_flags_more_values(seed):
    x = _sample(seed)
    low = _run(x, 2.0)
    high = _run(x, 5.0)
    if np.any(high & ~low):
        pytest.fail(
            "a higher threshold flagged a value the lower one did not: the cut must be monotone"
        )


def test_output_has_one_flag_per_value_and_bool_dtype():
    out = _run([3.0, 1.0, 2.0])
    if out.shape != (3,):
        pytest.fail("shape mismatch: one flag per value, in the same order")


def test_integer_input_gives_the_same_flags_as_float_input():
    x_int = [1, 2, 3, 4, 100, 2, 3]
    if not np.array_equal(_run(x_int), _run(np.asarray(x_int, dtype=float))):
        pytest.fail("integer input must be treated as the same numbers in float")


def test_large_magnitude_offset_does_not_change_the_flags():
    base = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    if not np.array_equal(_run(base + 1e12), _run(base)):
        pytest.fail(
            "a large common offset changed the flags: the score must be computed around the median"
        )


def test_single_value_is_never_flagged():
    if _run([7.0]).tolist() != [False]:
        pytest.fail("a single value equals its own median, so it must not be flagged")


def test_input_is_not_mutated():
    x = np.array([9.0, 1.0, 4.0, 8.0, 3.0, 50.0])
    before = x.copy()
    _run(x)
    if not np.array_equal(x, before):
        pytest.fail("the input array was modified or sorted in place: copy before computing")


@pytest.mark.parametrize(
    ("x", "threshold"),
    [
        ([[1.0, 2.0], [3.0, 4.0]], 3.5),
        ([], 3.5),
        ([1.0, float("nan"), 3.0], 3.5),
        ([1.0, float("inf"), 3.0], 3.5),
        ([1.0, 2.0, 3.0], 0),
        ([1.0, 2.0, 3.0], -2.0),
        ([1.0, 2.0, 3.0], float("nan")),
        ([1.0, 2.0, 3.0], float("inf")),
    ],
    ids=[
        "two-dimensional",
        "empty",
        "nan",
        "infinite",
        "zero-threshold",
        "negative-threshold",
        "nan-threshold",
        "infinite-threshold",
    ],
)
def test_invalid_inputs_raise_value_error(x, threshold):
    _expect_value_error(
        lambda: robust_zscore_outliers(x, threshold),
        "an invalid outlier request was accepted: check the rank, the values and the threshold",
    )


def test_midpoint_of_two_huge_same_sign_values_does_not_overflow():
    out = _run([1.6e308, 1.7e308], threshold=0.5)
    if out.tolist() != [True, True]:
        pytest.fail(
            "the median of two large finite values must not overflow: its midpoint is 1.65e308"
        )


def test_raw_deviation_that_overflows_still_gets_its_modified_score():
    x = [-1.7e308, -0.5e308, 1.0e308, 1.5e308, 1.7e308]
    if _run(x, threshold=3.5).any():
        pytest.fail("an overflowing raw deviation with a small modified score must not be flagged")
    if _run(x, threshold=2.5).tolist() != [True, False, False, False, False]:
        pytest.fail("the overflowing value has |M| of about 2.6, so it must be flagged above 2.5")


def test_large_finite_values_keep_representable_modified_scores():
    out = _run([-1e308, 1e308], threshold=0.5)
    if out.tolist() != [True, True]:
        pytest.fail("large finite magnitudes must not overflow the median or MAD and erase flags")


def test_extreme_tail_does_not_erase_representable_spread_near_the_median():
    got = _run([1e-300, 2e-300, 3e-300, 1e308, 1e308], threshold=0.5)
    if got.tolist() != [True, False, False, True, True]:
        pytest.fail(
            "an extreme tail must not erase small but representable deviations near the median"
        )


def test_wide_spread_with_representable_deviations_flags_nothing():
    got = _run([-1e308, 0.0, 1e308, 1e308], threshold=3.5)
    if got.any():
        pytest.fail(
            "avoid overflow in raw deviations when the final modified score is representable"
        )


def test_midpoint_of_a_small_and_a_huge_positive_value_does_not_overflow():
    # The exact median is 1.1e308, which is finite, but summing the two middle values overflows.
    out = _run([5e307, 1.7e308], threshold=0.5)
    assert out.tolist() == [True, True], (
        "the median of two large positive values overflowed: compute the midpoint as left plus half"
        " the gap when the sum could overflow"
    )


def test_midpoint_of_two_large_negative_values_with_different_magnitudes_does_not_overflow():
    # The exact median is -9e307, which is finite, but summing the two middle values overflows.
    out = _run([-1.7e308, -1e307], threshold=0.5)
    assert out.tolist() == [True, True], (
        "the median of two large negative values overflowed: the midpoint of a negative pair must"
        " be formed without a sum that can overflow, whichever value has the larger magnitude"
    )


def test_default_threshold_is_three_point_five():
    # The modified score of the last value is about 4.0: above the default 3.5, below 7.
    x = np.array([-1.0, 0.0, 0.0, 1.0, 6.0])
    try:
        out = robust_zscore_outliers(x)
    except Exception as exc:
        pytest.fail(f"a call without a threshold must work; raised {type(exc).__name__}")
    assert np.asarray(out).tolist() == [False, False, False, False, True], (
        "with the default threshold a modified score of about 4 must be flagged: the default"
        " threshold is 3.5"
    )
