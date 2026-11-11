import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from submission import ks_drift


def _require(ok, mistake):
    """Fail with the mistake's name only. Hypothesis and the grader show this first line."""
    if not ok:
        raise AssertionError(mistake)


def _run(call, mistake):
    try:
        result = call()
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from None
    _require(
        isinstance(result, tuple) and len(result) == 3,
        "return the KS statistic, p-value and drift flag as a three-entry tuple",
    )
    _require(
        isinstance(result[0], float)
        and isinstance(result[1], float)
        and isinstance(result[2], bool),
        "return floating-point KS statistic and p-value, and a bool drift flag",
    )
    return result


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from None
    raise AssertionError(mistake)


def _gaps(reference, current):
    """Explicit empirical-CDF gaps at every pooled value, with ties counted together."""
    pooled = sorted({float(v) for v in reference} | {float(v) for v in current})

    def ecdf(sample, x):
        return sum(1 for value in sample if value <= x) / len(sample)

    return [ecdf(reference, x) - ecdf(current, x) for x in pooled]


def _limit_p(statistic, n, m):
    return float(stats.kstwobign.sf(math.sqrt(n * m / (n + m)) * statistic))


def _check_statistic(statistic, expected, reference, current):
    """Compare with the scipy statistic computed before the learner ran."""
    if math.isclose(statistic, expected, rel_tol=1e-12, abs_tol=1e-12):
        return
    gaps = _gaps(reference, current)
    if math.isclose(statistic, max(gaps), rel_tol=1e-12, abs_tol=1e-12) or math.isclose(
        statistic, max(-g for g in gaps), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AssertionError(
            "statistic counts only one direction of the gap: use the largest absolute difference"
        )
    raise AssertionError(
        "statistic differs from the two-sample KS distance: check the CDF gaps and ties"
    )


def _check_p_value(p_value, expected, statistic, n, m):
    """Compare with the limiting p-value computed before the learner ran."""
    if math.isclose(p_value, expected, rel_tol=1e-9, abs_tol=1e-12):
        return
    one_size = [float(stats.kstwobign.sf(math.sqrt(size) * statistic)) for size in (n, m)]
    if any(math.isclose(p_value, value, rel_tol=1e-9, abs_tol=0.0) for value in one_size):
        raise AssertionError(
            "p-value uses one sample size: the effective size must be n * m / (n + m)"
        )
    no_root = float(stats.kstwobign.sf(n * m / (n + m) * statistic))
    if math.isclose(p_value, no_root, rel_tol=1e-9, abs_tol=0.0):
        raise AssertionError("lambda must be sqrt(n_eff) * D: the square root is missing")
    raise AssertionError("p-value differs from the limiting Kolmogorov distribution: check lambda")


def _sample(rng, size, shift, spread):
    return rng.normal(loc=shift, scale=spread, size=size)


@pytest.mark.parametrize(
    ("seed", "n", "m", "shift", "spread"),
    [
        (0, 50, 60, 0.0, 1.0),
        (1, 300, 120, 0.4, 1.0),
        (2, 40, 400, -0.6, 2.0),
        (3, 1000, 1000, 0.1, 1.0),
        (4, 7, 9, 1.5, 1.0),
    ],
    ids=["similar", "shifted-unequal", "wider-current", "large-samples", "tiny-samples"],
)
def test_statistic_matches_scipy_on_seeded_samples(seed, n, m, shift, spread):
    rng = np.random.default_rng(seed)
    reference = _sample(rng, n, 0.0, 1.0)
    current = _sample(rng, m, shift, spread)
    expected_statistic = float(stats.ks_2samp(reference, current).statistic)
    statistic, _, _ = _run(lambda: ks_drift(reference, current), "ks_drift raised on valid samples")
    _check_statistic(statistic, expected_statistic, reference, current)


@pytest.mark.parametrize("seed", [10, 11, 12], ids=["ties-a", "ties-b", "ties-c"])
def test_statistic_matches_scipy_with_heavy_ties(seed):
    rng = np.random.default_rng(seed)
    reference = rng.integers(0, 5, size=80).astype(np.float64)
    current = rng.integers(1, 7, size=50).astype(np.float64)
    expected_statistic = float(stats.ks_2samp(reference, current).statistic)
    statistic, _, _ = _run(lambda: ks_drift(reference, current), "ks_drift raised on tied samples")
    _check_statistic(statistic, expected_statistic, reference, current)


def test_tied_values_are_counted_as_one_block():
    statistic, _, _ = _run(
        lambda: ks_drift([1.0, 1.0, 2.0, 2.0], [1.0, 2.0, 2.0, 3.0]),
        "ks_drift raised on a tied example",
    )
    _require(
        math.isclose(statistic, 0.25, rel_tol=1e-12),
        "tied values must be counted together: a tied block is one step of each CDF, not several",
    )


@pytest.mark.parametrize(
    ("seed", "n", "m", "shift"),
    [
        (20, 40, 25, 0.0),
        (21, 200, 150, 0.2),
        (22, 60, 600, 0.3),
        (23, 500, 500, 0.05),
        (24, 12, 30, 0.8),
    ],
    ids=[
        "small-equal-ish",
        "medium-unequal",
        "unequal-large-current",
        "large-small-shift",
        "tiny-reference",
    ],
)
def test_p_value_matches_the_limiting_distribution_at_the_effective_size(seed, n, m, shift):
    rng = np.random.default_rng(seed)
    reference = _sample(rng, n, 0.0, 1.0)
    current = _sample(rng, m, shift, 1.0)
    expected_statistic = float(stats.ks_2samp(reference, current).statistic)
    expected_p = _limit_p(expected_statistic, n, m)
    statistic, p_value, _ = _run(
        lambda: ks_drift(reference, current), "ks_drift raised on valid samples"
    )
    _require(
        math.isclose(statistic, expected_statistic, rel_tol=1e-12, abs_tol=1e-12),
        "statistic differs from the two-sample KS distance before the p-value can be checked",
    )
    _check_p_value(p_value, expected_p, statistic, n, m)


def test_identical_samples_have_zero_statistic_unit_p_value_and_no_drift():
    values = np.arange(10.0)
    statistic, p_value, drifted = _run(
        lambda: ks_drift(values, values.copy()), "ks_drift raised on identical samples"
    )
    _require(statistic == 0.0, "identical samples must give a statistic of exactly zero")
    _require(
        math.isclose(p_value, 1.0, rel_tol=0.0, abs_tol=1e-12),
        "a zero statistic must give a p-value of exactly one: lambda = 0 needs its own case",
    )
    _require(drifted is False, "identical samples must not be flagged as drift")


def test_one_sided_shift_is_detected_with_the_full_statistic():
    rng = np.random.default_rng(30)
    reference = _sample(rng, 300, 0.0, 1.0)
    current = _sample(rng, 300, -0.5, 1.0)
    expected_statistic = float(stats.ks_2samp(reference, current).statistic)
    statistic, _, _ = _run(lambda: ks_drift(reference, current), "ks_drift raised on valid samples")
    _check_statistic(statistic, expected_statistic, reference, current)
    _require(
        statistic > 0.1, "a half-standard-deviation shift must give a clearly positive statistic"
    )


def test_disjoint_samples_have_statistic_one_and_drift():
    reference = np.arange(200.0)
    current = np.arange(200.0) + 1000.0
    expected_p = _limit_p(1.0, 200, 200)
    statistic, p_value, drifted = _run(
        lambda: ks_drift(reference, current), "ks_drift raised on disjoint samples"
    )
    _require(statistic == 1.0, "disjoint samples must give the largest possible statistic, 1")
    _check_p_value(p_value, expected_p, statistic, 200, 200)
    _require(drifted is True, "disjoint samples must be flagged as drift")


def test_drift_flag_follows_the_p_value_against_alpha():
    rng = np.random.default_rng(31)
    reference = _sample(rng, 150, 0.0, 1.0)
    current = _sample(rng, 90, 0.35, 1.0)
    expected_p = _limit_p(float(stats.ks_2samp(reference, current).statistic), 150, 90)
    _, p_value, _ = _run(lambda: ks_drift(reference, current), "ks_drift raised on valid samples")
    for alpha in (0.01, 0.05, 0.2, 0.5):
        _, _, drifted = _run(
            lambda alpha=alpha: ks_drift(reference, current, alpha=alpha),
            "ks_drift raised with a custom alpha",
        )
        _require(
            drifted == (expected_p < alpha),
            "drifted must be True exactly when the p-value is below alpha",
        )
    _require(p_value > 0.0, "the p-value must be positive for these samples")


def test_drift_flag_is_strict_when_alpha_equals_the_p_value():
    reference = np.arange(1.0, 11.0)
    current = np.arange(6.0, 16.0)
    _, p_value, _ = _run(lambda: ks_drift(reference, current), "ks_drift raised on valid samples")
    _, _, at_p = _run(
        lambda: ks_drift(reference, current, alpha=p_value), "ks_drift raised with alpha = p"
    )
    _, _, just_above = _run(
        lambda: ks_drift(reference, current, alpha=float(np.nextafter(p_value, 1.0))),
        "ks_drift raised with alpha just above p",
    )
    _require(at_p is False, "drifted must be False when p equals alpha: the rule is p < alpha")
    _require(just_above is True, "drifted must be True when p is just below alpha")


def test_p_value_series_converges_for_a_small_statistic():
    reference = np.arange(2000.0)
    current = np.arange(2000.0)
    current[1000] = 1000.5  # one value moves past its neighbour: D = 1 / 2000
    expected_p = _limit_p(1 / 2000, 2000, 2000)
    statistic, p_value, _ = _run(
        lambda: ks_drift(reference, current), "ks_drift raised on valid samples"
    )
    _require(
        math.isclose(statistic, 5e-4, rel_tol=1e-9),
        "one moved value out of 2000 must give the statistic 1 / 2000",
    )
    _check_p_value(p_value, expected_p, statistic, 2000, 2000)


def test_p_value_is_accurate_just_below_lambda_one():
    reference = np.arange(1.0, 11.0)
    current = np.arange(5.0, 15.0)  # D = 0.4, lambda = sqrt(5) * 0.4, about 0.894
    expected_p = _limit_p(float(stats.ks_2samp(reference, current).statistic), 10, 10)
    statistic, p_value, _ = _run(
        lambda: ks_drift(reference, current), "ks_drift raised on valid samples"
    )
    _check_p_value(p_value, expected_p, statistic, 10, 10)


@pytest.mark.parametrize(
    "alpha", [0.0, 1.0, -0.1, 1.5], ids=["zero", "one", "negative", "above-one"]
)
def test_alpha_outside_the_open_interval_raises_value_error(alpha):
    _expect_value_error(
        lambda: ks_drift([1.0, 2.0, 3.0], [1.0, 2.0], alpha=alpha),
        "an alpha outside (0, 1) was accepted: alpha must be strictly between 0 and 1",
    )


def test_empty_sample_raises_value_error():
    _expect_value_error(
        lambda: ks_drift([], [1.0, 2.0]),
        "an empty reference sample was accepted: it must contain at least one value",
    )


@pytest.mark.parametrize(
    ("reference", "current"),
    [([1.0, float("nan"), 3.0], [1.0, 2.0]), ([1.0, 2.0], [float("-inf"), 2.0])],
    ids=["nan-in-reference", "infinity-in-current"],
)
def test_non_finite_values_raise_value_error(reference, current):
    _expect_value_error(
        lambda: ks_drift(reference, current),
        "a NaN or infinite value was accepted: samples must be finite",
    )


def test_two_dimensional_input_raises_value_error():
    _expect_value_error(
        lambda: ks_drift(np.ones((3, 2)), [1.0, 2.0]),
        "a two-dimensional sample was accepted: samples must be one-dimensional",
    )


def test_integer_input_gives_float_statistic_and_p_value():
    statistic, p_value, drifted = _run(
        lambda: ks_drift([1, 2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9, 10]),
        "ks_drift raised on integer input",
    )
    _require(isinstance(statistic, float) and isinstance(p_value, float), "outputs must be floats")
    _require(isinstance(drifted, bool), "drifted must be a bool")


def test_inputs_are_not_mutated():
    reference = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    current = np.array([2.0, 6.0, 5.0, 3.0])
    r_before, c_before = reference.copy(), current.copy()
    _run(lambda: ks_drift(reference, current), "ks_drift raised on valid samples")
    _require(np.array_equal(reference, r_before), "reference was modified in place")
    _require(np.array_equal(current, c_before), "current was modified in place")


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(
        st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=40
    ),
    st.lists(
        st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=40
    ),
)
def test_property_is_symmetric_bounded_and_matches_scipy(reference, current):
    expected_statistic = float(stats.ks_2samp(reference, current).statistic)
    statistic, p_value, _ = _run(
        lambda: ks_drift(reference, current), "ks_drift raised on valid lists"
    )
    swapped_statistic, swapped_p, _ = _run(
        lambda: ks_drift(current, reference), "ks_drift raised on the swapped lists"
    )
    _require(0.0 <= statistic <= 1.0, "the statistic must lie in [0, 1]")
    _require(0.0 <= p_value <= 1.0, "the p-value must lie in [0, 1]")
    _require(
        math.isclose(statistic, swapped_statistic, rel_tol=1e-12, abs_tol=1e-12),
        "swapping the samples changed the statistic: the test must be symmetric",
    )
    _require(
        math.isclose(p_value, swapped_p, rel_tol=1e-12, abs_tol=1e-12),
        "swapping the samples changed the p-value: the test must be symmetric",
    )
    _check_statistic(statistic, expected_statistic, reference, current)


def test_p_value_stays_accurate_for_a_tiny_statistic():
    reference = np.array([0.0] + [1.0] * 1999)
    current = np.array([0.0] + [1.0] * 2000)  # one extra copy of 1.0: D = 1 / (2000 * 2001)
    scipy_statistic = float(stats.ks_2samp(reference, current).statistic)
    expected_p = _limit_p(scipy_statistic, 2000, 2001)
    statistic, p_value, _ = _run(
        lambda: ks_drift(reference, current), "ks_drift raised on valid samples"
    )
    _require(
        math.isclose(statistic, scipy_statistic, rel_tol=1e-9, abs_tol=1e-15),
        "statistic must be the two-sample KS distance on this pair",
    )
    _require(
        math.isclose(p_value, expected_p, rel_tol=1e-9, abs_tol=1e-9),
        "for a tiny statistic from nearly identical samples the p-value must still follow "
        "the limiting distribution: the formula is wrong for very small lambda",
    )
