import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from submission import discrete_moments

SEEDS = st.integers(min_value=0, max_value=10**6)
SIZES = st.integers(min_value=1, max_value=12)


def _random_distribution(seed, n):
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=10.0, size=n)
    weights = rng.uniform(0.01, 1.0, size=n)
    return values, weights / weights.sum()


def _moments(values, probs, mistake):
    try:
        return discrete_moments(values, probs)
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, SIZES)
def test_matches_scipy_discrete_distribution_on_random_inputs(seed, n):
    values, probs = _random_distribution(seed, n)
    dist = stats.rv_discrete(values=(values, probs))
    got = _moments(
        values, probs, "a valid distribution was rejected or failed: check the weighted sums"
    )
    np.testing.assert_allclose(
        got,
        [dist.mean(), dist.var(), dist.std()],
        rtol=1e-9,
        atol=1e-12,
        err_msg="mean, variance or std differ from scipy.stats.rv_discrete: check the weights "
        "and the squared deviations",
    )


def test_variance_stays_accurate_when_every_value_is_shifted_by_a_billion():
    base = np.array([0.0, 1.0, 2.5])
    probs = np.array([0.2, 0.5, 0.3])
    expected = stats.rv_discrete(values=(base, probs))
    shifted = base + 1e9  # built before any call, so nothing a learner does to base can change it
    _, small_variance, _ = _moments(base, probs, "the variance of small values failed")
    np.testing.assert_allclose(
        small_variance,
        expected.var(),
        rtol=1e-9,
        err_msg="the variance is wrong even for small values: check the weights and the squared "
        "deviations",
    )
    mean, variance, _ = _moments(
        shifted,
        probs,
        "variance lost precision at a large offset: the E[X^2] - mean^2 shortcut cancels digits",
    )
    np.testing.assert_allclose(
        variance,
        expected.var(),
        rtol=1e-9,
        err_msg="variance lost precision at a large offset: the E[X^2] - mean^2 shortcut "
        "cancels digits, so the variance is wrong",
    )
    np.testing.assert_allclose(
        mean,
        1e9 + expected.mean(),
        rtol=1e-12,
        err_msg="the mean of a shifted distribution should move by exactly the shift",
    )


def test_probabilities_that_sum_to_one_up_to_rounding_are_accepted():
    # np.sum of these four probabilities is 0.9999999999999999 in float64.
    mean, _, _ = _moments(
        [0.0, 1.0, 2.0, 3.0],
        [0.7, 0.1, 0.1, 0.1],
        "probabilities that sum to 1 up to float rounding must be accepted: compare the sum "
        "with a tolerance, not with ==",
    )
    np.testing.assert_allclose(
        mean,
        0.6,
        rtol=1e-9,
        atol=1e-12,
        err_msg="the mean is wrong for probabilities that sum to 1 up to rounding",
    )


def test_probabilities_off_by_a_real_amount_are_rejected():
    _raised_value_error(
        lambda: discrete_moments([0.0, 1.0], [0.5, 0.500001]),
        "probabilities summing to slightly more than 1 are not a distribution: the tolerance "
        "is 1e-9, so raise ValueError",
    )


def test_probabilities_off_by_a_small_real_amount_are_rejected():
    _raised_value_error(
        lambda: discrete_moments([0.0, 1.0], [0.5, 0.500000005]),
        "probabilities summing to slightly more than 1 are not a distribution: the tolerance "
        "is 1e-9, so raise ValueError",
    )


def test_nan_probability_raises_value_error():
    _raised_value_error(
        lambda: discrete_moments([0.0, 1.0], [0.5, np.nan]),
        "a NaN probability must raise ValueError: check isfinite before the sum test",
    )


def test_infinite_value_raises_value_error():
    _raised_value_error(
        lambda: discrete_moments([0.0, np.inf], [0.5, 0.5]),
        "an infinite value has no mean: raise ValueError",
    )


def test_negative_probability_raises_value_error():
    _raised_value_error(
        lambda: discrete_moments([0.0, 1.0, 2.0], [0.75, -0.25, 0.5]),
        "a negative probability is not allowed: raise ValueError even when the sum is 1",
    )


def test_length_mismatch_raises_value_error():
    _raised_value_error(
        lambda: discrete_moments([0.0, 1.0, 2.0], [0.5, 0.5]),
        "values and probabilities of different lengths: raise ValueError instead of truncating",
    )


def test_empty_distribution_raises_value_error():
    _raised_value_error(
        lambda: discrete_moments([], []),
        "an empty distribution has no mean: raise ValueError",
    )


def test_two_dimensional_input_raises_value_error():
    _raised_value_error(
        lambda: discrete_moments([[0.0, 1.0]], [[0.5, 0.5]]),
        "values must be 1-D: raise ValueError for a 2-D array",
    )


def test_integer_values_return_three_floats():
    result = _moments(
        [1, 2], [0.25, 0.75], "integer values must be accepted and converted to float"
    )
    assert len(result) == 3, "return a (mean, variance, std) tuple of three values"
    assert all(isinstance(value, float) for value in result), "return plain floats for all three"
    np.testing.assert_allclose(
        result,
        [1.75, 0.1875, np.sqrt(0.1875)],
        rtol=1e-12,
        err_msg="integer values gave the wrong moments: check the weighted mean",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    SEEDS,
    SIZES,
    st.sampled_from([-3.0, 0.5, 2.0]),
    st.sampled_from([-7.0, 0.0, 4.0]),
)
def test_affine_map_shifts_mean_and_scales_variance_by_slope_squared(seed, n, slope, shift):
    values, probs = _random_distribution(seed, n)
    image = slope * values + shift
    mean, variance, _ = _moments(values, probs, "a valid distribution was rejected or failed")
    mean2, variance2, _ = _moments(image, probs, "a valid affine image was rejected or failed")
    np.testing.assert_allclose(
        mean2,
        slope * mean + shift,
        rtol=1e-9,
        atol=1e-9,
        err_msg="E[aX + b] should equal a E[X] + b: linearity of expectation",
    )
    np.testing.assert_allclose(
        variance2,
        slope**2 * variance,
        rtol=1e-9,
        atol=1e-9,
        err_msg="Var(aX + b) should equal a^2 Var(X): the shift must drop out and the slope "
        "must be squared",
    )


def test_inputs_are_not_mutated():
    values = np.array([0.0, 1.0, 3.0])
    # Sums to 1 only within the tolerance, so an in-place normalisation would change it.
    probs = np.array([0.5, 0.25, 0.25 + 5e-10])
    values_before, probs_before = values.copy(), probs.copy()
    discrete_moments(values, probs)
    np.testing.assert_array_equal(
        values, values_before, err_msg="values was modified in place: centre a copy, not the input"
    )
    np.testing.assert_array_equal(
        probs,
        probs_before,
        err_msg="probs was modified in place: do not normalise the caller's array",
    )


def test_a_single_probability_does_not_broadcast_over_several_values():
    _raised_value_error(
        lambda: discrete_moments([0.0, 1.0, 2.0], [1.0]),
        "values and probabilities of different lengths must raise ValueError, even when one "
        "side has length 1",
    )


def test_zero_probability_entries_are_accepted():
    mean, variance, _ = _moments(
        [0.0, 1.0, 2.0],
        [0.0, 0.5, 0.5],
        "a zero probability is non-negative and must be accepted",
    )
    np.testing.assert_allclose(
        [mean, variance],
        [1.5, 0.25],
        rtol=1e-12,
        err_msg="a zero-probability outcome adds nothing: the mean and variance are those of the "
        "remaining outcomes",
    )


def test_probabilities_off_by_one_and_a_half_parts_per_billion_are_rejected():
    _raised_value_error(
        lambda: discrete_moments([0.0, 1.0], [0.5, 0.5 + 1.5e-9]),
        "probabilities whose sum is 1.5e-9 away from 1 are outside the 1e-9 tolerance and must "
        "raise ValueError",
    )


def test_variance_uses_the_probabilities_as_given_not_rescaled():
    # Contract: variance = sum p (x - mean)^2 with the probabilities as given. Here
    # p = [0.5, 0.5 + 9e-10] sums to 1 + 9e-10, inside the tolerance, so the variance
    # must not be divided by that total. A 9e-10 relative change needs a tight check.
    values = np.array([0.0, 1e6])
    probs = np.array([0.5, 0.5 + 9e-10])
    mean = probs[0] * values[0] + probs[1] * values[1]
    expected = probs[0] * (values[0] - mean) ** 2 + probs[1] * (values[1] - mean) ** 2
    _, variance, _ = _moments(
        values, probs, "a distribution whose sum is within tolerance of 1 failed"
    )
    np.testing.assert_allclose(
        variance,
        expected,
        rtol=1e-12,
        atol=0.0,
        err_msg="variance: use sum p (x - mean)^2 with the probabilities as given, no rescaling",
    )


def test_probability_just_below_zero_raises_value_error():
    # Contract: every probability is non-negative, so -1e-13 must raise, even though
    # it is within float noise of zero. The probabilities sum to 1, so only the sign fails.
    # The variance stays positive here, so a learner who leaves out the sign check returns
    # a number instead of failing on a square root of a negative variance.
    values = [0.0, 1.0, 2.0]
    probs = [-1e-13, 0.5, 0.5 + 1e-13]
    _raised_value_error(
        lambda: discrete_moments(values, probs),
        "a negative probability, even a tiny one, is not a distribution: raise ValueError",
    )
