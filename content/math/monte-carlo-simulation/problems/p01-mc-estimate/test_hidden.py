"""Hidden tests: an independent recomputation, exact and seeded checks, and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or an expected
answer.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from submission import mc_estimate

from dogma_data import load, sample_rows
from dogma_data import rng as data_rng

Z95 = float(stats.norm.ppf(0.975))


def _identity(x):
    return x


def _square(x):
    return np.asarray(x) ** 2


@functools.cache
def _population(name):
    """A real population to sample from, sorted so its order is the same in every run.

    house-values: the median house value of 6,000 California districts, in units of 100,000
    dollars, skewed and capped at 5. wine-alcohol: the alcohol content of 178 wines, in
    percent. petal-length: the petal length of 150 irises, in centimetres, with two modes.
    """
    if name == "house-values":
        _, values = sample_rows("california_housing", data_rng("p01-mc-estimate:values"), 6000)
    elif name == "wine-alcohol":
        values = load("wine").X[:, 0]
    else:
        values = load("iris").X[:, 2]
    return np.sort(np.asarray(values, dtype=np.float64))


def _resample(name):
    """A sampler that draws n members of a real population, with replacement."""

    def draw(rng, n):
        values = _population(name)
        return values[rng.integers(0, values.shape[0], size=n)]

    return draw


_house_value_draws = _resample("house-values")
SAMPLERS = {name: _resample(name) for name in ("house-values", "wine-alcohol", "petal-length")}
INTEGRANDS = {"identity": _identity, "square": _square, "exp": np.exp}


def _unpack(result):
    """Return (estimate, standard error, lo, hi) as floats, or fail naming the return shape."""
    try:
        estimate, standard_error, (lo, hi) = result
        return float(estimate), float(standard_error), float(lo), float(hi)
    except (TypeError, ValueError) as exc:
        pytest.fail(
            "return (estimate, standard_error, (lo, hi)): a 3-tuple whose last item is a "
            f"2-tuple ({type(exc).__name__})"
        )


def _mc(f, sample, n, *, seed, confidence=0.95):
    """Call the learner on valid inputs. A crash is reported as a mistake, not a traceback."""
    try:
        return mc_estimate(f, sample, n, seed=seed, confidence=confidence)
    except Exception as exc:
        pytest.fail(f"valid inputs must not raise: mc_estimate raised {type(exc).__name__}")


def _raises_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def _reference(f, sample, n, seed, confidence):
    """Independent recomputation: explicit sum of squares and scipy's normal quantile."""
    rng = np.random.default_rng(seed)
    y = np.asarray(f(sample(rng, n)), dtype=np.float64)
    mean = float(y.sum() / n)
    sum_squares = float(sum((value - mean) ** 2 for value in y))
    se = math.sqrt(sum_squares / (n - 1)) / math.sqrt(n)
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    return [mean, se, mean - z * se, mean + z * se]


@settings(max_examples=120, deadline=None, report_multiple_bugs=False)
@given(
    sampler=st.sampled_from(sorted(SAMPLERS)),
    integrand=st.sampled_from(sorted(INTEGRANDS)),
    n=st.integers(min_value=2, max_value=300),
    seed=st.integers(min_value=0, max_value=10**6),
    confidence=st.floats(min_value=0.5, max_value=0.999),
)
def test_matches_independent_recomputation_on_random_problems(
    sampler, integrand, n, seed, confidence
):
    f, sample = INTEGRANDS[integrand], SAMPLERS[sampler]
    got = _unpack(_mc(f, sample, n, seed=seed, confidence=confidence))
    np.testing.assert_allclose(
        got,
        _reference(f, sample, n, seed, confidence),
        rtol=1e-9,
        atol=1e-12,
        err_msg="estimate, standard error or interval differs from the recomputation: check the "
        "n - 1 divisor, the sqrt(n) and the normal quantile",
    )


def test_known_sample_gives_exact_mean_standard_error_and_interval():
    def ramp(rng, n):
        return np.arange(1.0, n + 1)

    estimate, se, lo, hi = _unpack(_mc(_identity, ramp, 4, seed=0))
    np.testing.assert_allclose(
        estimate,
        2.5,
        rtol=1e-12,
        err_msg="the estimate is not the mean of f over the draws: "
        "sample must be called once, with n",
    )
    np.testing.assert_allclose(
        se,
        math.sqrt(5 / 3) / 2,
        rtol=1e-12,
        err_msg="the standard error must use the sample sd with divisor n - 1, divided by sqrt(n)",
    )
    np.testing.assert_allclose(
        [lo, hi],
        [2.5 - Z95 * se, 2.5 + Z95 * se],
        rtol=1e-9,
        err_msg="the interval must be the estimate plus or minus z times the standard error",
    )


def test_sample_is_called_exactly_once_with_all_n_draws():
    calls = []

    def recording_sample(rng, n):
        calls.append(n)
        return _house_value_draws(rng, n)

    _unpack(_mc(_identity, recording_sample, 50, seed=3))
    assert calls == [50], (
        "sample must be called exactly once with the full n, not in batches and not twice"
    )


def test_f_is_applied_once_to_the_whole_sample_array():
    shapes = []

    def recording_f(x):
        shapes.append(np.shape(x))
        return _square(x)

    _unpack(_mc(recording_f, _house_value_draws, 40, seed=1))
    assert shapes == [(40,)], (
        "f must be applied once to the whole (n,) array, not element by element or twice"
    )


def test_sample_array_is_not_reordered_in_place():
    # Unsorted on purpose: sorting an already sorted array in place changes nothing, so it passes.
    held = np.array([0.7, 0.1, 0.9, 0.3, 0.0, 0.6, 0.2, 0.8, 0.5])
    snapshot = held.copy()

    def sample_held(rng, n):
        return held  # the caller keeps this array, in this order

    _unpack(_mc(lambda x: x**2, sample_held, 9, seed=0))
    np.testing.assert_array_equal(
        held, snapshot, err_msg="the array returned by sample was reordered: work on a copy"
    )


def test_sample_array_returned_by_sample_is_not_modified():
    held = np.linspace(0.0, 1.0, 9)
    snapshot = held.copy()

    def sample_held(rng, n):
        return held  # the caller keeps this array; the estimator must leave it alone

    _unpack(_mc(lambda x: x**2, sample_held, 9, seed=0))
    np.testing.assert_array_equal(
        held, snapshot, err_msg="the array returned by sample was modified: work on a copy"
    )


def test_same_seed_reproduces_the_result_and_another_seed_changes_it():
    first = _unpack(_mc(_square, _house_value_draws, 500, seed=11))
    again = _unpack(_mc(_square, _house_value_draws, 500, seed=11))
    other = _unpack(_mc(_square, _house_value_draws, 500, seed=12))
    assert first == again, (
        "the same seed must give identical output: build the generator from seed in the function"
    )
    assert first[0] != other[0], "a different seed must change the draws: check that seed is used"


def test_interval_covers_the_truth_at_the_nominal_rate_across_seeds():
    # The truth is known exactly: the mean of the squared values over the whole real population.
    truth = float(np.mean(_population("house-values") ** 2))
    hits = 0
    for seed in range(1000):
        _, _, lo, hi = _unpack(_mc(_square, _house_value_draws, 200, seed=seed))
        hits += lo <= truth <= hi
    window = 4 * math.sqrt(0.95 * 0.05 / 1000)
    assert abs(hits / 1000 - 0.95) <= window, (
        "interval coverage is far from the nominal 95%: check the normal quantile z and the "
        "standard error"
    )


def test_expected_exp_of_a_standard_normal_matches_the_closed_form():
    estimate, se, lo, hi = _unpack(
        _mc(np.exp, lambda rng, n: rng.standard_normal(n), 200_000, seed=1)
    )
    truth = math.exp(0.5)
    assert lo <= truth <= hi, (
        "the 95% interval should cover E[exp(Z)] = e^(1/2): check the interval and f"
    )
    assert abs(estimate - truth) <= 4 * se, (
        "the estimate of E[exp(Z)] is far from e^(1/2): check that f is applied to the draws"
    )


def test_standard_error_shrinks_like_one_over_root_n():
    ratios = []
    for seed in range(200):
        _, se_small, _, _ = _unpack(_mc(_identity, _house_value_draws, 1000, seed=seed))
        _, se_large, _, _ = _unpack(_mc(_identity, _house_value_draws, 4000, seed=seed))
        ratios.append(se_large / se_small)
    assert abs(float(np.mean(ratios)) - 0.5) <= 0.02, (
        "quadrupling n must halve the standard error: check the division by sqrt(n)"
    )


def test_standard_error_is_accurate_at_a_large_offset():
    def shifted_draws(rng, n):
        return 1e9 + _house_value_draws(rng, n)

    _, se, _, _ = _unpack(_mc(_identity, shifted_draws, 1000, seed=5))
    values = 1e9 + _house_value_draws(np.random.default_rng(5), 1000)
    expected = float(np.std(values, ddof=1)) / math.sqrt(1000)
    np.testing.assert_allclose(
        se,
        expected,
        rtol=1e-6,
        err_msg="standard error lost precision at a large offset: use deviations from the mean, "
        "not E[y^2] - E[y]^2",
    )


def test_constant_integrand_has_zero_standard_error_and_a_point_interval():
    def constant(x):
        return np.full(np.shape(x), 2.0)

    estimate, se, lo, hi = _unpack(_mc(constant, _house_value_draws, 10, seed=0))
    assert estimate == 2.0, "a constant integrand has that constant as its expectation"
    assert se == 0.0, "no variation in f means a standard error of exactly zero"
    assert lo == hi == 2.0, "with zero standard error the interval collapses onto the estimate"


@pytest.mark.parametrize("n", [0, 1, -3], ids=["no-draws", "one-draw", "negative"])
def test_sample_size_below_two_raises_value_error(n):
    _raises_value_error(
        lambda: mc_estimate(_identity, _house_value_draws, n, seed=0),
        "fewer than two draws has no sample standard deviation: raise ValueError before "
        "dividing by n - 1",
    )


@pytest.mark.parametrize(
    "confidence", [0.0, 1.0, 1.5, -0.2], ids=["zero", "one", "above-one", "negative"]
)
def test_confidence_outside_the_open_unit_interval_raises_value_error(confidence):
    _raises_value_error(
        lambda: mc_estimate(_identity, _house_value_draws, 20, seed=0, confidence=confidence),
        "a confidence level must be strictly between 0 and 1: raise ValueError",
    )


def test_return_value_is_a_three_tuple_with_a_pair_for_the_interval():
    result = _mc(_identity, _house_value_draws, 20, seed=0)
    assert isinstance(result, tuple) and len(result) == 3, (
        "return a 3-tuple: (estimate, standard_error, (lo, hi))"
    )
    assert isinstance(result[2], tuple) and len(result[2]) == 2, (
        "the third item must be the (lo, hi) pair, as a tuple"
    )
    assert all(isinstance(v, float) for v in (result[0], result[1], *result[2])), (
        "the estimate, the standard error and both bounds must be plain floats"
    )


def test_default_confidence_gives_a_95_percent_interval():
    def ramp(rng, n):
        return np.arange(1.0, n + 1)

    estimate, se, lo, hi = _unpack(mc_estimate(_identity, ramp, 4, seed=0))  # no confidence given
    np.testing.assert_allclose(
        [lo, hi],
        [estimate - Z95 * se, estimate + Z95 * se],
        rtol=1e-9,
        err_msg="when confidence is not given, the interval must be the 95% one: check the default",
    )


def test_seed_is_used_only_to_build_the_generator_and_leaves_numpy_global_state_alone():
    # Contract: "seed ... is used only to build rng". The global NumPy generator must not be
    # reseeded or advanced by the call, so its next draw is the same as with no call at all.
    np.random.seed(12345)
    expected_next_draw = np.random.random()
    np.random.seed(12345)
    _mc(_identity, _house_value_draws, 50, seed=7)
    actual_next_draw = np.random.random()
    assert actual_next_draw == expected_next_draw, (
        "seed is used only to build rng: the global NumPy generator was reseeded or advanced "
        "by the call"
    )
