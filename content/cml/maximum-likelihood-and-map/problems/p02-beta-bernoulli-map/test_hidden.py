import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import beta_bernoulli_map

from dogma_data import rng, sample_rows

GRID = np.linspace(1e-6, 1 - 1e-6, 200_001)
COUNTS = st.integers(min_value=1, max_value=30)
PRIORS = st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False)


def _grid_mode(k, n, alpha, beta):
    """Independent check: the grid point that maximises the log-posterior."""
    a, b = alpha + k, beta + n - k
    log_posterior = (a - 1) * np.log(GRID) + (b - 1) * np.log1p(-GRID)
    return GRID[int(np.argmax(log_posterior))]


def _estimates(k, n, alpha, beta, mistake):
    try:
        return beta_bernoulli_map(k, n, alpha, beta)
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


@pytest.mark.parametrize(
    ("k", "n", "alpha", "beta"),
    [
        (None, 10, 2, 2),
        (None, 20, 3, 0.5),
        (None, 4, 1.5, 2.5),
        (None, 12, 2, 2),
        (None, 3, 3, 5),
        (0, 10, 2, 2),
    ],
    ids=["case-a", "case-b", "case-c", "case-d", "case-e", "case-f"],
)
def test_interior_map_matches_grid_search_of_the_log_posterior(k, n, alpha, beta):
    if k is None:
        # Real Bernoulli data: the number of benign tumours among n drawn from breast_cancer.
        # The priors are parameters of the model and stay as written.
        _, benign = sample_rows("breast_cancer", rng(f"beta-bernoulli-map-p02:trials:{n}"), n, 1)
        k = int(benign.sum())
    _, map_estimate = _estimates(k, n, alpha, beta, "a valid interior case was rejected or failed")
    assert abs(map_estimate - _grid_mode(k, n, alpha, beta)) <= 1e-4, (
        "MAP differs from the grid maximiser of the log-posterior: check a and b"
    )


def test_zero_successes_with_weak_prior_puts_map_at_zero():
    for alpha, beta in ((0.5, 2.0), (1.0, 2.0)):
        _, map_estimate = _estimates(
            0, 5, alpha, beta, "a valid boundary case was rejected or failed"
        )
        assert map_estimate == pytest.approx(0.0, abs=1e-12), (
            "when a = alpha + k is at most 1 the posterior peaks at 0: MAP should be 0.0"
        )
        assert _grid_mode(0, 5, alpha, beta) <= 1e-5, (
            "the grid search should also peak at the lower boundary"
        )


@pytest.mark.parametrize(
    ("k", "n", "alpha", "beta"),
    [(5, 5, 1.0, 0.5), (4, 4, 2.0, 1.0)],
    ids=["case-a", "case-b"],
)
def test_all_successes_with_weak_beta_puts_map_at_one(k, n, alpha, beta):
    _, map_estimate = _estimates(k, n, alpha, beta, "a valid boundary case was rejected or failed")
    assert map_estimate == pytest.approx(1.0, abs=1e-12), (
        "when b = beta + n - k is at most 1 the posterior peaks at 1: MAP is 1.0"
    )
    assert _grid_mode(k, n, alpha, beta) >= 1 - 1e-5, (
        "the grid search should peak at the upper boundary"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(COUNTS, st.data())
def test_uniform_prior_map_equals_mle(n, data):
    k = data.draw(st.integers(min_value=0, max_value=n))
    mle, map_estimate = _estimates(k, n, 1.0, 1.0, "a uniform-prior case was rejected or failed")
    assert map_estimate == pytest.approx(mle, abs=1e-12), (
        "with a uniform prior the posterior mode equals the observed proportion k / n"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.floats(min_value=1.0, max_value=6.0),
    st.floats(min_value=1.0, max_value=6.0),
    COUNTS,
    st.data(),
)
def test_interior_mode_is_the_mode_not_the_posterior_mean(alpha, beta, n, data):
    k = data.draw(st.integers(min_value=0, max_value=n))
    _, map_estimate = _estimates(k, n, alpha, beta, "a valid interior case was rejected or failed")
    assert map_estimate == pytest.approx((alpha + k - 1) / (alpha + beta + n - 2), rel=1e-12), (
        "the MAP is the posterior mode (a - 1) / (a + b - 2), not the posterior mean a / (a + b)"
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(PRIORS, PRIORS, COUNTS, st.data())
def test_map_always_lies_in_the_unit_interval(alpha, beta, n, data):
    k = data.draw(st.integers(min_value=0, max_value=n))
    _, map_estimate = _estimates(k, n, alpha, beta, "a valid case was rejected or failed")
    assert math.isfinite(map_estimate) and 0.0 <= map_estimate <= 1.0, (
        "the MAP is a probability: keep it in [0, 1] and check the boundary cases"
    )


def test_map_approaches_mle_as_the_data_grow():
    n, k = 1_000_000, 300_000
    mle, map_estimate = _estimates(k, n, 2.0, 2.0, "a large sample was rejected or failed")
    assert abs(map_estimate - mle) < 1e-5, (
        "with a fixed prior, MAP should converge to the MLE as the number of flips grows"
    )


def test_zero_trials_raises_value_error():
    _raised_value_error(
        lambda: beta_bernoulli_map(0, 0, 2.0, 3.0),
        "an empty sample has no maximum likelihood estimate: raise ValueError",
    )


@pytest.mark.parametrize("successes", [-1, 11], ids=["below-range", "above-range"])
def test_successes_outside_zero_to_trials_raise_value_error(successes):
    _raised_value_error(
        lambda: beta_bernoulli_map(successes, 10, 1.0, 1.0),
        "successes must satisfy 0 <= successes <= trials: raise ValueError",
    )


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [(0.0, 1.0), (1.0, -2.0), (float("nan"), 1.0), (1.0, float("inf"))],
    ids=["zero-prior", "negative-prior", "nan-prior", "inf-prior"],
)
def test_invalid_prior_parameters_raise_value_error(alpha, beta):
    _raised_value_error(
        lambda: beta_bernoulli_map(3, 10, alpha, beta),
        "a prior shape must be a finite number greater than 0: raise ValueError",
    )


def test_boolean_count_raises_value_error():
    _raised_value_error(
        lambda: beta_bernoulli_map(True, 1, 1.0, 1.0),
        "a bool is not a whole-number count: raise ValueError, even though True == 1",
    )


def test_fractional_trials_raise_value_error():
    _raised_value_error(
        lambda: beta_bernoulli_map(2, 10.5, 1.0, 1.0),
        "a fractional number of trials is not a whole number: raise ValueError",
    )


def test_fractional_successes_raise_value_error():
    _raised_value_error(
        lambda: beta_bernoulli_map(2.5, 10, 1.0, 1.0),
        "a fractional count of successes is not a whole number: raise ValueError",
    )


def test_whole_float_counts_are_accepted():
    mle, map_estimate = _estimates(3.0, 10.0, 2, 2, "whole-number floats were rejected")
    assert mle == pytest.approx(0.3, abs=1e-12), "3.0 successes out of 10.0 trials is an MLE of 0.3"
    assert map_estimate == pytest.approx(1 / 3, abs=1e-12), "the MAP should match the integer call"


@pytest.mark.parametrize(
    ("k", "n"), [(1, 4), (2, 3), (0, 7), (7, 7)], ids=["case-a", "case-b", "case-c", "case-d"]
)
def test_mle_is_the_observed_proportion(k, n):
    mle, _ = _estimates(k, n, 2.0, 2.0, "a valid case was rejected or failed")
    assert mle == pytest.approx(k / n, abs=1e-12), (
        "the MLE is the observed proportion k / n, computed with true division"
    )


def test_returns_two_python_floats():
    result = _estimates(3, 10, 2, 2, "a valid case was rejected or failed")
    assert len(result) == 2, "return the pair (mle, map_estimate)"
    assert all(isinstance(value, float) for value in result), (
        "return plain floats for both estimates"
    )
