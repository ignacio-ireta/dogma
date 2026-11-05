from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import posterior

SEEDS = st.integers(min_value=0, max_value=10**6)
LIK = [[0.9, 0.1], [0.2, 0.8]]
ORACLE_HINT = (
    "posterior differs from the exact Bayes update: multiply the prior by the likelihood of every "
    "observation, then normalise over the hypotheses"
)
UNDERFLOW_HINT = (
    "long sequence: the posterior is 0/0 = NaN because the likelihoods underflow float64. Multiply "
    "in log space, and subtract the maximum log before exp (log-sum-exp)"
)
SEQUENCE_HINT = (
    "updating in two steps must equal one batch update: observations are conditionally "
    "independent given the hypothesis, so each posterior becomes the next prior"
)
ORDER_HINT = (
    "the posterior must not depend on the order of the observations (they are i.i.d. given the "
    "hypothesis); an update that treats the sequence positionally is a bug"
)
ZERO_PRIOR_HINT = (
    "a zero prior must keep an exactly zero posterior: a probability floor or clip gives the "
    "hypothesis weight it was never allowed to have"
)
ZERO_LIKELIHOOD_HINT = (
    "an observation with likelihood 0 under a hypothesis must eliminate it exactly; the zero "
    "must not be replaced by a small epsilon"
)
MUTATION_HINT = "inputs must not be modified in place: work on copies of prior, likelihood and data"
TOLERANCE_HINT = (
    "a probability vector must sum to 1 within 1e-9: a looser check accepts a model that is not "
    "a distribution"
)


def _weights(prior, likelihood, observations):
    """Unnormalised exact weights prior_i * prod_t likelihood[i, o_t], as Fractions."""
    out = []
    for i, p in enumerate(prior):
        w = p
        for o in observations:
            w = w * likelihood[i][o]
        out.append(w)
    return out


def _exact(prior, likelihood, observations):
    """Exact posterior in rational arithmetic: no rounding and no underflow to worry about."""
    weights = _weights(prior, likelihood, observations)
    total = sum(weights)
    return [float(w / total) for w in weights]


def _fraction_model(seed, h, o, t):
    """Probabilities that are multiples of 1/10, so the exact answer is a rational number.

    Draws again until the evidence is positive, so the update is defined.
    """
    rng = np.random.default_rng(seed)
    while True:
        prior = [Fraction(int(c), 10) for c in rng.multinomial(10, [1 / h] * h)]
        lik = [[Fraction(int(c), 10) for c in rng.multinomial(10, [1 / o] * o)] for _ in range(h)]
        obs = rng.integers(0, o, size=t).tolist()
        if sum(_weights(prior, lik, obs)) > 0:
            return prior, lik, obs


def _positive_model(rng, h, o):
    """Every probability is positive, so the evidence is always positive."""
    return rng.dirichlet(np.ones(h)), rng.dirichlet(np.ones(o), size=h)


def _fail(why):
    """The one place these tests raise, so a Hypothesis run reports a single named failure."""
    raise AssertionError(why)


def _outputs(prior, likelihood, observations, what):
    """Return (posterior, None), or (None, the mistake). Never raises, so each test has one site."""
    try:
        return posterior(prior, likelihood, observations), None
    except ValueError as exc:
        return None, f"{what}: a valid model was rejected with ValueError ({exc})"
    except Exception as exc:
        return None, f"{what} (the function raised {type(exc).__name__})"


def _solve(prior, likelihood, observations, what):
    """Call the function on a model that must be accepted. A crash is reported as a mistake."""
    got, problem = _outputs(prior, likelihood, observations, what)
    if problem is not None:
        _fail(problem)
    return got


def _expect_value_error(prior, likelihood, observations, why):
    try:
        posterior(prior, likelihood, observations)
    except ValueError:
        return
    except Exception as exc:
        _fail(f"{why} (the function raised {type(exc).__name__} instead of ValueError)")
    _fail(why)


@pytest.mark.parametrize(
    ("seed", "h", "o", "t"), [(s, 2 + s % 2, 2 + (s // 2) % 2, 1 + s % 6) for s in range(10)]
)
def test_matches_exact_rational_bayes_update_on_short_sequences(seed, h, o, t):
    prior, lik, obs = _fraction_model(seed, h, o, t)
    got = _solve(
        [float(p) for p in prior],
        [[float(x) for x in row] for row in lik],
        obs,
        ORACLE_HINT,
    )
    np.testing.assert_allclose(
        got,
        _exact(prior, lik, obs),
        rtol=1e-9,
        atol=1e-300,
        err_msg=ORACLE_HINT,
    )


def test_two_thousand_alternating_observations_match_exact_posterior():
    prior = [Fraction(1, 2), Fraction(1, 2)]
    lik = [[Fraction(9, 10), Fraction(1, 10)], [Fraction(2, 10), Fraction(8, 10)]]
    obs = [0, 1] * 1000
    got = _solve([0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]], obs, UNDERFLOW_HINT)
    assert np.isfinite(got).all(), UNDERFLOW_HINT
    np.testing.assert_allclose(
        got,
        _exact(prior, lik, obs),
        rtol=1e-9,
        atol=1e-300,
        err_msg=UNDERFLOW_HINT,
    )


@pytest.mark.parametrize("seed", [3, 4])
def test_fifteen_hundred_random_observations_match_exact_posterior(seed):
    prior, lik, obs = _fraction_model(seed, 3, 3, 1500)
    got = _solve(
        [float(p) for p in prior],
        [[float(x) for x in row] for row in lik],
        obs,
        UNDERFLOW_HINT,
    )
    assert np.isfinite(got).all(), UNDERFLOW_HINT
    np.testing.assert_allclose(
        got,
        _exact(prior, lik, obs),
        rtol=1e-9,
        atol=1e-300,
        err_msg=UNDERFLOW_HINT,
    )


@settings(max_examples=60, deadline=None)
@given(SEEDS, st.integers(min_value=1, max_value=20))
def test_updating_in_two_steps_equals_one_batch_update(seed, t):
    rng = np.random.default_rng(seed)
    prior, lik = _positive_model(rng, 3, 3)
    obs = rng.integers(0, 3, size=t).tolist()
    cut = t // 2
    batch, problem = _outputs(prior, lik, obs, SEQUENCE_HINT)
    if problem is None:
        first, problem = _outputs(prior, lik, obs[:cut], SEQUENCE_HINT)
    if problem is None:
        two_steps, problem = _outputs(first, lik, obs[cut:], SEQUENCE_HINT)
    if problem is None and not np.allclose(two_steps, batch, rtol=1e-9, atol=1e-300):
        problem = SEQUENCE_HINT
    assert problem is None, problem


@settings(max_examples=60, deadline=None)
@given(SEEDS, st.integers(min_value=2, max_value=15))
def test_order_of_observations_does_not_change_the_posterior(seed, t):
    rng = np.random.default_rng(seed)
    prior, lik = _positive_model(rng, 2, 3)
    obs = rng.integers(0, 3, size=t)
    shuffled = rng.permutation(obs)
    original, problem = _outputs(prior, lik, obs.tolist(), ORDER_HINT)
    if problem is None:
        reordered, problem = _outputs(prior, lik, shuffled.tolist(), ORDER_HINT)
    if problem is None and not np.allclose(original, reordered, rtol=1e-9, atol=1e-300):
        problem = ORDER_HINT
    assert problem is None, problem


@settings(max_examples=60, deadline=None)
@given(SEEDS, st.integers(min_value=0, max_value=12))
def test_posterior_is_a_float64_distribution_over_hypotheses(seed, t):
    rng = np.random.default_rng(seed)
    prior, lik = _positive_model(rng, 4, 2)
    obs = rng.integers(0, 2, size=t).tolist()
    got, problem = _outputs(
        prior, lik, obs, "the posterior must be a distribution over the hypotheses"
    )
    if problem is None:
        if got.dtype != np.float64:
            problem = "posterior must be float64"
        elif got.shape != (4,):
            problem = f"posterior shape must be (h,) = (4,), got {got.shape}"
        elif (got < 0).any():
            problem = "posterior probabilities must be non-negative"
        elif not np.allclose(got.sum(), 1.0, rtol=1e-9, atol=0.0):
            problem = "posterior must sum to 1 over the hypotheses"
    assert problem is None, problem


def test_zero_prior_hypothesis_keeps_exactly_zero_posterior():
    got = _solve([0.0, 1.0], [[0.99, 0.01], [0.05, 0.95]], [0, 0, 0], ZERO_PRIOR_HINT)
    assert got[0] == 0.0, ZERO_PRIOR_HINT
    np.testing.assert_allclose(got[1], 1.0, rtol=1e-12, err_msg=ZERO_PRIOR_HINT)


def test_zero_likelihood_eliminates_hypothesis_exactly():
    got = _solve([0.5, 0.5], [[1.0, 0.0], [0.5, 0.5]], [0, 1], ZERO_LIKELIHOOD_HINT)
    assert got[0] == 0.0, ZERO_LIKELIHOOD_HINT
    np.testing.assert_allclose(got[1], 1.0, rtol=1e-12, err_msg=ZERO_LIKELIHOOD_HINT)


def test_two_positive_tests_match_exact_rational_value():
    got = _solve([0.01, 0.99], [[0.99, 0.01], [0.05, 0.95]], [0, 0], ORACLE_HINT)
    np.testing.assert_allclose(
        got,
        [float(Fraction(99, 124)), float(Fraction(25, 124))],
        rtol=1e-12,
        err_msg="two positive tests with prior 1% and likelihood ratio 19.8 each: the posterior "
        "is 99/124 for disease",
    )


def test_impossible_observations_raise_value_error():
    _expect_value_error(
        [1.0, 0.0],
        [[1.0, 0.0], [0.5, 0.5]],
        [1],
        "impossible observations must raise ValueError, not return NaN or a uniform vector",
    )


def test_negative_observation_index_is_rejected_not_wrapped_around():
    _expect_value_error(
        [0.5, 0.5],
        LIK,
        [-1],
        "a negative index must raise ValueError: NumPy silently wraps -1 to the last column",
    )


def test_observation_index_past_last_category_is_rejected():
    _expect_value_error(
        [0.5, 0.5],
        LIK,
        [2],
        "an observation index outside [0, o) must raise ValueError",
    )


def test_non_integer_observation_is_rejected():
    _expect_value_error(
        [0.5, 0.5],
        LIK,
        [0.5],
        "observations must be integer category indices: 0.5 must raise ValueError",
    )


def test_prior_must_sum_to_one():
    _expect_value_error(
        [0.3, 0.3],
        LIK,
        [0],
        "a prior that sums to 0.6 must raise ValueError",
    )


def test_prior_sum_just_outside_tolerance_is_rejected_on_both_sides():
    _expect_value_error(
        [0.3, 0.7 + 1e-6],
        LIK,
        [0],
        "a prior summing to 1 + 1e-6 must raise ValueError (outside the 1e-9 tolerance). "
        + TOLERANCE_HINT,
    )
    _expect_value_error(
        [0.3, 0.7 - 1e-6],
        LIK,
        [0],
        "a prior summing to 1 - 1e-6 must raise ValueError (outside the 1e-9 tolerance). "
        + TOLERANCE_HINT,
    )


def test_negative_prior_is_rejected_even_when_it_sums_to_one():
    _expect_value_error(
        [1.5, -0.5],
        LIK,
        [0],
        "a negative prior must raise ValueError, even though it sums to 1",
    )


def test_likelihood_row_must_sum_to_one_over_observations():
    _expect_value_error(
        [0.5, 0.5],
        [[0.9, 0.3], [0.2, 0.8]],
        [0],
        "each likelihood row must sum to 1 over observations: a row summing to 1.2 must raise",
    )


def test_likelihood_row_sum_just_outside_tolerance_is_rejected():
    _expect_value_error(
        [0.5, 0.5],
        [[0.9, 0.1 + 1e-6], [0.2, 0.8]],
        [0],
        "a likelihood row summing to 1 + 1e-6 must raise ValueError (outside the 1e-9 tolerance). "
        + TOLERANCE_HINT,
    )


def test_sums_within_rounding_are_accepted_and_inputs_are_left_unchanged():
    prior = np.array([0.5, 0.5 + 5e-10])
    lik = np.array([[0.9, 0.1 + 5e-10], [0.2, 0.8]])
    obs = np.array([0, 1])
    prior_before, lik_before = prior.copy(), lik.copy()
    got = _solve(
        prior,
        lik,
        obs,
        "a sum within 1e-9 of 1 must be accepted: the tolerance is too strict",
    )
    assert np.isfinite(got).all(), "a sum within rounding must still give a finite posterior"
    np.testing.assert_array_equal(
        prior, prior_before, err_msg=MUTATION_HINT + " (prior was renormalised in place)"
    )
    np.testing.assert_array_equal(
        lik, lik_before, err_msg=MUTATION_HINT + " (likelihood was renormalised in place)"
    )


def test_prior_length_must_match_likelihood_rows():
    _expect_value_error(
        [0.2, 0.3, 0.5],
        LIK,
        [0],
        "prior length 3 with 2 likelihood rows must raise ValueError (shape mismatch)",
    )


def test_one_dimensional_likelihood_is_rejected():
    _expect_value_error(
        [0.5, 0.5],
        [0.9, 0.1],
        [0],
        "a 1-D likelihood must raise ValueError: it must be (h, o)",
    )


def test_nan_probability_is_rejected_even_in_an_unused_column():
    # The NaN sits in column 1, which the data never reads. Only an explicit finiteness check
    # catches it; a NaN that the data touched would raise ValueError for another reason.
    _expect_value_error(
        [0.5, 0.5],
        [[0.9, 0.1], [0.2, np.nan]],
        [0],
        "a NaN probability must raise ValueError even where the data does not read it: NaN "
        "fails every comparison, so a sign or sum check alone lets it through",
    )


def test_nan_prior_is_rejected():
    _expect_value_error(
        [np.nan, 1.0],
        LIK,
        [0],
        "a NaN in the prior must raise ValueError: NaN fails every comparison",
    )


def test_inputs_are_not_mutated():
    prior = np.array([0.3, 0.7])
    lik = np.array([[0.9, 0.1], [0.2, 0.8]])
    obs = np.array([0, 1, 1])
    prior_before, lik_before, obs_before = prior.copy(), lik.copy(), obs.copy()
    _solve(prior, lik, obs, MUTATION_HINT)
    np.testing.assert_array_equal(prior, prior_before, err_msg=MUTATION_HINT)
    np.testing.assert_array_equal(lik, lik_before, err_msg=MUTATION_HINT)
    np.testing.assert_array_equal(obs, obs_before, err_msg=MUTATION_HINT)


SPREAD_HINT = (
    "a very wide spread of finite log-weights: normalise against the peak (largest log-weight), "
    "not the mean; centring on the mean overflows exp, and the finite weights must survive"
)


def test_wide_spread_of_finite_log_weights_normalises_against_the_peak():
    # Three hypotheses, uniform prior, and every observed category has positive likelihood (the zero
    # sits in an unobserved column). Their log-weights are about
    # 0, -700 and -1500 nats, so the middle weight is representable (exp(-700) is a normal float)
    # while the last underflows. Centring on the mean would put the top term near exp(733), which
    # overflows float64.
    prior = [1 / 3, 1 / 3, 1 / 3]
    lik = [
        [1.0, 0.0],
        [np.exp(-175.0), 1.0 - np.exp(-175.0)],
        [np.exp(-375.0), 1.0 - np.exp(-375.0)],
    ]
    got = _solve(prior, lik, [0, 0, 0, 0], SPREAD_HINT)
    assert np.isfinite(got).all(), SPREAD_HINT
    np.testing.assert_allclose(got[0], 1.0, rtol=1e-12, err_msg=SPREAD_HINT)
    assert got[1] == pytest.approx(np.exp(-700.0), rel=1e-9, abs=0.0), SPREAD_HINT
    assert got[2] == 0.0, SPREAD_HINT


MEDIAN_HINT = (
    "a peak far above the median log-weight: normalise against the peak (largest log-weight), not "
    "the median; centring on the median overflows exp, and the finite weights must survive"
)


def test_peak_far_above_the_median_log_weight_normalises_against_the_peak():
    # Four hypotheses, uniform prior. Four observed categories of 0 give log-weights of about
    # 0, -705, -1000 and -1500 nats. The median is near -850, so centring on it would put the
    # peak near exp(850), which overflows float64. The -705 weight is a normal float and survives.
    prior = [0.25, 0.25, 0.25, 0.25]
    lik = [
        [1.0, 0.0],
        [np.exp(-176.25), 1.0 - np.exp(-176.25)],
        [np.exp(-250.0), 1.0 - np.exp(-250.0)],
        [np.exp(-375.0), 1.0 - np.exp(-375.0)],
    ]
    got = _solve(prior, lik, [0, 0, 0, 0], MEDIAN_HINT)
    assert np.isfinite(got).all(), MEDIAN_HINT
    np.testing.assert_allclose(got[0], 1.0, rtol=1e-12, err_msg=MEDIAN_HINT)
    assert got[1] == pytest.approx(np.exp(-705.0), rel=1e-9, abs=0.0), MEDIAN_HINT
    assert got[2] == 0.0 and got[3] == 0.0, MEDIAN_HINT


def test_negative_likelihood_entry_is_rejected_even_when_its_row_sums_to_one():
    _expect_value_error(
        [0.5, 0.5],
        [[1.1, -0.1], [0.2, 0.8]],
        [0],
        "a negative likelihood entry must raise ValueError, even when its row sums to 1 and the "
        "data never reads that column",
    )


def test_likelihood_row_summing_below_one_is_rejected():
    _expect_value_error(
        [0.5, 0.5],
        [[0.5, 0.3], [0.2, 0.8]],
        [0],
        "a likelihood row summing to 0.8 must raise ValueError: rows must sum to 1 on both sides",
    )


def test_prior_length_one_with_two_likelihood_rows_raises_value_error():
    _expect_value_error(
        [1.0],
        [[0.9, 0.1], [0.2, 0.8]],
        [0],
        "a prior of length 1 with two likelihood rows must raise ValueError: the shapes disagree",
    )


def test_single_hypothesis_returns_probability_one():
    got = _solve([1.0], [[0.5, 0.5]], [0, 1], "a single hypothesis is valid (h >= 1)")
    np.testing.assert_allclose(
        got,
        [1.0],
        rtol=1e-12,
        err_msg="with one hypothesis the posterior must be [1.0]: do not reject a prior "
        "of length 1",
    )


def test_negative_prior_is_rejected_when_there_are_no_observations():
    _expect_value_error(
        [1.5, -0.5],
        LIK,
        [],
        "a negative prior must raise ValueError even when there is no evidence to multiply it into",
    )


def test_nan_prior_is_rejected_when_there_are_no_observations():
    _expect_value_error(
        [np.nan, 1.0],
        LIK,
        [],
        "a NaN in the prior must raise ValueError even when there is no evidence",
    )
