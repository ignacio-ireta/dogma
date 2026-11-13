"""Hidden tests: a Bayes-route oracle, a seeded simulation, identities and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm
from submission import prior_shift_adjust

from dogma_data import load, rng, sample_rows

VALUE_MISTAKE = (
    "corrected probability differs from the Bayes route: multiply the prior odds by the likelihood "
    "ratio, then convert back, using the deployment prior"
)


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def _solve(probs, train_prior, deploy_prior):
    """Call the function on input that must be accepted. A crash is a contract mistake."""
    try:
        result = prior_shift_adjust(probs, train_prior, deploy_prior)
    except Exception as exc:
        pytest.fail(f"valid input raised {type(exc).__name__}: check the odds ratio and the bounds")
    assert isinstance(result, np.ndarray), "return a NumPy array of corrected probabilities"
    assert np.issubdtype(result.dtype, np.floating), "corrected probabilities need floating dtype"
    assert result.shape == np.asarray(probs).shape, "preserve the one-dimensional input shape"
    return result


def _bayes_route(probs, train_prior, deploy_prior):
    """Independent route: recover the likelihood ratio from each training probability.

    Then apply Bayes' rule with the deployment prior.
    """
    p = np.asarray(probs, dtype=float)
    likelihood_ratio = (p / (1 - p)) / (train_prior / (1 - train_prior))
    return deploy_prior * likelihood_ratio / (deploy_prior * likelihood_ratio + 1 - deploy_prior)


@pytest.mark.parametrize("seed", range(5), ids=[f"case-{s}" for s in range(5)])
def test_matches_the_bayes_route_on_random_probabilities(seed):
    # Real classifier probabilities: forty breast_cancer tumours scored by the direction from
    # the malignant mean to the benign mean (four standardised features), passed through a
    # sigmoid. The training prior is the benign share of the draw; the deployment prior is the
    # scenario and stays random.
    gen = rng(f"prior-shift-p02:bayes-route:{seed}")
    X, label = sample_rows("breast_cancer", gen, 40, 4, standardize=True)
    score = X @ (X[label == 1].mean(axis=0) - X[label == 0].mean(axis=0))
    probs = np.clip(1.0 / (1.0 + np.exp(-score)), 0.001, 0.999)
    train_prior = float(np.clip(label.mean(), 0.02, 0.98))
    deploy_prior = float(gen.uniform(0.02, 0.98))
    expected = _bayes_route(probs, train_prior, deploy_prior)
    got = _solve(probs, train_prior, deploy_prior)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12, err_msg=VALUE_MISTAKE)


def test_simulation_at_fixed_seed_is_calibrated_to_the_deployment_base_rate():
    train_prior, deploy_prior = 0.5, 0.05
    # Real tumours resampled at the deployment rate, one malignant in twenty. The feature is
    # the worst symmetry of a tumour (breast_cancer), and the model is the pair of Gaussians
    # fitted to it in each class, which follows the data closely enough to be calibrated.
    tumours = load("breast_cancer")
    malignant = np.sort(tumours.X[tumours.y == 0, 28])
    benign = np.sort(tumours.X[tumours.y == 1, 28])
    rng = np.random.default_rng(12345)
    n = 200_000
    y = (rng.random(n) < deploy_prior).astype(int)
    x = np.where(
        y == 1,
        malignant[rng.integers(0, malignant.size, n)],
        benign[rng.integers(0, benign.size, n)],
    )
    f1 = norm.pdf(x, malignant.mean(), malignant.std())
    f0 = norm.pdf(x, benign.mean(), benign.std())
    p_train = train_prior * f1 / (train_prior * f1 + (1 - train_prior) * f0)
    p_deploy_true = deploy_prior * f1 / (deploy_prior * f1 + (1 - deploy_prior) * f0)
    corrected = _solve(p_train, train_prior, deploy_prior)
    np.testing.assert_allclose(corrected, p_deploy_true, rtol=0, atol=1e-9, err_msg=VALUE_MISTAKE)
    observed_rate = float(y.mean())
    assert abs(float(corrected.mean()) - observed_rate) < 0.003, (
        "the mean corrected probability must match the deployment base rate: "
        "the odds were not shifted"
    )
    assert abs(float(p_train.mean()) - deploy_prior) > 0.1, (
        "the simulation must show a real prior shift"
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(st.lists(st.floats(0.0, 1.0), min_size=0, max_size=20), st.floats(0.01, 0.99))
def test_equal_priors_are_the_identity(probs, prior):
    got = _solve(probs, prior, prior)
    np.testing.assert_allclose(
        got,
        np.asarray(probs, dtype=float),
        rtol=0,
        atol=1e-12,
        err_msg=("equal priors must return the input unchanged: the ratio must be 1"),
    )


def test_endpoints_stay_exactly_zero_and_one():
    got = _solve([0.0, 1.0], 0.3, 0.02)
    assert got[0] == 0.0, "a probability of 0 must map to exactly 0"
    assert got[1] == 1.0, "a probability of 1 must map to exactly 1, with no division by zero"
    assert not np.any(np.isnan(got)), "the endpoints must not produce NaN"


@pytest.mark.parametrize(
    ("train_prior", "deploy_prior"),
    [(float(np.nextafter(0.0, 1.0)), 0.9), (0.99, float(np.nextafter(0.0, 1.0)))],
    ids=["tiny-training-prior", "tiny-deployment-prior"],
)
def test_extreme_valid_priors_match_high_precision_bayes(train_prior, deploy_prior):
    probs = np.array([0.0, 1e-320, 0.2, 0.5, np.nextafter(1.0, 0.0), 1.0])
    with localcontext() as context:
        context.prec = 100
        train, deploy = Decimal.from_float(train_prior), Decimal.from_float(deploy_prior)
        expected = []
        for value in probs:
            p = Decimal.from_float(float(value))
            positive = deploy * (1 - train) * p
            negative = (1 - deploy) * train * (1 - p)
            expected.append(float(positive / (positive + negative)))
    got = _solve(probs, train_prior, deploy_prior)
    assert np.all(np.isfinite(got)), (
        "valid priors near zero must not overflow or produce NaN: compute the odds shift stably"
    )
    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-9,
        atol=1e-12,
        err_msg="extreme prior correction differs from high-precision Bayes",
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.floats(0.0, 1.0), min_size=1, max_size=30),
    st.floats(0.01, 0.99),
    st.floats(0.01, 0.99),
)
def test_outputs_are_probabilities_and_non_decreasing_in_the_input(
    probs, train_prior, deploy_prior
):
    p = np.sort(np.asarray(probs, dtype=float))
    got = _solve(p, train_prior, deploy_prior)
    assert np.all((got >= 0) & (got <= 1)), "corrected values must lie in [0, 1]"
    assert np.all(np.diff(got) >= -1e-15), "the correction must not reverse the order of the input"


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(st.floats(0.001, 0.999), st.floats(0.01, 0.99), st.floats(0.01, 0.99))
def test_swapping_the_priors_undoes_the_correction(p, a, b):
    once = _solve([p], a, b)
    back = _solve(once, b, a)
    assert math.isclose(float(back[0]), p, rel_tol=1e-9, abs_tol=1e-12), (
        "adjusting from a to b and then from b to a must return the original probability"
    )


@pytest.mark.parametrize(
    ("probs", "train_prior", "deploy_prior", "mistake"),
    [
        (
            [0.5],
            0.0,
            0.1,
            "a training prior of 0 was accepted: priors must be strictly between 0 and 1",
        ),
        (
            [0.5],
            0.3,
            1.0,
            "a deployment prior of 1 was accepted: priors must be strictly between 0 and 1",
        ),
        (
            [0.5],
            0.3,
            float("nan"),
            "a NaN prior was accepted: priors must be strictly between 0 and 1",
        ),
        (
            [float("nan")],
            0.3,
            0.2,
            "a NaN probability was accepted: probabilities must lie in [0, 1]",
        ),
        ([1.2], 0.3, 0.2, "a probability above 1 was accepted: probabilities must lie in [0, 1]"),
        (
            [[0.1, 0.2]],
            0.3,
            0.2,
            "a two-dimensional input was accepted: probs must be one-dimensional",
        ),
    ],
    ids=[
        "train-zero",
        "deploy-one",
        "nan-prior",
        "nan-probability",
        "above-one",
        "two-dimensional",
    ],
)
def test_invalid_input_raises_value_error(probs, train_prior, deploy_prior, mistake):
    _expect_value_error(lambda: prior_shift_adjust(probs, train_prior, deploy_prior), mistake)


def test_empty_input_returns_an_empty_array():
    got = _solve([], 0.2, 0.3)
    assert got.shape == (0,), "an empty input must return an empty array, not an error"


def test_input_array_is_not_mutated():
    probs = np.array([0.1, 0.6, 0.95])
    before = probs.copy()
    _solve(probs, 0.4, 0.05)
    assert np.array_equal(probs, before), "probs was changed: return a new array"


def test_read_only_input_arrays_are_accepted_and_left_alone():
    probs = np.array([0.6, 0.1, 0.95])
    for array in (probs,):
        array.setflags(write=False)
    try:
        prior_shift_adjust(probs, 0.4, 0.05)
    except Exception as exc:
        if "read-only" in str(exc):
            pytest.fail(
                "an input array was written to in place: inputs must not be mutated, so work "
                "on a copy",
                pytrace=False,
            )
        pytest.fail(
            "a valid call failed when its input arrays were read-only: a correct solution "
            "only reads its inputs",
            pytrace=False,
        )
