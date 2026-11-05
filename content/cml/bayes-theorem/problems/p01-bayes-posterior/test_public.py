import numpy as np
from submission import posterior


def test_positive_medical_test_gives_one_in_six():
    got = posterior([0.01, 0.99], [[0.99, 0.01], [0.05, 0.95]], [0])
    np.testing.assert_allclose(
        got,
        [1 / 6, 5 / 6],
        rtol=1e-9,
        err_msg="1% prevalence, 99% sensitivity, 5% false-positive rate: P(disease | +) = 1/6, "
        "so the prior must multiply the likelihood before normalising",
    )


def test_two_observations_of_the_same_category_multiply_likelihoods():
    got = posterior([0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]], [0, 0])
    np.testing.assert_allclose(
        got,
        [0.81 / 0.85, 0.04 / 0.85],
        rtol=1e-9,
        err_msg="i.i.d. observations: the likelihoods of both observations must multiply",
    )


def test_empty_observation_sequence_returns_the_prior():
    got = posterior([0.2, 0.8], [[0.9, 0.1], [0.2, 0.8]], [])
    np.testing.assert_allclose(
        got, [0.2, 0.8], rtol=1e-12, err_msg="no data: the posterior must equal the prior"
    )


def test_impossible_observations_raise_value_error():
    try:
        posterior([1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]], [1])
    except ValueError:
        return
    raise AssertionError(
        "impossible observations must raise ValueError, not return NaN or a uniform vector"
    )
