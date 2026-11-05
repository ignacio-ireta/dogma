import numpy as np
import pytest
from submission import discrete_moments


def test_fair_die_matches_textbook_mean_and_variance():
    mean, variance, std = discrete_moments([1, 2, 3, 4, 5, 6], [1 / 6] * 6)
    np.testing.assert_allclose(
        [mean, variance, std],
        [3.5, 35 / 12, np.sqrt(35 / 12)],
        atol=1e-9,
        err_msg="fair die: the mean should be 3.5 and the variance 35/12 around that mean",
    )


def test_biased_coin_variance_is_p_times_one_minus_p():
    mean, variance, std = discrete_moments([0, 1], [0.7, 0.3])
    np.testing.assert_allclose(
        [mean, variance, std],
        [0.3, 0.3 * 0.7, np.sqrt(0.3 * 0.7)],
        atol=1e-9,
        err_msg="Bernoulli(p): mean p and variance p(1 - p); weight each squared deviation",
    )


def test_certain_outcome_has_zero_variance():
    mean, variance, std = discrete_moments([5.0], [1.0])
    np.testing.assert_allclose(
        [mean, variance, std],
        [5.0, 0.0, 0.0],
        atol=1e-12,
        err_msg="a single outcome with probability 1 has no spread",
    )


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_probabilities_that_do_not_sum_to_one_raise_value_error():
    _raised_value_error(
        lambda: discrete_moments([0, 1], [0.6, 0.3]),
        "probabilities summing to 0.9 are not a distribution: raise ValueError",
    )
