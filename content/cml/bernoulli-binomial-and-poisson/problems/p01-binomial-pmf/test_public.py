import numpy as np
from submission import binomial_pmf


def test_three_fair_coins_give_one_eighth_three_eighths():
    np.testing.assert_allclose(
        binomial_pmf([0, 1, 2, 3], 3, 0.5),
        [0.125, 0.375, 0.375, 0.125],
        rtol=1e-12,
        err_msg="three fair coins: P(k) = C(3, k) / 8, so the pmf must be 1/8, 3/8, 3/8, 1/8",
    )


def test_values_outside_zero_to_n_have_probability_zero():
    try:
        got = binomial_pmf([-1, 11], 10, 0.3)
    except Exception as exc:
        raise AssertionError(
            f"k outside [0, n] must have probability exactly 0.0, not an error ({exc})"
        ) from exc
    np.testing.assert_array_equal(
        got,
        [0.0, 0.0],
        err_msg="k outside [0, n] must have probability exactly 0.0",
    )


def test_zero_trials_is_a_certain_zero():
    try:
        got = binomial_pmf(0, 0, 0.5)
    except Exception as exc:
        raise AssertionError(
            f"n = 0 trials: the only possible count is 0, with probability 1 ({exc})"
        ) from exc
    np.testing.assert_allclose(
        got,
        1.0,
        rtol=1e-12,
        err_msg="n = 0 trials: the only possible count is 0, with probability 1",
    )


def test_non_integer_k_raises_value_error():
    try:
        binomial_pmf(2.5, 10, 0.3)
    except ValueError:
        return
    raise AssertionError("a non-integer k such as 2.5 must raise ValueError")
