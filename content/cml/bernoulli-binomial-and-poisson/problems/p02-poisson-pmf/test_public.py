import numpy as np
from submission import poisson_pmf


def test_rate_two_gives_the_textbook_values():
    np.testing.assert_allclose(
        poisson_pmf([0, 1, 2], 2.0),
        [0.13533528, 0.27067057, 0.27067057],
        rtol=1e-7,
        err_msg="Poisson(2): P(k) = e^-2 2^k / k!, so P(0) = e^-2 and P(1) = P(2) = 2 e^-2",
    )


def test_negative_count_has_probability_zero():
    got = poisson_pmf(-1, 2.0)
    assert got == 0.0, "a negative count is impossible: its probability must be exactly 0.0"


def test_zero_rate_puts_all_mass_on_zero():
    try:
        got = poisson_pmf(0, 0.0)
    except Exception as exc:
        raise AssertionError(
            f"rate 0 means no events: P(0) = 1 exactly, without taking log(0) ({exc})"
        ) from exc
    assert got == 1.0, "rate 0 means no events: P(0) = 1 exactly, without taking log(0)"


def test_large_count_past_170_factorial_stays_finite():
    try:
        got = poisson_pmf(200, 180.0)
    except Exception as exc:
        raise AssertionError(
            "k = 200 needs 200!, which overflows float64 (171! is the first that does): "
            f"use lgamma in log space ({exc})"
        ) from exc
    assert np.isfinite(got), "P(200) for rate 180 must be finite: k! overflows float64 at k = 171"
    np.testing.assert_allclose(
        got,
        0.00965174,
        rtol=1e-6,
        err_msg="computed directly, k! and lam^k overflow: work in log space",
    )
