import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from submission import poisson_pmf

RTOL = 1e-9
ATOL = 1e-14
ORACLE_HINT = (
    "pmf differs from scipy.stats.poisson.pmf, or the function crashed on large counts: the term "
    "is k log(lam) - lam - lgamma(k + 1), and a float k! overflows past k = 170"
)
OVERFLOW_HINT = (
    "counts past 170 overflow float64 if k! is formed as a float (170! is the largest finite "
    "factorial): use lgamma(k + 1) in log space"
)


def _fail(why):
    """The one place these tests raise, so a Hypothesis run reports a single named failure."""
    raise AssertionError(why)


def _call(k, lam):
    """Return (pmf, None), or (None, exception name). Never raises, so there is one failure site."""
    try:
        return poisson_pmf(k, lam), None
    except Exception as exc:
        return None, type(exc).__name__


def _check_pmf(k, lam, expected, what, *, rtol=RTOL):
    got, crash = _call(k, lam)
    if crash is not None:
        _fail(f"{what} (the function raised {crash})")
    if np.shape(got) != np.shape(expected) or not np.allclose(got, expected, rtol=rtol, atol=ATOL):
        _fail(what)


def _expect_value_error(k, lam, why):
    try:
        poisson_pmf(k, lam)
    except ValueError:
        return
    _fail(why)


@settings(max_examples=60, deadline=None)
@given(
    st.floats(min_value=0.01, max_value=60.0, allow_nan=False),
    st.integers(min_value=0, max_value=200),
)
def test_matches_scipy_poisson_pmf_for_moderate_rates(lam, k_max):
    k = np.arange(k_max + 1)
    _check_pmf(k, lam, stats.poisson.pmf(k, lam), ORACLE_HINT)


@pytest.mark.parametrize("lam", [180.0, 1000.0, 5000.0])
def test_matches_scipy_for_large_rates_and_counts(lam):
    k = np.unique(np.linspace(0, 3 * lam, 301).astype(int))
    _check_pmf(k, lam, stats.poisson.pmf(k, lam), ORACLE_HINT + " (large rate)")


def test_counts_past_170_factorial_do_not_overflow():
    k = np.array([170, 171, 172, 200, 400])
    _check_pmf(k, 180.0, stats.poisson.pmf(k, 180.0), OVERFLOW_HINT)


@pytest.mark.parametrize(("lam", "k_max"), [(0.5, 40), (7.5, 80), (50.0, 300)])
def test_pmf_sums_to_one_over_the_support(lam, k_max):
    k = np.arange(k_max + 1)
    got, crash = _call(k, lam)
    if crash is not None:
        _fail(f"the pmf over 0..{k_max} must be computed (the function raised {crash})")
    np.testing.assert_allclose(
        got.sum(),
        1.0,
        rtol=0,
        atol=1e-12,
        err_msg="the pmf over the support must sum to 1: check the normalising terms",
    )


@pytest.mark.parametrize("lam", [0.7, 7.5, 120.0])
def test_mean_and_variance_are_both_lambda(lam):
    k = np.arange(int(lam * 5) + 40)
    got, crash = _call(k, lam)
    if crash is not None:
        _fail(f"mean and variance need the pmf (the function raised {crash})")
    mean = float((k * got).sum())
    var = float(((k - mean) ** 2 * got).sum())
    np.testing.assert_allclose(
        mean, lam, rtol=1e-9, err_msg="the Poisson mean must equal lam: the pmf is shaped wrong"
    )
    np.testing.assert_allclose(
        var, lam, rtol=1e-9, err_msg="the Poisson variance must equal lam (mean = variance)"
    )


def test_zero_rate_puts_all_mass_on_zero_exactly():
    k = np.arange(4)
    got, crash = _call(k, 0.0)
    if crash is not None:
        _fail(
            "rate 0 must give P(0) = 1 exactly, without taking log(0) "
            f"(the function raised {crash})"
        )
    assert got[0] == 1.0, "rate 0: P(0) must be exactly 1.0, so a zero rate needs its own branch"
    assert (got[1:] == 0.0).all(), "rate 0: every positive count must have probability exactly 0"


def test_negative_counts_have_probability_exactly_zero():
    got, crash = _call([-5, -1, -0], 3.0)
    if crash is not None:
        _fail(f"negative counts must have probability 0.0 (the function raised {crash})")
    assert got[0] == 0.0 and got[1] == 0.0, "a negative count must have probability exactly 0.0"
    np.testing.assert_allclose(
        got[2], np.exp(-3.0), rtol=1e-12, err_msg="the count 0 must have probability e^-lam"
    )


def test_output_shape_follows_the_shape_of_k():
    k = np.array([[0, 1, 2], [3, 4, 5]])
    _check_pmf(k, 2.5, stats.poisson.pmf(k, 2.5), "a 2-D k must be accepted and vectorised")


def test_scalar_k_returns_a_zero_dimensional_result():
    got, crash = _call(4, 2.5)
    if crash is not None:
        _fail(f"a scalar k must be accepted (the function raised {crash})")
    assert np.shape(got) == (), f"a scalar k must give a 0-d result, got shape {np.shape(got)}"
    np.testing.assert_allclose(
        got,
        stats.poisson.pmf(4, 2.5),
        rtol=RTOL,
        atol=ATOL,
        err_msg="a scalar k must give the Poisson pmf value, not a wrong number",
    )


def test_integral_float_k_gives_the_same_pmf_as_integer_k():
    a, crash_a = _call(np.array([2.0, 3.0]), 4.0)
    b, crash_b = _call(np.array([2, 3]), 4.0)
    if crash_a or crash_b:
        _fail("integral floats and integers must both be accepted")
    np.testing.assert_allclose(a, b, rtol=1e-15, err_msg="integral floats must match integer k")


@pytest.mark.parametrize("lam", [-1.0, float("nan"), float("inf")])
def test_invalid_rate_raises_value_error(lam):
    _expect_value_error(
        2,
        lam,
        f"lam = {lam} must raise ValueError: a rate must be finite and non-negative",
    )


def test_non_integer_k_raises_value_error():
    _expect_value_error(np.array([0.5]), 2.0, "k = 0.5 is not an integer: raise ValueError")


def test_inputs_are_not_mutated():
    k = np.array([0, 1, 2, 3])
    before = k.copy()
    _call(k, 2.0)
    np.testing.assert_array_equal(k, before, err_msg="input k was modified in place")


def test_non_integer_entry_in_object_array_raises_value_error():
    _expect_value_error(
        np.array([1, 2.5], dtype=object),
        2.0,
        "k holds a non-integer entry (2.5) in an object array: raise ValueError",
    )


def test_infinite_count_raises_value_error():
    _expect_value_error(
        np.array([0.0, -np.inf]),
        2.0,
        "an infinite count next to a finite one is not an integer: raise ValueError",
    )


def test_one_non_integer_among_integers_raises_value_error():
    _expect_value_error(
        np.array([1.0, 2.5]),
        2.0,
        "a single non-integer entry must raise ValueError, even next to integral values",
    )
