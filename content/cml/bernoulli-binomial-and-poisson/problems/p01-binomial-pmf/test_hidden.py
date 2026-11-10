import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from submission import binomial_pmf

RTOL = 1e-7  # matches meta.yaml: lgamma(10^6) is about 1.3e7, so the pmf carries about 2e-9 error
ATOL = 1e-14
ORACLE_HINT = (
    "pmf differs from scipy.stats.binom.pmf: compute log C(n, k) with lgamma, then add "
    "k log p and (n - k) log(1 - p)"
)
LARGE_HINT = (
    "n = 10^6: the pmf is wrong or not finite. Factorials, C(n, k) and p^k as floats overflow or "
    "underflow (171! already overflows float64), so work in log space with lgamma, and check each "
    "lgamma argument (lgamma(n + 1) is log n!)"
)
MOMENT_HINT = "the pmf does not have mean n p and variance n p (1 - p): check the shape over k"


def _fail(why):
    """The one place these tests raise, so a Hypothesis run reports a single named failure."""
    raise AssertionError(why)


def _call(k, n, p):
    """Return (pmf, None), or (None, exception name). Never raises, so there is one failure site."""
    try:
        return binomial_pmf(k, n, p), None
    except Exception as exc:
        return None, type(exc).__name__


def _pmf(k, n, p, what):
    got, crash = _call(k, n, p)
    if crash is not None:
        _fail(f"{what} (the function raised {crash})")
    return got


def _check_pmf(k, n, p, expected, what, *, rtol=RTOL):
    got, crash = _call(k, n, p)
    if crash is not None:
        _fail(f"{what} (the function raised {crash})")
    if np.shape(got) != np.shape(expected) or not np.allclose(got, expected, rtol=rtol, atol=ATOL):
        _fail(what)


def _expect_value_error(k, n, p, why):
    try:
        binomial_pmf(k, n, p)
    except ValueError:
        return
    raise AssertionError(why)


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.integers(min_value=0, max_value=300),
    st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
)
def test_matches_scipy_binom_pmf_for_small_and_medium_n(n, p):
    k = np.arange(n + 1)
    _check_pmf(k, n, p, stats.binom.pmf(k, n, p), ORACLE_HINT)


@pytest.mark.parametrize(
    "p", [0.5, 1e-6, 0.999], ids=["fair-coin", "rare-success", "common-success"]
)
def test_matches_scipy_at_one_million_trials(p):
    n = 10**6
    k = np.concatenate(
        [np.arange(0, 5), np.linspace(0, n, 201).astype(int), np.arange(n - 4, n + 1)]
    )
    _check_pmf(k, n, p, stats.binom.pmf(k, n, p), LARGE_HINT)


def test_pmf_sums_to_one_over_the_support_at_one_million_trials():
    n = 10**6
    total = _pmf(np.arange(n + 1), n, 0.3, LARGE_HINT).sum()
    np.testing.assert_allclose(
        total,
        1.0,
        rtol=0,
        atol=1e-7,
        err_msg="the pmf over 0..n must sum to 1 at n = 10^6 (within the log-space accuracy)",
    )


@pytest.mark.parametrize(
    ("n", "p"),
    [(200, 0.37), (1000, 0.02), (7, 0.9)],
    ids=["medium-trials", "many-trials-rare-success", "few-trials-likely-success"],
)
def test_mean_and_variance_match_np_and_np_one_minus_p(n, p):
    k = np.arange(n + 1)
    pmf = _pmf(k, n, p, MOMENT_HINT)
    mean = float((k * pmf).sum())
    var = float(((k - mean) ** 2 * pmf).sum())
    np.testing.assert_allclose(mean, n * p, rtol=1e-9, err_msg=MOMENT_HINT + " (mean is wrong)")
    np.testing.assert_allclose(
        var, n * p * (1 - p), rtol=1e-9, err_msg=MOMENT_HINT + " (variance is wrong)"
    )


def test_p_zero_puts_all_mass_on_zero_exactly():
    got = _pmf(np.arange(6), 5, 0.0, "p = 0 edge case must be handled")
    assert got[0] == 1.0, (
        "p = 0 must give P(X = 0) = 1 exactly: log(0) is -inf, so p = 0 needs its own branch"
    )
    assert (got[1:] == 0.0).all(), "p = 0 must give probability exactly 0 for every k > 0"


def test_p_one_puts_all_mass_on_n_exactly():
    got = _pmf(np.arange(6), 5, 1.0, "p = 1 edge case must be handled")
    assert got[5] == 1.0, (
        "p = 1 must give P(X = n) = 1 exactly: log(1 - p) = log(0) needs its own branch"
    )
    assert (got[:5] == 0.0).all(), "p = 1 must give probability exactly 0 for every k < n"


def test_k_outside_the_support_has_probability_exactly_zero():
    got = _pmf([-3, -1, 11, 12, 1000], 10, 0.3, "k outside [0, n] must be handled")
    assert (got == 0.0).all(), (
        "k outside [0, n] must have probability exactly 0.0, not an error and not NaN"
    )


def test_zero_trials_is_a_point_mass_at_zero():
    np.testing.assert_allclose(
        _pmf(0, 0, 0.37, "n = 0 edge case"), 1.0, rtol=1e-12, err_msg="n = 0: P(0) = 1"
    )
    assert _pmf(0, 0, 0.0, "n = 0 with p = 0") == 1.0, "n = 0, p = 0: P(0) = 1 exactly"


def test_output_shape_follows_the_shape_of_k():
    k = np.array([[0, 1, 2], [3, 4, 5]])
    got = _pmf(k, 5, 0.4, "a 2-D k must be accepted and vectorised")
    assert got.shape == (2, 3), f"output shape must match k's shape (2, 3), got {got.shape}"
    np.testing.assert_allclose(
        got,
        stats.binom.pmf(k, 5, 0.4),
        rtol=RTOL,
        atol=ATOL,
        err_msg="vectorised over k: every element needs its own pmf value",
    )


def test_scalar_k_returns_a_zero_dimensional_result():
    got = _pmf(3, 10, 0.5, "a scalar k must be accepted")
    assert np.shape(got) == (), f"a scalar k must give a 0-d result, got shape {np.shape(got)}"
    np.testing.assert_allclose(
        got,
        stats.binom.pmf(3, 10, 0.5),
        rtol=RTOL,
        atol=ATOL,
        err_msg="a scalar k must give the binomial pmf value, not a wrong number",
    )


def test_result_is_float64():
    got = _pmf(np.array([0, 1, 2]), 4, 0.5, "an integer k must be accepted")
    assert got.dtype == np.float64, f"pmf must be float64, got {got.dtype}"


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.integers(min_value=0, max_value=200),
    st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
)
def test_symmetry_swapping_successes_and_failures(n, p):
    k = np.arange(n + 1)
    # pmf(k; n, p) = pmf(n - k; n, 1 - p): the mirror image under swapping success and failure.
    _check_pmf(
        k,
        n,
        p,
        stats.binom.pmf(n - k, n, 1 - p),
        "symmetry pmf(k; n, p) = pmf(n - k; n, 1 - p) fails: check the log(p) and log(1 - p) terms",
        rtol=1e-9,
    )


def test_non_integer_n_raises_value_error():
    _expect_value_error(3, 2.5, 0.5, "n = 2.5 is not an integer: raise ValueError")


def test_negative_n_raises_value_error():
    _expect_value_error(0, -1, 0.5, "n must be non-negative: n = -1 must raise ValueError")


def test_p_outside_the_unit_interval_raises_value_error():
    _expect_value_error(1, 10, 1.5, "p = 1.5 must raise ValueError: p is a probability")
    _expect_value_error(1, 10, -0.1, "p = -0.1 must raise ValueError: p is a probability")


def test_nan_p_raises_value_error():
    _expect_value_error(
        1,
        10,
        float("nan"),
        "a NaN p must raise ValueError: NaN passes a 'p < 0 or p > 1' check, so write the test "
        "as a positive range check",
    )


def test_non_integer_k_raises_value_error():
    _expect_value_error(np.array([0.5]), 10, 0.3, "k = 0.5 is not an integer: raise ValueError")


def test_integral_float_k_gives_the_same_pmf_as_integer_k():
    np.testing.assert_allclose(
        _pmf(np.array([2.0, 3.0]), 6, 0.4, "integral floats must be accepted"),
        _pmf(np.array([2, 3]), 6, 0.4, "integers must be accepted"),
        rtol=1e-15,
        err_msg="integral floats (2.0, 3.0) must give the same pmf as integers",
    )


def test_inputs_are_not_mutated():
    k = np.array([0, 1, 2, 3])
    before = k.copy()
    _pmf(k, 3, 0.5, "a valid call must succeed before checking inputs")
    np.testing.assert_array_equal(k, before, err_msg="input k was modified in place")


def test_object_array_holding_a_non_integer_value_raises_value_error():
    _expect_value_error(
        np.array([2.5], dtype=object),
        10,
        0.3,
        "k contains the non-integer value 2.5 in an object array: raise ValueError, do "
        "not truncate it",
    )


def test_mixed_array_with_one_non_integer_entry_raises_value_error():
    _expect_value_error(
        np.array([2.0, 2.5]),
        10,
        0.3,
        "a k array holding one non-integer entry must raise ValueError, even if another "
        "entry is integral",
    )


def test_integer_k_beyond_32_bits_is_outside_the_support_and_has_probability_zero():
    # 2**32 + 3 is an integer outside [0, 10]: its probability must be exactly 0, not the pmf at 3.
    got = _pmf(2**32 + 3, 10, 0.3, "a k far outside [0, n] must be handled")
    assert float(got) == 0.0, (
        "k outside [0, n] must have probability exactly 0.0: a wide integer was truncated into the "
        "support"
    )


def test_negative_n_with_p_at_zero_or_one_raises_value_error():
    _expect_value_error(
        0,
        -1,
        0.0,
        "n = -1 must raise ValueError for every p: n must be a non-negative integer (p = 0 case)",
    )
    _expect_value_error(
        0,
        -1,
        1.0,
        "n = -1 must raise ValueError for every p: n must be a non-negative integer (p = 1 case)",
    )
