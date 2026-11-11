from fractions import Fraction

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from statsmodels.tsa.stattools import acf as sm_acf
from submission import acf

from dogma_data import load_series


def _require(ok, mistake):
    """Fail with the mistake's name only. Hypothesis and the grader show this first line."""
    if not ok:
        raise AssertionError(mistake)


def _run(call, mistake):
    try:
        result = call()
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from None
    _require(
        isinstance(result, np.ndarray) and result.dtype == np.float64 and result.ndim == 1,
        "acf must return a one-dimensional float64 NumPy array",
    )
    return result


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from None
    raise AssertionError(mistake)


def _close(actual, expected, mistake, *, rtol=1e-9, atol=1e-9):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    _require(actual.shape == expected.shape, f"shape mismatch: {mistake}")
    _require(bool(np.allclose(actual, expected, rtol=rtol, atol=atol)), mistake)


def _loop_reference(values, nlags):
    """Explicit sums, one lag at a time, with no NumPy helpers: an independent formula."""
    n = len(values)
    mean = sum(values) / n
    d = [v - mean for v in values]
    denominator = sum(v * v for v in d)
    return np.array(
        [sum(d[t] * d[t + k] for t in range(n - k)) / denominator for k in range(nlags + 1)]
    )


@pytest.mark.parametrize(
    ("n", "nlags", "seed", "offset"),
    [(10, 3, 0, 0.0), (50, 10, 1, 0.0), (200, 40, 2, 0.0), (120, 20, 3, 1e3), (30, 29, 4, -5.0)],
    ids=["short", "medium", "long", "large-mean", "all-lags"],
)
def test_matches_statsmodels_biased_acf(n, nlags, seed, offset):
    values = np.random.default_rng(seed).normal(size=n) + offset
    reference = sm_acf(values, nlags=nlags, adjusted=False, fft=False)
    result = _run(lambda: acf(values, nlags), "acf raised on a valid series")
    _require(result.shape == (nlags + 1,), "acf must return one entry per lag from 0 to nlags")
    _close(
        result,
        reference,
        "autocorrelation differs from the biased estimator: divide every lag by the same n * var",
        atol=1e-10,
    )


def test_alternating_series_matches_closed_form():
    # Mean 0 and sum of squares n, so r_k = (n - k) * (-1)^k / n for every lag.
    n = 10
    values = np.array([(-1.0) ** t for t in range(n)])
    lags = np.arange(n)
    expected = (n - lags) * (-1.0) ** lags / n
    result = _run(lambda: acf(values, n - 1), "acf raised on an alternating series")
    _close(
        result,
        expected,
        "alternating series autocorrelation is wrong: lag k must be (n - k)(-1)^k / n, so divide "
        "by the total sum of squares and not by the number of pairs",
    )


def test_lag_zero_is_exactly_one_on_random_series():
    values = np.random.default_rng(7).normal(size=80)
    result = _run(lambda: acf(values, 15), "acf raised on a valid series")
    _require(
        result[0] == 1.0, "lag 0 must be exactly one: the numerator and denominator must match"
    )


def test_shift_and_scale_leave_the_autocorrelation_unchanged():
    values = np.random.default_rng(8).normal(size=60)
    base = _run(lambda: acf(values, 12), "acf raised on a valid series")
    shifted = _run(lambda: acf(values + 250.0, 12), "acf raised on a shifted series")
    scaled = _run(lambda: acf(values * 0.001, 12), "acf raised on a scaled series")
    _close(shifted, base, "adding a constant changed the autocorrelation: centre the series")
    _close(scaled, base, "scaling the series changed the autocorrelation: it must be scale-free")


def test_nlags_zero_returns_only_lag_zero():
    result = _run(lambda: acf([1.0, 4.0, 2.0], 0), "acf raised for nlags = 0")
    _require(result.shape == (1,) and result[0] == 1.0, "nlags = 0 must return [1.0]")


def test_nlags_at_the_maximum_returns_every_lag():
    values = np.random.default_rng(9).normal(size=12)
    reference = sm_acf(values, nlags=11, adjusted=False, fft=False)
    result = _run(lambda: acf(values, 11), "acf raised for nlags = n - 1")
    _require(result.shape == (12,), "nlags = n - 1 must return n entries")
    _close(result, reference, "last lags differ")


@pytest.mark.parametrize(
    "values",
    [[2.0, 2.0, 2.0, 2.0, 2.0], [0.1] * 7],
    ids=["repeated-integer", "repeated-decimal"],
)
def test_constant_series_raises_value_error(values):
    _expect_value_error(
        lambda: acf(values, 2),
        "a constant series was accepted: its autocorrelation is undefined, raise ValueError",
    )


@pytest.mark.parametrize(
    "nlags",
    [-1, 5, 2.0, True],
    ids=["negative", "length", "float", "bool"],
)
def test_invalid_nlags_raises_value_error(nlags):
    _expect_value_error(
        lambda: acf([1.0, 3.0, 2.0, 5.0, 4.0], nlags),
        "an invalid nlags was accepted: nlags must be an integer from 0 to len(x) - 1",
    )


def test_series_too_short_for_lags_raises_value_error():
    _expect_value_error(
        lambda: acf([1.0], 0),
        "a series with one value was accepted: it needs at least two values",
    )


@pytest.mark.parametrize(
    "values",
    [[1.0, float("nan"), 3.0], [1.0, float("inf"), 3.0]],
    ids=["nan", "infinity"],
)
def test_non_finite_input_raises_value_error(values):
    _expect_value_error(
        lambda: acf(values, 1),
        "a NaN or infinite value was accepted: x must contain only finite numbers",
    )


def test_two_dimensional_input_raises_value_error():
    _expect_value_error(
        lambda: acf(np.ones((4, 2)), 1),
        "a two-dimensional input was accepted: x must be one-dimensional",
    )


def _exact_acf(values, nlags):
    """Autocorrelations in exact rational arithmetic from the float values as given."""
    xs = [Fraction(float(v)) for v in values]
    n = len(xs)
    mean = sum(xs) / n
    d = [x - mean for x in xs]
    denominator = sum(v * v for v in d)
    return np.array(
        [float(sum(d[t] * d[t + k] for t in range(n - k)) / denominator) for k in range(nlags + 1)]
    )


def test_nearly_constant_series_is_not_rejected_as_constant():
    steps = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float64)
    values = 1.0 + 1e-13 * steps
    expected = _exact_acf(values, 3)
    result = _run(lambda: acf(values, 3), "acf rejected a series that is not exactly constant")
    _close(
        result,
        expected,
        "a series that is not exactly constant must get its autocorrelations, not an error",
        atol=1e-9,
    )


def test_integer_input_returns_float_autocorrelations():
    result = _run(lambda: acf([1, 2, 3, 4, 5, 7], 2), "acf raised on integer input")
    _require(result.dtype.kind == "f", "autocorrelations must be floating-point for integer input")


def test_inputs_are_not_mutated():
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    before = values.copy()
    _run(lambda: acf(values, 3), "acf raised on a valid series")
    _require(np.array_equal(values, before), "the input array was modified in place")


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(
        st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=40
    ),
    st.data(),
)
def test_property_matches_the_explicit_sums_and_is_bounded_by_one(values, data):
    arr = np.asarray(values, dtype=np.float64)
    assume(bool(np.ptp(arr) > 1e-6))
    nlags = data.draw(st.integers(min_value=0, max_value=arr.size - 1))
    expected = _loop_reference(list(arr), nlags)
    result = _run(lambda: acf(values, nlags), "acf raised on a valid list")
    _close(
        result,
        expected,
        "autocorrelation disagrees with the explicit sums of products",
        atol=1e-9,
    )
    _require(bool(np.all(np.abs(result) <= 1.0 + 1e-9)), "autocorrelations must lie in [-1, 1]")


def test_varying_two_dimensional_input_raises_value_error():
    _expect_value_error(
        lambda: acf(np.arange(4.0).reshape(2, 2), 1),
        "a two-dimensional input with varying values was accepted: x must be one-dimensional",
    )


@pytest.fixture(scope="module")
def nile_volumes():
    # Annual Nile flow volumes: integer-valued, with a level change near 1898.
    return load_series("nile")


@pytest.fixture(scope="module")
def sunspot_counts():
    # Yearly sunspot activity: an 11-year cycle, stored as float64 by the source.
    return load_series("sunspots")


def test_matches_statsmodels_on_integer_valued_nile_volumes(nile_volumes):
    values = np.round(nile_volumes).astype(np.int64)
    reference = sm_acf(values.astype(np.float64), nlags=20, adjusted=False, fft=False)
    result = _run(lambda: acf(values, 20), "acf raised on integer-valued real data")
    _close(
        result,
        reference,
        "integer-valued series gets the wrong autocorrelation: the mean must be exact, not an "
        "integer floor division",
    )


def test_matches_float64_formula_on_float32_storage_of_sunspots(sunspot_counts):
    stored = sunspot_counts.astype(np.float32)
    reference = sm_acf(stored.astype(np.float64), nlags=20, adjusted=False, fft=False)
    result = _run(lambda: acf(stored, 20), "acf raised on a float32 array")
    _close(
        result,
        reference,
        "float32 input gives float32-rounded autocorrelations: convert to float64 before the sums",
        atol=1e-9,
    )


def test_huge_readings_of_opposite_sign_give_finite_autocorrelations():
    # Two readings near the float limit with opposite signs, plus a zero and a one. The
    # autocorrelations are real numbers in [-1, 1], and lag 0 is exactly one. The test does not
    # depend on how the arithmetic is ordered.
    values = np.array([1e308, -1e308, 0.0, 1.0])
    result = _run(lambda: acf(values, 3), "acf raised on huge readings of opposite sign")
    _require(
        bool(np.all(np.isfinite(result))),
        "huge readings of opposite sign made the autocorrelation infinite or NaN: the scaling "
        "step before the products is wrong",
    )
    _require(result[0] == 1.0, "lag 0 must be exactly one, even for huge readings")
    _require(
        bool(np.all(np.abs(result) <= 1.0 + 1e-9)),
        "an autocorrelation is never larger than one in size, even for huge readings",
    )
