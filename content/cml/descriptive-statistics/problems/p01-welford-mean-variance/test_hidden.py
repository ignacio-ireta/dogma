import math
import time
import tracemalloc

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import welford


def _stream(values):
    """A one-shot generator over ``values``: it can be consumed only once."""
    for value in values:
        yield float(value)


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@pytest.mark.parametrize(("n", "seed"), [(2, 0), (3, 1), (10, 2), (100, 3), (1000, 4)])
def test_matches_numpy_mean_and_sample_variance(n, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=3.0, scale=2.0, size=n)
    count, mean, variance = welford(values)
    assert count == n, "count disagrees with the number of values"
    np.testing.assert_allclose(
        mean,
        np.mean(values),
        rtol=1e-12,
        atol=1e-12,
        err_msg="mean differs from np.mean: check the running-mean update",
    )
    np.testing.assert_allclose(
        variance,
        np.var(values, ddof=1),
        rtol=1e-9,
        atol=1e-12,
        err_msg="variance differs from np.var(ddof=1): divide M2 by n - 1, not n",
    )


@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=40)
)
def test_property_agrees_with_numpy_on_any_finite_list(values):
    arr = np.asarray(values, dtype=np.float64)
    _, mean, variance = welford(values)
    scale = float(np.max(np.abs(arr))) ** 2 + 1.0
    np.testing.assert_allclose(
        mean,
        arr.mean(),
        rtol=1e-9,
        atol=1e-9 * (1 + np.max(np.abs(arr))),
        err_msg="mean differs from np.mean on a random list",
    )
    np.testing.assert_allclose(
        variance,
        arr.var(ddof=1),
        rtol=1e-9,
        atol=1e-9 * scale,
        err_msg="variance differs from np.var(ddof=1) on a random list",
    )


def test_large_offset_does_not_destroy_the_variance():
    rng = np.random.default_rng(5)
    values = 1e9 + rng.normal(size=1000)
    _, _, variance = welford(values)
    np.testing.assert_allclose(
        variance,
        np.var(values, ddof=1),
        rtol=1e-6,
        err_msg="variance lost precision at a large offset: use Welford's update, not "
        "E[x^2] - E[x]^2 from raw sums",
    )


def test_generator_input_matches_the_array_result():
    values = np.random.default_rng(6).normal(size=200)
    from_array = welford(values)
    from_stream = welford(_stream(values))
    assert from_stream[0] == from_array[0] == 200, "generator input must be counted in full"
    np.testing.assert_allclose(from_stream[1], from_array[1], rtol=1e-12)
    np.testing.assert_allclose(
        from_stream[2],
        from_array[2],
        rtol=1e-12,
        err_msg="a generator gave a different variance: the input must be read in one pass",
    )


def test_streams_the_input_in_constant_memory():
    count = 300_000
    rng = np.random.default_rng(7)
    values = rng.normal(size=count)
    tracemalloc.start()
    try:
        n, _, variance = welford(_stream(values))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert n == count, "every streamed value must be counted"
    np.testing.assert_allclose(variance, np.var(values, ddof=1), rtol=1e-9)
    assert peak < 4 * 1024 * 1024, (
        "the input is being stored: keep only the count, mean and M2 while streaming"
    )


def test_constant_values_have_exactly_zero_variance():
    n, mean, variance = welford([7.0] * 5)
    assert n == 5
    assert mean == 7.0, "the mean of identical values is that value"
    assert variance == 0.0, "identical values have no spread: the variance must be exactly 0"


def test_single_negative_value_has_nan_variance_and_its_mean():
    n, mean, variance = welford([-3.5])
    assert n == 1, "count of one value must be 1"
    assert mean == -3.5, "mean of one value is that value"
    assert math.isnan(variance), "one value must give a nan variance, not 0.0 or another number"


def test_integer_input_returns_float_mean_and_variance():
    n, mean, variance = welford([1, 2, 3, 4])
    assert isinstance(n, int), "count must be an int"
    assert isinstance(mean, float), "mean must be a float even for integer input"
    assert isinstance(variance, float), "variance must be a float even for integer input"
    np.testing.assert_allclose(mean, 2.5, atol=1e-12)
    np.testing.assert_allclose(variance, 5.0 / 3.0, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-3, 1e6])
def test_scaling_the_data_scales_the_variance_by_the_square(scale):
    values = np.random.default_rng(8).normal(size=50)
    _, _, base = welford(values)
    _, _, scaled = welford(values * scale)
    np.testing.assert_allclose(
        scaled,
        base * scale**2,
        rtol=1e-9,
        err_msg="variance of c*x must be c^2 times the variance of x",
    )


def test_empty_generator_raises_value_error():
    _expect_value_error(
        lambda: welford(_stream([])),
        "an empty stream was accepted: raise ValueError when no values are seen",
    )


def test_long_stream_agrees_with_numpy():
    values = np.random.default_rng(9).normal(loc=-4.0, scale=0.5, size=20_000)
    start = time.perf_counter()
    n, mean, variance = welford(_stream(values))
    assert time.perf_counter() - start < 10, "a single pass over 20k values should be fast"
    assert n == 20_000
    np.testing.assert_allclose(mean, np.mean(values), rtol=1e-9)
    np.testing.assert_allclose(variance, np.var(values, ddof=1), rtol=1e-9)
