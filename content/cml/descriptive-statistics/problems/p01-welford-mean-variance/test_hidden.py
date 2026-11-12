import math
import time
import tracemalloc

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import welford

from dogma_data import rng, sample_rows, sample_window


def _real_normal(gen, size=None, loc=0.0, scale=1.0):
    """Real numbers where this test used to draw from a normal distribution.

    Standardised breast_cancer measurements (mean 0 and standard deviation 1 over the dataset,
    skewed, with a few values far out), drawn by ``gen``. A vector is one real measurement
    over that many tumours, a matrix is real tumours by real measurements, and any other
    shape is filled from whole rows. ``loc`` and ``scale`` shift and stretch them as before.
    """
    shape = () if size is None else tuple(int(side) for side in np.atleast_1d(size))
    total = int(np.prod(shape)) if shape else 1
    if len(shape) == 1 and 1 <= shape[0] <= 569:
        values = sample_rows("breast_cancer", gen, shape[0], 1, standardize=True)[0][:, 0]
    elif len(shape) == 2 and 1 <= shape[0] <= 569 and 1 <= shape[1] <= 30:
        values = sample_rows("breast_cancer", gen, shape[0], shape[1], standardize=True)[0]
    else:
        rows = max(1, min(569, -(-total // 30)))
        drawn = sample_rows("breast_cancer", gen, rows, standardize=True)[0]
        values = np.resize(drawn.ravel(), total).reshape(shape)
    values = values * scale + loc
    return values if shape else float(values)


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


@pytest.mark.parametrize(
    ("n", "seed"), [(2, 0), (3, 1), (10, 2), (100, 3), (1000, 4)], ids=list("abcde")
)
def test_matches_numpy_mean_and_sample_variance(n, seed):
    # Real block groups of california_housing: average rooms per household, heavy-tailed.
    values = sample_rows("california_housing", rng(f"welford-p01:numpy:{seed}"), n)[0][:, 2]
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


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
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
    values = 1e9 + _real_normal(rng, 1000)
    _, _, variance = welford(values)
    np.testing.assert_allclose(
        variance,
        np.var(values, ddof=1),
        rtol=1e-6,
        err_msg="variance lost precision at a large offset: use Welford's update, not "
        "E[x^2] - E[x]^2 from raw sums",
    )


def test_generator_input_matches_the_array_result():
    values = sample_window("sunspots", rng("welford-p01:stream"), 200)  # real yearly counts
    from_array = welford(values)
    from_stream = welford(_stream(values))
    assert from_stream[0] == from_array[0] == 200, "generator input must be counted in full"
    np.testing.assert_allclose(
        from_stream[1], from_array[1], rtol=1e-12, err_msg="a generator gave a different mean"
    )
    np.testing.assert_allclose(
        from_stream[2],
        from_array[2],
        rtol=1e-12,
        err_msg="a generator gave a different variance: the input must be read in one pass",
    )


def test_streams_the_input_in_constant_memory():
    count = 300_000
    rng = np.random.default_rng(7)
    values = _real_normal(rng, count)
    tracemalloc.start()
    try:
        n, _, variance = welford(_stream(values))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert n == count, "every streamed value must be counted"
    np.testing.assert_allclose(
        variance, np.var(values, ddof=1), rtol=1e-9, err_msg="streamed variance differs from np.var"
    )
    assert peak < 4 * 1024 * 1024, (
        "the input is being stored: keep only the count, mean and M2 while streaming"
    )


def test_constant_values_have_exactly_zero_variance():
    n, mean, variance = welford([7.0] * 5)
    assert n == 5, "count must equal the number of values streamed"
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
    np.testing.assert_allclose(mean, 2.5, atol=1e-12, err_msg="integer input gives the wrong mean")
    np.testing.assert_allclose(
        variance, 5.0 / 3.0, atol=1e-12, err_msg="integer input gives the wrong variance"
    )


@pytest.mark.parametrize("scale", [1e-3, 1e6], ids=["small-scale", "large-scale"])
def test_scaling_the_data_scales_the_variance_by_the_square(scale):
    values = sample_window("nile", rng("welford-p01:scale"), 50)  # real annual river flow
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
    values = _real_normal(np.random.default_rng(9), 20_000, loc=-4.0, scale=0.5)
    start = time.perf_counter()
    n, mean, variance = welford(_stream(values))
    assert time.perf_counter() - start < 10, "a single pass over 20k values should be fast"
    assert n == 20_000, "count must equal the number of values streamed"
    np.testing.assert_allclose(
        mean, np.mean(values), rtol=1e-9, err_msg="long-stream mean differs from np.mean"
    )
    np.testing.assert_allclose(
        variance,
        np.var(values, ddof=1),
        rtol=1e-9,
        err_msg="long-stream variance differs from np.var",
    )
