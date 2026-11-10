import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import rolling_mean_std

from dogma_data import load_series


def _require(ok, mistake):
    """Fail with the mistake's name only. Hypothesis and the grader show this first line."""
    if not ok:
        raise AssertionError(mistake)


def _run(call, mistake):
    try:
        return call()
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from None


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from None
    raise AssertionError(mistake)


def _unpack(result, n, mistake):
    _require(isinstance(result, tuple) and len(result) == 2, f"{mistake}: return (means, stds)")
    _require(
        all(isinstance(value, np.ndarray) and value.dtype == np.float64 for value in result),
        "return two float64 NumPy arrays, preserving the required output dtype",
    )
    means, stds = np.asarray(result[0]), np.asarray(result[1])
    _require(
        means.shape == (n,) and stds.shape == (n,),
        f"{mistake}: means and stds must have one entry per value of x",
    )
    return means, stds


def _close(actual, expected, mistake, *, rtol=1e-9, atol=1e-9, nan_mistake=None):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    _require(actual.shape == expected.shape, f"shape mismatch: {mistake}")
    _require(
        np.array_equal(np.isnan(actual), np.isnan(expected)),
        nan_mistake or f"NaN positions differ from the expected ones: {mistake}",
    )
    finite = ~np.isnan(expected)
    _require(np.allclose(actual[finite], expected[finite], rtol=rtol, atol=atol), mistake)


def _two_pass(values, window, min_periods):
    """Per-window two-pass reference: each window is summarised directly, with no running state."""
    n = values.size
    means = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    for i in range(n):
        segment = values[max(0, i - window + 1) : i + 1]
        if segment.size >= min_periods:
            means[i] = segment.mean()
            if segment.size > 1:
                stds[i] = segment.std(ddof=1)
    return means, stds


def _pandas_reference(values, window, min_periods):
    rolling = pd.Series(values, dtype=np.float64).rolling(window, min_periods=min_periods)
    return rolling.mean().to_numpy(), rolling.std().to_numpy()


MEAN_NAN = "a mean must be NaN exactly when its window holds fewer than min_periods values"
STD_NAN = "a std must be NaN for one value and for fewer than min_periods values"
PANDAS_CASES = [(60, 5, None), (120, 10, 3), (80, 1, None), (30, 45, 1), (200, 37, 37)]


@pytest.mark.parametrize(
    ("n", "window", "min_periods"),
    PANDAS_CASES,
    ids=["short", "partial", "window-one", "long-window", "full-only"],
)
def test_matches_pandas_rolling_on_random_series(n, window, min_periods):
    values = np.random.default_rng(n + window).normal(loc=2.0, scale=3.0, size=n)
    need = window if min_periods is None else min_periods
    ref_means, ref_stds = _pandas_reference(values, window, need)
    result = _run(
        lambda: rolling_mean_std(values, window, min_periods=min_periods),
        "rolling_mean_std raised on a valid series",
    )
    means, stds = _unpack(result, n, "rolling_mean_std")
    _close(
        means,
        ref_means,
        "rolling mean differs from pandas: check the trailing window and the removal step",
        nan_mistake=MEAN_NAN,
    )
    _close(
        stds,
        ref_stds,
        "rolling std differs from pandas: check the ddof=1 denominator and the removal step",
        nan_mistake=STD_NAN,
    )


def test_window_of_two_spread_is_gap_over_root_two():
    values = np.random.default_rng(21).normal(size=50)
    closed_spread = np.abs(np.diff(values)) / np.sqrt(2.0)
    closed_mean = (values[1:] + values[:-1]) / 2.0
    means, stds = _unpack(
        _run(lambda: rolling_mean_std(values, 2), "rolling_mean_std raised on a valid series"),
        50,
        "rolling_mean_std",
    )
    _require(np.isnan(stds[0]), "the first position holds one value: its spread must be NaN")
    _close(
        stds[1:],
        closed_spread,
        "a pair's spread must be |difference| / sqrt(2): divide by n - 1 = 1, not by n = 2",
    )
    _close(
        means[1:],
        closed_mean,
        "a pair's mean must be the average of its two values",
    )


def test_earlier_outputs_do_not_depend_on_later_values():
    rng = np.random.default_rng(22)
    values = rng.normal(size=60)
    changed = values.copy()
    changed[30:] = rng.normal(loc=50.0, scale=10.0, size=30)
    first = _unpack(
        _run(lambda: rolling_mean_std(values, 5, min_periods=1), "rolling_mean_std raised"),
        60,
        "rolling_mean_std",
    )
    second = _unpack(
        _run(lambda: rolling_mean_std(changed, 5, min_periods=1), "rolling_mean_std raised"),
        60,
        "rolling_mean_std",
    )
    _close(
        first[0][:30],
        second[0][:30],
        "a later value changed an earlier mean: each window must end at its own position",
    )
    _close(
        first[1][:30],
        second[1][:30],
        "a later value changed an earlier spread: each window must end at its own position",
    )


def test_large_common_offset_over_long_series_keeps_the_spread():
    values = 1e9 + np.random.default_rng(5).normal(size=20_000)
    expected_stds = _two_pass(values, 50, 50)[1][49:]
    _, stds = _unpack(
        _run(lambda: rolling_mean_std(values, 50), "rolling_mean_std raised on a valid series"),
        values.size,
        "rolling_mean_std",
    )
    _close(
        stds[49:],
        expected_stds,
        "spread drifted at a large common offset over a long series: centre the values before "
        "the running update",
        rtol=1e-6,
        atol=0.0,
    )


def test_constant_series_has_zero_spread_and_its_own_mean():
    values = np.full(40, 0.1)
    means, stds = _unpack(
        _run(lambda: rolling_mean_std(values, 7, min_periods=1), "rolling_mean_std raised"),
        40,
        "rolling_mean_std",
    )
    _close(means, values, "the mean of constant values must be that value")
    _require(
        bool(np.all(np.abs(stds[1:]) <= 1e-12)),
        "constant values must have zero spread: the running sums cancel imperfectly",
    )


def test_repeated_pairs_have_numerically_zero_spread_at_window_two():
    values = np.repeat(np.random.default_rng(31).normal(size=12) * 3.7, 2)
    _, stds = _unpack(
        _run(lambda: rolling_mean_std(values, 2), "rolling_mean_std raised on a valid series"),
        values.size,
        "rolling_mean_std",
    )
    bound = 1e-6 * (1.0 + float(np.max(np.abs(values))))
    _require(
        bool(np.all(np.abs(stds[1::2]) <= bound)),
        "a pair of equal values must have numerically zero spread: clamp the rounding-negative M2 "
        "before the square root",
    )


def test_window_of_one_returns_the_values_and_nan_spread():
    values = np.random.default_rng(23).normal(size=12)
    means, stds = _unpack(
        _run(lambda: rolling_mean_std(values, 1), "rolling_mean_std raised on a valid series"),
        12,
        "rolling_mean_std",
    )
    _close(means, values, "a window of one must return each value as its own mean")
    _require(bool(np.isnan(stds).all()), "a window of one value has no sample spread: expected NaN")


def test_integer_input_gives_floating_point_outputs():
    result = _run(
        lambda: rolling_mean_std([1, 2, 3, 4, 5], 3, min_periods=1),
        "rolling_mean_std raised on integer input",
    )
    means, stds = _unpack(result, 5, "rolling_mean_std")
    _require(
        means.dtype.kind == "f" and stds.dtype.kind == "f",
        "outputs must be floating-point arrays even when the input holds integers",
    )


@pytest.mark.parametrize(
    "bad_values",
    [[1.0, float("nan"), 3.0], [1.0, float("inf"), 3.0]],
    ids=["nan", "infinity"],
)
def test_non_finite_input_raises_value_error(bad_values):
    _expect_value_error(
        lambda: rolling_mean_std(bad_values, 2),
        "a NaN or infinite value was accepted: x must contain only finite numbers",
    )


def test_two_dimensional_input_raises_value_error():
    _expect_value_error(
        lambda: rolling_mean_std(np.ones((3, 2)), 2),
        "a two-dimensional input was accepted: x must be one-dimensional",
    )


@pytest.mark.parametrize(
    ("window", "min_periods"),
    [(0, None), (2.5, None), (True, None), (3, 4), (3, 0)],
    ids=[
        "window-zero",
        "window-fraction",
        "window-bool",
        "min-periods-above-window",
        "min-periods-zero",
    ],
)
def test_invalid_window_or_min_periods_raises_value_error(window, min_periods):
    _expect_value_error(
        lambda: rolling_mean_std([1.0, 2.0, 3.0, 4.0], window, min_periods=min_periods),
        "an invalid window or min_periods was accepted: window must be an integer of at least 1 "
        "and min_periods must lie between 1 and window",
    )


def test_empty_input_gives_empty_outputs():
    means, stds = _unpack(
        _run(lambda: rolling_mean_std([], 3), "rolling_mean_std raised on empty input"),
        0,
        "rolling_mean_std",
    )
    _require(means.size == 0 and stds.size == 0, "empty input must give empty outputs")


def test_inputs_are_not_mutated():
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    before = values.copy()
    _run(lambda: rolling_mean_std(values, 2), "rolling_mean_std raised on a valid series")
    _require(np.array_equal(values, before), "the input array was modified in place")


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False), max_size=40),
    st.integers(min_value=1, max_value=12),
    st.booleans(),
)
def test_property_agrees_with_per_window_statistics(values, window, partial):
    min_periods = 1 if partial else None
    need = window if min_periods is None else min_periods
    arr = np.asarray(values, dtype=np.float64)
    ref_means, ref_stds = _two_pass(arr, window, need)
    result = _run(
        lambda: rolling_mean_std(values, window, min_periods=min_periods),
        "rolling_mean_std raised on a valid list",
    )
    means, stds = _unpack(result, arr.size, "rolling_mean_std")
    scale = 1.0 + (float(np.max(np.abs(arr))) if arr.size else 0.0)
    _close(
        means,
        ref_means,
        "rolling mean disagrees with the per-window mean",
        atol=1e-9 * scale,
        nan_mistake=MEAN_NAN,
    )
    _close(
        stds,
        ref_stds,
        "rolling spread disagrees with the per-window spread",
        atol=1e-8 * scale,
        nan_mistake=STD_NAN,
    )


@pytest.fixture(scope="module")
def nile_volumes():
    # Annual Nile flow volumes: integer-valued, with a level change near 1898.
    return load_series("nile")


def test_matches_pandas_on_integer_valued_nile_volumes_across_the_level_change(nile_volumes):
    values = np.round(nile_volumes).astype(np.int64)
    ref_means, ref_stds = _pandas_reference(values.astype(np.float64), 10, 10)
    means, stds = _unpack(
        _run(lambda: rolling_mean_std(values, 10), "rolling_mean_std raised on integer input"),
        values.size,
        "rolling_mean_std",
    )
    _close(
        means,
        ref_means,
        "integer input gives rounded window means: keep the fractional mean of the integers",
        nan_mistake=MEAN_NAN,
    )
    _close(
        stds,
        ref_stds,
        "rolling std differs from pandas on integer input: check the ddof=1 denominator",
        nan_mistake=STD_NAN,
    )
