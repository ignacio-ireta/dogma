import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from submission import seasonal_naive_forecast_and_mase

from dogma_data import rng as data_rng
from dogma_data import sample_rows, sample_window


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


def _reference(train, test, season):
    """The formulas written as explicit loops, with train positions counted from 1."""
    n = len(train)
    y = {t + 1: float(train[t]) for t in range(n)}
    forecast = []
    for h in range(1, len(test) + 1):
        position = n + h - season * math.ceil(h / season)
        forecast.append(y[position])
    scale_terms = [abs(y[t] - y[t - season]) for t in range(season + 1, n + 1)]
    scale = sum(scale_terms) / len(scale_terms)
    errors = [abs(float(test[h]) - forecast[h]) for h in range(len(test))]
    if scale == 0.0:
        return np.array(forecast), math.nan  # undefined: the property test skips these inputs
    return np.array(forecast), (sum(errors) / len(errors)) / scale


def _unpack(result, horizon, mistake):
    _require(isinstance(result, tuple) and len(result) == 2, f"{mistake}: return (forecast, mase)")
    _require(
        isinstance(result[0], np.ndarray) and result[0].dtype == np.float64,
        "forecast must be a float64 NumPy array",
    )
    forecast = np.asarray(result[0], dtype=np.float64)
    _require(forecast.shape == (horizon,), f"{mistake}: forecast must have one value per test step")
    _require(isinstance(result[1], float), f"{mistake}: MASE must be a number")
    return forecast, float(result[1])


@pytest.mark.parametrize(
    ("n", "season", "horizon", "seed"),
    [(30, 1, 5, 0), (24, 3, 9, 1), (40, 4, 13, 2), (60, 7, 30, 3), (36, 12, 12, 4)],
    ids=["season-one", "season-three", "season-four", "long-horizon", "season-twelve"],
)
def test_matches_explicit_formulas_on_seeded_series(n, season, horizon, seed):
    # One real run of monthly sea-surface temperatures, split in time order: the series has a
    # yearly season, so the season length matters to the forecast and to the scale.
    series = sample_window("elnino", data_rng(f"mase-p03:formulas:{seed}"), n + horizon)
    train, test = series[:n], series[n:]
    ref_forecast, ref_mase = _reference(train, test, season)
    forecast, mase = _unpack(
        _run(
            lambda: seasonal_naive_forecast_and_mase(train, test, season),
            "seasonal_naive_forecast_and_mase raised on a valid series",
        ),
        horizon,
        "seasonal_naive_forecast_and_mase",
    )
    _require(
        bool(np.allclose(forecast, ref_forecast, rtol=1e-12, atol=1e-12)),
        "forecast differs from the seasonal-naive rule: check the seasonal index of each step",
    )
    _require(
        math.isclose(mase, ref_mase, rel_tol=1e-9, abs_tol=1e-12),
        "MASE differs: divide the test error by the mean of the n - season seasonal differences",
    )


def test_horizon_longer_than_a_season_repeats_the_last_complete_season():
    train = np.array([4.0, 9.0, 1.0, 7.0, 5.0, 2.0, 8.0, 6.0, 3.0, 0.0, 11.0, 2.5])
    expected = np.tile(train[-4:], 3)[:10]
    forecast, _ = _unpack(
        _run(
            lambda: seasonal_naive_forecast_and_mase(train, np.zeros(10), 4),
            "seasonal_naive_forecast_and_mase raised on a valid series",
        ),
        10,
        "seasonal_naive_forecast_and_mase",
    )
    _require(
        bool(np.array_equal(forecast, expected)),
        "a horizon longer than one season must repeat the last complete season, not drift",
    )


def test_season_one_repeats_the_last_value():
    train = np.array([3.0, 8.0, 5.0, 6.0])
    forecast, _ = _unpack(
        _run(
            lambda: seasonal_naive_forecast_and_mase(train, np.zeros(4), 1),
            "seasonal_naive_forecast_and_mase raised on a valid series",
        ),
        4,
        "seasonal_naive_forecast_and_mase",
    )
    _require(bool(np.all(forecast == 6.0)), "season one must repeat the last value of train")


def test_shifting_and_scaling_train_and_test_together_leaves_mase_unchanged():
    series = sample_window("elnino", data_rng("mase-p03:shift-and-scale"), 30)
    train, test = series[:24], series[24:]
    base_f, base_m = _unpack(
        _run(lambda: seasonal_naive_forecast_and_mase(train, test, 4), "call raised"), 6, "base"
    )
    shift_f, shift_m = _unpack(
        _run(lambda: seasonal_naive_forecast_and_mase(train + 40.0, test + 40.0, 4), "call raised"),
        6,
        "shifted",
    )
    scale_f, scale_m = _unpack(
        _run(lambda: seasonal_naive_forecast_and_mase(train * 3.5, test * 3.5, 4), "call raised"),
        6,
        "scaled",
    )
    _require(
        bool(np.allclose(shift_f, base_f + 40.0, rtol=1e-12, atol=1e-12)),
        "shifting train and test by a constant must shift the forecast by that constant",
    )
    _require(
        math.isclose(shift_m, base_m, rel_tol=1e-9),
        "MASE changed under a shift of train and test: the scale must be a difference, not a level",
    )
    _require(
        bool(np.allclose(scale_f, base_f * 3.5, rtol=1e-12, atol=1e-12)),
        "scaling train and test by a factor must scale the forecast by that factor",
    )
    _require(
        math.isclose(scale_m, base_m, rel_tol=1e-9),
        "MASE changed under a common rescaling: it must be scale-free",
    )


def test_forecast_depends_on_training_values_only():
    rng = np.random.default_rng(11)
    train = sample_window("sunspots", data_rng("mase-p03:training-only"), 20)
    first_f, _ = _unpack(
        _run(
            lambda: seasonal_naive_forecast_and_mase(train, _real_normal(rng, 5), 5),
            "call raised",
        ),
        5,
        "first",
    )
    second_f, _ = _unpack(
        _run(
            lambda: seasonal_naive_forecast_and_mase(train, _real_normal(rng, 5) * 100.0, 5),
            "call raised",
        ),
        5,
        "second",
    )
    _require(
        bool(np.array_equal(first_f, second_f)),
        "the forecast changed when only the test values changed: the forecast must use train alone",
    )


def test_scale_does_not_depend_on_the_test_values():
    series = sample_window("co2", data_rng("mase-p03:two-tests"), 26)  # weekly CO2, trending
    train, test_a = series[:18], series[18:22]
    test_b = series[22:] * 2.0
    forecast, _ = _reference(train, test_a, 3)
    error_a = float(np.mean(np.abs(test_a - forecast)))
    error_b = float(np.mean(np.abs(test_b - forecast)))
    _, mase_a = _unpack(
        _run(lambda: seasonal_naive_forecast_and_mase(train, test_a, 3), "call raised"), 4, "a"
    )
    _, mase_b = _unpack(
        _run(lambda: seasonal_naive_forecast_and_mase(train, test_b, 3), "call raised"), 4, "b"
    )
    _require(
        math.isclose(mase_a / mase_b, error_a / error_b, rel_tol=1e-9),
        "the MASE ratio must equal the error ratio: the scale must come from train alone",
    )


def test_zero_scale_training_series_raises_value_error():
    _expect_value_error(
        lambda: seasonal_naive_forecast_and_mase([2.0, 5.0, 7.0] * 4, [1.0, 2.0], 3),
        "a perfectly periodic training series was accepted: the MASE scale is zero",
    )


def test_training_series_must_cover_one_season_difference():
    _expect_value_error(
        lambda: seasonal_naive_forecast_and_mase([1.0, 2.0, 4.0], [1.0], 3),
        "a training series with no seasonal difference was accepted: need season + 1 values",
    )


@pytest.mark.parametrize(
    ("season", "label"),
    [(0, "zero"), (-2, "negative"), (2.5, "fraction"), (True, "bool")],
    ids=["zero", "negative", "fraction", "bool"],
)
def test_invalid_season_raises_value_error(season, label):
    _expect_value_error(
        lambda: seasonal_naive_forecast_and_mase([1.0, 3.0, 2.0, 5.0], [1.0], season),
        f"an invalid season ({label}) was accepted: season must be an integer of at least 1",
    )


def test_empty_test_raises_value_error():
    _expect_value_error(
        lambda: seasonal_naive_forecast_and_mase([1.0, 3.0, 2.0, 5.0], [], 2),
        "an empty test was accepted: at least one forecast step is required",
    )


@pytest.mark.parametrize(
    ("train", "test"),
    [([1.0, float("nan"), 3.0, 2.0], [1.0]), ([1.0, 3.0, 2.0, 5.0], [float("inf")])],
    ids=["nan-in-train", "infinite-in-test"],
)
def test_non_finite_values_raise_value_error(train, test):
    _expect_value_error(
        lambda: seasonal_naive_forecast_and_mase(train, test, 2),
        "a NaN or infinite value was accepted: inputs must be finite",
    )


def test_two_dimensional_input_raises_value_error():
    _expect_value_error(
        lambda: seasonal_naive_forecast_and_mase(np.ones((4, 2)), [1.0], 2),
        "a two-dimensional train was accepted: train must be one-dimensional",
    )


def test_integer_input_gives_a_float_forecast():
    forecast, mase = _unpack(
        _run(
            lambda: seasonal_naive_forecast_and_mase([1, 3, 2, 5, 4], [4, 4], 2),
            "call raised on integer input",
        ),
        2,
        "integer input",
    )
    _require(forecast.dtype.kind == "f", "the forecast must be floating-point for integer input")
    _require(math.isfinite(mase), "MASE must be finite for this training series")


def test_inputs_are_not_mutated():
    train = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    test = np.array([2.0, 6.0])
    train_before, test_before = train.copy(), test.copy()
    _run(lambda: seasonal_naive_forecast_and_mase(train, test, 2), "call raised on a valid series")
    _require(np.array_equal(train, train_before), "train was modified in place")
    _require(np.array_equal(test, test_before), "test was modified in place")


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.integers(min_value=1, max_value=6),
    st.lists(
        st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False), min_size=7, max_size=30
    ),
    st.lists(
        st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=12
    ),
)
def test_property_matches_the_explicit_formulas(season, train_values, test_values):
    train = np.asarray(train_values, dtype=np.float64)
    test = np.asarray(test_values, dtype=np.float64)
    assume(train.size >= season + 1)
    ref_forecast, ref_mase = _reference(train, test, season)
    assume(math.isfinite(ref_mase))
    forecast, mase = _unpack(
        _run(
            lambda: seasonal_naive_forecast_and_mase(train, test, season),
            "seasonal_naive_forecast_and_mase raised on a valid list",
        ),
        test.size,
        "seasonal_naive_forecast_and_mase",
    )
    _require(
        bool(np.allclose(forecast, ref_forecast, rtol=1e-12, atol=1e-12)),
        "forecast differs from the seasonal-naive rule on a random list",
    )
    _require(
        math.isclose(mase, ref_mase, rel_tol=1e-9, abs_tol=1e-12),
        "MASE differs from the explicit formula on a random list",
    )


def test_all_zero_training_series_raises_value_error():
    _expect_value_error(
        lambda: seasonal_naive_forecast_and_mase([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 2.0], 2),
        "an all-zero training series was accepted: the MASE scale is zero, so raise ValueError",
    )
