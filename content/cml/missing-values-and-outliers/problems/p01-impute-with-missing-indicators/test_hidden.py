"""Hidden tests: scikit-learn's SimpleImputer(add_indicator=True) as the oracle, plus layout, dtype,
mutation and validation cases. Each failure names the likely mistake first."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.impute import SimpleImputer
from submission import impute_with_indicators

from dogma_data import load_frame

SETTINGS = settings(max_examples=40, deadline=None, report_multiple_bugs=False)


def _run(df, numeric, strategy="median"):
    try:
        return impute_with_indicators(df, numeric, strategy)
    except Exception as exc:
        pytest.fail(
            f"valid imputation request crashed with {type(exc).__name__}: "
            "check the fill, the indicator columns and the column names"
        )


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def _table(seed):
    rng = np.random.default_rng(seed)
    rows = int(rng.integers(3, 25))
    data = {}
    for name in ["a", "b", "c"]:
        values = rng.normal(size=rows) * 4 + 1
        values[rng.random(rows) < 0.3] = np.nan
        data[name] = values
    data["complete"] = rng.normal(size=rows)  # never missing, so it gets no indicator
    data["count"] = rng.integers(0, 9, size=rows)  # integer dtype, never missing
    df = pd.DataFrame(data)
    df.loc[0, ["a", "b", "c"]] = [2.0, 3.0, 4.0]  # every named column keeps an observed value
    return df


@SETTINGS
@given(st.integers(0, 10_000), st.sampled_from(["median", "mean"]))
def test_values_and_indicators_match_sklearn_simple_imputer(seed, strategy):
    df = _table(seed)
    numeric = ["a", "b", "c"]
    oracle = SimpleImputer(strategy=strategy, add_indicator=True).fit_transform(
        df[numeric].to_numpy()
    )
    out = _run(df, numeric, strategy)
    gapped = [name for name in numeric if df[name].isna().any()]
    got_values = out[numeric].to_numpy(dtype=np.float64)
    got_indicators = out[[f"{name}_was_missing" for name in gapped]].to_numpy(dtype=np.float64)
    if not np.allclose(got_values, oracle[:, : len(numeric)], rtol=1e-9, atol=1e-9):
        pytest.fail(
            "imputed values differ: fill each gap with the statistic of the observed values"
        )
    if not np.array_equal(got_indicators, oracle[:, len(numeric) :]):
        pytest.fail("indicator values differ: 1 exactly where the input was missing")


def test_layout_has_indicators_only_for_gapped_columns_in_named_order():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, 6.0], "z": [np.nan, 1.0, 2.0]})
    out = _run(df, ["z", "x", "y"])
    if list(out.columns) != ["x", "y", "z", "z_was_missing", "x_was_missing"]:
        pytest.fail(
            "column layout differs: indicators come last, in the order the names were given"
        )


def test_complete_column_keeps_its_dtype_and_gets_no_indicator():
    df = pd.DataFrame({"n": [1, 2, 3], "m": [np.nan, 2.0, 3.0]})
    out = _run(df, ["n", "m"])
    if "n_was_missing" in out.columns:
        pytest.fail("a column without missing values must not get an indicator")
    if out["n"].dtype != df["n"].dtype:
        pytest.fail("a complete column must keep its original dtype")


def test_indicator_dtype_is_int8():
    df = pd.DataFrame({"x": [1.0, np.nan]})
    out = _run(df, ["x"])
    if out["x_was_missing"].dtype != np.int8:
        pytest.fail("indicators must be int8, not a float or a bool column")


def test_input_frame_is_not_mutated():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [np.nan, 2.0, 2.0]})
    before = df.copy()
    _run(df, ["x", "y"])
    if not df.equals(before):
        pytest.fail("the input frame was modified: build a new frame instead of filling in place")


def test_even_count_median_and_single_value_column():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 4.0, 10.0, np.nan], "y": [7.0, np.nan, np.nan, np.nan, np.nan]}
    )
    out = _run(df, ["x", "y"])
    if not np.isclose(out.loc[4, "x"], 3.0):
        pytest.fail("an even count needs the mean of the two middle values as the median")
    if not np.allclose(out["y"].tolist(), [7.0, 7.0, 7.0, 7.0, 7.0]):
        pytest.fail("a single observed value must fill every gap with itself")


def test_large_magnitude_values_fill_without_precision_loss():
    base = 1e15
    df = pd.DataFrame({"x": [base + 1.0, np.nan, base + 3.0, base + 5.0]})
    out = _run(df, ["x"])
    if not np.isclose(out.loc[1, "x"], base + 3.0, rtol=1e-15, atol=0):
        pytest.fail("the fill is off at large magnitude: compute the median of the observed values")


def test_boolean_column_is_not_numeric_here():
    df = pd.DataFrame({"flag": [True, False, True]})
    _expect_value_error(
        lambda: impute_with_indicators(df, ["flag"]),
        "a boolean column was accepted as numeric: booleans are not numeric in this contract",
    )


def test_existing_indicator_name_clashes_only_when_an_indicator_would_be_created():
    df = pd.DataFrame({"x": [1.0, 2.0], "x_was_missing": [0, 0]})
    out = _run(df, ["x"])
    if list(out.columns) != ["x", "x_was_missing"]:
        pytest.fail("no gap means no indicator, so an existing name must not raise")


def test_empty_frame_and_empty_name_list_return_copies():
    empty = pd.DataFrame({"x": pd.Series([], dtype=float)})
    out = _run(empty, ["x"])
    if out.shape != (0, 1) or "x_was_missing" in out.columns:
        pytest.fail("an empty frame with no gaps must come back unchanged")
    df = pd.DataFrame({"x": [1.0, np.nan]})
    if list(_run(df, []).columns) != ["x"]:
        pytest.fail("an empty list of names must add no indicator columns")


@pytest.mark.parametrize(
    ("frame", "numeric", "strategy"),
    [
        (pd.DataFrame({"x": [1.0, np.nan]}), ["x"], "mode"),
        (pd.DataFrame({"x": [1.0, np.nan]}), ["y"], "median"),
        (pd.DataFrame({"x": ["a", None]}), ["x"], "median"),
        (pd.DataFrame({"x": [1.0, np.nan]}), ["x", "x"], "median"),
        (pd.DataFrame({"x": [np.nan, np.nan]}), ["x"], "median"),
        (pd.DataFrame({"x": [1.0, np.nan], "x_was_missing": [0, 0]}), ["x"], "median"),
    ],
    ids=[
        "unknown-strategy",
        "missing-column",
        "non-numeric",
        "named-twice",
        "no-observed-value",
        "indicator-name-clash",
    ],
)
def test_invalid_requests_raise_value_error(frame, numeric, strategy):
    _expect_value_error(
        lambda: impute_with_indicators(frame, numeric, strategy),
        "an invalid imputation request was accepted: check the strategy, the names and the data",
    )


@pytest.fixture(scope="module")
def penguins():
    return load_frame("penguins")


@pytest.mark.parametrize("strategy", ["median", "mean"], ids=["median", "mean"])
def test_matches_simple_imputer_on_real_penguin_measurements(penguins, strategy):
    # Real measurements with gaps on the same rows, in grams and millimetres.
    numeric = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    oracle = SimpleImputer(strategy=strategy, add_indicator=True).fit_transform(
        penguins[numeric].to_numpy(dtype=np.float64)
    )
    out = _run(penguins, numeric, strategy)
    got_values = out[numeric].to_numpy(dtype=np.float64)
    got_indicators = out[[f"{name}_was_missing" for name in numeric]].to_numpy(dtype=np.float64)
    if not np.allclose(got_values, oracle[:, :4], rtol=1e-9, atol=1e-9):
        pytest.fail(
            "imputed penguin measurements differ: fill each gap with the observed statistic"
        )
    if not np.array_equal(got_indicators, oracle[:, 4:]):
        pytest.fail("indicator values differ: 1 exactly on the rows where the measurement was NaN")


def test_gaps_in_unnamed_columns_stay_as_they_are_on_real_penguins(penguins):
    # Only body_mass_g is named. Other gapped columns (including the text column sex) keep their
    # NaN values, get no indicator, and an all-missing column that is not named raises nothing.
    frame = penguins.assign(unused=np.nan)
    out = _run(frame, ["body_mass_g"])
    if list(out.columns) != [*frame.columns, "body_mass_g_was_missing"]:
        pytest.fail(
            "column layout differs: only named gapped columns get an indicator, and the rest"
            " keep their columns"
        )
    for name in frame.columns:
        if name == "body_mass_g":
            continue
        if not out[name].equals(frame[name]):
            pytest.fail(f"an unnamed column was changed: {name} must be copied unchanged")
    if out["body_mass_g"].isna().any():
        pytest.fail("the named gap in body_mass_g was not filled")


def test_returned_frame_keeps_the_input_index():
    df = pd.DataFrame({"age": [20.0, np.nan, 40.0]}, index=[10, 20, 30])
    out = _run(df, ["age"])
    if not out.index.equals(df.index):
        pytest.fail("the returned frame must keep the index of the input frame, not a new one")
