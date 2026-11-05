import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from submission import impute_with_indicators


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


def test_gap_is_filled_with_median_and_indicator_is_appended():
    df = pd.DataFrame({"age": [20.0, np.nan, 40.0, 30.0], "income": [1, 2, 3, 4]})
    out = _run(df, ["age", "income"])
    if list(out.columns) != ["age", "income", "age_was_missing"]:
        pytest.fail("column layout differs: originals first, indicators only for columns with gaps")
    if out["age"].tolist() != [20.0, 30.0, 40.0, 30.0]:
        pytest.fail("the gap must take the median of the observed ages, 30")
    if out["age_was_missing"].tolist() != [0, 1, 0, 0] or out["age_was_missing"].dtype != np.int8:
        pytest.fail("the indicator must be an int8 column of 0 and 1 for the missing rows")


def test_mean_strategy_and_unknown_strategy():
    df = pd.DataFrame({"age": [20.0, np.nan, 40.0, 30.0]})
    out = _run(df, ["age"], strategy="mean")
    if out["age"].tolist() != [20.0, 30.0, 40.0, 30.0]:
        pytest.fail("the mean strategy must fill with the mean of the observed values")
    _expect_value_error(
        lambda: impute_with_indicators(df, ["age"], strategy="max"),
        "an unknown strategy was accepted: use 'median' or 'mean'",
    )


def test_frame_without_gaps_gets_no_indicator_and_keeps_dtypes():
    df = pd.DataFrame({"n": [1, 2, 3], "m": [0.5, 1.5, 2.5]})
    out = _run(df, ["n", "m"])
    if list(out.columns) != ["n", "m"]:
        pytest.fail("a frame without gaps must come back with the same columns and no indicators")
    if out["n"].dtype != df["n"].dtype:
        pytest.fail("a complete column must keep its original dtype")
