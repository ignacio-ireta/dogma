"""Hidden tests: a scikit-learn ColumnTransformer as the oracle on random mixed tables,
plus leakage, rule and validation cases. Each failure names the likely mistake first."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from submission import fit_transform_pipeline

NUMERIC = ["age", "income", "count", "const"]
CATEGORICAL = ["city", "plan"]
SETTINGS = settings(max_examples=40, deadline=None, report_multiple_bugs=False)


def _oracle(train, test):
    ct = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
    )
    fitted = ct.fit(train)
    return fitted.transform(train), fitted.transform(test)


def _frames(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 30))
    m = int(rng.integers(1, 15))

    def make(rows, shift):
        age = rng.normal(size=rows) * 3 + shift
        age[rng.random(rows) < 0.25] = np.nan
        income = rng.lognormal(size=rows)
        income[rng.random(rows) < 0.2] = np.nan
        count = rng.integers(0, 9, size=rows)
        city = rng.choice(["north", "south", "east"], size=rows).astype(object)
        city[rng.random(rows) < 0.2] = np.nan
        plan = np.where(rng.random(rows) < 0.5, "basic", "pro").astype(object)
        plan[rng.random(rows) < 0.1] = np.nan
        return pd.DataFrame(
            {
                "age": age,
                "income": income,
                "count": count,
                "city": city,
                "plan": plan,
                "const": np.full(rows, 7.0),
            }
        )

    train = make(n, 0.0)
    test = make(m, 4.0)
    test.loc[test.index[::3], "city"] = "west"  # an unseen category
    # Every listed column keeps one observed training value, so the fit is defined.
    train.loc[0, ["age", "income", "city", "plan"]] = [2.0, 1.0, "north", "basic"]
    return train, test


def _call(train, test, numeric, categorical):
    before_train, before_test = train.copy(), test.copy()
    try:
        result = fit_transform_pipeline(train, test, numeric, categorical)
    except Exception as exc:
        pytest.fail(
            f"valid preprocessing request crashed with {type(exc).__name__}: "
            "check missing-value fills, unseen categories and column names"
        )
    if not train.equals(before_train) or not test.equals(before_test):
        pytest.fail("the caller's table was modified: fill copies of the frames, not the frames")
    if not (isinstance(result, tuple) and len(result) == 2):
        pytest.fail("return the train output and the test output as a pair")
    for array in result:
        if not isinstance(array, np.ndarray) or array.dtype != np.float64 or array.ndim != 2:
            pytest.fail("both outputs must be 2-D float64 NumPy arrays")
    return result


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@SETTINGS
@given(st.integers(0, 10_000))
def test_matches_column_transformer_on_random_mixed_tables(seed):
    train, test = _frames(seed)
    want_train, want_test = _oracle(train, test)
    got_train, got_test = _call(train, test, NUMERIC, CATEGORICAL)
    if got_train.shape != want_train.shape or got_test.shape != want_test.shape:
        pytest.fail("output shape differs: check the numeric columns and the category blocks")
    if not np.allclose(got_train, want_train, rtol=1e-9, atol=1e-9):
        pytest.fail("train output differs from the oracle: fit imputers and scalers on train only")
    if not np.allclose(got_test, want_test, rtol=1e-9, atol=1e-9):
        pytest.fail(
            "test output differs from the oracle: use the training statistics and categories"
        )


@SETTINGS
@given(st.integers(0, 10_000))
def test_test_table_never_changes_the_fitted_training_output(seed):
    train, test = _frames(seed)
    shifted = test.copy()
    shifted["age"] = 1000.0
    shifted["city"] = "elsewhere"
    first, _ = _call(train, test, NUMERIC, CATEGORICAL)
    second, _ = _call(train, shifted, NUMERIC, CATEGORICAL)
    if not np.array_equal(first, second):
        pytest.fail("the test table changed the training output: the fit must use train rows only")


def test_missing_test_values_take_the_training_median_and_mode():
    train = pd.DataFrame({"age": [1.0, 2.0, 3.0, 100.0], "city": ["a", "a", "b", "a"]})
    test = pd.DataFrame({"age": [np.nan], "city": [np.nan]})
    _, out = _call(train, test, ["age"], ["city"])
    median = 2.5
    mean = np.mean([1.0, 2.0, 3.0, 100.0])
    scale = np.std([1.0, 2.0, 3.0, 100.0])
    if not np.allclose(out[0, 0], (median - mean) / scale):
        pytest.fail("a missing test numeric must take the training median before scaling")
    if out[0, 1:].tolist() != [1.0, 0.0]:
        pytest.fail("a missing test category must take the training mode")


def test_unseen_test_categories_give_all_zero_blocks():
    train = pd.DataFrame({"city": ["north", "south", "north"]})
    test = pd.DataFrame({"city": ["west", "south"]})
    _, out = _call(train, test, [], ["city"])
    if out.tolist() != [[0.0, 0.0], [0.0, 1.0]]:
        pytest.fail("an unseen category must give an all-zero block, not a new column")


def test_mode_tie_in_training_resolves_to_the_smallest_value():
    train = pd.DataFrame({"plan": ["zeta", "alpha", "zeta", "alpha", np.nan]})
    test = pd.DataFrame({"plan": [np.nan]})
    out_train, out_test = _call(train, test, [], ["plan"])
    if out_train[4].tolist() != [1.0, 0.0]:
        pytest.fail("the tied training mode must be the smallest training category")
    if out_test.tolist() != [[1.0, 0.0]]:
        pytest.fail("a missing test value must take the same tie-resolved training mode")


def test_column_order_is_numeric_then_each_categorical_block_in_given_order():
    train = pd.DataFrame({"x": [0.0, 2.0], "c1": ["b", "a"], "c2": ["q", "p"]})
    test = pd.DataFrame({"x": [1.0], "c1": ["a"], "c2": ["p"]})
    out_train, out_test = _call(train, test, ["x"], ["c2", "c1"])
    if out_train.shape != (2, 5):
        pytest.fail("shape mismatch: one numeric column plus two blocks of two categories")
    if out_train[0].tolist() != [-1.0, 0.0, 1.0, 0.0, 1.0]:
        pytest.fail(
            "column order differs: numeric first, then category blocks in the requested order"
        )
    if out_test[0].tolist() != [0.0, 1.0, 0.0, 1.0, 0.0]:
        pytest.fail("test row differs: check the block order and the training categories")


def test_constant_numeric_column_is_centered_with_unit_scale():
    train = pd.DataFrame({"k": [7.0, 7.0, 7.0]})
    test = pd.DataFrame({"k": [9.0, np.nan]})
    out_train, out_test = _call(train, test, ["k"], [])
    if out_train.tolist() != [[0.0], [0.0], [0.0]]:
        pytest.fail("a constant training column must be centered to zero")
    if out_test.tolist() != [[2.0], [0.0]]:
        pytest.fail("a constant training column must keep scale 1.0 for the test rows")


def test_constant_numeric_column_with_an_inexact_mean_keeps_unit_scale():
    train = pd.DataFrame({"k": [0.1, 0.1, 0.1]})
    test = pd.DataFrame({"k": [0.3, np.nan]})
    out_train, out_test = _call(train, test, ["k"], [])
    if np.max(np.abs(out_train)) > 1e-9:
        pytest.fail(
            "an exactly constant training column must be centred to zero, even when its float mean "
            "is inexact: test max and min for equality, not the spread"
        )
    np.testing.assert_allclose(
        out_test,
        [[0.2], [0.0]],
        atol=1e-9,
        err_msg="a constant column keeps scale 1.0 for held-out rows too",
    )


def test_empty_test_table_gives_zero_rows_with_the_full_width():
    train = pd.DataFrame({"age": [1.0, 3.0], "city": ["a", "b"]})
    test = pd.DataFrame({"age": pd.Series([], dtype=float), "city": pd.Series([], dtype=object)})
    out_train, out_test = _call(train, test, ["age"], ["city"])
    if out_test.shape != (0, out_train.shape[1]):
        pytest.fail("shape mismatch: an empty test table keeps the full column width")


@pytest.mark.parametrize(
    ("numeric", "categorical"),
    [(["age"], []), ([], ["city"])],
    ids=["numeric-all-missing", "categorical-all-missing"],
)
def test_a_column_with_no_training_value_raises_value_error(numeric, categorical):
    train = pd.DataFrame({"age": [np.nan, np.nan], "city": [np.nan, np.nan]})
    test = pd.DataFrame({"age": [1.0], "city": ["a"]})
    _expect_value_error(
        lambda: fit_transform_pipeline(train, test, numeric, categorical),
        "a column with no observed training value was accepted: raise ValueError",
    )


@pytest.mark.parametrize(
    ("numeric", "categorical"),
    [([], []), (["age"], ["age"]), (["age", "age"], [])],
    ids=["nothing-named", "named-twice-across-lists", "named-twice-in-one-list"],
)
def test_invalid_column_requests_raise_value_error(numeric, categorical):
    train = pd.DataFrame({"age": [1.0, 2.0]})
    test = pd.DataFrame({"age": [1.0]})
    _expect_value_error(
        lambda: fit_transform_pipeline(train, test, numeric, categorical),
        "an invalid column request was accepted: name each column once, and at least one",
    )


def test_inputs_are_not_mutated():
    train, test = _frames(3)
    before_train, before_test = train.copy(), test.copy()
    _call(train, test, NUMERIC, CATEGORICAL)
    if not train.equals(before_train):
        pytest.fail("the training table was modified: fill copies, not the caller's frame")
    if not test.equals(before_test):
        pytest.fail("the test table was modified: fill copies, not the caller's frame")


@pytest.mark.parametrize("magnitude", [1e-200, 1e200], ids=["small-scale", "large-scale"])
def test_numeric_scaling_keeps_representable_results_at_extreme_magnitudes(magnitude):
    train = pd.DataFrame({"x": np.array([0.0, 1.0, 2.0]) * magnitude})
    test = pd.DataFrame({"x": [3.0 * magnitude]})
    got_train, got_test = _call(train, test, ["x"], [])
    expected_train = np.array([[-1.0], [0.0], [1.0]]) / np.sqrt(2.0 / 3.0)
    expected_test = np.array([[2.0 / np.sqrt(2.0 / 3.0)]])
    if not np.allclose(got_train, expected_train, rtol=1e-9, atol=1e-9):
        pytest.fail("numeric training scale underflowed or overflowed: rescale before squaring")
    if not np.allclose(got_test, expected_test, rtol=1e-9, atol=1e-9):
        pytest.fail("held-out values need the finite nonzero training scale at extreme magnitudes")


def test_single_row_training_column_is_centred_with_unit_scale():
    train = pd.DataFrame({"k": [5.0]})
    test = pd.DataFrame({"k": [6.0]})
    out_train, out_test = _call(train, test, ["k"], [])
    if out_train.tolist() != [[0.0]]:
        pytest.fail("a training column with one observed value is constant: centre it to zero")
    if out_test.tolist() != [[1.0]]:
        pytest.fail(
            "held-out values must use the training mean and scale 1.0 for a constant column"
        )


def test_nearly_constant_training_column_is_standardized_to_finite_values():
    values = np.array([1.0 - 2.0**-52, 1.0, 1.0, 1.0, 1.0, 1.0])
    train = pd.DataFrame({"x": values})
    test = pd.DataFrame({"x": [1.0]})
    out_train, out_test = _call(train, test, ["x"], [])
    if not (np.all(np.isfinite(out_train)) and np.all(np.isfinite(out_test))):
        pytest.fail(
            "a non-constant training column gave non-finite standardized values: its scale is the "
            "population standard deviation, which is finite and nonzero here"
        )
    np.testing.assert_allclose(
        out_train[:, 0],
        (values - values.mean()) / values.std(),
        rtol=1e-6,
        atol=1e-6,
        err_msg="a non-constant training column must be standardized by its population std",
    )
