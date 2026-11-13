"""Hidden tests: a scikit-learn ColumnTransformer as the oracle on random mixed tables,
plus leakage, rule and validation cases. Each failure names the likely mistake first."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from submission import fit_transform_pipeline

from dogma_data import load_frame, sample_rows
from dogma_data import rng as data_rng

NUMERIC = ["age", "income", "count", "const"]
CATEGORICAL = ["city", "plan"]
SETTINGS = settings(max_examples=40, deadline=None, report_multiple_bugs=False)


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


def _real_unit(gen, size=None):
    """Real numbers between 0 and 1 where this test used to draw uniform ones.

    Each is the share of a standard normal that lies below a standardised breast_cancer
    measurement, so the values keep the order, the ties and the skew of real measurements.
    """
    return ndtr(_real_normal(gen, size))


def _real_ints(gen, low, high, size=None):
    """Real whole numbers from ``low`` up to ``high - 1``: the band a real measurement falls in.

    The range is cut into ``high - low`` bands of equal normal probability, and each value is
    the band of one standardised measurement. Real measurements are skewed, so the bands are
    not equally full, and neighbouring values tie as real categories and grades do.
    """
    bands = np.floor(_real_unit(gen, size) * (high - low))
    return (low + np.minimum(bands, high - low - 1)).astype(np.int64)


def _real_shares(gen, alpha, size=None):
    """Real probability vectors: positive real numbers (see `_real_unit`) divided by their sum."""
    k = int(np.size(alpha))
    shape = (k,) if size is None else (int(size), k)
    weights = _real_unit(gen, shape) + 1e-3
    return weights / weights.sum(axis=-1, keepdims=True)


@functools.cache
def _adult():
    return load_frame("adult")


def _oracle(train, test, numeric=NUMERIC, categorical=CATEGORICAL):
    ct = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    fitted = ct.fit(train)
    return fitted.transform(train), fitted.transform(test)


def _frames(seed):
    """Train and test frames of real adult census rows, with gaps placed by the draw.

    age: age in years. income: the census weight of the row in thousands, a right-skewed
    amount (the table holds no income in dollars). count: capital gain in dollars, an integer
    column that is zero for nine people in ten. city: the
    workclass, nine levels from very common to almost unseen, "?" among them. plan: the sex
    column. The values are real; about a fifth of each listed column is removed at random,
    because the pipeline is about gaps and the source marks them with "?" instead.
    """
    rng = data_rng(f"numeric-categorical-pipeline-p01:frames:{seed}")
    n = int(rng.integers(4, 30))
    m = int(rng.integers(1, 15))
    adult = _adult()

    def make(rows, shift):
        picked = adult.iloc[rng.choice(len(adult), size=rows, replace=False)]
        age = picked["age"].to_numpy(dtype=np.float64) + shift
        age[_real_unit(rng, rows) < 0.25] = np.nan
        income = picked["fnlwgt"].to_numpy(dtype=np.float64) / 1000.0
        income[_real_unit(rng, rows) < 0.2] = np.nan
        count = picked["capital.gain"].to_numpy(dtype=np.int64)
        city = picked["workclass"].to_numpy(dtype=object).copy()
        city[_real_unit(rng, rows) < 0.2] = np.nan
        plan = picked["sex"].to_numpy(dtype=object).copy()
        plan[_real_unit(rng, rows) < 0.1] = np.nan
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
    train.loc[0, ["age", "income", "city", "plan"]] = [39.0, 180.0, "Private", "Male"]
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


ADULT_NUMERIC = ["age", "fnlwgt", "hours.per.week", "capital.gain"]
ADULT_CATEGORICAL = ["workclass", "occupation", "native.country"]


@pytest.fixture(scope="module")
def adult_split():
    # Real census rows. "?" marks a missing answer in this source, so it becomes NaN here.
    frame = load_frame("adult")
    for column in ADULT_CATEGORICAL:
        frame[column] = frame[column].replace("?", np.nan)
    return frame.iloc[:4000].reset_index(drop=True), frame.iloc[4000:6000].reset_index(drop=True)


def test_matches_column_transformer_on_real_adult_with_rare_and_unseen_countries(adult_split):
    # native.country has about 40 levels, several seen once or twice in train, and test has a
    # country that train never saw. capital.gain is constant (zero) in this slice.
    train, test = adult_split
    want_train, want_test = _oracle(train, test, ADULT_NUMERIC, ADULT_CATEGORICAL)
    got_train, got_test = _call(train, test, ADULT_NUMERIC, ADULT_CATEGORICAL)
    if got_train.shape != want_train.shape or got_test.shape != want_test.shape:
        pytest.fail("output shape differs on real census columns: one block per training level")
    if not np.allclose(got_train, want_train, rtol=1e-9, atol=1e-9):
        pytest.fail(
            "train output differs on real census columns: keep every training level, rare or not"
        )
    if not np.allclose(got_test, want_test, rtol=1e-9, atol=1e-9):
        pytest.fail(
            "test output differs on real census columns: unseen levels are all zero, and missing"
            " test answers take the training mode"
        )


def test_empty_string_is_an_ordinary_category_not_a_missing_value():
    # Missing means NaN only. An empty string is a text value, so it is its own category,
    # and sorted distinct categories put "" before "north" and "south".
    train = pd.DataFrame({"city": ["north", "", "north", "south"]})
    test = pd.DataFrame({"city": ["", "north"]})
    out_train, out_test = _call(train, test, [], ["city"])
    if out_train.tolist() != [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]:
        pytest.fail("an empty string is a category of its own, not a missing value to be filled")
    if out_test.tolist() != [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]:
        pytest.fail("a test empty string must match its own training category, not the mode")


def test_frames_with_a_shuffled_index_give_the_same_arrays():
    train, test = _frames(3)
    want_train, want_test = _call(train, test, NUMERIC, CATEGORICAL)
    moved_train, moved_test = train.copy(), test.copy()
    moved_train.index = pd.Index(range(1000 + len(train), 1000, -1))
    moved_test.index = pd.Index(range(500 + len(test), 500, -1))
    got_train, got_test = _call(moved_train, moved_test, NUMERIC, CATEGORICAL)
    same = np.array_equal(got_train, want_train) and np.array_equal(got_test, want_test)
    assert same, (
        "the arrays change with the index labels of the frames, as after a shuffled split: "
        "assign by position, or reset the index of a copy first"
    )
