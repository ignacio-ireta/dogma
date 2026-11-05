import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import expanding_target_mean


def _encode(categories, target):
    """Call the learner's function and return a float64 array, naming the mistake if it crashes."""
    try:
        out = expanding_target_mean(categories, target)
    except ValueError:
        pytest.fail("a valid input raised ValueError: check the length and finiteness rules")
    except Exception as exc:
        pytest.fail(f"a valid input raised {type(exc).__name__}: check the running state")
    out = np.asarray(out)
    if out.dtype != np.float64:
        pytest.fail("the encoding must be a float64 array, one entry per row")
    return out


def _require(condition, mistake):
    if not condition:
        pytest.fail(mistake)


def _brute_force(categories, target):
    """Independent reference: for each row, scan the earlier rows of its category."""
    target = np.asarray(target, dtype=np.float64)
    out = np.full(len(categories), np.nan)
    for i, label in enumerate(categories):
        earlier = [target[j] for j in range(i) if categories[j] == label]
        if earlier:
            out[i] = sum(earlier) / len(earlier)
    return out


def _pandas_reference(categories, target):
    """The groupby, shift and expanding idiom: earlier rows of the same category only."""
    df = pd.DataFrame({"c": list(categories), "y": np.asarray(target, dtype=np.float64)})
    shifted = df.groupby("c")["y"].shift(1)
    return shifted.groupby(df["c"]).transform(lambda s: s.expanding().mean()).to_numpy()


def _assert_encoding(got, expected, mistake):
    _require(got.shape == expected.shape, "the encoding must have one entry per row")
    _require(
        bool(np.array_equal(np.isnan(got), np.isnan(expected))),
        mistake + ": NaN only where no earlier row shares the category",
    )
    finite = ~np.isnan(expected)
    _require(bool(np.allclose(got[finite], expected[finite], rtol=1e-9, atol=1e-12)), mistake)


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=30),
    st.integers(min_value=0, max_value=10_000),
    st.data(),
)
def test_a_rows_encoding_ignores_its_own_and_every_later_target(cats, seed, data):
    n = len(cats)
    i = data.draw(st.integers(min_value=0, max_value=n - 1), label="row")
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    changed = y.copy()
    changed[i:] = 100.0 * rng.normal(size=n - i)
    before = _encode(cats, y)
    after = _encode(cats, changed)
    # Tolerance, not exact equality: summation order may shift the last bits, but real leakage
    # moves the encoding by an order of the target's size.
    _require(
        bool(np.allclose(before[: i + 1], after[: i + 1], rtol=1e-9, atol=1e-12, equal_nan=True)),
        "leakage: a row's encoding changed when its own or a later target changed",
    )


def test_first_occurrence_is_nan_and_second_equals_the_first_target():
    cats = ["p", "q", "p", "r", "q", "p"]
    y = np.array([0.5, -1.0, 2.5, 4.0, 7.0, 9.0])
    got = _encode(cats, y)
    _require(
        bool(np.isnan(got[0]) and np.isnan(got[1]) and np.isnan(got[3])),
        "the first row of each category must be NaN: it has no earlier row of its own category",
    )
    _require(
        got[2] == 0.5,
        "the second row of a category must equal its first target exactly: keep every earlier row"
        " of that category in the running state",
    )
    _require(
        got[4] == -1.0,
        "the second row of a category must equal its first target exactly: keep every earlier row"
        " of that category in the running state",
    )


def test_alternating_categories_keep_their_running_state_across_the_series():
    cats = ["x", "y"] * 12
    y = np.arange(24, dtype=np.float64) ** 1.5
    expected = _pandas_reference(cats, y)
    got = _encode(cats, y)
    seen = set()
    for i, label in enumerate(cats):
        if label in seen:
            _require(
                not np.isnan(got[i]),
                "a row whose category appeared earlier got no encoding: keep the state for the"
                " whole series, not only within runs of adjacent rows",
            )
        seen.add(label)
    _assert_encoding(
        got,
        expected,
        "an encoding differs from the expanding mean of earlier same-category rows",
    )


@pytest.mark.parametrize("seed", [0, 1, 2], ids=["seed-zero", "seed-one", "seed-two"])
def test_integer_categories_match_the_pandas_expanding_idiom(seed):
    rng = np.random.default_rng(seed)
    cats = rng.integers(0, 4, size=40).tolist()
    y = rng.normal(size=40)
    expected = _pandas_reference(cats, y)
    _assert_encoding(
        _encode(cats, y),
        expected,
        "an integer-category encoding differs from the expanding mean of earlier same-category"
        " rows",
    )


def test_string_categories_match_the_pandas_expanding_idiom():
    rng = np.random.default_rng(3)
    cats = [str(v) for v in rng.choice(["red", "green", "blue", "teal"], size=60)]
    y = rng.normal(loc=2.0, size=60)
    expected = _pandas_reference(cats, y)
    _assert_encoding(
        _encode(cats, y),
        expected,
        "a string-category encoding differs from the expanding mean: key the state by label",
    )


def test_matches_brute_force_reference_on_mixed_labels():
    rng = np.random.default_rng(4)
    cats = [int(v) if v % 2 else f"s{v}" for v in rng.integers(0, 5, size=30)]
    y = rng.normal(size=30)
    expected = _brute_force(cats, y)
    _assert_encoding(
        _encode(cats, y),
        expected,
        "mixed int and string labels must be treated as distinct categories",
    )


def test_integer_and_string_labels_that_print_alike_stay_distinct():
    # The integer 1 and the string "1" are different categories, so row 3 averages only row 1.
    expected = np.array([np.nan, np.nan, 1.0, 2.0])
    got = _encode([1, "1", 1, "1"], [1.0, 2.0, 3.0, 4.0])
    _assert_encoding(
        got,
        expected,
        "the integer 1 and the string '1' are different categories: do not convert labels to text",
    )


def test_large_magnitude_targets_match_the_reference():
    rng = np.random.default_rng(5)
    cats = rng.integers(0, 3, size=50).tolist()
    y = 1e12 + rng.normal(size=50)
    expected = _brute_force(cats, y)
    _assert_encoding(
        _encode(cats, y),
        expected,
        "large-magnitude targets: the encoding is off beyond tolerance",
    )


def test_small_positive_magnitude_targets_match_the_reference():
    rng = np.random.default_rng(6)
    cats = rng.integers(0, 3, size=50).tolist()
    y = 1e-9 * rng.uniform(1.0, 2.0, size=50)
    expected = _brute_force(cats, y)
    got = _encode(cats, y)
    finite = ~np.isnan(expected)
    _require(
        bool(np.array_equal(np.isnan(got), np.isnan(expected))),
        "small-magnitude targets: NaN only where no earlier row shares the category",
    )
    # Relative tolerance only: an absolute epsilon of 1e-12 would hide errors at this scale.
    _require(
        bool(np.allclose(got[finite], expected[finite], rtol=1e-9, atol=0.0)),
        "small-magnitude targets: the encoding is off beyond relative tolerance; do not add an"
        " absolute constant to the mean",
    )


def test_empty_input_returns_an_empty_array():
    got = _encode([], [])
    _require(got.shape == (0,), "empty input must return an empty array, not an error")


@pytest.mark.parametrize(
    ("categories", "target"),
    [
        (["a", "a"], [1.0, float("nan")]),
        (["a", "b"], [float("inf"), 2.0]),
        (["a", "b"], [[1.0], [2.0]]),
    ],
    ids=["nan-target", "inf-target", "two-dimensional-target"],
)
def test_invalid_target_raises_value_error(categories, target):
    try:
        expanding_target_mean(categories, target)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid target raised {type(exc).__name__} instead of ValueError")
    pytest.fail("a non-finite or non-1-D target must raise ValueError")


def test_inputs_are_not_mutated():
    cats = ["a", "b", "a", "c"]
    # Zero and negative targets, so an in-place clip or fill of the caller's array is visible.
    y = np.array([1.0, -2.0, 0.0, 4.0])
    cats_before, y_before = list(cats), y.copy()
    expanding_target_mean(cats, y)
    _require(cats == cats_before, "the categories were modified in place")
    _require(bool(np.array_equal(y, y_before)), "the target array was modified in place")


@pytest.mark.parametrize(
    "categories",
    [np.array([["a"], ["a"]]), [["a"], ["a"]]],
    ids=["array-rank", "nested-list-rank"],
)
def test_two_dimensional_categories_raise_value_error(categories):
    try:
        expanding_target_mean(categories, [1.0, 2.0])
    except ValueError:
        return
    except Exception:
        pytest.fail("non-1-D categories must raise ValueError rather than fail as dictionary keys")
    pytest.fail("non-1-D categories were accepted: validate the label rank")


def test_finite_repeated_large_targets_do_not_overflow_the_historical_mean():
    got = _encode(["x", "x", "x"], [1e308, 1e308, 1e308])
    _assert_encoding(
        got,
        np.array([np.nan, 1e308, 1e308]),
        "historical means must stay finite when every earlier target has the same finite value",
    )


def test_opposite_sign_huge_targets_average_without_overflow():
    # Two large targets of opposite sign: their mean is near zero and must stay finite.
    got = _encode(["x", "x", "x"], [-1.5e308, 1.5e308, 0.0])
    _require(
        bool(np.isfinite(got[2]) and abs(got[2]) <= 1e-9 * 1.5e308),
        "the mean of large targets with opposite signs must stay finite and near zero: do not"
        " form a difference between a running mean and a target that can overflow",
    )
