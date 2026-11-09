import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.decomposition import PCA
from submission import components_for_variance

from dogma_data import load, rng
from dogma_data.synth import correlated_mixed_scale

SEEDS = st.integers(min_value=0, max_value=10**6)
ROWS = st.integers(min_value=3, max_value=12)
COLS = st.integers(min_value=2, max_value=5)
THRESHOLDS = st.sampled_from([0.25, 0.5, 0.8, 0.9, 0.95, 1.0])
MARGIN = 1e-12  # the contract's rounding margin; mixed-scale data has ratios near 1e-10


def _data(seed, n, d):
    """Correlated columns on scales from 0.01 to 1000, as real measurements have."""
    return correlated_mixed_scale(rng(f"components-p02:data:{seed}:{n}:{d}"), n, d)


def _call(X, threshold, mistake):
    try:
        return components_for_variance(X, threshold)
    except Exception as exc:
        pytest.fail(
            f"{mistake}; raised {type(exc).__name__} instead: return the position of the first "
            "cumulative ratio that reaches the threshold, counted from one"
        )


def _check(ok, mistake):
    if not ok:
        pytest.fail(mistake)


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def _independent_ratios(X):
    """Explained-variance ratios from the eigenvalues of the sample covariance (no SVD involved)."""
    cov = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return eigenvalues / eigenvalues.sum()


def _expected_count(X, threshold):
    cumulative = np.cumsum(_independent_ratios(X))
    return int(np.flatnonzero(cumulative >= threshold - MARGIN)[0]) + 1


@pytest.mark.parametrize(
    ("seed", "n", "d", "threshold"),
    [(1, 30, 4, 0.3), (2, 25, 6, 0.6), (3, 40, 3, 0.85), (4, 12, 5, 0.95)],
    ids=["low-threshold", "middle", "high", "very-high"],
)
def test_matches_scikit_learn_threshold_rule_on_random_data(seed, n, d, threshold):
    X = _data(seed, n, d)
    reference = PCA(n_components=threshold, svd_solver="full").fit(X).n_components_
    _check(
        _call(X, threshold, "a valid random matrix was rejected or failed") == reference,
        "the component count differs from scikit-learn's threshold rule: check the cumulative "
        "ratios, their order and the divisor used in the ratios",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS, THRESHOLDS)
def test_returns_the_smallest_count_that_reaches_the_threshold(seed, n, d, threshold):
    X = _data(seed, n, d)
    result = _call(X, threshold, "a valid random matrix was rejected or failed")
    _check(
        isinstance(result, (int, np.integer)) and not isinstance(result, bool),
        "the result must be an integer count of components",
    )
    _check(
        int(result) == _expected_count(X, threshold),
        "the count is not the smallest k whose cumulative ratio reaches the threshold: check the "
        "cumulative sum and the position you return",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    SEEDS,
    ROWS,
    COLS,
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
)
def test_count_never_decreases_as_the_threshold_rises(seed, n, d, first, second):
    X = _data(seed, n, d)
    low, high = sorted([first, second])
    count_low = _call(X, low, "a valid random matrix was rejected or failed")
    count_high = _call(X, high, "a valid random matrix was rejected or failed")
    _check(
        count_low <= count_high,
        "a higher threshold needs at least as many components: the cumulative sum must be "
        "compared with the threshold, not inverted",
    )


def test_threshold_of_one_on_rank_two_data_returns_the_rank():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 4))
    _check(
        _call(X, 1.0, "rank-two data was rejected or failed") == 2,
        (
            "a rank-deficient matrix must reach full explained variance without ret"
            "aining null components: allow the stated rounding margin"
        ),
    )


def test_exact_tie_on_a_symmetric_case_counts_as_reached():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    _check(
        _call(X, 0.5, "the symmetric case was rejected or failed") == 1,
        "a cumulative ratio exactly equal to the threshold must count as reaching it",
    )


def test_threshold_just_above_the_tie_needs_the_second_component():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    _check(
        _call(X, 0.5 + 1e-6, "the symmetric case was rejected or failed") == 2,
        "a threshold just above the first cumulative ratio needs the second component",
    )


def test_worked_example_tie_at_three_quarters_counts_as_reached():
    X = np.array([[-3.0, -1.0], [0.0, -2.0], [0.0, 3.0]])
    _check(
        _call(X, 0.75, "the worked example was rejected or failed") == 1,
        (
            "the rounding margin must absorb last-digit error when the cumulative v"
            "ariance exactly reaches the threshold"
        ),
    )


def test_large_common_offset_does_not_change_the_count():
    X = 100.0 + np.random.default_rng(3).normal(size=(50, 3))
    _check(
        _call(X, 0.9, "the offset matrix was rejected or failed") == 3,
        "adding a common offset to every column must not change the variance structure: centre "
        "the columns before the SVD",
    )


@pytest.mark.parametrize(
    "threshold",
    [0.0, -0.2, 1.5, float("nan")],
    ids=["zero", "negative", "above-one", "nan"],
)
def test_threshold_outside_zero_to_one_raises_value_error(threshold):
    _expect_value_error(
        lambda: components_for_variance(_data(0, 6, 3), threshold),
        "a threshold must be a real number in (0, 1]: raise ValueError for other values",
    )


def test_boolean_threshold_raises_value_error():
    _expect_value_error(
        lambda: components_for_variance(_data(0, 6, 3), True),
        "a boolean is not a threshold: raise ValueError",
    )


@pytest.mark.parametrize(
    "X",
    [np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0]]), np.zeros((0, 3)), np.zeros((4, 0))],
    ids=["one-dimensional", "one-row", "no-rows", "no-columns"],
)
def test_malformed_matrix_raises_value_error(X):
    _expect_value_error(
        lambda: components_for_variance(X, 0.5),
        "a matrix must be 2-D with at least two rows and one column: raise ValueError",
    )


def test_nan_entry_raises_value_error():
    _expect_value_error(
        lambda: components_for_variance([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]], 0.5),
        "a NaN entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_constant_data_raises_value_error_because_no_ratio_exists():
    _expect_value_error(
        lambda: components_for_variance(np.full((4, 3), 2.0), 0.5),
        "constant data has zero total variance, so no ratio exists: raise ValueError",
    )


def test_integer_input_gives_the_same_count_as_float_input():
    X_int = np.array([[1, 0, 2], [3, 1, 2], [5, 4, 9], [2, 2, 1], [0, 3, 4]])
    expected = _expected_count(X_int.astype(np.float64), 0.8)
    _check(
        int(_call(X_int, 0.8, "integer input was rejected or failed")) == expected,
        "integer input must give the same count as its float64 version",
    )


def test_inputs_are_not_mutated():
    X = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 4.0]])
    X_before = X.copy()
    _call(X, 0.5, "a valid matrix was rejected or failed")
    _check(
        bool(np.array_equal(X, X_before)),
        "the input matrix was modified in place: centre a copy, not X",
    )


def test_two_observations_are_a_valid_input():
    _check(
        _call([[1.0, 2.0], [3.0, 5.0]], 0.5, "two observations are valid input") == 1,
        "two observations give one component that carries all the variance: n >= 2 is valid",
    )


def test_single_variable_is_a_valid_input():
    _check(
        _call([[1.0], [2.0], [4.0]], 1.0, "a single variable is valid input") == 1,
        "one variable is one component with ratio 1: d >= 1 is valid",
    )


def test_cumulative_ratio_more_than_the_rounding_margin_below_does_not_reach_the_threshold():
    # The first ratio is about 0.5 + 1.0e-12, so the threshold 0.5 + 2.5e-12 is 1.5e-12 above it.
    # That is outside the 1e-12 rounding margin, so one component is not enough.
    c = np.sqrt(1.0 - 4e-12)
    X = [[1.0, 0.0], [-1.0, 0.0], [0.0, c], [0.0, -c]]
    _check(
        _call(X, 0.5 + 2.5e-12, "the near-tie case was rejected or failed") == 2,
        "a cumulative ratio 1.5e-12 below the threshold is outside the 1e-12 rounding margin: "
        "it must not count as reaching the threshold",
    )


@pytest.fixture(scope="module")
def wine_tiny():
    """Raw wine scaled down by an exact power of two, 2**-40: its total variance is about 1e-19."""
    return load("wine").X * 2.0**-40


def test_tiny_scale_wine_is_not_constant_and_needs_the_real_count(wine_tiny):
    # Proline carries about 99.8% of the variance at any scale, so one component reaches 0.95.
    reference = PCA(n_components=0.95, svd_solver="full").fit(wine_tiny).n_components_
    try:
        count = components_for_variance(wine_tiny, 0.95)
    except ValueError:
        pytest.fail(
            "tiny-scale data was treated as constant: a tiny positive total variance is not "
            "constant, so test the total for exact zero, not against an absolute tolerance",
            pytrace=False,
        )
    _check(
        count == reference,
        "tiny-scale data gave the wrong count: a tiny positive total variance is not constant, so "
        "the constant-column check must be an exact zero test, not an absolute tolerance",
    )
