import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.decomposition import PCA
from submission import pca

from dogma_data import load, rng
from dogma_data.synth import correlated_mixed_scale

SEEDS = st.integers(min_value=0, max_value=10**6)
ROWS = st.integers(min_value=3, max_value=12)
COLS = st.integers(min_value=2, max_value=5)


def _data(seed, n, d):
    """Correlated columns on scales from 0.03 to 30, as real measurements have.

    Three decades of scale, not five: an eigh-based solution squares the condition number and
    cannot meet the 1e-9 comparisons on data spanning 1e10 in variance (measured: 7 of 12 draws
    fail), while the SVD route and every route the Constraints allow pass on this range.
    """
    return correlated_mixed_scale(
        rng(f"pca-p01:data:{seed}:{n}:{d}"), n, d, scale_range=(0.03, 30.0)
    )


def _call(X, k, mistake):
    try:
        return pca(X, k)
    except Exception as exc:
        pytest.fail(
            f"{mistake}; raised {type(exc).__name__} instead: check that components is (k, d) "
            "and transformed is (n, k), built from the SVD's right singular vectors"
        )


def _field(result, key):
    if not isinstance(result, dict) or key not in result:
        pytest.fail(f"the result has no {key!r} entry: return a dict with exactly the four keys")
    try:
        return np.asarray(result[key], dtype=np.float64)
    except (TypeError, ValueError):
        pytest.fail(f"the {key!r} entry is not a numeric array")


def _check(ok, mistake):
    if not ok:
        pytest.fail(mistake)


def _check_close(actual, expected, mistake):
    _check(
        actual.shape == np.shape(expected)
        and bool(np.allclose(actual, expected, rtol=1e-9, atol=1e-9)),
        mistake,
    )


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def _determined_axes(n, d, k):
    """Axes with nonzero variance are determined by the data; the others are any completion."""
    return min(k, n - 1, d)


@pytest.mark.parametrize(
    ("seed", "n", "d", "k"),
    [(1, 8, 3, 2), (2, 4, 6, 3), (3, 7, 6, 6), (4, 10, 4, 1)],
    ids=["tall-two-of-three", "wide-three-of-four", "full-rank-all", "single-axis"],
)
def test_matches_scikit_learn_full_svd_pca_including_signs(seed, n, d, k):
    X = _data(seed, n, d)
    result = _call(X, k, "pca failed on a valid random matrix")
    ref = PCA(n_components=k, svd_solver="full").fit(X)
    m = _determined_axes(n, d, k)
    _check_close(
        _field(result, "components")[:m],
        ref.components_[:m],
        "components do not match the principal axes with the sign rule: check the right singular "
        "vectors, their order and the sign of each row",
    )
    _check_close(
        _field(result, "explained_variance"),
        ref.explained_variance_,
        "explained variances differ from the sample variances (divisor n - 1) of the projections",
    )
    _check_close(
        _field(result, "explained_variance_ratio"),
        ref.explained_variance_ratio_,
        "explained-variance ratios differ: divide by the total variance of all columns",
    )
    _check_close(
        _field(result, "transformed"),
        ref.transform(X),
        "transformed scores differ from the centred data projected onto the axes",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS, st.data())
def test_axes_are_orthonormal_and_ordered_by_explained_variance(seed, n, d, data):
    X = _data(seed, n, d)
    k = data.draw(st.integers(min_value=1, max_value=min(n, d)))
    result = _call(X, k, "pca failed on a valid random matrix")
    components = _field(result, "components")
    _check(
        bool(np.allclose(components @ components.T, np.eye(k), atol=1e-9)),
        "the components are not orthonormal unit rows: they must be rows of the SVD's Vt",
    )
    variance = _field(result, "explained_variance")
    _check(
        bool(np.all(np.diff(variance) <= 1e-12 * max(1.0, variance[0]))),
        "components are not ordered by decreasing explained variance: keep the SVD order",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS, st.data())
def test_largest_entry_of_every_axis_is_positive(seed, n, d, data):
    X = _data(seed, n, d)
    k = data.draw(st.integers(min_value=1, max_value=min(n, d)))
    components = _field(_call(X, k, "pca failed on a valid random matrix"), "components")
    for row in components:
        _check(
            row[np.argmax(np.abs(row))] > 0,
            "a component's largest-magnitude entry is negative: apply the sign rule to every row",
        )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS, st.data())
def test_explained_variance_is_sample_variance_of_the_projections(seed, n, d, data):
    X = _data(seed, n, d)
    k = data.draw(st.integers(min_value=1, max_value=min(n, d)))
    result = _call(X, k, "pca failed on a valid random matrix")
    scores = _field(result, "transformed")
    _check_close(
        np.var(scores, axis=0, ddof=1),
        _field(result, "explained_variance"),
        "explained variances are not the sample variances (divisor n - 1) of the scores: "
        "check the squared singular values and the divisor",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS, st.data())
def test_transformed_scores_are_centred_data_times_components(seed, n, d, data):
    X = _data(seed, n, d)
    k = data.draw(st.integers(min_value=1, max_value=min(n, d)))
    result = _call(X, k, "pca failed on a valid random matrix")
    _check_close(
        _field(result, "transformed"),
        (X - X.mean(axis=0)) @ _field(result, "components").T,
        "transformed scores are not the centred data times the components: centre the columns "
        "and use the same signed axes as the components",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS, st.data())
def test_adding_a_constant_to_each_column_changes_nothing(seed, n, d, data):
    X = _data(seed, n, d)
    k = data.draw(st.integers(min_value=1, max_value=min(n, d)))
    offset = np.random.default_rng(seed + 1).normal(scale=50.0, size=d)
    base = _call(X, k, "pca failed on a valid random matrix")
    shifted = _call(X + offset, k, "pca failed on a valid shifted matrix")
    m = _determined_axes(n, d, k)
    for key in ("explained_variance", "explained_variance_ratio", "transformed"):
        _check_close(
            _field(shifted, key),
            _field(base, key),
            "adding a constant to the columns changed the output: "
            "centre each column before the SVD",
        )
    _check_close(
        _field(shifted, "components")[:m],
        _field(base, "components")[:m],
        "adding a constant to the columns changed the axes: centre each column before the SVD",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, COLS, st.data())
def test_ratios_divide_by_the_total_variance_of_all_columns(seed, n, d, data):
    X = _data(seed, n, d)
    k = data.draw(st.integers(min_value=1, max_value=min(n, d)))
    result = _call(X, k, "pca failed on a valid random matrix")
    total = np.var(X, axis=0, ddof=1).sum()
    _check_close(
        _field(result, "explained_variance_ratio"),
        _field(result, "explained_variance") / total,
        "explained-variance ratios use the wrong total: divide by the sum of all column variances",
    )


@pytest.mark.parametrize(("n", "d"), [(7, 3), (5, 5)], ids=["tall", "square"])
def test_keeping_every_axis_preserves_distances_from_the_mean(n, d):
    X = _data(11 + n, n, d)
    result = _call(X, d, "pca failed when keeping every axis")
    scores = _field(result, "transformed")
    centred = X - X.mean(axis=0)
    _check_close(
        np.linalg.norm(scores, axis=1),
        np.linalg.norm(centred, axis=1),
        "keeping every axis must preserve each observation's distance from the mean: the axes "
        "must form an orthonormal basis",
    )


def test_rank_one_data_puts_all_variance_on_the_first_axis():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, 1)) @ rng.normal(size=(1, 3))
    result = _call(X, 2, "pca failed on rank-one data")
    variance = _field(result, "explained_variance")
    ratio = _field(result, "explained_variance_ratio")
    _check(
        bool(variance[1] <= 1e-12 * max(1.0, variance[0])),
        "rank-one data must give the second axis zero variance: check the singular values and "
        "their order",
    )
    _check_close(ratio[:1], [1.0], "rank-one data must give the first axis ratio 1")


def test_constant_data_gives_zero_variances_ratios_and_scores():
    result = _call(np.full((4, 3), 2.0), 2, "pca failed on constant data")
    for key in ("explained_variance", "explained_variance_ratio", "transformed"):
        values = _field(result, key)
        _check(
            bool(np.all(values == 0.0)),
            "constant data has no variance: the ratios must be exactly 0.0, not NaN or rounding "
            "noise, and the scores must be zero",
        )
    components = _field(result, "components")
    _check(
        bool(np.allclose(components @ components.T, np.eye(2), atol=1e-9)),
        "constant data still needs orthonormal component rows",
    )


def test_integer_input_returns_float64_arrays_with_documented_shapes():
    X = np.array([[1, 0, 2], [3, 1, 2], [5, 4, 9], [2, 2, 1], [0, 3, 4]])
    result = _call(X, 2, "pca failed on integer input")
    expected_shapes = {
        "components": (2, 3),
        "explained_variance": (2,),
        "explained_variance_ratio": (2,),
        "transformed": (5, 2),
    }
    for key, shape in expected_shapes.items():
        value = result[key] if isinstance(result, dict) and key in result else None
        _check(
            value is not None and np.asarray(value).shape == shape,
            f"shape mismatch in {key!r}: check the axes count, the variable count and the "
            "observation count",
        )
        _check(
            np.asarray(value).dtype == np.float64,
            f"integer input must give float64 values in {key!r}, not an integer dtype",
        )


@pytest.mark.parametrize(
    ("X", "k"),
    [
        (np.array([1.0, 2.0, 3.0]), 1),
        (np.array([[1.0, 2.0]]), 1),
        (np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]), 1),
        (np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 5.0]]), 1),
        (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]), 0),
        (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]), 3),
        (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]), True),
        (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]), 1.0),
    ],
    ids=[
        "one-dimensional",
        "one-row",
        "nan",
        "infinite",
        "zero-components",
        "too-many-components",
        "boolean-components",
        "float-components",
    ],
)
def test_invalid_input_or_component_count_raises_value_error(X, k):
    _expect_value_error(
        lambda: pca(X, k),
        "invalid input must raise ValueError: check dimensions, row count, finiteness and "
        "that n_components is an integer in range",
    )


def test_inputs_are_not_mutated():
    X = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 4.0]])
    X_before = X.copy()
    _call(X, 1, "pca failed on a valid matrix")
    _check(
        bool(np.array_equal(X, X_before)),
        "the input matrix was modified in place: centre a copy, not X",
    )


def test_two_observations_are_a_valid_fit():
    result = _call([[1.0, 2.0], [3.0, 5.0]], 1, "two observations are valid input for PCA")
    components = _field(result, "components")
    _check(
        components.shape == (1, 2) and bool(np.isclose(np.linalg.norm(components), 1.0, atol=1e-9)),
        "two observations must give one unit-length principal axis: n >= 2 is valid",
    )


def test_single_variable_is_a_valid_fit_that_explains_everything():
    result = _call([[1.0], [2.0], [4.0]], 1, "a single variable is valid input for PCA")
    components = _field(result, "components")
    ratio = _field(result, "explained_variance_ratio")
    _check(
        components.shape == (1, 1) and bool(np.allclose(components, [[1.0]], atol=1e-9)),
        "one variable has one axis, [[1.0]] with a positive sign: d >= 1 is valid",
    )
    _check(
        bool(np.allclose(ratio, [1.0], atol=1e-9)),
        "one variable explains all of its own variance, so the ratio must be 1",
    )


def test_entry_outside_the_tie_band_decides_the_sign_by_magnitude():
    # The two entries of the axis differ by a relative 1.5e-9, which is outside the 1e-9 tie band.
    # The largest entry (index 1) is therefore the only one that counts, and it must be positive.
    s = 1.0 - 1.5e-9
    components = _field(
        _call([[s, -1.0], [-s, 1.0]], 1, "pca failed on rank-one data"), "components"
    )
    _check(
        components[0, 1] > 0,
        "the largest-magnitude entry lies outside the tie band of the other entry, so it alone "
        "sets the sign and must be positive",
    )


@pytest.fixture(scope="module")
def wine_tiny():
    """Raw wine (proline near 1000, hue near 0.7) scaled down by an exact power of two, 2**-40."""
    return load("wine").X * 2.0**-40


def _relative_close(actual, expected, mistake):
    """Relative agreement only: an absolute tolerance would hide every value at this scale."""
    try:
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0.0, err_msg=mistake)
    except AssertionError:
        pytest.fail(mistake, pytrace=False)


def test_tiny_scale_wine_keeps_its_real_ratios_and_is_not_treated_as_constant(wine_tiny):
    # The total variance of this data is about 1e-19, far below any absolute epsilon. Its ratios
    # are the ordinary ones, so they must agree with scikit-learn on the same numbers.
    result = _call(wine_tiny, 3, "pca failed on a valid tiny-scale matrix")
    ref = PCA(n_components=3, svd_solver="full").fit(wine_tiny)
    _relative_close(
        _field(result, "explained_variance_ratio"),
        ref.explained_variance_ratio_,
        "tiny-scale data gave the wrong ratios: a tiny positive total variance is not constant, "
        "so test the total for exact zero, not against an absolute tolerance",
    )
    _relative_close(
        _field(result, "explained_variance"),
        ref.explained_variance_,
        "tiny-scale data gave the wrong explained variances: keep the scale of the data",
    )
    _relative_close(
        _field(result, "components")[:3],
        ref.components_,
        "tiny-scale data gave the wrong axes: the axes are scale-free, so they must match",
    )


def test_integer_input_is_centred_by_its_exact_mean():
    # Column 0 is 1, 2, 4 with mean 7/3. Its centred values -4/3, -1/3, 5/3 have squares summing
    # to 14/3, so the sample variance of that column (divisor 2) is 7/3. Column 1 is constant.
    X = np.array([[1, 0], [2, 0], [4, 0]])
    result = _call(X, 1, "pca failed on integer input with a fractional column mean")
    _check_close(
        _field(result, "explained_variance"),
        np.array([7 / 3]),
        "explained variance is wrong for integer input: centre each column by its exact mean, "
        "which can be a fraction, not by a rounded whole number",
    )


def test_near_tie_inside_the_band_makes_the_lower_index_positive():
    # The two entries of the axis differ by a relative 1e-10, which is inside the 1e-9 tie band.
    # They tie, so the lowest index decides and the first entry must be positive.
    s = 1.0 - 1e-10
    components = _field(
        _call([[s, -1.0], [-s, 1.0]], 1, "pca failed on rank-one data"), "components"
    )
    _check(
        components[0, 0] > 0,
        "entries within the relative tie band must count as tied, so the lowest index decides "
        "the sign and must be positive",
    )
