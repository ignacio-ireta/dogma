import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from submission import cosine_similarity_matrix

from dogma_data import rng, sample_rows

SEEDS = st.integers(min_value=0, max_value=10**6)
ROWS = st.integers(min_value=1, max_value=7)
DIMS = st.integers(min_value=1, max_value=6)


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


def _random_matrix(seed, rows, cols, scale=1.0):
    """Rows of standardised real data with both signs: breast_cancer, or digits when wider."""
    gen = rng(f"cosine-p01:data:{seed}")
    name = "breast_cancer" if cols <= 30 else "digits"
    X, _ = sample_rows(name, gen, rows, cols, standardize=True)
    return X * scale


def _must_raise_value_error(call, mistake):
    """Fail with the kind of mistake, instead of pytest's generic 'DID NOT RAISE'."""
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(f"{mistake} (raised {type(exc).__name__} instead)") from exc
    raise AssertionError(mistake)


@pytest.mark.parametrize(
    ("n", "m", "d"),
    [(5, 4, 3), (1, 1, 1), (6, 2, 40), (3, 3, 2)],
    ids=["general-shapes", "single-vector", "many-features", "equal-counts"],
)
def test_matches_scikit_learn_on_random_matrices(n, m, d):
    A = _random_matrix(seed=n * 100 + d, rows=n, cols=d)
    B = _random_matrix(seed=m * 100 + d + 1, rows=m, cols=d)
    np.testing.assert_allclose(
        cosine_similarity_matrix(A, B),
        sklearn_cosine(A, B),
        rtol=1e-9,
        atol=1e-9,
        err_msg=f"values differ from scikit-learn for shapes ({n}, {d}) and ({m}, {d}): "
        "check the normalisation and which matrix is transposed",
    )


def test_zero_rows_follow_scikit_learn_zero_convention():
    A = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]])
    B = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, 0.5, 3.0]])
    np.testing.assert_allclose(
        cosine_similarity_matrix(A, B),
        sklearn_cosine(A, B),
        atol=1e-12,
        err_msg="any pair with a zero-norm row must score 0.0, as in scikit-learn",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, DIMS)
def test_self_similarity_is_symmetric(seed, n, d):
    A = _random_matrix(seed, n, d)
    S = cosine_similarity_matrix(A, A)
    np.testing.assert_allclose(
        S,
        S.T,
        atol=1e-12,
        err_msg="S(A, A) must be symmetric: a row and column were mixed up",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, DIMS)
def test_self_similarity_has_unit_diagonal_for_nonzero_rows(seed, n, d):
    A = _random_matrix(seed, n, d)
    diagonal = np.diag(cosine_similarity_matrix(A, A))
    np.testing.assert_allclose(
        diagonal,
        1.0,
        atol=1e-9,
        err_msg="a nonzero row must have cosine 1 with itself: check the normalisation",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, ROWS, DIMS)
def test_positive_row_scaling_leaves_similarities_unchanged(seed, n, m, d):
    rng = np.random.default_rng(seed + 1)
    A = _random_matrix(seed, n, d)
    B = _random_matrix(seed + 2, m, d)
    row_scale = 0.01 + (100.0 - 0.01) * _real_unit(rng, (n, 1))
    np.testing.assert_allclose(
        cosine_similarity_matrix(row_scale * A, B),
        cosine_similarity_matrix(A, B),
        rtol=1e-9,
        atol=1e-9,
        err_msg="scaling a row by a positive factor must not change its cosines",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, ROWS, DIMS)
def test_swapping_arguments_transposes_the_result(seed, n, m, d):
    A = _random_matrix(seed, n, d)
    B = _random_matrix(seed + 3, m, d)
    np.testing.assert_allclose(
        cosine_similarity_matrix(B, A),
        cosine_similarity_matrix(A, B).T,
        atol=1e-12,
        err_msg="S(B, A) should equal S(A, B) transposed: check the output orientation",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, ROWS, ROWS, DIMS, st.floats(min_value=1e-6, max_value=1e6))
def test_values_stay_within_minus_one_and_one(seed, n, m, d, scale):
    A = _random_matrix(seed, n, d, scale=scale)
    B = _random_matrix(seed + 4, m, d, scale=scale)
    S = cosine_similarity_matrix(A, B)
    assert np.all(S <= 1.0) and np.all(S >= -1.0), (
        "similarities must lie in [-1, 1]: a missing clip or a wrong normalisation"
    )


def test_similarities_are_clipped_to_exactly_minus_one_and_one():
    # Unit-vector dot products of [1, 1, 1] with itself round to 1.0000000000000002, and
    # those of [-1, -1, -1] to -1.0000000000000002, so an unclipped result leaves [-1, 1].
    S = cosine_similarity_matrix([[1, 1, 1]], [[1, 1, 1], [-1, -1, -1]])
    assert np.all(S <= 1.0) and np.all(S >= -1.0), (
        "a similarity outside [-1, 1] is a rounding overshoot: clip the result to [-1, 1]"
    )


def test_opposite_vectors_give_minus_one_regardless_of_length():
    np.testing.assert_allclose(
        cosine_similarity_matrix([[1, 2]], [[-2, -4]]),
        [[-1.0]],
        atol=1e-12,
        err_msg="opposite directions must give -1, whatever the lengths",
    )


def test_parallel_vectors_give_one_regardless_of_length():
    np.testing.assert_allclose(
        cosine_similarity_matrix([[0.001, 0.002, 0.0]], [[500.0, 1000.0, 0.0]]),
        [[1.0]],
        atol=1e-12,
        err_msg="parallel vectors must give 1 even when their lengths differ by 10^6",
    )


def test_integer_input_returns_float64_matrix_of_shape_n_by_m():
    S = cosine_similarity_matrix([[1, 0], [1, 1], [0, 1]], [[1, 0], [0, 2]])
    assert isinstance(S, np.ndarray), "return a NumPy array, not a list"
    assert S.shape == (3, 2), f"shape mismatch: expected (3, 2) for 3 and 2 rows, got {S.shape}"
    assert S.dtype == np.float64, f"dtype should be float64 for integer input, got {S.dtype}"


def test_one_dimensional_input_raises_value_error():
    _must_raise_value_error(
        lambda: cosine_similarity_matrix([1, 2], [[1, 2]]),
        "a 1-D input must raise ValueError: check ndim before comparing shapes",
    )


def test_empty_input_raises_value_error():
    _must_raise_value_error(
        lambda: cosine_similarity_matrix(np.zeros((0, 3)), np.ones((2, 3))),
        "an empty input must raise ValueError: check for zero rows or columns before normalising",
    )


def test_nan_entry_raises_value_error():
    _must_raise_value_error(
        lambda: cosine_similarity_matrix([[1.0, np.nan]], [[1.0, 0.0]]),
        "a NaN entry must raise ValueError: validate with np.isfinite before any arithmetic",
    )


def test_huge_entries_do_not_overflow_in_the_norm():
    np.testing.assert_allclose(
        cosine_similarity_matrix([[3e200, 4e200]], [[1.0, 0.0]]),
        [[0.6]],
        rtol=1e-12,
        err_msg="entries near 1e200 overflow when squared: rescale each row before its norm",
    )


def test_tiny_entries_do_not_underflow_to_a_zero_vector():
    np.testing.assert_allclose(
        cosine_similarity_matrix([[3e-200, 4e-200]], [[0.0, 1.0]]),
        [[0.8]],
        rtol=1e-12,
        err_msg="a row of tiny but nonzero entries was treated as the zero vector",
    )


def test_inputs_are_not_mutated():
    A = np.array([[0.0, 0.0], [1.0, 2.0]])
    B = np.array([[3.0, -1.0]])
    A_copy, B_copy = A.copy(), B.copy()
    cosine_similarity_matrix(A, B)
    np.testing.assert_array_equal(A, A_copy, err_msg="input A was modified in place")
    np.testing.assert_array_equal(B, B_copy, err_msg="input B was modified in place")


def test_nan_in_the_second_matrix_raises_value_error():
    _must_raise_value_error(
        lambda: cosine_similarity_matrix([[1.0, 0.0]], [[1.0, np.nan]]),
        "a NaN entry in B must raise ValueError: check every input, not only the first",
    )


def test_empty_second_matrix_raises_value_error():
    _must_raise_value_error(
        lambda: cosine_similarity_matrix(np.ones((2, 3)), np.zeros((0, 3))),
        "an empty B must raise ValueError: check for zero rows or columns in both inputs",
    )


def test_subnormal_entries_are_not_treated_as_a_zero_vector():
    # Row (3e-310, 4e-310) is nonzero, so it has a unit direction (0.6, 0.8).
    # Its dot product with the unit vector (1, 0) is 0.6.
    np.testing.assert_allclose(
        cosine_similarity_matrix([[3e-310, 4e-310]], [[1.0, 0.0]]),
        [[0.6]],
        rtol=1e-9,
        err_msg="a nonzero row with subnormal entries was treated as the zero vector",
    )


def test_large_integer_entries_are_not_wrapped():
    # 2**40 is a real number. Its direction is (1, 1), so the cosine with (1, 0) is 1/sqrt(2).
    np.testing.assert_allclose(
        cosine_similarity_matrix([[2**40, 2**40]], [[1, 0]]),
        [[1 / np.sqrt(2)]],
        atol=1e-9,
        err_msg="large integer entries were wrapped or truncated: keep integers exact as float64",
    )
