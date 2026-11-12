import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import best_rank_k

from dogma_data import rng, sample_rows

SEEDS = st.integers(min_value=0, max_value=10**6)
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


def _matrix(seed, m, n):
    """m rows and n columns of standardised wine: a real data matrix with correlated columns."""
    A, _ = sample_rows("wine", rng(f"best-rank-k-p01:data:{seed}"), m, n, standardize=True)
    return A


def _call(A, k, mistake):
    try:
        return best_rank_k(A, k)
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")


def _unpack(result, shape):
    """Return (A_k, error) after checking the tuple layout and the shape of A_k."""
    if not isinstance(result, tuple) or len(result) != 2:
        pytest.fail("the result must be a tuple (A_k, error)")
    A_k, error = result
    try:
        A_k = np.asarray(A_k, dtype=np.float64)
        error = float(error)
    except (TypeError, ValueError):
        pytest.fail("A_k must be a numeric matrix and error a real number")
    _check(
        A_k.shape == tuple(shape),
        "shape mismatch: A_k must have the same (m, n) shape as the input A",
    )
    return A_k, error


def _check(ok, mistake):
    if not ok:
        pytest.fail(mistake)


def _close(actual, expected, mistake, atol):
    _check(bool(np.allclose(actual, expected, rtol=1e-9, atol=atol)), mistake)


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def _scale(A):
    return max(1.0, float(np.linalg.norm(A)))


def _eigen_reference(A, k):
    """Truncation and discarded squared singular values from the eigen-decomposition of A^T A."""
    m, n = A.shape
    eigenvalues, eigenvectors = np.linalg.eigh(A.T @ A)
    order = np.argsort(eigenvalues)[::-1]
    V_k = eigenvectors[:, order[:k]]
    tail = np.clip(eigenvalues[order[k : min(m, n)]], 0.0, None)
    return A @ V_k @ V_k.T, float(tail.sum())


@pytest.mark.parametrize(
    ("seed", "m", "n", "k"),
    [(1, 5, 3, 1), (2, 3, 6, 2), (3, 4, 4, 2), (4, 6, 2, 1)],
    ids=["tall", "wide", "square", "very-tall"],
)
def test_matches_eigen_reference_on_random_matrices(seed, m, n, k):
    A = _matrix(seed, m, n)
    A_k, error = _unpack(_call(A, k, "a valid random matrix was rejected or failed"), A.shape)
    reference, tail = _eigen_reference(A, k)
    _close(
        A_k,
        reference,
        "A_k differs from the rank-k truncation: keep the k largest singular terms, each with "
        "its own left and right singular vectors",
        atol=1e-9 * _scale(A),
    )
    _check(
        abs(error**2 - tail) <= 1e-8 * _scale(A) ** 2,
        "the error is not the square root of the discarded squared singular values: check which "
        "singular values you sum, and that you square them",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, DIMS, DIMS, st.data())
def test_error_equals_frobenius_norm_of_the_residual(seed, m, n, data):
    A = _matrix(seed, m, n)
    k = data.draw(st.integers(min_value=0, max_value=min(m, n)))
    A_k, error = _unpack(_call(A, k, "a valid random matrix was rejected or failed"), A.shape)
    _check(
        abs(error - float(np.linalg.norm(A - A_k))) <= 1e-9 * _scale(A),
        "the error does not equal the Frobenius norm of A - A_k: the error must describe the "
        "residual you actually return",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, DIMS, DIMS, st.data())
def test_squared_error_equals_sum_of_discarded_eigenvalues_of_gram_matrix(seed, m, n, data):
    A = _matrix(seed, m, n)
    k = data.draw(st.integers(min_value=0, max_value=min(m, n)))
    _, error = _unpack(_call(A, k, "a valid random matrix was rejected or failed"), A.shape)
    _, tail = _eigen_reference(A, k)
    _check(
        abs(error**2 - tail) <= 1e-8 * _scale(A) ** 2,
        "the squared error must equal the sum of the discarded singular values squared (the "
        "discarded eigenvalues of A^T A)",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, DIMS, DIMS, st.data())
def test_no_random_rank_k_matrix_is_closer_than_the_truncation(seed, m, n, data):
    A = _matrix(seed, m, n)
    k = data.draw(st.integers(min_value=0, max_value=min(m, n)))
    _, error = _unpack(_call(A, k, "a valid random matrix was rejected or failed"), A.shape)
    rng = np.random.default_rng(seed + 7)
    competitor = _real_normal(rng, (m, k)) @ _real_normal(rng, (k, n))
    _check(
        float(np.linalg.norm(A - competitor)) >= error - 1e-9 * _scale(A),
        "a rank-k matrix is closer to A than the truncation: keep the largest singular values, "
        "not the smallest",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, DIMS, DIMS, st.data())
def test_truncation_has_at_most_k_nonzero_singular_values(seed, m, n, data):
    A = _matrix(seed, m, n)
    k = data.draw(st.integers(min_value=0, max_value=min(m, n)))
    A_k, _ = _unpack(_call(A, k, "a valid random matrix was rejected or failed"), A.shape)
    singular = np.linalg.svd(A_k, compute_uv=False)
    _check(
        bool(np.all(singular[k:] <= 1e-9 * _scale(A))),
        "A_k has more than k non-zero singular values: keep exactly the first k singular terms",
    )


def test_zero_matrix_gives_zero_truncation_and_zero_error():
    A = np.zeros((3, 4))
    A_k, error = _unpack(_call(A, 2, "the zero matrix was rejected or failed"), A.shape)
    _check(
        bool(np.all(np.abs(A_k) <= 1e-12)),
        "the zero matrix has only zero singular values, so its truncation must be zero",
    )
    _check(abs(error) <= 1e-12, "the zero matrix has zero error for every k")


def test_rank_one_input_is_reproduced_by_one_term_with_zero_error():
    rng = np.random.default_rng(21)
    A = np.outer(_real_normal(rng, 5), _real_normal(rng, 3))
    A_k, error = _unpack(_call(A, 1, "a rank-one input was rejected or failed"), A.shape)
    _close(
        A_k,
        A,
        "a rank-one input is reproduced exactly by its single singular term",
        atol=1e-9 * _scale(A),
    )
    _check(
        error <= 1e-9 * _scale(A),
        "rank-one input with k = 1: the error must vanish to rounding precision (about 1e-9 of the "
        "norm); a square root of a rounding residue is too coarse",
    )


@pytest.mark.parametrize(("m", "n"), [(4, 3), (3, 5)], ids=["tall", "wide"])
def test_keeping_min_of_rows_and_columns_reproduces_the_input(m, n):
    A = _matrix(31 + m, m, n)
    A_k, error = _unpack(_call(A, min(m, n), "a full-rank truncation was rejected"), A.shape)
    _close(
        A_k,
        A,
        "keeping min(m, n) singular terms must reproduce A",
        atol=1e-9 * _scale(A),
    )
    _check(error <= 1e-9 * _scale(A), "keeping min(m, n) terms leaves no error")


@pytest.mark.parametrize(("m", "n"), [(4, 3), (2, 5)], ids=["tall", "wide"])
def test_k_zero_gives_zero_matrix_with_error_equal_to_full_norm(m, n):
    A = _matrix(41 + m, m, n)
    A_k, error = _unpack(_call(A, 0, "k = 0 was rejected or failed"), A.shape)
    _check(bool(np.all(np.abs(A_k) <= 1e-12)), "k = 0 must give the zero matrix")
    _check(
        abs(error - float(np.linalg.norm(A))) <= 1e-9 * _scale(A),
        "with k = 0 the error is the Frobenius norm of A",
    )


def test_repeated_singular_values_keep_the_error_identity_and_rank():
    rng = np.random.default_rng(51)
    Q = np.linalg.qr(_real_normal(rng, (3, 3)))[0]
    A = 3.0 * Q  # all singular values are equal, so the best rank-one matrix is not unique
    A_k, error = _unpack(
        _call(A, 1, "a matrix with repeated singular values was rejected"), A.shape
    )
    _check(
        abs(error - 3.0 * np.sqrt(2.0)) <= 1e-9 * 3.0,
        "with repeated singular values the error is the square root of the discarded squared "
        "singular values: check which ones you discard",
    )
    _check(
        abs(error - float(np.linalg.norm(A - A_k))) <= 1e-9 * 3.0,
        "the error does not equal the Frobenius norm of A - A_k for this repeated spectrum",
    )
    singular = np.linalg.svd(A_k, compute_uv=False)
    _check(
        bool(singular[1] <= 1e-9 * 3.0),
        "A_k has more singular terms than requested: keep exactly the k singular terms "
        "you were given",
    )


@pytest.mark.parametrize("scale", [1e100, 1e-100], ids=["huge", "tiny"])
def test_scaling_the_input_scales_the_truncation_and_the_error(scale):
    A0 = _matrix(61, 4, 3)
    base_k, base_error = _unpack(_call(A0, 2, "the reference matrix was rejected"), A0.shape)
    A = scale * A0
    A_k, error = _unpack(_call(A, 2, "a scaled matrix was rejected or failed"), A.shape)
    _close(
        A_k,
        scale * base_k,
        "scaling A by c must scale A_k by c: the truncation must not depend on an absolute size",
        atol=1e-9 * scale * _scale(A0),
    )
    _check(
        abs(error - scale * base_error) <= 1e-9 * scale * _scale(A0),
        "scaling A by c must scale the error by |c|: check the square root and the sum",
    )


def test_integer_input_returns_float64_matrix_of_input_shape_and_real_error():
    A = np.array([[1, 0, 2, 3, 1], [0, 4, 1, 0, 2]])
    result = _call(A, 1, "integer input was rejected or failed")
    _, error = _unpack(result, A.shape)
    _check(
        np.asarray(result[0]).dtype == np.float64,
        "integer input must give a float64 matrix for A_k, not an integer dtype",
    )
    _check(np.isfinite(error), "the error must be a finite real number")


@pytest.mark.parametrize(
    ("A", "k"),
    [
        (np.array([1.0, 2.0, 3.0]), 1),
        (np.zeros((0, 3)), 0),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), 1),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), 1),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), -1),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 3),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), True),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 1.0),
    ],
    ids=[
        "one-dimensional",
        "empty",
        "nan",
        "infinite",
        "negative-k",
        "k-above-min-shape",
        "boolean-k",
        "float-k",
    ],
)
def test_invalid_matrix_or_rank_raises_value_error(A, k):
    _expect_value_error(
        lambda: best_rank_k(A, k),
        "invalid input must raise ValueError: check dimensions, finiteness and that k is an "
        "integer between 0 and min(m, n)",
    )


def test_inputs_are_not_mutated():
    A = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 4.0]])
    A_before = A.copy()
    _call(A, 1, "a valid matrix was rejected or failed")
    _check(
        bool(np.array_equal(A, A_before)),
        "the input matrix was modified in place: decompose a copy, not A",
    )


def test_rank_above_the_smaller_side_of_a_wide_matrix_raises_value_error():
    _expect_value_error(
        lambda: best_rank_k([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 3),
        "a rank above min(m, n) was accepted: k must be at most the smaller dimension",
    )


def test_matrix_with_no_columns_raises_value_error():
    _expect_value_error(
        lambda: best_rank_k(np.zeros((2, 0)), 0),
        "a matrix with no columns was accepted: A must have at least one row and one column",
    )
