import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import pseudo_inverse

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
    A, _ = sample_rows("wine", rng(f"pseudo-inverse-p02:data:{seed}"), m, n, standardize=True)
    return A


def _call(A, mistake, **kwargs):
    try:
        return pseudo_inverse(A, **kwargs)
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")


def _as_matrix(result, shape):
    try:
        matrix = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        pytest.fail("the result must be a numeric matrix")
    _check(
        matrix.shape == tuple(shape),
        "shape mismatch: the pseudo-inverse of an (m, n) matrix must have shape (n, m)",
    )
    return matrix


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


def _scale(M):
    return max(1.0, float(np.max(np.abs(M))))


@pytest.mark.parametrize(
    ("seed", "m", "n"),
    [(1, 5, 3), (2, 2, 6), (3, 4, 4)],
    ids=["tall", "wide", "square"],
)
def test_matches_numpy_pseudo_inverse_on_random_full_rank_matrices(seed, m, n):
    A = _matrix(seed, m, n)
    result = _as_matrix(_call(A, "a valid full-rank matrix was rejected or failed"), (n, m))
    reference = np.linalg.pinv(A, rcond=1e-15)
    _close(
        result,
        reference,
        "values differ from the reference pseudo-inverse: check the SVD factors, their order and "
        "the transposes",
        atol=1e-9 * _scale(reference),
    )


@pytest.mark.parametrize(
    ("seed", "m", "n", "r"),
    [(4, 5, 4, 2), (5, 3, 6, 2), (6, 4, 4, 3)],
    ids=["tall-rank-two", "wide-rank-two", "square-rank-three"],
)
def test_matches_numpy_on_rank_deficient_matrices_under_an_explicit_cutoff(seed, m, n, r):
    rng = np.random.default_rng(seed)
    A = _real_normal(rng, (m, r)) @ _real_normal(rng, (r, n))
    result = _as_matrix(
        _call(A, "a rank-deficient matrix was rejected or failed", rcond=1e-10), (n, m)
    )
    reference = np.linalg.pinv(A, rcond=1e-10)
    _close(
        result,
        reference,
        "rank-deficient input: drop the singular values at or below the cutoff, "
        "and invert the rest",
        atol=1e-9 * _scale(reference),
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(SEEDS, DIMS, DIMS, st.data())
def test_penrose_identities_hold_on_random_rank_deficient_matrices(seed, m, n, data):
    r = data.draw(st.integers(min_value=1, max_value=min(m, n)))
    rng = np.random.default_rng(seed)
    A = _real_normal(rng, (m, r)) @ _real_normal(rng, (r, n))
    P = _as_matrix(_call(A, "a valid rank-deficient matrix was rejected", rcond=1e-10), (n, m))
    atol_A = 1e-8 * _scale(A)
    atol_P = 1e-8 * _scale(P)
    _close(
        A @ P @ A,
        A,
        "A A+ A must equal A: the result is not a generalised inverse (check the factor order)",
        atol=atol_A,
    )
    _close(
        P @ A @ P,
        P,
        "A+ A A+ must equal A+: the result is not a generalised inverse (check the transposes)",
        atol=atol_P,
    )
    _close(
        A @ P,
        (A @ P).T,
        "A A+ must be symmetric: the first projection condition fails",
        atol=atol_A,
    )
    _close(
        P @ A,
        (P @ A).T,
        "A+ A must be symmetric: the second projection condition fails",
        atol=atol_A,
    )


def test_default_cutoff_keeps_a_singular_value_that_is_a_small_fraction_of_the_largest():
    A = np.diag([1.0, 1e-14])
    result = _as_matrix(_call(A, "a diagonal matrix was rejected or failed"), (2, 2))
    _close(
        result,
        np.diag([1.0, 1e14]),
        "the default cutoff is too coarse: a singular value that is a small fraction of "
        "the largest one must be kept by default",
        atol=1e-9,
    )


def test_default_cutoff_drops_a_singular_value_below_rcond_times_the_largest():
    # 5e-16 is below the default cutoff 1e-15 * 1, so the default must treat it as zero.
    A = np.diag([1.0, 5e-16])
    result = _as_matrix(_call(A, "a diagonal matrix was rejected or failed"), (2, 2))
    _close(
        result,
        np.diag([1.0, 0.0]),
        "the default cutoff is too fine: a singular value below 1e-15 times the largest must be "
        "dropped by default",
        atol=1e-9,
    )


def test_cutoff_is_relative_to_the_largest_singular_value_for_tiny_matrices():
    A = 1e-16 * np.eye(2)
    result = _as_matrix(_call(A, "a tiny matrix was rejected or failed"), (2, 2))
    _close(
        result,
        1e16 * np.eye(2),
        "the cutoff must be relative to the largest singular value: "
        "a tiny matrix must not be zeroed",
        atol=1e-9 * 1e16,
    )


def test_cutoff_drops_a_singular_value_below_rcond_times_the_largest():
    A = np.diag([1000.0, 0.5])
    result = _as_matrix(_call(A, "a diagonal matrix was rejected or failed", rcond=1e-3), (2, 2))
    _close(
        result,
        np.diag([1e-3, 0.0]),
        "the cutoff must be rcond times the largest singular value: a singular value below it must "
        "be dropped",
        atol=1e-9,
    )


def test_singular_value_equal_to_the_cutoff_is_dropped():
    A = np.diag([1.0, 0.5])
    result = _as_matrix(_call(A, "a diagonal matrix was rejected or failed", rcond=0.5), (2, 2))
    _close(
        result,
        np.diag([1.0, 0.0]),
        "a singular value exactly equal to the cutoff must be dropped: keep only values strictly "
        "above it",
        atol=1e-9,
    )


def test_rank_one_matrix_has_the_closed_form_pseudo_inverse():
    rng = np.random.default_rng(17)
    u = _real_normal(rng, 4)
    v = _real_normal(rng, 3)
    A = np.outer(u, v)
    result = _as_matrix(_call(A, "a rank-one matrix was rejected or failed"), (3, 4))
    expected = np.outer(v, u) / ((u @ u) * (v @ v))
    _close(
        result,
        expected,
        "rank-one matrix: the pseudo-inverse is v u^T / (|u|^2 |v|^2): check the factor order and "
        "the transposes",
        atol=1e-9 * _scale(expected),
    )


def test_zero_matrix_gives_zero_matrix_of_transposed_shape():
    result = _as_matrix(_call(np.zeros((2, 3)), "the zero matrix was rejected or failed"), (3, 2))
    _check(
        bool(np.all(np.isfinite(result)) and np.all(result == 0.0)),
        "the zero matrix has no kept singular values, so its pseudo-inverse is zero "
        "(no inf or NaN)",
    )


@pytest.mark.parametrize(
    ("A", "kwargs"),
    [
        (np.array([1.0, 2.0, 3.0]), {}),
        (np.zeros((0, 3)), {}),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), {}),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), {}),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), {"rcond": -1e-3}),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), {"rcond": float("nan")}),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), {"rcond": True}),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), {"rcond": float("inf")}),
    ],
    ids=[
        "one-dimensional",
        "empty",
        "nan",
        "infinite",
        "negative-rcond",
        "nan-rcond",
        "boolean-rcond",
        "infinite-rcond",
    ],
)
def test_invalid_matrix_or_rcond_raises_value_error(A, kwargs):
    _expect_value_error(
        lambda: pseudo_inverse(A, **kwargs),
        "invalid input must raise ValueError: check dimensions, finiteness, and that rcond is a "
        "finite number >= 0",
    )


def test_rcond_is_keyword_only():
    A = np.diag([1.0, 0.5])
    try:
        pseudo_inverse(A, 0.5)
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"a positional rcond raised {type(exc).__name__}; rcond must be keyword-only")
    pytest.fail("rcond must be keyword-only: a positional second argument must be rejected")


def test_integer_input_gives_float64_pseudo_inverse_of_transposed_shape():
    A = np.array([[1, 2, 0], [0, 1, 3]])
    raw = _call(A, "integer input was rejected or failed")
    result = _as_matrix(raw, (3, 2))
    _check(
        np.asarray(raw).dtype == np.float64,
        "integer input must give a float64 pseudo-inverse, not an integer dtype",
    )
    _close(
        result,
        np.linalg.pinv(A.astype(np.float64), rcond=1e-15),
        "integer input gives different values than its float64 version",
        atol=1e-9,
    )


@pytest.mark.parametrize("scale", [1e100, 1e-100], ids=["huge", "tiny"])
def test_scaling_the_input_scales_the_pseudo_inverse_by_the_reciprocal(scale):
    A0 = _matrix(71, 3, 2)
    P0 = _as_matrix(_call(A0, "the reference matrix was rejected or failed"), (2, 3))
    P = _as_matrix(_call(scale * A0, "a scaled matrix was rejected or failed"), (2, 3))
    _close(
        P,
        P0 / scale,
        "scaling A by c must scale its pseudo-inverse by 1/c: the cutoff must be relative to the "
        "largest singular value",
        atol=1e-9 * _scale(P0) / scale,
    )


def test_inputs_are_not_mutated():
    A = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 4.0]])
    A_before = A.copy()
    _call(A, "a valid matrix was rejected or failed")
    _check(
        bool(np.array_equal(A, A_before)),
        "the input matrix was modified in place: decompose a copy, not A",
    )


def test_default_cutoff_keeps_a_singular_value_just_above_its_default_threshold():
    # 1.5e-15 is above the default cutoff 1e-15 * 1, so it is inverted, not dropped.
    A = np.diag([1.0, 1.5e-15])
    result = _as_matrix(_call(A, "a diagonal matrix was rejected or failed"), (2, 2))
    _close(
        result,
        np.diag([1.0, 1.0 / 1.5e-15]),
        "the default cutoff is too coarse: a singular value above 1e-15 times the largest must "
        "be kept by default",
        atol=1e-9,
    )


def test_cutoff_is_relative_to_the_largest_singular_value_not_the_frobenius_norm():
    # Four singular values equal 10 and one equals 1.5. The cutoff is 0.1 * 10 = 1, so 1.5 is
    # kept. A cutoff taken from the Frobenius norm (about 20) would drop it.
    A = np.diag([10.0, 10.0, 10.0, 10.0, 1.5])
    result = _as_matrix(_call(A, "a diagonal matrix was rejected or failed", rcond=0.1), (5, 5))
    _close(
        result,
        np.diag([0.1, 0.1, 0.1, 0.1, 1.0 / 1.5]),
        "the cutoff must be rcond times the largest singular value, not the Frobenius norm: a "
        "singular value above it must be kept",
        atol=1e-9,
    )


def test_zero_rcond_is_a_valid_cutoff_and_keeps_every_singular_value():
    result = _as_matrix(
        _call(np.diag([2.0, 4.0]), "rcond = 0 is a valid cutoff and was rejected", rcond=0.0),
        (2, 2),
    )
    _close(
        result,
        np.diag([0.5, 0.25]),
        "rcond = 0 must keep every non-zero singular value and invert it",
        atol=1e-12,
    )
