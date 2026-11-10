import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from submission import vector_norm

FINITE = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
VECTORS = arrays(np.float64, st.integers(min_value=1, max_value=8), elements=FINITE)
EXPONENTS = st.sampled_from([1.0, 2.0, 3.5, math.inf])


@st.composite
def same_length_pairs(draw):
    """Two vectors x and y of one shared length n >= 1."""
    n = draw(st.integers(min_value=1, max_value=8))
    x = draw(arrays(np.float64, n, elements=FINITE))
    y = draw(arrays(np.float64, n, elements=FINITE))
    return x, y


def _assert_close(actual, desired, rtol=1e-7, atol=0.0, err_msg=""):
    """Like np.testing.assert_allclose, but the message is the first line of the failure."""
    got = np.asarray(actual, dtype=np.float64)
    want = np.asarray(desired, dtype=np.float64)
    if want.ndim == 0:
        want = np.broadcast_to(want, got.shape)
    if got.shape != want.shape:
        raise AssertionError(f"{err_msg}: shape {got.shape}, expected {want.shape}")
    if not np.allclose(got, want, rtol=rtol, atol=atol):
        worst = float(np.max(np.abs(got - want)))
        raise AssertionError(f"{err_msg} (largest difference {worst:.3g})")


def _assert_equal(actual, desired, err_msg=""):
    """Exact array equality, with the message first, as in _assert_close."""
    if not np.array_equal(np.asarray(actual), np.asarray(desired)):
        raise AssertionError(err_msg)


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_matches_numpy_norm_on_random_vectors_for_common_exponents():
    rng = np.random.default_rng(0)
    for p in [1.0, 2.0, 3.5, np.inf]:
        for n in [1, 2, 5, 8]:
            x = rng.normal(size=n)
            _assert_close(
                vector_norm(x, p),
                np.linalg.norm(x, ord=p),
                rtol=1e-9,
                err_msg="norm differs from numpy's norm: check the 1/p root",
            )


def test_zero_vector_has_zero_norm_for_every_exponent():
    for p in [1.0, 2.0, 3.5, math.inf]:
        assert vector_norm([0.0, 0.0, 0.0], p) == 0.0, (
            "the zero vector has norm 0 for every p; "
            "dividing by the largest entry would divide by zero here"
        )


def test_single_entry_norm_is_its_absolute_value():
    for p in [1.0, 2.0, 3.5, math.inf]:
        assert vector_norm([-5.0], p) == pytest.approx(5.0, rel=1e-12), (
            "a one-entry vector has norm |x| for every p"
        )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(VECTORS, EXPONENTS, st.floats(min_value=-100.0, max_value=100.0, allow_nan=False))
def test_norm_is_homogeneous_in_its_scalar(x, p, alpha):
    lhs = vector_norm(alpha * x, p)
    rhs = abs(alpha) * vector_norm(x, p)
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-12), (
        "homogeneity fails: scaling a vector by a must scale its norm by |a|"
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(same_length_pairs(), EXPONENTS)
def test_triangle_inequality_holds(pair, p):
    x, y = pair
    lhs = vector_norm(x + y, p)
    rhs = vector_norm(x, p) + vector_norm(y, p)
    assert lhs <= rhs * (1 + 1e-9) + 1e-12, (
        "triangle inequality violated: the norm of a sum must not exceed the sum of the norms"
    )


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(
    VECTORS,
    st.sampled_from([1.0, 1.5, 2.0, 3.0, math.inf]),
    st.sampled_from([1.0, 2.0, 4.0, math.inf]),
)
def test_larger_exponent_never_gives_a_larger_norm(x, p, q):
    assume(p <= q)
    assert vector_norm(x, q) <= vector_norm(x, p) * (1 + 1e-9) + 1e-12, (
        "norms must not grow as p grows: check the exponent and the 1/p root"
    )


def test_norms_are_ordered_linf_le_l2_le_l1():
    x = np.array([3.0, -4.0, 0.5])
    linf, l2, l1 = vector_norm(x, math.inf), vector_norm(x, 2), vector_norm(x, 1)
    assert linf <= l2 <= l1, f"expected ||x||_inf <= ||x||_2 <= ||x||_1, got {linf}, {l2}, {l1}"


def test_huge_entries_do_not_overflow_for_the_euclidean_norm():
    result = vector_norm([3e200, 4e200], 2)
    assert math.isfinite(result), "L2 norm overflowed: square the entries after scaling by the max"
    assert result == pytest.approx(5e200, rel=1e-12), (
        "the L2 norm of (3e200, 4e200) is 5e200; squaring first overflows float64"
    )


def test_huge_entries_do_not_overflow_for_a_non_integer_exponent():
    result = vector_norm([1e200, 1e200], 3.5)
    assert math.isfinite(result), "the p-norm overflowed for large entries; scale before powering"
    assert result == pytest.approx(1e200 * 2 ** (1 / 3.5), rel=1e-12), (
        "two equal large entries a have norm a * 2**(1/p) for any p"
    )


def test_tiny_entries_do_not_underflow_to_zero():
    assert vector_norm([3e-200, 4e-200], 2) == pytest.approx(5e-200, rel=1e-12), (
        "the L2 norm of (3e-200, 4e-200) is 5e-200; squaring underflows to zero"
    )
    assert vector_norm([1e-200, 1e-200], 3.5) == pytest.approx(
        1e-200 * 2 ** (1 / 3.5), rel=1e-12
    ), "tiny equal entries underflow when raised to a power; scale by the max before powering"


@pytest.mark.parametrize(
    "bad_p", [0.5, 0.0, -1.0, float("nan")], ids=["fraction-below-one", "zero", "negative", "nan"]
)
def test_exponent_below_one_or_nan_raises_value_error(bad_p):
    _raised_value_error(
        lambda: vector_norm([1.0, 2.0], bad_p),
        "an exponent below 1, or NaN, does not define a norm: raise ValueError, do not compute",
    )


def test_matrix_input_raises_value_error():
    _raised_value_error(
        lambda: vector_norm([[1.0, 2.0], [3.0, 4.0]], 2),
        "a 2-D input is not a vector: raise ValueError instead of flattening it silently",
    )


def test_empty_vector_raises_value_error():
    _raised_value_error(
        lambda: vector_norm([], 2),
        "the norm of an empty vector is not defined here: raise ValueError",
    )


def test_input_vector_is_not_mutated():
    x = np.array([3.0, -4.0, 1.0])
    before = x.copy()
    vector_norm(x, 2)
    _assert_equal(x, before, err_msg="input vector was modified in place")
