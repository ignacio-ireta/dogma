import functools

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from submission import vector_projection


@functools.cache
def _tables():
    """Three real tables to take vectors from.

    The 178 wines in their own units (all positive, from hundredths to thousands). The same
    wines standardised (both signs, comparable scales). And 200 handwritten digits as the
    shortfall of each pixel from full ink, intensity minus 16: whole numbers from -16 to 0,
    with an exact 0 wherever a pixel is fully inked.
    """
    # Imported here: this topic is also checked as a copy outside the repository, where the
    # import order of a first-party package differs.
    from dogma_data import rng as data_rng  # noqa: PLC0415
    from dogma_data import sample_rows  # noqa: PLC0415

    raw, _ = sample_rows("wine", data_rng("p01-vector-projection:wines"), 178)
    scaled, _ = sample_rows("wine", data_rng("p01-vector-projection:wines"), 178, standardize=True)
    pixels, _ = sample_rows("digits", data_rng("p01-vector-projection:digits"), 200)
    return raw, scaled, pixels - 16.0


@st.composite
def vector_pairs(draw):
    """Two real rows of one table over the same run of one to eight columns."""
    n = draw(st.integers(min_value=1, max_value=8))
    table = _tables()[draw(st.integers(min_value=0, max_value=2))]
    first = draw(st.integers(min_value=0, max_value=table.shape[1] - n))
    rows = draw(st.lists(st.integers(0, table.shape[0] - 1), min_size=2, max_size=2, unique=True))
    a = table[rows[0], first : first + n].copy()
    b = table[rows[1], first : first + n].copy()
    assume(float(b @ b) > 1e-6)
    return a, b


def _reference(a, b):
    """Independent formula: project onto the unit vector u = b/|b|, then scale."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u = b / np.sqrt(np.dot(b, b))
    return np.dot(a, u) * u


def _scale(a, b):
    return np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)) + 1.0


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def _project(a, b, mistake):
    try:
        return vector_projection(a, b)
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")


@settings(max_examples=80, deadline=None, report_multiple_bugs=False)
@given(vector_pairs())
def test_matches_independent_reference_on_random_vectors(pair):
    a, b = pair
    np.testing.assert_allclose(
        vector_projection(a, b),
        _reference(a, b),
        rtol=1e-7,
        atol=1e-9 * (np.sqrt(np.dot(a, a)) + 1.0),
        err_msg="values differ from the reference projection (a·b / b·b)·b",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(vector_pairs())
def test_residual_is_orthogonal_to_b(pair):
    a, b = pair
    residual = a - vector_projection(a, b)
    assert abs(float(np.dot(residual, b))) <= 1e-8 * _scale(a, b), (
        "the residual a - proj_b(a) should be orthogonal to b"
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(vector_pairs())
def test_projection_is_idempotent(pair):
    a, b = pair
    once = vector_projection(a, b)
    np.testing.assert_allclose(
        vector_projection(once, b),
        once,
        rtol=1e-7,
        atol=1e-9 * (np.sqrt(np.dot(a, a)) + 1.0),
        err_msg="projecting a projection again should not change it",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(vector_pairs(), st.sampled_from([-3.0, 0.5, 7.0]))
def test_rescaling_b_does_not_change_the_projection(pair, factor):
    a, b = pair
    np.testing.assert_allclose(
        vector_projection(a, factor * b),
        vector_projection(a, b),
        rtol=1e-7,
        atol=1e-9 * (np.sqrt(np.dot(a, a)) + 1.0),
        err_msg="the projection depends on the line through b, not on b's length or sign",
    )


def test_returns_float64_array_of_shape_n_for_integer_input():
    out = vector_projection([1, 2, 3], [2, 0, 0])
    assert isinstance(out, np.ndarray), "return a NumPy array, not a list"
    assert out.shape == (3,), f"shape mismatch: expected (3,), got {out.shape}"
    assert out.dtype == np.float64, f"dtype should be float64 even for int input, got {out.dtype}"
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0], err_msg="integer arithmetic truncated")


def test_length_mismatch_raises_value_error():
    _raised_value_error(
        lambda: vector_projection([1, 2, 3], [1, 2]),
        "vectors of different lengths have no projection: raise ValueError",
    )


def test_two_dimensional_input_raises_value_error():
    _raised_value_error(
        lambda: vector_projection([[1, 2], [3, 4]], [[1, 0], [0, 1]]),
        "a 2-D input is not a vector: raise ValueError instead of computing a matrix product",
    )


def test_empty_vectors_raise_value_error():
    _raised_value_error(
        lambda: vector_projection([], []),
        "empty vectors have no line to project onto: raise ValueError",
    )


def test_large_magnitudes_do_not_overflow():
    a = np.array([3e100, 4e100])
    b = np.array([1e100, 0.0])
    np.testing.assert_allclose(
        vector_projection(a, b),
        [3e100, 0.0],
        rtol=1e-12,
        err_msg="large magnitudes should still project correctly",
    )


def test_huge_nonzero_b_does_not_overflow_in_its_squared_length():
    # b·b overflows to inf here, which turns the coefficient into nan.
    result = _project(
        [1e160, 1e160],
        [1e160, 0.0],
        "a huge nonzero b must project correctly: rescale b before forming b·b",
    )
    np.testing.assert_allclose(
        result,
        [1e160, 0.0],
        rtol=1e-12,
        err_msg="a huge nonzero b gave inf or nan: rescale b before forming b·b",
    )


def test_tiny_nonzero_b_is_not_the_zero_vector():
    # b·b underflows to 0 here, so testing b·b == 0 wrongly rejects a nonzero b.
    result = _project(
        [3e-170, 4e-170],
        [1e-170, 0.0],
        "a tiny nonzero b is not the zero vector: rescale b before forming b·b",
    )
    np.testing.assert_allclose(
        result,
        [3e-170, 0.0],
        rtol=1e-12,
        err_msg="a tiny nonzero b gave the wrong projection: rescale b before forming b·b",
    )


def test_tiny_magnitudes_do_not_underflow_to_zero_division():
    a = np.array([3e-100, 4e-100])
    b = np.array([0.0, 2e-100])
    np.testing.assert_allclose(
        vector_projection(a, b),
        [0.0, 4e-100],
        rtol=1e-12,
        err_msg="tiny magnitudes are valid input, not the zero vector",
    )


def test_inputs_are_not_mutated():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, 0.0])
    a_copy, b_copy = a.copy(), b.copy()
    vector_projection(a, b)
    np.testing.assert_array_equal(a, a_copy, err_msg="input a was modified in place")
    np.testing.assert_array_equal(b, b_copy, err_msg="input b was modified in place")


def test_float_b_whose_largest_entry_is_not_one_is_not_mutated():
    a = np.array([1.0, 2.0, 3.0])
    # Largest entry is 2, so any scaling of b by it would change the caller's array.
    b = np.array([0.0, 2.0, 0.0])
    b_copy = b.copy()
    # a·b / b·b = 4 / 4 = 1, so the projection is b itself: [0, 2, 0].
    np.testing.assert_allclose(
        vector_projection(a, b),
        [0.0, 2.0, 0.0],
        atol=1e-12,
        err_msg="wrong projection onto a float64 b that is not a unit direction",
    )
    np.testing.assert_array_equal(
        b,
        b_copy,
        err_msg="input b was modified in place when its largest entry is not 1",
    )


def test_zero_dimensional_input_raises_value_error():
    _raised_value_error(
        lambda: vector_projection(np.array(3.0), [1.0, 2.0]),
        "an input that is not one-dimensional must raise ValueError",
    )
