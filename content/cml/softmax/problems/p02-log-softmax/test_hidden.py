import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import log_softmax as scipy_log_softmax
from submission import log_softmax

from dogma_data import rng, sample_rows

FINITE = st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False)


def _real_scores(key, shape):
    """Real numbers of the given shape, standing in for logits.

    The values are standardised breast_cancer measurements, tripled: skewed, correlated along
    the last axis and with a few entries far larger than the rest, as the scores of a trained
    classifier are.
    """
    shape = tuple(shape)
    rows = math.prod(shape[:-1])
    drawn, _ = sample_rows("breast_cancer", rng(key), rows, shape[-1], standardize=True)
    return 3.0 * drawn.reshape(shape)


def _call(logits, **kwargs):
    """Call log_softmax with NumPy warnings turned into errors.

    Hidden failures show only the first line of the message, so each line names the mistake and
    never includes the input, its shape or the expected value.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = log_softmax(logits, **kwargs)
            if not isinstance(got, np.ndarray) or got.dtype != np.float64:
                pytest.fail("return a float64 NumPy array without caller-side conversion")
            return got
    except RuntimeWarning:
        pytest.fail(
            "log_softmax overflowed, or took the log of a zero probability, on finite logits: "
            "compute the log-sum-exp from the maximum-subtracted logits"
        )
    except Exception as exc:
        pytest.fail(
            f"log_softmax raised {type(exc).__name__} on a valid input: check that each reduction "
            "keeps its axis (keepdims=True) and that the shapes broadcast as intended"
        )


def _expect_value_error(logits, **kwargs):
    try:
        log_softmax(logits, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"invalid input must raise ValueError; the function raised {type(exc).__name__}"
        )
    pytest.fail("invalid input must raise ValueError, not return values")


@pytest.mark.parametrize(
    ("shape", "axis"),
    [((5,), -1), ((4, 6), -1), ((4, 6), 0), ((2, 3, 4), 1), ((2, 3, 4), -1)],
    ids=["vector", "matrix-last", "matrix-first", "cube-middle", "cube-last"],
)
def test_random_inputs_match_an_independent_reference(shape, axis):
    z = _real_scores(f"log-softmax-p02:reference:{len(shape)}:{axis}", shape)
    expected = scipy_log_softmax(z, axis=axis)
    np.testing.assert_allclose(
        _call(z, axis=axis),
        expected,
        rtol=1e-12,
        atol=1e-12,
        err_msg="log_softmax disagrees with the reference: check the normalisation axis and the "
        "log-sum-exp",
    )


def test_square_matrix_is_normalised_along_each_axis_correctly():
    z = _real_scores("log-softmax-p02:square", (3, 3)) / 3.0
    for axis in (0, -1):
        expected = scipy_log_softmax(z, axis=axis)
        np.testing.assert_allclose(
            _call(z, axis=axis),
            expected,
            rtol=1e-12,
            atol=1e-12,
            err_msg="a square matrix gives the wrong log-probabilities: the maximum and the sum "
            "must keep their axis (keepdims=True)",
        )


@pytest.mark.parametrize(
    "z",
    [
        np.array([1e4, -1e4, 0.0]),
        np.array([-1e4, -1e4 + 2.0]),
        np.array([[2e4, 0.0], [1e4, 1e4]]),
    ],
    ids=["mixed-extremes", "all-negative", "matrix-extremes"],
)
def test_extreme_logits_match_reference_and_stay_finite(z):
    expected = scipy_log_softmax(z, axis=-1)
    got = _call(z, axis=-1)
    assert np.all(np.isfinite(got)), (
        "extreme logits gave -inf or NaN: a tail log-probability must stay finite"
    )
    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-12,
        atol=1e-9,
        err_msg="extreme logits gave the wrong log-probabilities: use the log-sum-exp, and never "
        "take the log of a probability that has been rounded",
    )


def test_tail_log_probability_is_exact_without_rounding_the_probability_first():
    got = _call(np.array([0.0, -800.0]), axis=-1)
    np.testing.assert_allclose(
        got,
        [0.0, -800.0],
        rtol=1e-12,
        atol=1e-12,
        err_msg="a tail log-probability must come from the logits, not a rounded probability",
    )


def test_rows_with_very_different_scales_do_not_affect_each_other():
    z = np.array([[1e4, 1e4 - 1.0], [0.0, 1.0], [-800.0, -800.0]])
    expected_rows = [scipy_log_softmax(row) for row in z]
    got = _call(z, axis=-1)
    for expected_row, out in zip(expected_rows, got, strict=True):
        np.testing.assert_allclose(
            out,
            expected_row,
            rtol=1e-12,
            atol=1e-9,
            err_msg="rows were normalised together: take the maximum per slice along the axis",
        )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.lists(FINITE, min_size=2, max_size=8), st.floats(min_value=-1e3, max_value=1e3))
def test_adding_a_constant_to_a_slice_leaves_its_log_probabilities_unchanged(values, shift):
    z = np.array(values)
    np.testing.assert_allclose(
        _call(z + shift),
        _call(z),
        rtol=1e-12,
        atol=1e-9,
        err_msg="log_softmax must be invariant to adding a constant to every logit in a slice",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.lists(FINITE, min_size=1, max_size=10))
def test_exponentials_sum_to_one_and_every_entry_is_at_most_zero(values):
    got = _call(np.array(values))
    assert np.all(got <= 1e-15), "log-probabilities must be at most zero"
    assert abs(np.exp(got).sum() - 1.0) <= 1e-12, "exp of log_softmax must sum to 1"


def test_integer_input_returns_float64_and_keeps_the_shape():
    got = _call(np.array([[1, 2, 3], [0, 0, 0]]), axis=-1)
    assert got.dtype == np.float64, "dtype should be float64 even for integer logits"
    assert got.shape == (2, 3), "shape mismatch: log_softmax must keep the shape of its input"


@pytest.mark.parametrize(
    ("logits", "kwargs"),
    [
        (np.array([1.0, np.nan]), {}),
        (np.array([1.0, np.inf]), {}),
        (np.array([-np.inf, 0.0]), {}),
        (np.zeros((3, 0)), {}),
        (np.float64(2.0), {}),
        (np.array([1.0, 2.0]), {"axis": 2}),
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
        "empty-axis",
        "scalar",
        "axis-out-of-range",
    ],
)
def test_invalid_inputs_raise_value_error(logits, kwargs):
    _expect_value_error(logits, **kwargs)


def test_input_array_is_not_mutated():
    z = np.array([[1e4, 0.0], [-3.0, 2.0]])
    before = z.copy()
    _call(z, axis=-1)
    np.testing.assert_array_equal(z, before, err_msg="the input array was modified in place")


def test_common_huge_offset_does_not_erase_the_log_normalizer():
    got = _call(np.full((2, 3), 1e20), axis=-1)
    np.testing.assert_allclose(
        got,
        -np.log(3.0),
        rtol=1e-12,
        atol=1e-12,
        err_msg="a common huge offset erased the log normalizer: stay in shifted coordinates",
    )


def test_opposite_float64_extremes_return_the_representable_limits_without_warnings():
    largest = np.finfo(np.float64).max
    got = _call(np.array([largest, -largest]))
    np.testing.assert_array_equal(
        got,
        [0.0, -np.inf],
        err_msg="an unrepresentable tail log-probability must be -inf without a warning",
    )


def test_list_inputs_give_the_same_result_as_arrays():
    logits = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 0.25]])
    try:
        from_lists = log_softmax(logits.tolist())
    except Exception:
        pytest.fail(
            "a valid call with Python lists failed: the inputs are array-like, so convert them "
            "with np.asarray before using array methods",
            pytrace=False,
        )
    from_arrays = log_softmax(logits)
    np.testing.assert_allclose(
        from_lists,
        from_arrays,
        rtol=1e-9,
        atol=1e-12,
        err_msg="lists and arrays of the same numbers gave different results",
    )
