import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import expit
from scipy.special import softmax as scipy_softmax
from submission import softmax

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
    """Call softmax with NumPy warnings turned into errors.

    Hidden failures show only the first line of the message, so each line names the mistake and
    never includes the input, its shape or the expected value.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = softmax(logits, **kwargs)
            if not isinstance(got, np.ndarray) or got.dtype != np.float64:
                pytest.fail("return a float64 NumPy array without caller-side conversion")
            return got
    except RuntimeWarning:
        pytest.fail(
            "softmax overflowed or raised an invalid-value warning on finite logits: subtract the "
            "maximum along the axis before exponentiating"
        )
    except Exception as exc:
        pytest.fail(
            f"softmax raised {type(exc).__name__} on a valid input: check that each reduction "
            "keeps "
            "its axis (keepdims=True) and that the shapes broadcast as intended"
        )


def _expect_value_error(logits, **kwargs):
    try:
        softmax(logits, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"invalid input must raise ValueError; the function raised {type(exc).__name__}"
        )
    pytest.fail("invalid input must raise ValueError, not return a distribution")


@pytest.mark.parametrize(
    ("shape", "axis"),
    [((5,), -1), ((4, 6), -1), ((4, 6), 0), ((2, 3, 4), 1), ((2, 3, 4), -1)],
    ids=["vector", "matrix-last", "matrix-first", "cube-middle", "cube-last"],
)
def test_random_inputs_match_an_independent_reference(shape, axis):
    z = _real_scores(f"softmax-p01:reference:{len(shape)}:{axis}", shape)
    expected = scipy_softmax(z, axis=axis)
    np.testing.assert_allclose(
        _call(z, axis=axis),
        expected,
        rtol=1e-12,
        atol=1e-15,
        err_msg="softmax disagrees with the reference: check the normalisation axis and the "
        "exponentials",
    )


def test_square_matrix_is_normalised_row_by_row_not_by_a_misaligned_maximum():
    z = _real_scores("softmax-p01:square", (3, 3)) / 3.0
    expected_last = scipy_softmax(z, axis=-1)
    expected_first = scipy_softmax(z, axis=0)
    np.testing.assert_allclose(
        _call(z, axis=-1),
        expected_last,
        rtol=1e-12,
        atol=1e-15,
        err_msg="a square matrix normalised along its last axis gives the wrong answer: the "
        "maximum must keep its axis (keepdims=True), or it broadcasts along the wrong dimension",
    )
    np.testing.assert_allclose(
        _call(z, axis=0),
        expected_first,
        rtol=1e-12,
        atol=1e-15,
        err_msg="a square matrix normalised along axis 0 gives the wrong answer: the maximum "
        "must keep its axis (keepdims=True)",
    )


@pytest.mark.parametrize(
    "z",
    [
        np.array([1e4, -1e4, 0.0]),
        np.array([-1e4, -1e4 + 2.0]),
        np.array([[1e4, 1e4], [-1e4, 0.0]]),
    ],
    ids=["mixed-extremes", "all-negative", "matrix-extremes"],
)
def test_extreme_logits_match_reference_without_warnings(z):
    expected = scipy_softmax(z, axis=-1)
    got = _call(z, axis=-1)
    assert np.all(np.isfinite(got)), "extreme logits produced NaN or inf"
    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-12,
        atol=1e-15,
        err_msg="extreme logits gave the wrong distribution: subtract the maximum, and never add "
        "a floor to the probabilities",
    )


def test_rows_with_very_different_scales_do_not_affect_each_other():
    z = np.array([[1e4, 1e4 - 1.0], [0.0, 1.0], [-800.0, -800.0]])
    expected_rows = [scipy_softmax(row) for row in z]
    got = _call(z, axis=-1)
    for expected_row, out in zip(expected_rows, got, strict=True):
        np.testing.assert_allclose(
            out,
            expected_row,
            rtol=1e-12,
            atol=1e-15,
            err_msg="rows were normalised together: the maximum must be taken per slice along "
            "the axis, not over the whole array",
        )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.lists(FINITE, min_size=2, max_size=8), st.floats(min_value=-1e3, max_value=1e3))
def test_adding_a_constant_to_a_slice_leaves_it_unchanged(values, shift):
    z = np.array(values)
    np.testing.assert_allclose(
        _call(z + shift),
        _call(z),
        rtol=1e-12,
        atol=1e-15,
        err_msg="softmax must be invariant to adding a constant to every logit in a slice",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.lists(FINITE, min_size=1, max_size=10))
def test_each_slice_is_a_non_negative_distribution(values):
    got = _call(np.array(values))
    assert np.all(got >= 0.0), "softmax values must be non-negative"
    assert abs(got.sum() - 1.0) <= 1e-12, "softmax values must sum to 1 along the axis"


@pytest.mark.parametrize(
    "temperature", [1e-3, 0.5, 1.0, 100.0], ids=["cold", "half", "unit", "hot"]
)
def test_temperature_scaling_matches_the_reference(temperature):
    z = np.array([[2.0, 0.5, -1.0, 1.5], [0.0, 0.0, 1.0, -2.0]])
    expected = scipy_softmax(z / temperature, axis=-1)
    np.testing.assert_allclose(
        _call(z / temperature, axis=-1),
        expected,
        rtol=1e-10,
        atol=1e-15,
        err_msg="temperature-scaled softmax disagrees with the reference",
    )


def test_cold_temperature_concentrates_on_the_unique_maximum():
    z = np.array([2.0, 0.0, -1.0, 1.0])
    got = _call(z / 1e-3)
    np.testing.assert_allclose(
        got,
        [1.0, 0.0, 0.0, 0.0],
        atol=1e-12,
        err_msg="a cold softmax must approach the one-hot maximum",
    )


def test_two_class_softmax_is_the_logistic_function():
    z = np.array([-30.0, -5.0, 0.0, 5.0, 30.0])
    got = _call(np.column_stack([z, np.zeros_like(z)]), axis=-1)
    np.testing.assert_allclose(
        got[:, 0],
        expit(z),
        rtol=1e-10,
        atol=1e-15,
        err_msg="a two-class softmax on [z, 0] must equal the logistic function of z",
    )
    np.testing.assert_allclose(
        got[:, 1],
        expit(-z),
        rtol=1e-10,
        atol=1e-15,
        err_msg="the second class must equal sigma(-z)",
    )


def test_integer_input_returns_float64_and_keeps_the_shape():
    got = _call(np.array([[1, 2, 3], [0, 0, 0]]), axis=-1)
    assert got.dtype == np.float64, "dtype should be float64 even for integer logits"
    assert got.shape == (2, 3), "shape mismatch: softmax must keep the shape of its input"


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


def test_single_entry_slices_are_all_ones():
    got = _call(np.array([[3.0], [-7.0]]), axis=-1)
    np.testing.assert_allclose(
        got,
        [[1.0], [1.0]],
        rtol=0.0,
        atol=0.0,
        err_msg="one logit per slice must give probability 1",
    )


def test_input_array_is_not_mutated():
    z = np.array([[1e4, 0.0], [-3.0, 2.0]])
    before = z.copy()
    _call(z, axis=-1)
    np.testing.assert_array_equal(z, before, err_msg="the input array was modified in place")


def test_opposite_float64_extremes_saturate_without_subtraction_overflow_warning():
    largest = np.finfo(np.float64).max
    np.testing.assert_array_equal(
        _call(np.array([largest, -largest])),
        [1.0, 0.0],
        err_msg="an unrepresentably negative shifted logit must safely give zero probability",
    )


def test_cold_temperature_splits_mass_uniformly_between_tied_maxima():
    np.testing.assert_allclose(
        _call(np.array([2.0, 2.0, -1.0]) / 1e-3),
        [0.5, 0.5, 0.0],
        atol=1e-12,
        err_msg="cold temperature must preserve a uniform split across tied maximum logits",
    )
