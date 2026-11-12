from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from submission import dense_forward

from dogma_data import rng as data_rng
from dogma_data import sample_rows

FINITE = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


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


def _real_rows(gen, n, d):
    """n real rows with d standardised features: breast_cancer, or digits when d is above 30.

    The features are correlated and skewed, and many values repeat, as measured data does.
    """
    name = "breast_cancer" if d <= 30 else "digits"
    return sample_rows(name, gen, n, d, standardize=True)[0]


def _call(X, W, b, mistake):
    """Call dense_forward on a valid input. The message never includes the inputs or shapes."""
    try:
        result = dense_forward(X, W, b)
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__}")
    assert isinstance(result, np.ndarray), "return a NumPy array, not another container"
    assert result.dtype == np.float64, "output dtype must be float64"
    assert result.shape == (np.shape(X)[0], np.shape(W)[1]), (
        "output shape must preserve the batch axis and output-unit axis"
    )
    return result


def _expect_value_error(X, W, b, mistake):
    try:
        dense_forward(X, W, b)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead of ValueError")
    pytest.fail(mistake)


def _assert_close(actual, desired, rtol=1e-10, atol=1e-10, err_msg=""):
    """Like np.testing.assert_allclose, but the first line of the failure names the mistake."""
    got = np.asarray(actual, dtype=np.float64)
    want = np.asarray(desired, dtype=np.float64)
    if got.shape != want.shape:
        raise AssertionError(f"{err_msg}: shape mismatch")
    if not np.allclose(got, want, rtol=rtol, atol=atol):
        raise AssertionError(err_msg)


def _loop_reference(X, W, b):
    """Y[i, j] = b[j] + sum_k X[i, k] * W[k, j], one entry at a time (no matrix product)."""
    n, d_in = X.shape
    d_out = W.shape[1]
    out = np.zeros((n, d_out))
    for i in range(n):
        for j in range(d_out):
            total = float(b[j])
            for k in range(d_in):
                total += float(X[i, k]) * float(W[k, j])
            out[i, j] = total
    return out


@pytest.mark.parametrize(
    ("n", "d_in", "d_out"),
    [(1, 3, 2), (4, 4, 4), (3, 2, 5), (6, 3, 2), (2, 7, 9)],
    ids=["single-row", "square", "wide-output", "tall-batch", "many-units"],
)
def test_random_batches_match_an_explicit_per_entry_reference(n, d_in, d_out):
    rng = data_rng(f"dense-forward-p01:batches:{n}:{d_in}:{d_out}")
    X = _real_rows(rng, n, d_in)
    W = _real_normal(rng, (d_in, d_out))
    b = _real_normal(rng, d_out)
    _assert_close(
        _call(X, W, b, "the batch result could not be computed"),
        _loop_reference(X, W, b),
        err_msg="result differs from the per-entry reference: check the product axes and the bias",
    )


def test_square_weights_are_applied_as_x_times_w_not_transposed():
    rng = data_rng("dense-forward-p01:square")
    X = _real_rows(rng, 5, 3)
    W = _real_normal(rng, (3, 3))
    b = np.zeros(3)
    _assert_close(
        _call(X, W, b, "square weights could not be applied"),
        _loop_reference(X, W, b),
        err_msg="square weights were applied transposed: the layout is (d_in, d_out), so use X @ W",
    )


def test_bias_is_added_once_per_output_unit_to_every_row():
    # n equals d_out here: a bias broadcast along the wrong axis is silent in this case.
    rng = data_rng("dense-forward-p01:bias")
    X = _real_rows(rng, 3, 2)
    W = _real_normal(rng, (2, 3))
    b = np.array([10.0, 20.0, 30.0])
    _assert_close(
        _call(X, W, b, "the bias case could not be computed"),
        _loop_reference(X, W, b),
        err_msg="bias on the wrong axis: b has one entry per output unit, shared by all rows",
    )


@pytest.mark.parametrize(
    ("bias_shape"),
    [(1, 3), (3, 1)],
    ids=["row-shaped", "column-shaped"],
)
def test_bias_must_be_exactly_one_dimensional(bias_shape):
    X = np.zeros((3, 2))
    W = np.zeros((2, 3))
    _expect_value_error(
        X,
        W,
        np.zeros(bias_shape),
        "a row- or column-shaped bias was broadcast instead of rejected: "
        "b must be exactly (d_out,)",
    )


@pytest.mark.parametrize(
    ("X", "W"),
    [(np.zeros(3), np.zeros((3, 2))), (np.zeros((2, 3)), np.zeros(3))],
    ids=["x-1d", "w-1d"],
)
def test_one_dimensional_x_or_w_raises_value_error(X, W):
    _expect_value_error(X, W, np.zeros(2), "a 1-D X or W was accepted instead of rejected")


def test_pytorch_layout_weight_raises_value_error_on_feature_mismatch():
    # PyTorch stores (d_out, d_in) = (4, 3). Passing it as (d_in, d_out) must not be accepted.
    X = np.zeros((2, 3))
    W_torch_layout = np.zeros((4, 3))
    _expect_value_error(
        X,
        W_torch_layout,
        np.zeros(4),
        "a weight in PyTorch (out, in) layout was accepted; transpose it to (d_in, d_out) first",
    )


def test_empty_batch_gives_zero_rows_of_output_width():
    got = _call(np.zeros((0, 3)), np.ones((3, 4)), np.arange(4.0), "empty batch failed")
    assert got.shape == (0, 4), "shape mismatch: no examples should give (0, d_out)"


def test_zero_width_inputs_return_the_bias_on_every_row():
    # An empty sum over zero features is 0, so the output is the bias repeated per row.
    b = np.array([1.5, -2.0, 0.25])
    got = _call(np.zeros((4, 0)), np.zeros((0, 3)), b, "zero-width features failed")
    _assert_close(got, np.tile(b, (4, 1)), err_msg="an empty feature sum is zero: output the bias")


def test_bias_length_mismatch_is_rejected_even_when_broadcastable():
    _expect_value_error(
        np.zeros((2, 3)),
        np.zeros((3, 4)),
        np.ones(1),
        "bias length must equal output width even when NumPy could broadcast it",
    )


def test_zero_output_width_preserves_the_batch_axis():
    got = _call(np.ones((3, 2)), np.empty((2, 0)), np.empty(0), "zero output width failed")
    assert got.shape == (3, 0), "zero output width must preserve the batch axis"


def test_integer_inputs_return_float64():
    X = np.array([[1, 2], [3, 4]])
    W = np.array([[1, 0], [0, 2]])
    b = np.array([1, -1])
    got = _call(X, W, b, "integer input failed")
    assert isinstance(got, np.ndarray), "return a NumPy array, not a list"
    assert got.dtype == np.float64, "dtype mismatch: integer input must return float64"
    _assert_close(got, [[2.0, 3.0], [4.0, 7.0]], err_msg="integer input gave wrong values")


def test_large_and_small_magnitudes_keep_float64_precision():
    rng = data_rng("dense-forward-p01:large")
    X = _real_rows(rng, 4, 3)
    W = _real_normal(rng, (3, 2))
    b = _real_normal(rng, 2)
    _assert_close(
        _call(X * 1e8, W, b * 1e8, "large magnitudes failed"),
        _loop_reference(X * 1e8, W, b * 1e8),
        rtol=1e-12,
        atol=0.0,
        err_msg="precision lost at large magnitudes: compute in float64",
    )
    _assert_close(
        _call(X * 1e-8, W, b * 1e-8, "small magnitudes failed"),
        _loop_reference(X * 1e-8, W, b * 1e-8),
        rtol=1e-12,
        atol=1e-30,
        err_msg="precision lost at small magnitudes: compute in float64",
    )


def test_inputs_are_not_mutated():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    W = np.array([[0.5, -1.0], [2.0, 0.0]])
    b = np.array([0.25, -0.75])
    X0, W0, b0 = X.copy(), W.copy(), b.copy()
    _call(X, W, b, "input preservation case failed")
    np.testing.assert_array_equal(X, X0, err_msg="X was modified in place")
    np.testing.assert_array_equal(W, W0, err_msg="W was modified in place")
    np.testing.assert_array_equal(b, b0, err_msg="b was modified in place")


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    arrays(np.float64, (3, 2), elements=FINITE),
    arrays(np.float64, (2, 3), elements=FINITE),
    arrays(np.float64, (3,), elements=FINITE),
)
def test_each_row_depends_only_on_its_own_example(X, W, b):
    # Dense layers act row by row: the output for one row must not depend on the other rows.
    full = _call(X, W, b, "the batch could not be computed")
    for i in range(X.shape[0]):
        single = _call(X[i : i + 1], W, b, "a single row could not be computed")
        _assert_close(
            full[i],
            single[0],
            rtol=1e-12,
            atol=1e-12,
            err_msg="a row's output depends on the other rows in the batch",
        )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    arrays(np.float64, (4, 3), elements=FINITE),
    arrays(np.float64, (3, 2), elements=FINITE),
    arrays(np.float64, (2,), elements=FINITE),
)
def test_bias_is_the_difference_between_with_and_without_bias(X, W, b):
    # The bias enters once, unscaled by the inputs: Y(b) - Y(0) must equal b on every row.
    with_bias = _call(X, W, b, "the batch with bias could not be computed")
    without = _call(X, W, np.zeros(2), "the batch without bias could not be computed")
    _assert_close(
        with_bias - without,
        np.tile(b, (4, 1)),
        rtol=1e-10,
        atol=1e-10,
        err_msg="the bias is not added once to each row: Y(b) - Y(0) should equal b",
    )


def test_list_inputs_are_accepted_as_array_like():
    # X, W and b given as plain Python lists are array-like, so they must give the same answer
    # as arrays. Row [1, 2] times W = [[1, 0], [0, 3]] is [1, 6]; adding b = [0.5, -1]
    # gives [1.5, 5].
    got = _call([[1.0, 2.0]], [[1.0, 0.0], [0.0, 3.0]], [0.5, -1.0], "list inputs failed")
    _assert_close(got, [[1.5, 5.0]], err_msg="list inputs gave wrong values")
