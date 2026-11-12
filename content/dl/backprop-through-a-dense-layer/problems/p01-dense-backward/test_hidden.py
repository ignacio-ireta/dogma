from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import dense_backward

from dogma_data import rng as data_rng
from dogma_data import sample_rows


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


def checked(call):
    try:
        result = call()
    except Exception:
        pytest.fail("valid input failed: honor the documented interface", pytrace=False)
    if not isinstance(result, tuple | list) or len(result) != 3:
        pytest.fail("return three gradient arrays: dX, dW and db", pytrace=False)
    if any(not isinstance(array, np.ndarray) for array in result):
        pytest.fail("return NumPy gradient arrays", pytrace=False)
    if any(array.dtype != np.float64 for array in result):
        pytest.fail("return float64 gradient arrays", pytrace=False)
    return result


def rejects(call):
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError", pytrace=False)
    pytest.fail("invalid input must raise ValueError", pytrace=False)


def close(actual, expected):
    if np.shape(actual) != np.shape(expected):
        pytest.fail("output shape differs from contract", pytrace=False)
    if not np.allclose(actual, expected, atol=1e-8, rtol=1e-6):
        pytest.fail("numerical result differs from the independent oracle", pytrace=False)


@settings(max_examples=25, deadline=None, report_multiple_bugs=False)
@given(st.integers(1, 5), st.integers(1, 4), st.integers(1, 4), st.integers(0, 999))
def test_random_scalar_loop_oracle(n, d, k, seed):
    rng = data_rng(f"dense-backward-p01:loop-oracle:{seed}")
    X, W, G = _real_rows(rng, n, d), _real_normal(rng, (d, k)), _real_normal(rng, (n, k))
    dx = np.zeros_like(X)
    dw = np.zeros_like(W)
    db = np.zeros(k)
    for i in range(n):
        for j in range(d):
            for c in range(k):
                dx[i, j] += G[i, c] * W[j, c]
                dw[j, c] += X[i, j] * G[i, c]
        for c in range(k):
            db[c] += G[i, c]
    for mode, scale in [("sum", 1), ("mean", n)]:
        out = checked(lambda mode=mode: dense_backward(X, W, G, mode))
        for got, want in zip(out, (dx / scale, dw / scale, db / scale), strict=True):
            close(got, want)


def test_every_gradient_matches_finite_difference():
    rng = data_rng("dense-backward-p01:finite-differences")
    X, W, G = _real_rows(rng, 2, 3), _real_normal(rng, (3, 2)), _real_normal(rng, (2, 2))
    b = np.array([0.2, -0.4])
    for mode, scale in [("sum", 1), ("mean", 2)]:
        grads = checked(lambda mode=mode: dense_backward(X, W, G, mode))
        for variable, grad in zip((X, W, b), grads, strict=True):
            numeric = np.zeros_like(variable)
            for index in np.ndindex(variable.shape):
                old = variable[index]
                variable[index] = old + 1e-5
                plus = float(np.sum((X @ W + b) * G)) / scale
                variable[index] = old - 1e-5
                minus = float(np.sum((X @ W + b) * G)) / scale
                variable[index] = old
                numeric[index] = (plus - minus) / 2e-5
            close(grad, numeric)


def test_integer_strided_inputs_float64_and_no_mutation():
    X = np.arange(12).reshape(3, 4)[:, ::2]
    W = np.ones((2, 2), dtype=int)
    G = np.ones((3, 2), dtype=int)
    originals = [a.copy() for a in (X, W, G)]
    result = checked(lambda: dense_backward(X, W, G))  # type: ignore[arg-type]  # integer casting
    for out in result:
        assert out.dtype == np.float64, "return float64 gradients"
    for a, b in zip((X, W, G), originals, strict=True):
        assert np.array_equal(a, b), "input arrays must not be mutated"


def test_small_and_large_magnitudes():
    for scale in (1e-6, 1e6):
        dx, dw, db = checked(
            lambda scale=scale: dense_backward(
                np.array([[scale]]), np.array([[scale]]), np.array([[scale]])
            )
        )
        close(dx, [[scale * scale]])
        close(dw, [[scale * scale]])
        close(db, [scale])


def test_zero_upstream_is_zero():
    for out in checked(
        lambda: dense_backward(np.ones((2, 3)), np.ones((3, 4)), np.zeros((2, 4)), "mean")
    ):
        assert np.count_nonzero(out) == 0, "zero upstream must give zero gradients"


def test_invalid_shapes_values_and_policy():
    good = (np.ones((2, 3)), np.ones((3, 4)), np.ones((2, 4)))
    for bad in [
        np.empty((0, 3)),
        np.ones(3),
        np.full((2, 3), np.nan),
        np.full((2, 3), 1e7),
        np.ones((2, 3), dtype=complex),
    ]:
        rejects(lambda bad=bad: dense_backward(bad, good[1], good[2]))
    rejects(lambda: dense_backward(good[0], np.ones((4, 4)), good[2]))
    rejects(lambda: dense_backward(good[0], good[1], np.ones((1, 4))))
    rejects(lambda: dense_backward(*good, "median"))


def test_dimension_of_exactly_sixty_four_is_accepted():
    checked(lambda: dense_backward(np.ones((64, 1)), np.ones((1, 1)), np.ones((64, 1))))
    checked(lambda: dense_backward(np.ones((1, 64)), np.ones((64, 1)), np.ones((1, 1))))


def test_dimension_above_sixty_four_is_rejected():
    rejects(lambda: dense_backward(np.ones((65, 1)), np.ones((1, 1)), np.ones((65, 1))))
    rejects(lambda: dense_backward(np.ones((1, 65)), np.ones((65, 1)), np.ones((1, 1))))


def test_value_just_above_the_magnitude_bound_is_rejected():
    rejects(lambda: dense_backward(np.array([[1.5e6]]), np.ones((1, 1)), np.ones((1, 1))))


def test_large_negative_value_is_rejected_by_magnitude():
    rejects(lambda: dense_backward(np.array([[-1e7]]), np.ones((1, 1)), np.ones((1, 1))))


def test_one_oversized_entry_among_normal_ones_is_rejected():
    rejects(lambda: dense_backward(np.array([[1.0, 1e7]]), np.ones((2, 1)), np.ones((1, 1))))


def test_one_non_finite_entry_among_finite_ones_is_rejected():
    rejects(lambda: dense_backward(np.array([[1.0, np.nan]]), np.ones((2, 1)), np.ones((1, 1))))
