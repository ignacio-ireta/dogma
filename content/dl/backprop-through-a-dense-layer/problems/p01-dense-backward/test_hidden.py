from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import dense_backward


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
    rng = np.random.default_rng(seed)
    X, W, G = rng.normal(size=(n, d)), rng.normal(size=(d, k)), rng.normal(size=(n, k))
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
    rng = np.random.default_rng(83)
    X, W, G = rng.normal(size=(2, 3)), rng.normal(size=(3, 2)), rng.normal(size=(2, 2))
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
