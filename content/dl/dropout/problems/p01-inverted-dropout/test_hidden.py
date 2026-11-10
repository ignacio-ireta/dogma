from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import dropout


def checked(call):
    try:
        result = call()
    except Exception:
        pytest.fail("valid input failed: honor the documented interface", pytrace=False)
    if not isinstance(result, tuple | list) or len(result) != 3:
        pytest.fail("return output, input gradient and multiplier arrays", pytrace=False)
    if any(not isinstance(array, np.ndarray) for array in result):
        pytest.fail("return NumPy dropout arrays", pytrace=False)
    if any(array.dtype != np.float64 for array in result):
        pytest.fail("return float64 dropout arrays", pytrace=False)
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
@given(
    st.integers(1, 8),
    st.integers(1, 8),
    st.integers(0, 1000),
    st.sampled_from([0.0, 0.2, 0.5, 0.95]),
)
def test_random_seed_mask_oracle(n, d, seed, p):
    X = np.arange(n * d, dtype=float).reshape(n, d)
    G = X + 1
    expected = (np.random.default_rng(seed).random((n, d)) >= p).astype(float) / (1 - p)
    out, dx, mask = checked(lambda: dropout(X, G, p, True, seed))
    close(mask, expected)
    close(out, X * expected)
    close(dx, G * expected)


def test_fixed_mask_gradient_finite_difference():
    X = np.array([[0.2, -0.8], [1.4, 0.7]])
    G = np.array([[0.3, 0.5], [-0.2, 0.8]])
    _, dx, _ = checked(lambda: dropout(X, G, 0.4, True, 19))
    numeric = np.zeros_like(X)
    for index in np.ndindex(X.shape):
        old = X[index]
        X[index] = old + 1e-5
        plus = float(np.sum(checked(lambda: dropout(X, G, 0.4, True, 19))[0] * G))
        X[index] = old - 1e-5
        minus = float(np.sum(checked(lambda: dropout(X, G, 0.4, True, 19))[0] * G))
        X[index] = old
        numeric[index] = (plus - minus) / 2e-5
    close(dx, numeric)


def test_eval_is_identity_for_every_seed():
    X = np.array([[2, -1]], dtype=int)
    G = np.array([[3, 5]], dtype=int)
    out, dx, mask = checked(lambda: dropout(X, G, 0.95, False, 987))
    close(out, X)
    close(dx, G)
    close(mask, [[1.0, 1.0]])
    assert all(a.dtype == np.float64 for a in (out, dx, mask)), "return float64 arrays"
    assert not np.shares_memory(out, X) and not np.shares_memory(dx, G), (
        "return independent eval copies"
    )


def test_repeat_seed_global_rng_and_no_mutation():
    X = np.arange(12.0).reshape(3, 4)[:, ::2]
    G = np.ones_like(X)
    old = X.copy(), G.copy()
    state = np.random.get_state()
    first = checked(lambda: dropout(X, G, 0.3, True, 7))
    second = checked(lambda: dropout(X, G, 0.3, True, 7))
    for a, b in zip(first, second, strict=True):
        close(a, b)
    close(X, old[0])
    close(G, old[1])
    current = np.random.get_state()
    assert np.array_equal(state[1], current[1]) and state[2:] == current[2:], (
        "do not modify global RNG state"
    )


def test_float_eval_outputs_are_independent_storage():
    X = np.array([[0.2, -0.7]])
    G = np.array([[1.5, 2.5]])
    out, dx, mask = checked(lambda: dropout(X, G, 0.7, False, 12))
    arrays = (X, G, out, dx, mask)
    for i, left in enumerate(arrays):
        for right in arrays[i + 1 :]:
            assert not np.shares_memory(left, right), "return independent eval copies"
    out[0, 0] = 99.0
    dx[0, 0] = 99.0
    mask[0, 0] = 99.0
    close(X, [[0.2, -0.7]])
    close(G, [[1.5, 2.5]])


def test_large_small_magnitudes_and_all_dropped():
    X = np.array([[1e6, 1e-8]])
    G = -X
    out, dx, mask = checked(lambda: dropout(X, G, 0.95, True, 0))
    close(out, np.zeros_like(X))
    close(dx, np.zeros_like(G))
    close(mask, np.zeros_like(X))
    out, dx, _ = checked(lambda: dropout(X, G, 0.0, True, 0))
    close(out, X)
    close(dx, G)


def test_invalid_shapes_values_probability_and_mode():
    good = np.ones((2, 2))
    for bad in (
        np.empty((0, 2)),
        np.ones(2),
        np.full((2, 2), np.inf),
        np.full((2, 2), 1e7),
        np.ones((2, 2), dtype=complex),
    ):
        rejects(lambda bad=bad: dropout(bad, good, 0.5, True, 0))
    rejects(lambda: dropout(good, np.ones((1, 2)), 0.5, True, 0))
    for p in (-0.1, 1.0, np.nan):
        rejects(lambda p=p: dropout(good, good, p, True, 0))
    rejects(lambda: dropout(good, good, 0.5, 1, 0))  # type: ignore[arg-type]  # invalid mode
    rejects(lambda: dropout(good, good, 0.5, True, -1))


def test_seed_at_or_above_two_pow_32_is_rejected():
    good = np.ones((2, 2))
    rejects(lambda: dropout(good, good, 0.5, True, 2**32))


def test_negative_seed_is_rejected_in_evaluation_mode():
    good = np.ones((2, 2))
    rejects(lambda: dropout(good, good, 0.5, False, -1))


def test_boolean_seed_is_rejected():
    good = np.ones((2, 2))
    rejects(lambda: dropout(good, good, 0.5, True, True))


def test_largest_valid_seeds_are_accepted_and_match_the_draw():
    X = np.arange(6.0).reshape(2, 3)
    G = X + 1
    for seed in (2**31, 2**32 - 1):
        expected = (np.random.default_rng(seed).random(X.shape) >= 0.3).astype(float) / (1 - 0.3)
        _, _, mask = checked(lambda seed=seed: dropout(X, G, 0.3, True, seed))
        close(mask, expected)


def test_empty_one_dimensional_or_too_tall_matrices_are_rejected():
    for bad in (np.empty((0, 2)), np.ones(3), np.ones((65, 1))):
        rejects(lambda bad=bad: dropout(bad, bad, 0.5, True, 0))


def test_sixty_four_long_axis_is_accepted():
    for shape in ((64, 1), (1, 64)):
        X = np.ones(shape)
        G = np.ones(shape)
        out, dx, _mask = checked(lambda X=X, G=G: dropout(X, G, 0.0, True, 0))
        close(out, X)
        close(dx, G)


def test_partially_non_finite_matrix_is_rejected():
    rejects(lambda: dropout(np.array([[1.0, np.nan]]), np.ones((1, 2)), 0.5, True, 0))


def test_large_negative_entry_is_rejected_by_magnitude():
    rejects(lambda: dropout(np.array([[-1e7, 1.0]]), np.ones((1, 2)), 0.5, True, 0))


def test_one_oversized_entry_among_normal_ones_is_rejected():
    rejects(lambda: dropout(np.array([[1.0, 1e7]]), np.ones((1, 2)), 0.5, True, 0))


def test_draw_equal_to_drop_probability_is_kept():
    # The rule is M = (U >= p). An entry whose draw equals p exactly must be kept, not dropped.
    seed = 5
    U = np.random.default_rng(seed).random((3, 4))
    k = int(np.flatnonzero(U.ravel() <= 0.95)[0])
    p = float(U.ravel()[k])
    X = np.ones((3, 4))
    G = np.ones((3, 4))
    _, _, mask = checked(lambda: dropout(X, G, p, True, seed))
    expected = (p <= U).astype(float) / (1 - p)
    if not np.allclose(mask, expected, atol=1e-8, rtol=1e-6):
        pytest.fail(
            "a draw equal to the drop probability must be kept: the rule is draw >= p",
            pytrace=False,
        )


def test_entry_just_above_the_magnitude_limit_is_rejected():
    # Contract: entries have absolute value at most 1e6, so an entry of 1.5e6 must be rejected.
    rejects(lambda: dropout(np.array([[1.5e6, 1.0]]), np.ones((1, 2)), 0.5, True, 0))
