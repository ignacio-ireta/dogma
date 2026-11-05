from __future__ import annotations

import numpy as np
import pytest
from submission import train_mlp


def checked(call):
    try:
        result = call()
    except Exception:
        pytest.fail("valid input failed: honor the documented interface", pytrace=False)
    if not isinstance(result, tuple | list) or len(result) != 2:
        pytest.fail("return a parameter checkpoint and loss history", pytrace=False)
    params, losses = result
    if not isinstance(params, tuple | list) or len(params) != 4:
        pytest.fail("return four parameter arrays in W1, b1, W2, b2 order", pytrace=False)
    arrays = (*params, losses)
    if any(not isinstance(array, np.ndarray) for array in arrays):
        pytest.fail("return NumPy parameter arrays and loss history", pytrace=False)
    if any(array.dtype != np.float64 for array in arrays):
        pytest.fail("return float64 parameter arrays and loss history", pytrace=False)
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


def initial(d, h, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((d, h)) / np.sqrt(d),
        np.zeros(h),
        rng.standard_normal((h, 1)) / np.sqrt(h),
        np.zeros(1),
    )


def objective(X, y, params):
    a, b, c, d = params
    # Explicit neurons rather than a backward implementation.
    prediction = np.array(
        [
            sum(
                np.tanh(sum(row[j] * a[j, k] for j in range(X.shape[1])) + b[k]) * c[k, 0]
                for k in range(len(b))
            )
            + d[0]
            for row in X
        ]
    )
    return float(np.mean((prediction - y[:, 0]) ** 2) / 2)


def finite_grads(X, y, params):
    grads = []
    for p in params:
        grad = np.zeros_like(p)
        for index in np.ndindex(p.shape):
            old = p[index]
            p[index] = old + 1e-5
            plus = objective(X, y, params)
            p[index] = old - 1e-5
            minus = objective(X, y, params)
            p[index] = old
            grad[index] = (plus - minus) / 2e-5
        grads.append(grad)
    return grads


def test_initialization_and_initial_loss_oracle():
    X = np.array([[1.0, -1.0], [2.0, 0.5]])
    y = np.array([[0.3], [-0.2]])
    params, losses = checked(lambda: train_mlp(X, y, 3, 0, 0.1, 17))
    oracle = initial(2, 3, 17)
    for a, b in zip(params, oracle, strict=True):
        close(a, b)
    close(losses, [objective(X, y, oracle)])


def test_one_step_all_parameter_gradients_finite_difference():
    X = np.array([[0.1, 0.8], [-0.4, 0.2], [0.3, -0.9]])
    y = np.array([[0.2], [0.7], [-0.3]])
    before = initial(2, 2, 23)
    numerical = finite_grads(X, y, before)
    after, losses = checked(lambda: train_mlp(X, y, 2, 1, 0.03, 23))
    for old, new, grad in zip(before, after, numerical, strict=True):
        close((old - new) / 0.03, grad)
    close(losses, [objective(X, y, before), objective(X, y, after)])


def test_multiple_steps_match_numeric_gradient_descent():
    X = np.array([[-0.5], [0.8]])
    y = np.array([[0.1], [-0.2]])
    expected = initial(1, 2, 9)
    history = [objective(X, y, expected)]
    for _ in range(3):
        grads = finite_grads(X, y, expected)
        expected = tuple(p - 0.05 * g for p, g in zip(expected, grads, strict=True))
        history.append(objective(X, y, expected))
    got, losses = checked(lambda: train_mlp(X, y, 2, 3, 0.05, 9))
    for a, b in zip(got, expected, strict=True):
        close(a, b)
    close(losses, history)


def test_repeated_seed_and_no_global_rng_or_input_mutation():
    X = np.array([[1.0], [2.0]])
    y = np.array([[0.0], [1.0]])
    before = X.copy(), y.copy()
    state = np.random.get_state()
    one = checked(lambda: train_mlp(X, y, 2, 4, 0.1, 2))
    two = checked(lambda: train_mlp(X, y, 2, 4, 0.1, 2))
    for a, b in zip(one[0], two[0], strict=True):
        close(a, b)
    close(one[1], two[1])
    close(X, before[0])
    close(y, before[1])
    current = np.random.get_state()
    assert np.array_equal(state[1], current[1]) and state[2:] == current[2:], (
        "do not change global RNG state"
    )


def test_dtype_singleton_and_fit_constant():
    params, losses = checked(
        lambda: train_mlp(np.zeros((1, 1), dtype=int), np.ones((1, 1), dtype=int), 1, 80, 0.1, 0)
    )
    assert all(a.dtype == np.float64 for a in (*params, losses)), "return float64 arrays"
    assert losses[-1] < 0.001, "SGD should fit this bounded constant target"


def test_invalid_input_shapes_values_and_hyperparameters():
    X, y = np.ones((2, 2)), np.ones((2, 1))
    for bad in (
        np.empty((0, 2)),
        np.ones(2),
        np.full((2, 2), np.nan),
        np.full((2, 2), 11.0),
        np.ones((2, 2), dtype=complex),
    ):
        rejects(lambda bad=bad: train_mlp(bad, y, 2, 1, 0.1, 0))
    rejects(lambda: train_mlp(X, np.ones(2), 2, 1, 0.1, 0))
    for h, s, lr, seed in [
        (0, 1, 0.1, 0),
        (2, -1, 0.1, 0),
        (2, 201, 0.1, 0),
        (2, 1, 0.0, 0),
        (2, 1, np.nan, 0),
        (2, 1, 0.1, -1),
        (2, 1, 0.1, 1.5),
    ]:
        rejects(lambda h=h, s=s, lr=lr, seed=seed: train_mlp(X, y, h, s, lr, seed))


def test_nonfinite_training_trajectory_raises_value_error():
    with np.errstate(over="ignore", invalid="ignore"):
        rejects(lambda: train_mlp(np.full((2, 2), 10.0), np.full((2, 1), 10.0), 32, 200, 1.0, 0))


def test_hidden_width_at_upper_bound_is_accepted():
    params, _ = checked(lambda: train_mlp(np.ones((2, 2)), np.ones((2, 1)), 32, 0, 0.1, 0))
    assert params[0].shape == (2, 32), (
        "a 32-unit hidden layer is within the contract but was rejected"
    )


def test_seed_at_upper_bound_is_accepted_and_above_it_is_rejected():
    _, losses = checked(lambda: train_mlp(np.ones((2, 2)), np.ones((2, 1)), 2, 0, 0.1, 2**32 - 1))
    assert losses.shape == (1,), "the largest 32-bit seed is within the contract but was rejected"
    rejects(lambda: train_mlp(np.ones((2, 2)), np.ones((2, 1)), 2, 1, 0.1, 2**32))


def test_dimensions_at_the_32_limit_are_accepted():
    _, losses = checked(
        lambda: train_mlp(np.full((32, 1), 0.5), np.full((32, 1), 0.2), 1, 0, 0.1, 0)
    )
    assert losses.shape == (1,), "a 32-row training set is within the contract but was rejected"
    _, losses = checked(lambda: train_mlp(np.full((1, 32), 0.5), np.ones((1, 1)), 1, 0, 0.1, 0))
    assert losses.shape == (1,), "a 32-feature input is within the contract but was rejected"


def test_empty_dimensions_are_rejected_as_value_error():
    rejects(lambda: train_mlp(np.empty((2, 0)), np.ones((2, 1)), 2, 0, 0.1, 0))
    rejects(lambda: train_mlp(np.empty((0, 2)), np.empty((0, 1)), 2, 0, 0.1, 0))


def test_target_must_be_a_column_matching_the_rows():
    X = np.ones((2, 2))
    rejects(lambda: train_mlp(X, np.ones((1, 1)), 2, 0, 0.1, 0))
    rejects(lambda: train_mlp(X, np.ones(2), 2, 0, 0.1, 0))


def test_one_large_positive_entry_in_inputs_is_rejected():
    rejects(lambda: train_mlp(np.array([[11.0, 1.0], [0.5, 0.2]]), np.ones((2, 1)), 2, 0, 0.1, 0))


def test_one_large_negative_entry_in_inputs_is_rejected():
    rejects(lambda: train_mlp(np.array([[-11.0, 1.0], [0.5, 0.2]]), np.ones((2, 1)), 2, 0, 0.1, 0))
