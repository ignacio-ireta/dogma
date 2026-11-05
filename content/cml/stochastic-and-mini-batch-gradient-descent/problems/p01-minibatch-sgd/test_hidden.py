import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import minibatch_sgd


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def _data(seed, n, d):
    """Deterministic design with rows of norm about 1, and a random target."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) / np.sqrt(d)
    y = rng.normal(size=n)
    return X, y


def _reference(X, y, lr, batch_size, epochs, *, seed):
    """Row-by-row implementation of the protocol: each batch gradient is an explicit sum."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            rows = perm[start : start + batch_size]
            g = np.zeros(d)
            for i in rows:
                g += X[i] * (X[i] @ w - y[i])
            w = w - lr * g / len(rows)
    return w


@pytest.mark.parametrize(
    ("n", "d", "batch_size", "epochs", "seed"),
    [(4, 1, 2, 3, 0), (7, 2, 3, 5, 1), (10, 3, 4, 4, 2), (9, 2, 1, 3, 3), (12, 3, 5, 2, 4)],
)
def test_matches_row_by_row_reference_of_the_protocol(n, d, batch_size, epochs, seed):
    X, y = _data(seed, n, d)
    w = minibatch_sgd(X, y, 0.05, batch_size, epochs, seed=seed)
    np.testing.assert_allclose(
        w,
        _reference(X, y, 0.05, batch_size, epochs, seed=seed),
        rtol=1e-10,
        atol=1e-12,
        err_msg="weights differ from the protocol: check the permutation draw per epoch, "
        "the consecutive slices (keep the smaller last batch) and the batch-size divisor",
    )


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 1000),
    n=st.integers(1, 12),
    d=st.integers(1, 3),
    epochs=st.integers(0, 4),
    data=st.data(),
)
def test_any_batch_size_matches_the_reference(seed, n, d, epochs, data):
    batch_size = data.draw(st.integers(1, n))
    X, y = _data(seed + 1, n, d)
    np.testing.assert_allclose(
        minibatch_sgd(X, y, 0.05, batch_size, epochs, seed=seed),
        _reference(X, y, 0.05, batch_size, epochs, seed=seed),
        rtol=1e-10,
        atol=1e-12,
        err_msg="a batch size in range gives different weights from the protocol",
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batch_equal_to_n_is_full_batch_gradient_descent(seed):
    n, d, epochs, lr = 8, 3, 6, 0.1
    X, y = _data(seed, n, d)
    w_full = np.zeros(d)
    for _ in range(epochs):
        w_full = w_full - lr * (X.T @ (X @ w_full - y) / n)
    np.testing.assert_allclose(
        minibatch_sgd(X, y, lr, n, epochs, seed=seed),
        w_full,
        rtol=1e-10,
        atol=1e-12,
        err_msg="with batch_size = n, one update per epoch must use the full-data gradient",
    )


def test_one_step_matches_finite_difference_gradient_of_the_batch_loss():
    X, y = _data(9, 3, 2)
    lr = 0.1

    def loss(w):
        return 0.5 * np.mean((X @ w - y) ** 2)

    h = 1e-6
    fd_grad = np.array([(loss(h * e) - loss(-h * e)) / (2 * h) for e in np.eye(2)])
    w_after = minibatch_sgd(X, y, lr, 3, 1, seed=0)
    np.testing.assert_allclose(
        w_after,
        -lr * fd_grad,
        rtol=1e-6,
        atol=1e-9,
        err_msg="the batch gradient disagrees with finite differences of 1/2 mean((Xw-y)^2): "
        "check the factor 1/|b| and the transpose",
    )


def test_single_sample_step_is_lr_times_x_times_y():
    x = np.array([[1.0, -2.0]])
    y = np.array([3.0])
    np.testing.assert_allclose(
        minibatch_sgd(x, y, 0.1, 1, 1, seed=0),
        [0.3, -0.6],
        atol=1e-12,
        err_msg="one sample from zero: the update is lr * x * y",
    )


def test_noise_free_data_converges_to_the_least_squares_solution():
    n, d = 60, 3
    X, _ = _data(3, n, d)
    w_true = np.array([2.0, -1.0, 0.5])
    y = X @ w_true
    lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    w = minibatch_sgd(X, y, 0.2, 7, 200, seed=0)
    np.testing.assert_allclose(
        w,
        lstsq,
        atol=1e-6,
        err_msg="enough epochs on noise-free data must reach the least-squares fit",
    )


def test_same_seed_gives_identical_weights():
    X, y = _data(4, 9, 2)
    a = minibatch_sgd(X, y, 0.05, 2, 3, seed=123)
    b = minibatch_sgd(X, y, 0.05, 2, 3, seed=123)
    np.testing.assert_array_equal(a, b, err_msg="the same seed must reproduce the run exactly")


def test_different_seed_changes_the_shuffle_order():
    X, y = _data(42, 5, 2)
    a = minibatch_sgd(X, y, 0.2, 2, 1, seed=0)
    b = minibatch_sgd(X, y, 0.2, 2, 1, seed=1)
    assert not np.allclose(a, b), "the seed must control the shuffle, so one epoch should differ"


@pytest.mark.parametrize("c", [-2.0, 0.5, 7.0])
def test_scaling_targets_scales_the_weights_linearly(c):
    X, y = _data(6, 7, 2)
    np.testing.assert_allclose(
        minibatch_sgd(X, c * y, 0.05, 3, 4, seed=2),
        c * minibatch_sgd(X, y, 0.05, 3, 4, seed=2),
        rtol=1e-10,
        atol=1e-12,
        err_msg="the gradient is linear in y and w starts at zero, so scaling y scales w",
    )


def test_zero_epochs_returns_float64_zeros_of_length_d():
    X, y = _data(5, 4, 3)
    w = minibatch_sgd(X, y, 0.1, 2, 0, seed=0)
    assert w.shape == (3,), "shape mismatch: the weights have one entry per column of X"
    assert w.dtype == np.float64, "weights must be float64"
    np.testing.assert_array_equal(w, np.zeros(3))


@pytest.mark.parametrize("bad_batch", [0, -1, 9])
def test_batch_size_outside_one_to_n_raises_value_error(bad_batch):
    X, y = _data(7, 8, 2)
    try:
        minibatch_sgd(X, y, 0.1, bad_batch, 1, seed=0)
    except ValueError:
        return
    pytest.fail("a batch size outside 1..n was accepted: check the range of batch_size")


def test_shape_mismatches_raise_value_error():
    X, y = _data(8, 5, 2)
    _raised_value_error(
        lambda: minibatch_sgd(X[:, 0], y, 0.1, 2, 1, seed=0),
        "X must be 2-D: raise ValueError for a 1-D X instead of guessing a shape",
    )
    _raised_value_error(
        lambda: minibatch_sgd(X, y[:4], 0.1, 2, 1, seed=0),
        "y must have one entry per row of X: raise ValueError on a length mismatch",
    )
    _raised_value_error(
        lambda: minibatch_sgd(X, y.reshape(-1, 1), 0.1, 2, 1, seed=0),
        "a column-vector y must be rejected: y must be 1-D, so do not broadcast it",
    )


@pytest.mark.parametrize("bad_lr", [0.0, -0.1])
def test_non_positive_learning_rate_raises_value_error(bad_lr):
    X, y = _data(8, 5, 2)
    _raised_value_error(
        lambda: minibatch_sgd(X, y, bad_lr, 2, 1, seed=0),
        "a non-positive learning rate must raise ValueError: check lr > 0 before the first update",
    )


def test_negative_epochs_raises_value_error():
    X, y = _data(8, 5, 2)
    _raised_value_error(
        lambda: minibatch_sgd(X, y, 0.1, 2, -1, seed=0),
        "negative epochs must raise ValueError: check epochs >= 0",
    )


def test_integer_inputs_give_float64_weights_equal_to_float_inputs():
    X_int = np.array([[1, 0], [0, 2], [1, 1], [2, 1]])
    y_int = np.array([1, 2, 3, 4])
    w_int = minibatch_sgd(X_int, y_int, 0.05, 3, 3, seed=0)
    assert w_int.dtype == np.float64, "integer X and y must be converted to float64"
    np.testing.assert_allclose(
        w_int,
        minibatch_sgd(X_int.astype(float), y_int.astype(float), 0.05, 3, 3, seed=0),
        rtol=1e-12,
        err_msg="integer inputs gave different weights from the same values as floats",
    )


def test_inputs_are_not_mutated():
    X, y = _data(10, 9, 3)
    X_copy, y_copy = X.copy(), y.copy()
    minibatch_sgd(X, y, 0.05, 4, 3, seed=5)
    np.testing.assert_array_equal(X, X_copy, err_msg="X was modified: shuffle indices, not rows")
    np.testing.assert_array_equal(y, y_copy, err_msg="y was modified: shuffle indices, not rows")


def test_batch_size_below_one_raises_even_with_zero_epochs():
    X, y = _data(11, 8, 2)
    _raised_value_error(
        lambda: minibatch_sgd(X, y, 0.1, 0, 0, seed=0),
        "a batch size below 1 must raise ValueError even when there are no epochs: "
        "validate batch_size before the epoch loop",
    )
