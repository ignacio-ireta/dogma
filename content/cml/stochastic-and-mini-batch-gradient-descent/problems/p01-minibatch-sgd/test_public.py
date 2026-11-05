import numpy as np
import pytest
from submission import minibatch_sgd


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


X_TOY = [[1.0], [2.0], [3.0], [4.0]]
Y_TOY = [2.0, 4.0, 6.0, 8.0]


def test_single_full_batch_step_matches_hand_computation():
    w = minibatch_sgd(X_TOY, Y_TOY, lr=0.1, batch_size=4, epochs=1, seed=0)
    np.testing.assert_allclose(
        w,
        [1.5],
        atol=1e-12,
        err_msg="one full-batch step from zero: gradient is -15 and lr = 0.1, so w = 1.5",
    )


def test_two_batches_per_epoch_follow_the_permutation_protocol():
    perm = np.random.default_rng(0).permutation(4)
    w = np.zeros(1)
    for rows in (perm[:2], perm[2:]):
        Xb = np.array(X_TOY)[rows]
        yb = np.array(Y_TOY)[rows]
        w = w - 0.1 * (Xb.T @ (Xb @ w - yb) / len(rows))
    np.testing.assert_allclose(
        minibatch_sgd(X_TOY, Y_TOY, lr=0.1, batch_size=2, epochs=1, seed=0),
        w,
        rtol=1e-12,
        err_msg="the batches must be consecutive slices of rng.permutation(n) for this seed",
    )


def test_zero_epochs_returns_zero_weights():
    w = minibatch_sgd(X_TOY, Y_TOY, lr=0.1, batch_size=2, epochs=0, seed=0)
    np.testing.assert_array_equal(w, [0.0], err_msg="no epochs means no updates from zeros")


def test_batch_size_outside_one_to_n_raises_value_error():
    _raised_value_error(
        lambda: minibatch_sgd(X_TOY, Y_TOY, lr=0.1, batch_size=0, epochs=1, seed=0),
        "batch_size 0 must raise ValueError: batch_size must be at least 1",
    )
    _raised_value_error(
        lambda: minibatch_sgd(X_TOY, Y_TOY, lr=0.1, batch_size=5, epochs=1, seed=0),
        "batch_size above n must raise ValueError: batch_size must be at most n",
    )
