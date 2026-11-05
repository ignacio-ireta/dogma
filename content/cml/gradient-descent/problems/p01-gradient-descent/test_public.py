import numpy as np
import pytest
from submission import gradient_descent


def _parabola(x):
    return 2 * (x - 3)


def _run(grad, x0, lr, **kwargs):
    """Call the learner's function; a stray ValueError is reported as the mistake behind it."""
    try:
        return gradient_descent(grad, x0, lr, **kwargs)
    except ValueError as exc:
        pytest.fail(
            f"a step size inside the stability bound raised ValueError({exc}): "
            "check the update sign (x - lr * grad)"
        )


def test_one_dimensional_quadratic_reaches_its_minimum():
    x, n_iters = _run(_parabola, [0.0], lr=0.25)
    np.testing.assert_allclose(
        x,
        [3.0],
        atol=1e-6,
        err_msg="the iterates did not reach the minimum x = 3: check the update sign, and that the "
        "stop test compares the gradient norm with tol, not its square",
    )
    assert n_iters > 0, "a start away from the minimum needs at least one update"


def test_half_step_lands_on_minimum_in_one_update():
    x, n_iters = _run(_parabola, [0.0], lr=0.5)
    np.testing.assert_allclose(
        x,
        [3.0],
        atol=1e-12,
        err_msg="with lr = 1/2 one update lands exactly on x = 3: check the update scale",
    )
    assert n_iters == 1, "step count wrong: one update must be reported as 1"


def test_start_at_minimum_takes_no_steps():
    x, n_iters = gradient_descent(_parabola, [3.0], lr=0.25)
    assert n_iters == 0, "a start that already meets the tolerance must take 0 updates"
    np.testing.assert_allclose(x, [3.0], atol=1e-12)


def test_zero_learning_rate_raises_value_error():
    try:
        gradient_descent(_parabola, [0.0], lr=0.0)
    except ValueError:
        return
    pytest.fail("a zero learning rate must raise ValueError: check lr > 0 before the first update")


def test_step_past_stability_limit_raises_diverged():
    try:
        gradient_descent(_parabola, [0.0], lr=1.5)
    except ValueError as exc:
        if "diverged" in str(exc):
            return
        pytest.fail(f"a step past the stability limit must raise ValueError('diverged'); got {exc}")
    pytest.fail("a step past the stability limit returned a result: check for non-finite iterates")


def test_two_dimensional_quadratic_matches_linear_solve():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 1.0])
    x, _ = _run(lambda x: A @ x - b, [0.0, 0.0], lr=0.3)
    np.testing.assert_allclose(
        x,
        np.linalg.solve(A, b),
        atol=1e-6,
        err_msg="the minimiser of 1/2 x'Ax - b'x solves Ax = b: check the update",
    )
