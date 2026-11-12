import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from submission import gradient_descent

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


def _real_unit(gen, size=None):
    """Real numbers between 0 and 1 where this test used to draw uniform ones.

    Each is the share of a standard normal that lies below a standardised breast_cancer
    measurement, so the values keep the order, the ties and the skew of real measurements.
    """
    return ndtr(_real_normal(gen, size))


def _real_ints(gen, low, high, size=None):
    """Real whole numbers from ``low`` up to ``high - 1``: the band a real measurement falls in.

    The range is cut into ``high - low`` bands of equal normal probability, and each value is
    the band of one standardised measurement. Real measurements are skewed, so the bands are
    not equally full, and neighbouring values tie as real categories and grades do.
    """
    bands = np.floor(_real_unit(gen, size) * (high - low))
    return (low + np.minimum(bands, high - low - 1)).astype(np.int64)


def _real_shares(gen, alpha, size=None):
    """Real probability vectors: positive real numbers (see `_real_unit`) divided by their sum."""
    k = int(np.size(alpha))
    shape = (k,) if size is None else (int(size), k)
    weights = _real_unit(gen, shape) + 1e-3
    return weights / weights.sum(axis=-1, keepdims=True)


def _real_axes(gen, n):
    """The principal axes of real data: eigenvectors of the covariance of n wine measurements.

    Sixty wines and n standardised columns are drawn; the columns of the result are the
    orthonormal directions along which those real measurements vary.
    """
    X, _ = sample_rows("wine", gen, 60, n, standardize=True)
    return np.linalg.eigh(np.atleast_2d(np.cov(X, rowvar=False)))[1]


def _quadratic(seed, d):
    """Symmetric positive definite A with eigenvalues in [1, 10], and a vector b.

    The eigenvectors of A are the principal axes of real wine data and b is one real wine
    (d standardised measurements). The eigenvalues are set by the test, because the step sizes
    and iteration counts the tests use depend on them.
    """
    rng = data_rng(f"gradient-descent-p01:quadratic:{seed}:{d}")
    Q = _real_axes(rng, d)
    eigenvalues = 1.0 + (10.0 - 1.0) * _real_unit(rng, d)
    A = Q @ np.diag(eigenvalues) @ Q.T
    return (A + A.T) / 2, sample_rows("wine", rng, 1, d, standardize=True)[0][0]


def _gradient(A, b):
    """Gradient of f(x) = 1/2 x'Ax - b'x, which is Ax - b."""

    def grad(x):
        return A @ x - b

    return grad


def _objective(A, b, x):
    return 0.5 * x @ A @ x - b @ x


def _parabola(x):
    return 2 * (x - 3)


def _run(grad, x0, lr, **kwargs):
    """Call the learner's function; a stray ValueError is reported as the mistake behind it."""
    try:
        return gradient_descent(grad, x0, lr, **kwargs)
    except ValueError as exc:
        pytest.fail(
            f"a step size inside the stability bound raised ValueError({exc}): "
            "check the update sign (x - lr * grad) and that finite iterates are not rejected"
        )


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    pytest.fail(mistake)


@pytest.mark.parametrize(("d", "seed"), [(1, 1), (2, 2), (3, 3), (5, 4), (8, 5)], ids=list("abcde"))
def test_converges_to_linear_solve_on_random_quadratics(d, seed):
    A, b = _quadratic(seed, d)
    lr = 1.0 / np.linalg.eigvalsh(A).max()
    x, _ = _run(_gradient(A, b), np.zeros(d), lr)
    np.testing.assert_allclose(
        x,
        np.linalg.solve(A, b),
        atol=1e-6,
        err_msg="did not reach the minimiser A^-1 b: step against the gradient and keep "
        "stepping until the gradient-norm tolerance is met",
    )


@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@given(seed=st.integers(0, 10**6), d=st.integers(1, 4), fraction=st.floats(0.1, 1.9))
def test_any_step_inside_the_stability_bound_reaches_the_solution(seed, d, fraction):
    A, b = _quadratic(seed, d)
    lr = fraction / np.linalg.eigvalsh(A).max()
    x, _ = _run(_gradient(A, b), np.zeros(d), lr)
    np.testing.assert_allclose(
        x,
        np.linalg.solve(A, b),
        atol=1e-6,
        err_msg="a step size inside 2/lambda_max does not converge: "
        "the update or the stop rule is wrong",
    )


def test_step_beyond_stability_limit_raises_diverged():
    A, b = _quadratic(7, 3)
    lr = 2.5 / np.linalg.eigvalsh(A).max()
    _expect_value_error(
        lambda: gradient_descent(_gradient(A, b), np.zeros(3), lr),
        "a step size past 2/lambda_max returned a result: iterates that overflow must raise "
        "ValueError('diverged') instead of returning NaN or inf",
    )


def test_diverged_error_message_names_divergence():
    try:
        gradient_descent(_parabola, [0.0], 1.5)
    except ValueError as exc:
        assert "diverged" in str(exc), "the divergence error must carry the message 'diverged'"
    else:
        pytest.fail("a diverging run returned a result instead of raising ValueError")


def _reference_steps(lam, x_start, lr, tol):
    """The same rule as a plain loop: test the gradient, then step, counting the steps taken."""
    x, steps = x_start, 0
    while True:
        g = lam * x
        if abs(g) <= tol:
            return steps
        x = x - lr * g
        steps += 1


def _scaled_identity_grad(lam):
    def grad(x):
        return lam * x

    return grad


def test_exact_step_count_on_one_dimensional_quadratic():
    expected = _reference_steps(4.0, 1.0, 0.1, 1e-8)
    x, n_iters = _run(_scaled_identity_grad(4.0), [1.0], 0.1)
    assert n_iters == expected, (
        "step count is off by one: test the gradient before each update, "
        "so k updates are reported as k"
    )
    np.testing.assert_allclose(
        x, [0.0], atol=1e-7, err_msg="the one-dimensional run did not reach the minimum"
    )


def _reference_steps_nd(A, b, x_start, lr, tol):
    """Plain-loop reference for a quadratic in any dimension, with the Euclidean stop norm."""
    x, steps = np.array(x_start, dtype=np.float64), 0
    while True:
        g = A @ x - b
        if np.sqrt(np.sum(g * g)) <= tol:
            return steps
        x = x - lr * g
        steps += 1


def test_stop_rule_uses_the_euclidean_norm_of_the_gradient():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 1.0])
    expected = _reference_steps_nd(A, b, [0.0, 0.0], 0.3, 0.06)
    _, n_iters = _run(_gradient(A, b), np.zeros(2), 0.3, tol=0.06)
    assert n_iters == expected, (
        "the stop test must compare the Euclidean norm of the gradient with the tol argument: a "
        "different step count means the norm or the tolerance is wrong"
    )


def test_looser_tolerance_stops_after_exactly_fewer_steps():
    expected = _reference_steps(4.0, 1.0, 0.1, 1e-3)
    _, n_iters = _run(_scaled_identity_grad(4.0), [1.0], 0.1, tol=1e-3)
    assert n_iters == expected, "the tolerance argument is ignored or the stop test is misplaced"


def test_max_iters_caps_the_count_and_returns_the_last_iterate():
    x, n_iters = _run(_parabola, [0.0], 0.001, max_iters=7)
    assert n_iters == 7, "the max_iters cap must stop the loop and report that many updates"
    np.testing.assert_allclose(
        x,
        [3.0 - 3.0 * 0.998**7],
        rtol=1e-12,
        err_msg="after max_iters updates the returned iterate must be the one after exactly that "
        "many updates: an extra or a missing update changes it",
    )


def test_start_at_minimum_returns_a_copy_not_the_input():
    A, b = _quadratic(3, 2)
    x_star = np.linalg.solve(A, b)
    x0 = x_star.copy()
    x, n_iters = _run(_gradient(A, b), x0, 0.1)
    assert n_iters == 0, "a start that already meets the tolerance must take 0 updates"
    x[:] = 99.0
    np.testing.assert_array_equal(
        x0,
        x_star,
        err_msg="returned array shares memory with the start point: return a copy",
    )


@pytest.mark.parametrize("bad_lr", [0.0, -0.1, float("nan")], ids=["zero", "negative", "nan"])
def test_invalid_learning_rate_raises_even_when_start_is_optimal(bad_lr):
    _expect_value_error(
        lambda: gradient_descent(_parabola, [3.0], bad_lr),
        "an invalid learning rate was accepted at the minimum: validate lr before the first "
        "gradient step",
    )


def test_non_finite_start_point_raises_value_error():
    _expect_value_error(
        lambda: gradient_descent(_parabola, [float("nan")], 0.1),
        "a NaN start point was accepted: reject non-finite x0 before iterating",
    )


def test_input_start_point_is_not_mutated():
    A, b = _quadratic(11, 3)
    x0 = np.array([1.0, -2.0, 0.5])
    snapshot = x0.copy()
    _run(_gradient(A, b), x0, 0.1, max_iters=50)
    np.testing.assert_array_equal(
        x0,
        snapshot,
        err_msg="the caller's start point was modified in place: update a copy of x0",
    )


def test_integer_start_point_gives_a_float64_result():
    x, _ = _run(lambda x: x - 1.0, [1, 1], 0.5)
    assert x.dtype == np.float64, "an integer start must be converted to float64 before stepping"
    x, _ = _run(lambda x: x - 1.0, [1, 2], 0.5)
    assert x.dtype == np.float64, "an integer start must be converted to float64 before stepping"


def test_ill_conditioned_quadratic_needs_many_more_steps():
    steps = {}
    for kappa in (10.0, 100.0):
        A = np.diag([1.0, kappa])
        b = np.array([1.0, 1.0])
        _, steps[kappa] = _run(_gradient(A, b), np.zeros(2), 2.0 / (kappa + 1.0))
    assert steps[100.0] > 5 * steps[10.0], (
        "step count does not grow with the condition number: an ill-conditioned quadratic "
        "must take many more steps"
    )


def test_fixture_gradient_matches_central_differences():
    A, b = _quadratic(5, 4)
    x = np.array([0.3, -1.2, 0.7, 2.0])
    h = 1e-6
    numeric = np.array(
        [(_objective(A, b, x + h * e) - _objective(A, b, x - h * e)) / (2 * h) for e in np.eye(4)]
    )
    np.testing.assert_allclose(
        _gradient(A, b)(x),
        numeric,
        rtol=1e-6,
        atol=1e-6,
        err_msg="test fixture: the oracle gradient disagrees with central differences",
    )


def test_non_finite_start_point_raises_even_when_the_gradient_is_zero():
    # A zero gradient never fails on its own, so only the start-point check can raise here.
    def zero_grad(x):
        return np.zeros_like(x)

    for x0 in ([float("nan")], [0.0, float("inf")], [float("nan"), 1.0]):
        _expect_value_error(
            lambda x0=x0: gradient_descent(zero_grad, x0, 0.1),
            "a start point with any non-finite entry must raise ValueError before iterating",
        )


def test_iterate_that_overflows_on_the_last_update_raises_diverged():
    # The only allowed update overflows to infinity, so the divergence check must still run.
    _expect_value_error(
        lambda: gradient_descent(_parabola, [0.0], 1e308, max_iters=1),
        "an iterate that becomes infinite on the last update must raise ValueError('diverged')",
    )


def test_iterate_with_one_overflowing_coordinate_raises_diverged():
    # Only the first coordinate overflows; every coordinate of an iterate must be checked.
    _expect_value_error(
        lambda: gradient_descent(lambda x: np.array([-6.0, 0.0]), [0.0, 0.0], 1e308, max_iters=1),
        "an iterate with any non-finite coordinate must raise ValueError('diverged')",
    )


def test_default_update_budget_is_ten_thousand():
    # A constant gradient of norm 1 never meets the tolerance, so the default budget is used up.
    _, n_iters = gradient_descent(lambda x: np.array([1.0]), [0.0], 1e-6)
    assert n_iters == 10_000, "the default max_iters must allow exactly 10000 updates"
