import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import constrained_quadratic_min


def _close(actual, desired, *, err_msg, **tolerance):
    """assert_allclose whose failure shows err_msg as the first line, without the values."""
    try:
        np.testing.assert_allclose(actual, desired, err_msg=err_msg, **tolerance)
    except AssertionError:
        pytest.fail(err_msg, pytrace=False)


def _equal(actual, desired, *, err_msg):
    """assert_array_equal whose failure shows err_msg as the first line, without the values."""
    try:
        np.testing.assert_array_equal(actual, desired, err_msg=err_msg)
    except AssertionError:
        pytest.fail(err_msg, pytrace=False)


def _random_problem(seed):
    """Random positive definite Q and random full-row-rank A with m <= n (seeded)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    m = int(rng.integers(1, n + 1))
    M = rng.normal(size=(n, n))
    Q = M @ M.T + n * np.eye(n)
    c = rng.normal(size=n)
    A = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    return Q, c, A, b


def _null_space_oracle(Q, c, A, b):
    """Independent construction: particular solution + null-space reduced problem, no KKT matrix."""
    m, _ = A.shape
    x_p = np.linalg.lstsq(A, b, rcond=None)[0]  # satisfies A x_p = b (A has full row rank)
    _, _, vt = np.linalg.svd(A)
    Z = vt[m:].T  # columns span the null space of A
    if Z.shape[1] == 0:
        x = x_p
    else:
        y = np.linalg.solve(Z.T @ Q @ Z, -Z.T @ (Q @ x_p + c))
        x = x_p + Z @ y
    lam = np.linalg.lstsq(A.T, Q @ x + c, rcond=None)[0]  # stationarity: Qx + c = A'lam
    return x, lam


def _optimal_value(Q, c, A, b):
    x, _ = _null_space_oracle(Q, c, A, b)
    return 0.5 * x @ Q @ x + c @ x


def _call(Q, c, A, b):
    """Run the learner's function; a crash on valid input is a contract failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = constrained_quadratic_min(Q, c, A, b)
    except RuntimeWarning:
        pytest.fail("a NumPy warning was raised on valid input: check for overflow")
    except Exception as exc:
        pytest.fail(f"valid input must be accepted; the call raised {type(exc).__name__}")
    if not (isinstance(result, tuple) and len(result) == 2):
        pytest.fail("return a two-item tuple (x, lam)")
    x, lam = result
    if np.shape(x) != (len(c),):
        pytest.fail("shape mismatch: x must have one entry per variable")
    if np.shape(lam) != (len(b),):
        pytest.fail("shape mismatch: lam must have one entry per constraint row")
    return np.asarray(x, dtype=np.float64), np.asarray(lam, dtype=np.float64)


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def test_matches_independent_null_space_construction_on_random_problems():
    for seed in range(20):
        Q, c, A, b = _random_problem(seed)
        x, lam = _call(Q, c, A, b)
        want_x, want_lam = _null_space_oracle(Q, c, A, b)
        _close(
            x,
            want_x,
            rtol=1e-7,
            atol=1e-9,
            err_msg="minimiser differs from the constrained optimum: check the KKT block and rhs",
        )
        _close(
            lam,
            want_lam,
            rtol=1e-7,
            atol=1e-9,
            err_msg="multipliers differ from the reference: check the sign of the A' block",
        )


def test_multipliers_equal_the_sensitivity_of_the_optimal_value_to_b():
    h = 1e-5
    for seed in range(6):
        Q, c, A, b = _random_problem(seed)
        _, lam = _call(Q, c, A, b)
        slopes = []
        for j in range(len(b)):
            step = np.zeros(len(b))
            step[j] = h
            up = _optimal_value(Q, c, A, b + step)
            down = _optimal_value(Q, c, A, b - step)
            slopes.append((up - down) / (2 * h))
        _close(
            lam,
            slopes,
            rtol=1e-5,
            atol=1e-6,
            err_msg="multiplier is not dp*/db: the convention is lam = dp*/db, so check the sign",
        )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=5000))
def test_solution_is_feasible_and_satisfies_stationarity(seed):
    Q, c, A, b = _random_problem(seed)
    x, lam = _call(Q, c, A, b)
    scale = 1.0 + float(np.abs(b).max())
    assert float(np.max(np.abs(A @ x - b))) <= 1e-9 * scale, (
        "the returned point violates the equality constraints Ax = b"
    )
    grad_scale = 1.0 + float(np.abs(Q).max() * np.abs(x).max() + np.abs(c).max())
    assert float(np.max(np.abs(Q @ x + c - A.T @ lam))) <= 1e-9 * grad_scale, (
        "stationarity fails: Qx + c must equal A'lam at the solution"
    )


def test_rank_deficient_constraints_raise_value_error_even_when_a_solve_would_not_fail():
    # Row 2 is three times row 1, so the constraints are dependent. A plain solve of the KKT
    # system returns numbers of order 1e17 here instead of failing.
    A = np.array([[0.1, 0.7, 0.3], [0.3, 2.1, 0.9]])
    b = np.array([1.0, 3.0])
    _raised_value_error(
        lambda: constrained_quadratic_min(np.eye(3), np.zeros(3), A, b),
        "dependent constraint rows must raise ValueError: check the rank before solving",
    )


def test_indefinite_q_raises_value_error():
    Q = np.diag([1.0, -1.0])
    A = np.array([[1.0, 2.0]])
    _raised_value_error(
        lambda: constrained_quadratic_min(Q, np.zeros(2), A, np.array([1.0])),
        "Q is not positive definite, which the contract requires: raise ValueError",
    )


def test_scaling_the_objective_keeps_the_minimiser_and_scales_the_multipliers():
    Q, c, A, b = _random_problem(7)
    x, lam = _call(Q, c, A, b)
    for s in (1e6, 1e-6):
        xs, lams = _call(s * Q, s * c, A, b)
        _close(
            xs,
            x,
            rtol=1e-8,
            atol=1e-12,
            err_msg="scaling the objective moved the minimiser: x depends on f only up to scale",
        )
        _close(
            lams,
            s * lam,
            rtol=1e-8,
            atol=1e-10 * s,
            err_msg="multipliers did not scale with the objective: lam must satisfy Qx + c = A'lam",
        )


def test_tiny_and_huge_constraint_rows_are_valid_and_scale_multipliers_inversely():
    Q, c, A, b = _random_problem(11)
    x, lam = _call(Q, c, A, b)
    for s in (1e-9, 1e9):
        xs, lams = _call(Q, c, s * A, s * b)
        _close(
            xs,
            x,
            rtol=1e-7,
            atol=1e-9,
            err_msg="rescaling constraint rows moved the minimiser: the feasible set is unchanged",
        )
        _close(
            lams,
            lam / s,
            rtol=1e-6,
            atol=1e-9 / s,
            err_msg="valid small or large constraint rejected: use a relative rank test",
        )


def test_one_variable_problem_matches_the_hand_solution():
    # Minimise x^2 subject to x = 3. Stationarity gives 2x = lam, so x = 3 and lam = 6.
    x, lam = _call([[2.0]], [0.0], [[1.0]], [3.0])
    _close(x, [3.0], atol=1e-12, err_msg="one-variable minimiser is wrong")
    _close(lam, [6.0], atol=1e-12, err_msg="one-variable multiplier is wrong: lam = dp*/db = 2b")


def test_integer_inputs_give_float64_results():
    x, lam = constrained_quadratic_min([[2, 0], [0, 2]], [0, 0], [[1, 1]], [1])
    assert np.asarray(x).dtype == np.float64, "x must be float64 even for integer input"
    assert np.asarray(lam).dtype == np.float64, "lam must be float64 even for integer input"
    _close(x, [0.5, 0.5], atol=1e-12, err_msg="integer input gave wrong x")


def test_rounding_level_asymmetry_is_accepted_as_symmetric():
    # The contract says Q is symmetric up to rounding, so a 1e-15 mismatch must not be rejected.
    Q = np.array([[2.0, 1.0 + 1e-15], [1.0, 2.0]])
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    x, lam = _call(Q, np.zeros(2), A, b)
    want_x, want_lam = _null_space_oracle(0.5 * (Q + Q.T), np.zeros(2), A, b)
    _close(
        x,
        want_x,
        rtol=1e-7,
        atol=1e-9,
        err_msg="a Q symmetric up to rounding was rejected: use a symmetry tolerance relative to Q",
    )
    _close(
        lam,
        want_lam,
        rtol=1e-7,
        atol=1e-9,
        err_msg="a Q symmetric up to rounding was rejected: use a symmetry tolerance relative to Q",
    )


INVALID = [
    (
        "non-square-Q",
        np.ones((2, 3)),
        [0.0, 0.0],
        [[1.0, 1.0]],
        [1.0],
        "Q must be square: raise ValueError for a non-square matrix",
    ),
    (
        "linear-term-length",
        np.eye(2),
        [0.0, 0.0, 0.0],
        [[1.0, 1.0]],
        [1.0],
        "c must match the size of Q: raise ValueError on a length mismatch",
    ),
    (
        "constraint-columns",
        np.eye(2),
        [0.0, 0.0],
        [[1.0, 1.0, 1.0]],
        [1.0],
        "A must have as many columns as Q has rows: raise ValueError",
    ),
    (
        "constraint-rhs-length",
        np.eye(2),
        [0.0, 0.0],
        [[1.0, 1.0]],
        [1.0, 2.0],
        "b must have one entry per row of A: raise ValueError",
    ),
    (
        "no-constraints",
        np.eye(2),
        [0.0, 0.0],
        np.zeros((0, 2)),
        [],
        "A needs at least one row: raise ValueError",
    ),
    (
        "one-dimensional-A",
        np.eye(2),
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0],
        "A must be a 2-D matrix: raise ValueError for a vector",
    ),
    (
        "nan-in-Q",
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        [0.0, 0.0],
        [[1.0, 1.0]],
        [1.0],
        "non-finite entries must raise ValueError: check np.isfinite",
    ),
    (
        "infinite-linear-term",
        np.eye(2),
        [np.inf, 0.0],
        [[1.0, 1.0]],
        [1.0],
        "infinite entries in c must raise ValueError",
    ),
    (
        "asymmetric-Q",
        np.array([[2.0, 1.0], [0.0, 2.0]]),
        [0.0, 0.0],
        [[1.0, 1.0]],
        [1.0],
        "Q must be symmetric: raise ValueError for an asymmetric matrix",
    ),
    (
        "zero-Q",
        np.zeros((2, 2)),
        [0.0, 0.0],
        [[1.0, 1.0]],
        [1.0],
        "a zero Q is not positive definite: raise ValueError",
    ),
]


@pytest.mark.parametrize(
    ("Q", "c", "A", "b", "mistake"),
    [case[1:] for case in INVALID],
    ids=[case[0] for case in INVALID],
)
def test_invalid_inputs_raise_value_error(Q, c, A, b, mistake):
    _raised_value_error(lambda: constrained_quadratic_min(Q, c, A, b), mistake)


def test_inputs_are_not_mutated():
    Q, c, A, b = _random_problem(3)
    before = {"Q": Q.copy(), "c": c.copy(), "A": A.copy(), "b": b.copy()}
    _call(Q, c, A, b)
    for name, arr in (("Q", Q), ("c", c), ("A", A), ("b", b)):
        _equal(arr, before[name], err_msg=f"{name} was modified in place")


def test_rounding_level_asymmetry_is_accepted_when_Q_has_a_zero_entry():
    # Q is symmetric up to rounding, and one off-diagonal entry is exactly zero. The symmetry
    # tolerance must scale with the size of Q, not with its smallest entry.
    Q = np.array([[2.0, 0.0], [1e-15, 2.0]])
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    x, lam = _call(Q, np.zeros(2), A, b)
    want_x, want_lam = _null_space_oracle(0.5 * (Q + Q.T), np.zeros(2), A, b)
    _close(
        x,
        want_x,
        rtol=1e-7,
        atol=1e-9,
        err_msg="a Q symmetric up to rounding was rejected: scale the tolerance by the size of Q",
    )
    _close(
        lam,
        want_lam,
        rtol=1e-7,
        atol=1e-9,
        err_msg="a Q symmetric up to rounding was rejected: scale the tolerance by the size of Q",
    )


def test_negative_definite_q_raises_value_error():
    # Q = -I has determinant +1, but both eigenvalues are negative, so Q is not positive
    # definite. A sign test on det(Q) alone would accept it.
    Q = -np.eye(2)
    A = np.array([[1.0, 1.0]])
    _raised_value_error(
        lambda: constrained_quadratic_min(Q, np.zeros(2), A, np.array([1.0])),
        "negative definite Q is not positive definite: raise ValueError",
    )
