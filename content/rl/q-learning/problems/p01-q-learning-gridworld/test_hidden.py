from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import q_learning_update, train_q_learning


def _close(
    actual, desired, *, err_msg="values differ from the expected result", atol=0.0, rtol=1e-7
):
    """numpy.testing.assert_allclose, but the first failure line is the named message."""
    if np.shape(actual) != np.shape(desired):
        pytest.fail(f"{err_msg} (shape differs from the contract)", pytrace=False)
    if not np.allclose(actual, desired, atol=atol, rtol=rtol, equal_nan=False):
        pytest.fail(err_msg, pytrace=False)


def _equal(actual, desired, *, err_msg="values differ from the expected result"):
    """numpy.testing.assert_array_equal, but the first failure line is the named message."""
    if np.shape(actual) != np.shape(desired) or not np.array_equal(actual, desired):
        pytest.fail(err_msg, pytrace=False)


ATOL = 1e-8
RTOL = 1e-8
N_STATES, N_ACTIONS = 9, 4
GOAL = 8


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # a valid call must not fail, whatever the exception type
        pytest.fail(f"valid input raised {type(exc).__name__}: check the indices and the target")


def _rejects(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception type")
    pytest.fail("invalid input was accepted: check the argument validation")


def _ref_step(state, action):
    """Independent gridworld: moves are listed by name, and off-grid moves stay put."""
    row, col = divmod(state, 3)
    moves = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
    d_row, d_col = list(moves.values())[action]
    nr, nc = row + d_row, col + d_col
    nxt = nr * 3 + nc if (0 <= nr < 3 and 0 <= nc < 3) else state
    return nxt, (1.0 if nxt == GOAL else 0.0), nxt == GOAL


def _reference_run(n_episodes, seed, alpha, gamma, epsilon, max_steps=100):  # noqa: PLR0917
    """Independent reference loop with the same draw order, updating the table in place."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(n_episodes):
        s = 0
        for _ in range(max_steps):
            u = rng.random()
            if u < epsilon:
                a = int(rng.integers(N_ACTIONS))
            else:
                best = max(Q[s])
                a = next(i for i in range(N_ACTIONS) if Q[s, i] == best)
            s2, r, done = _ref_step(s, a)
            target = r if done else r + gamma * max(Q[s2])
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            if done:
                break
    return Q


def _distance_to_goal(state):
    row, col = divmod(state, 3)
    return abs(row - 2) + abs(col - 2)


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    st.integers(0, 8),
    st.integers(0, 3),
    st.integers(0, 8),
    st.floats(-10, 10, allow_nan=False),
    st.booleans(),
    st.floats(0.01, 1.0),
    st.floats(0.0, 1.0),
    st.integers(0, 10_000),
)
def test_single_update_matches_the_formula(s, a, s2, r, done, alpha, gamma, seed):  # noqa: PLR0917
    Q = np.random.default_rng(seed).normal(size=(N_STATES, N_ACTIONS))
    got = _call(q_learning_update, Q, s, a, r, s2, done, alpha, gamma)
    target = r if done else r + gamma * float(np.max(Q[s2]))
    want = Q[s, a] + alpha * (target - Q[s, a])
    _close(
        got[s, a],
        want,
        atol=ATOL,
        rtol=RTOL,
        err_msg="the update differs from the formula: check the max over the next state's "
        "actions and the step size on the error term",
    )


def test_terminal_update_does_not_bootstrap_even_with_huge_next_values():
    Q = np.zeros((N_STATES, N_ACTIONS))
    Q[GOAL] = 1e6
    out = _call(q_learning_update, Q, 7, 1, 2.0, GOAL, True, 0.25, 0.9)
    _close(
        out[7, 1],
        0.5,
        atol=ATOL,
        rtol=RTOL,
        err_msg="terminal update bootstrapped from the next state: use the reward alone",
    )


def test_non_terminal_update_with_the_same_values_does_bootstrap():
    Q = np.zeros((N_STATES, N_ACTIONS))
    Q[GOAL] = 1e6
    out = _call(q_learning_update, Q, 7, 1, 2.0, GOAL, False, 0.25, 0.9)
    assert out[7, 1] > 1e5, "non-terminal update must include gamma * max next value"


def test_update_does_not_modify_the_input_table():
    Q = np.random.default_rng(5).normal(size=(N_STATES, N_ACTIONS))
    original = Q.copy()
    _call(q_learning_update, Q, 2, 3, 1.0, 5, False, 0.5, 0.9)
    _equal(Q, original, err_msg="the input table was modified in place")


def test_only_the_visited_entry_changes():
    Q = np.random.default_rng(6).normal(size=(N_STATES, N_ACTIONS))
    out = _call(q_learning_update, Q, 4, 0, 1.0, 1, False, 0.3, 0.8)
    changed = np.argwhere(out != Q)
    assert changed.tolist() == [[4, 0]], "exactly one entry, Q[s, a], may change"


def test_invalid_update_arguments_rejected():
    Q = np.zeros((N_STATES, N_ACTIONS))
    for alpha in (0.0, 1.5, float("nan"), True):
        _rejects(q_learning_update, Q, 0, 0, 0.0, 1, False, alpha, 0.9)
    for gamma in (-0.1, 1.1, float("nan"), True):
        _rejects(q_learning_update, Q, 0, 0, 0.0, 1, False, 0.5, gamma)
    _rejects(q_learning_update, Q, 9, 0, 0.0, 1, False, 0.5, 0.9)
    _rejects(q_learning_update, Q, 0, -1, 0.0, 1, False, 0.5, 0.9)
    _rejects(q_learning_update, Q, 0, 0, 0.0, 99, False, 0.5, 0.9)
    _rejects(q_learning_update, Q, 0, 0, float("nan"), 1, False, 0.5, 0.9)
    _rejects(q_learning_update, np.zeros(N_STATES), 0, 0, 0.0, 1, False, 0.5, 0.9)
    _rejects(q_learning_update, np.full((2, 2), np.nan), 0, 0, 0.0, 1, False, 0.5, 0.9)


def test_full_run_matches_independent_reference_loop():
    for seed in (0, 1, 2):
        got = _call(train_q_learning, 150, seed, alpha=0.5, gamma=0.9, epsilon=0.3)
        want = _reference_run(150, seed, 0.5, 0.9, 0.3)
        _close(
            got,
            want,
            atol=ATOL,
            rtol=RTOL,
            err_msg="the run differs from the reference loop: check the draw order (rng.random "
            "on every choice, integers only when exploring), the terminal flag and the reset "
            "to state 0",
        )


def test_long_run_reaches_the_analytic_optimum_of_the_gridworld():
    # In this deterministic gridworld, Q*(s,a) = gamma ** d(s'), where d is the number of steps
    # from s' to the goal (d(goal) = 0) and s' is the state the action leads to.
    got = _call(train_q_learning, 20_000, 0, alpha=0.5, gamma=0.9, epsilon=0.5)
    for s in range(8):
        for a in range(N_ACTIONS):
            row, col = divmod(s, 3)
            d_row, d_col = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]
            nr, nc = row + d_row, col + d_col
            s2 = nr * 3 + nc if (0 <= nr < 3 and 0 <= nc < 3) else s
            want = 0.9 ** _distance_to_goal(s2)
            _close(
                got[s, a],
                want,
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"entry ({s}, {a}) is not the analytic optimum: check the target and "
                "the bootstrap at the goal",
            )


def test_step_cap_ends_episodes_before_the_goal_is_reachable():
    # The goal is 4 moves from the start, so with at most 3 steps per episode no reward can ever
    # be collected and the table must stay at zero. A loop that ignores max_steps would wander on
    # and reach the goal.
    got = _call(train_q_learning, 200, 3, max_steps=3)
    _equal(
        got,
        np.zeros((N_STATES, N_ACTIONS)),
        err_msg="the step cap was ignored: episodes continued past max_steps",
    )


def test_zero_discount_leaves_every_non_goal_entering_entry_at_zero():
    got = _call(train_q_learning, 300, 4, gamma=0.0, epsilon=0.5)
    entering_goal = {(5, 2), (7, 1)}  # down from 5 and right from 7 enter the goal
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if (s, a) in entering_goal:
                assert got[s, a] > 0.0, "a goal-entering pair must learn a positive value"
            else:
                assert got[s, a] == 0.0, (
                    "with gamma = 0 the target is the reward alone; a bootstrap leaked into the "
                    "table"
                )


def test_same_seed_gives_identical_tables():
    first = _call(train_q_learning, 40, 11)
    second = _call(train_q_learning, 40, 11)
    _equal(first, second, err_msg="the same seed must reproduce the run")


def test_table_is_float64_with_the_gridworld_shape():
    got = _call(train_q_learning, 3, 0)
    assert got.dtype == np.float64, "the Q table must be float64"
    assert got.shape == (N_STATES, N_ACTIONS), "the Q table must have shape (9, 4)"
    # With no episodes there are no updates, so the initial table itself must be float64.
    empty = _call(train_q_learning, 0, 0)
    assert empty.dtype == np.float64, "the zero table must be float64 even with no episodes"
    assert empty.shape == (N_STATES, N_ACTIONS), "the zero table must have shape (9, 4)"


def test_large_reward_gives_a_finite_update():
    Q = np.zeros((N_STATES, N_ACTIONS))
    out = _call(q_learning_update, Q, 0, 1, 1e300, 1, True, 1.0, 1.0)
    assert np.all(np.isfinite(out)), "a large reward must not overflow the update"


def test_tiny_reward_keeps_relative_accuracy():
    Q = np.zeros((N_STATES, N_ACTIONS))
    out = _call(q_learning_update, Q, 0, 1, 1e-200, 1, True, 0.5, 0.9)
    _close(out[0, 1], 5e-201, rtol=1e-9, err_msg="tiny rewards lost relative accuracy")


def test_invalid_training_arguments_rejected():
    for args in ((-1, 0), (1.5, 0), (3, -1), (3, True)):
        _rejects(train_q_learning, *args)
    _rejects(train_q_learning, 3, 0, 0.0)
    _rejects(train_q_learning, 3, 0, 0.5, 1.2)
    _rejects(train_q_learning, 3, 0, 0.5, 0.9, float("nan"))
    _rejects(train_q_learning, 3, 0, 0.5, 0.9, True)
    _rejects(train_q_learning, 3, 0, 0.5, 0.9, 0.3, 0)


def test_non_integer_state_index_is_rejected():
    Q = np.zeros((N_STATES, N_ACTIONS))
    _rejects(q_learning_update, Q, 1.5, 0, 0.0, 1, False, 0.5, 0.9)


def test_complex_table_is_rejected_as_not_real():
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.complex128)
    _rejects(q_learning_update, Q, 0, 0, 0.0, 1, False, 0.5, 0.9)


def test_table_with_one_nan_entry_is_rejected():
    Q = np.zeros((N_STATES, N_ACTIONS))
    Q[3, 2] = np.nan
    _rejects(q_learning_update, Q, 0, 0, 0.0, 1, False, 0.5, 0.9)


def test_zero_alpha_is_rejected_even_with_no_episodes():
    _rejects(train_q_learning, 0, 0, 0.0)


def test_zero_epsilon_is_accepted_for_training():
    got = _call(train_q_learning, 3, 0, epsilon=0.0)
    assert got.shape == (N_STATES, N_ACTIONS), (
        "epsilon = 0 is a valid rate: the table must be (9, 4)"
    )


def test_one_step_cap_only_updates_the_start_state():
    got = _call(train_q_learning, 5, 0, max_steps=1)
    _equal(
        got[1:],
        np.zeros((N_STATES - 1, N_ACTIONS)),
        err_msg="with a one-step cap every episode updates only the start state: other rows "
        "must stay zero",
    )


def test_default_step_cap_is_100_steps():
    _equal(
        _call(train_q_learning, 200, 9),
        _call(train_q_learning, 200, 9, max_steps=100),
        err_msg="the default max_steps must be 100: an episode that runs 100 steps was cut short",
    )


def test_single_state_table_is_a_valid_table():
    # A 1 x 2 table is a 2-D finite real array of shape (S, A) with S = 1, so it is valid.
    # Hand-worked: target = r = 1.0 (done), so Q[0, 1] = 0 + 0.5 * (1.0 - 0) = 0.5.
    Q = np.zeros((1, 2))
    out = _call(q_learning_update, Q, 0, 1, 1.0, 0, True, 0.5, 0.9)
    assert out.shape == (1, 2), "a one-state table must keep its shape"
    _close(
        out[0, 1],
        0.5,
        atol=ATOL,
        rtol=RTOL,
        err_msg="a valid one-state table must be updated like any other table",
    )


def test_integer_table_gives_a_float64_result():
    # The contract returns a float64 table whatever the dtype of Q. Hand-worked:
    # Q[7, 1] = 0 + 0.5 * (1.0 - 0) = 0.5, which an integer table would round away.
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    out = _call(q_learning_update, Q, 7, 1, 1.0, GOAL, True, 0.5, 0.9)
    assert out.dtype == np.float64, (
        "the returned table must be float64 even when Q is an integer array"
    )
    _close(
        out[7, 1],
        0.5,
        atol=ATOL,
        rtol=RTOL,
        err_msg="an integer input table must not round the update",
    )
