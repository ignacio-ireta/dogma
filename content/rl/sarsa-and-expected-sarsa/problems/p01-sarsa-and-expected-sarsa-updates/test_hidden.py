# Hidden tests: targets, oracles, properties and the input contract (CONTENT_SPEC §6.4).
# Failure messages name the kind of mistake, never hidden inputs or expected values.
# Most target checks use alpha = 1, so the new entry is the target and each test isolates one rule.
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from submission import expected_sarsa_update, sarsa_update

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


# Two states, three actions. In state 1 actions 1 and 2 tie for the largest value (3.0).
TIE_TABLE = [[0.0, 0.0, 0.5], [1.0, 3.0, 3.0]]
# Same shape, but state 1 has one greedy action (value 3.0).
UNIQUE_TABLE = [[0.0, 0.0, 0.5], [1.0, 3.0, 2.0]]


def _expect_value_error(call, mistake: str) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def _run(call, mistake: str):
    """Call the learner; a crash is reported with the mistake, not a raw traceback."""
    try:
        return call()
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__}")


def _policy_mixture(row: np.ndarray, epsilon: float) -> np.ndarray:
    """Independent form: (1 - epsilon) * uniform over greedy actions + epsilon * uniform."""
    row = np.asarray(row, dtype=np.float64)
    greedy = (row == row.max()).astype(np.float64)
    return (1.0 - epsilon) * greedy / greedy.sum() + epsilon / row.shape[0]


def _sarsa_target(Q, r, s_next, a_next, gamma):
    return float(r) if s_next is None else float(r + gamma * Q[s_next, a_next])


def _expected_target(Q, r, s_next, gamma, epsilon):
    if s_next is None:
        return float(r)
    row = np.asarray(Q, dtype=np.float64)[s_next]
    return float(r + gamma * np.dot(_policy_mixture(row, epsilon), row))


@st.composite
def transitions(draw):
    n_states = draw(st.integers(1, 3))
    n_actions = draw(st.integers(1, 4))
    Q = draw(arrays(np.float64, (n_states, n_actions), elements=st.floats(-5, 5, allow_nan=False)))
    s = draw(st.integers(0, n_states - 1))
    a = draw(st.integers(0, n_actions - 1))
    s_next = draw(st.one_of(st.none(), st.integers(0, n_states - 1)))
    a_next = draw(st.integers(0, n_actions - 1))
    r = draw(st.floats(-5, 5, allow_nan=False))
    alpha = draw(st.floats(0, 1))
    gamma = draw(st.floats(0, 1))
    epsilon = draw(st.floats(0, 1))
    return Q, s, a, r, s_next, a_next, alpha, gamma, epsilon


def test_sarsa_target_uses_the_action_taken_next():
    # alpha = 1 reads the target. Q[1, 0] = 1.0 and Q[1, 2] = 3.0, so the target is 1.5 or 3.5.
    low = sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 0, 1.0, 1.0)
    high = sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 2, 1.0, 1.0)
    assert np.isclose(low[0, 1], 1.5), (
        "Sarsa target must use Q[s_next, a_next] for the action taken"
    )
    assert np.isclose(high[0, 1], 3.5), (
        "Sarsa target must use Q[s_next, a_next] for the action taken"
    )


def test_sarsa_discounts_the_next_value_by_gamma():
    # gamma = 0.5 and r = 0: the target is 0.5 * Q[1, 2] = 1.5, not the undiscounted 3.0.
    out = sarsa_update(TIE_TABLE, 0, 1, 0.0, 1, 2, 1.0, 0.5)
    assert np.isclose(out[0, 1], 1.5), "the next-state value must be multiplied by gamma"


def test_sarsa_step_size_scales_the_move_toward_the_target():
    # From 0.0 toward target 2.0 with alpha = 0.25 gives 0.5.
    # Swapping alpha and gamma (alpha = 0.5, gamma = 0.25) gives 0.625.
    out = sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 2, 0.25, 0.5)
    assert np.isclose(out[0, 1], 0.5), "alpha must scale the move from Q[s, a] toward the target"


def test_expected_sarsa_matches_hand_computed_tie_case():
    # Greedy set {1, 2}, k = 2: pi = (0.1, 0.45, 0.45), so the expectation is 2.8.
    out = expected_sarsa_update(TIE_TABLE, 0, 1, 0.0, 1, 1.0, 1.0, 0.3)
    assert np.isclose(out[0, 1], 2.8), (
        "expected Sarsa weights: epsilon/A everywhere, plus (1-epsilon)/k on tied greedy actions"
    )


def test_expected_sarsa_matches_hand_computed_unique_greedy_case():
    # Unique greedy action 1: pi = (0.1, 0.8, 0.1), so the expectation is 2.7.
    out = expected_sarsa_update(UNIQUE_TABLE, 0, 1, 0.0, 1, 1.0, 1.0, 0.3)
    assert np.isclose(out[0, 1], 2.7), (
        "expected Sarsa must give the greedy action (1-epsilon) plus its epsilon/A share"
    )


def test_three_way_tie_splits_exploit_share_over_all_tied_actions():
    # Greedy set {1, 2, 3}, k = 3: pi = (0.1, 0.3, 0.3, 0.3), so the expectation is 4.6.
    table = [[0.0, 0.0, 0.0, 0.0], [1.0, 5.0, 5.0, 5.0]]
    out = expected_sarsa_update(table, 0, 0, 0.0, 1, 1.0, 1.0, 0.4)
    assert np.isclose(out[0, 0], 4.6), (
        "the exploit share (1-epsilon) must be divided by the number of tied greedy actions k"
    )


def test_values_that_differ_slightly_are_not_ties():
    # 3.000001 is the unique greedy action, so pi = (0.1, 0.1, 0.8) at epsilon = 0.3.
    # Treating it as tied with 3.0 would move the target by about 9e-8.
    table = [[0.0, 0.0, 0.5], [1.0, 3.0, 3.000001]]
    expected = 0.5 + 0.5 * (0.1 * 1.0 + 0.1 * 3.0 + 0.8 * 3.000001)
    out = expected_sarsa_update(table, 0, 1, 0.5, 1, 1.0, 0.5, 0.3)
    assert np.isclose(out[0, 1], expected, rtol=0.0, atol=1e-9), (
        "ties are exact equality: near-equal values must not share the exploit share"
    )


def test_epsilon_zero_matches_q_learning_max_target():
    out = expected_sarsa_update(TIE_TABLE, 0, 1, 0.0, 1, 1.0, 1.0, 0.0)
    assert np.isclose(out[0, 1], 3.0), (
        "at epsilon = 0 the expected target must be the Q-learning max target"
    )


def test_epsilon_one_matches_uniform_mean_target():
    # Uniform mean of (1, 3, 3) is 7/3.
    out = expected_sarsa_update(TIE_TABLE, 0, 1, 0.0, 1, 1.0, 1.0, 1.0)
    assert np.isclose(out[0, 1], 7.0 / 3.0), (
        "at epsilon = 1 the expected target must average all next-state actions uniformly"
    )


def test_expected_sarsa_discounts_by_gamma():
    # epsilon = 0, r = 0.5, gamma = 0.5: target = 0.5 + 0.5 * 3 = 2.0.
    out = expected_sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 1.0, 0.5, 0.0)
    assert np.isclose(out[0, 1], 2.0), "expected Sarsa must discount the next-state term by gamma"


def test_terminal_next_state_uses_the_reward_alone_in_both_methods():
    huge = [[0.0, 0.0, 0.0], [1e6, 1e6, 1e6]]
    terminal = "a terminal next state must not bootstrap or read a_next"
    got_sarsa = _run(lambda: sarsa_update(huge, 0, 2, 2.0, None, None, 0.25, 0.9), terminal)
    got_expected = _run(
        lambda: expected_sarsa_update(huge, 0, 2, 2.0, None, 0.25, 0.9, 0.5), terminal
    )
    assert np.isclose(got_sarsa[0, 2], 0.5), (
        "a terminal next state has no bootstrap: target must be r, with no next-state value"
    )
    assert np.isclose(got_expected[0, 2], 0.5), (
        "a terminal next state has no bootstrap: target must be r, with no next-state value"
    )


def test_random_tables_match_explicit_policy_oracle():
    rng = np.random.default_rng(11)
    for _ in range(80):
        n_states = int(rng.integers(1, 5))
        n_actions = int(rng.integers(1, 5))
        # Real values that tie: pixel intensities of handwritten digits, whole numbers from 0
        # to 16 with many zeros, centred on 8.
        Q, _ = sample_rows("digits", rng, n_states, n_actions)
        Q -= 8.0
        s = int(rng.integers(n_states))
        a = int(rng.integers(n_actions))
        s_next = None if rng.random() < 0.2 else int(rng.integers(n_states))
        r = float(_real_normal(rng))
        alpha, gamma, epsilon = (float(rng.random()) for _ in range(3))
        target = _expected_target(Q, r, s_next, gamma, epsilon)
        expected = Q.copy()
        expected[s, a] = Q[s, a] + alpha * (target - Q[s, a])
        _close(
            expected_sarsa_update(Q, s, a, r, s_next, alpha, gamma, epsilon),
            expected,
            rtol=1e-9,
            atol=1e-9,
            err_msg="expected Sarsa differs from the explicit epsilon-greedy policy average",
        )


@settings(report_multiple_bugs=False, max_examples=80, deadline=None)
@given(transitions())
def test_sarsa_update_is_a_convex_combination_of_old_value_and_target(case):
    Q, s, a, r, s_next, a_next, alpha, gamma, _ = case
    before = np.asarray(Q, dtype=np.float64)
    out = sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma)
    target = _sarsa_target(before, r, s_next, a_next, gamma)
    old = before[s, a]
    tol = 1e-9 * (1.0 + abs(old) + abs(target))
    assert min(old, target) - tol <= out[s, a] <= max(old, target) + tol, (
        "the updated value must lie between the old value and the target"
    )
    others = np.ones(before.shape, dtype=bool)
    others[s, a] = False
    _equal(out[others], before[others], err_msg="an entry other than (s, a) changed")


@settings(report_multiple_bugs=False, max_examples=80, deadline=None)
@given(transitions())
def test_expected_sarsa_update_is_a_convex_combination_of_old_value_and_target(case):
    Q, s, a, r, s_next, _, alpha, gamma, epsilon = case
    before = np.asarray(Q, dtype=np.float64)
    out = expected_sarsa_update(Q, s, a, r, s_next, alpha, gamma, epsilon)
    target = _expected_target(before, r, s_next, gamma, epsilon)
    old = before[s, a]
    tol = 1e-9 * (1.0 + abs(old) + abs(target))
    assert min(old, target) - tol <= out[s, a] <= max(old, target) + tol, (
        "the updated value must lie between the old value and the target"
    )
    others = np.ones(before.shape, dtype=bool)
    others[s, a] = False
    _equal(out[others], before[others], err_msg="an entry other than (s, a) changed")


def test_alpha_zero_leaves_the_table_unchanged_and_alpha_one_writes_the_target():
    _equal(
        sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 0, 0.0, 0.5),
        np.asarray(TIE_TABLE),
        err_msg="alpha = 0 must leave every entry unchanged",
    )
    out = sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 0, 1.0, 0.5)
    assert np.isclose(out[0, 1], 1.0), "alpha = 1 must write the target itself into Q[s, a]"


def test_invalid_arguments_raise_value_error():
    _expect_value_error(
        lambda: expected_sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 0.5, 0.5, 1.5),
        "epsilon above 1 must raise ValueError: parameters are probabilities in [0, 1]",
    )
    _expect_value_error(
        lambda: expected_sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 0.5, 0.5, float("nan")),
        "epsilon NaN must raise ValueError: parameters are probabilities in [0, 1]",
    )
    _expect_value_error(
        lambda: sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 0, -0.1, 0.5),
        "alpha below 0 must raise ValueError: alpha must lie in [0, 1]",
    )
    _expect_value_error(
        lambda: sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 0, 0.5, 1.2),
        "gamma above 1 must raise ValueError: gamma must lie in [0, 1]",
    )
    _expect_value_error(
        lambda: sarsa_update(TIE_TABLE, 0, 1, 0.5, 1, 3, 0.5, 0.5),
        "an action index outside the table must raise ValueError",
    )
    _expect_value_error(
        lambda: sarsa_update(TIE_TABLE, -1, 1, 0.5, 1, 0, 0.5, 0.5),
        "a negative state index must raise ValueError, not wrap around",
    )
    _expect_value_error(
        lambda: expected_sarsa_update(TIE_TABLE, 0, 1, 0.5, 2, 0.5, 0.5, 0.5),
        "a next state outside the table must raise ValueError",
    )
    _expect_value_error(
        lambda: expected_sarsa_update(TIE_TABLE, 0, 1, float("nan"), 1, 0.5, 0.5, 0.5),
        "a NaN reward must raise ValueError",
    )
    _expect_value_error(
        lambda: expected_sarsa_update(np.zeros((0, 3)), 0, 0, 0.0, None, 0.5, 0.5, 0.5),
        "an empty table must raise ValueError",
    )
    _expect_value_error(
        lambda: expected_sarsa_update(np.zeros(3), 0, 0, 0.0, None, 0.5, 0.5, 0.5),
        "Q must be two-dimensional: raise ValueError for a 1-D table",
    )


def test_inputs_are_not_mutated():
    table = np.array(TIE_TABLE, dtype=np.float64)
    before = table.copy()
    sarsa_update(table, 0, 1, 0.5, 1, 0, 0.5, 0.5)
    expected_sarsa_update(table, 0, 1, 0.5, 1, 0.5, 0.5, 0.3)
    _equal(table, before, err_msg="the input table was modified in place")


def test_outputs_are_float_tables_of_the_input_shape():
    int_table = [[0, 0, 1], [2, 2, 0]]
    out_sarsa = sarsa_update(int_table, 0, 2, 1, 1, 0, 0.5, 0.5)
    out_expected = expected_sarsa_update(int_table, 0, 2, 1, 1, 0.5, 0.5, 0.5)
    for out in (out_sarsa, out_expected):
        assert isinstance(out, np.ndarray), "return a NumPy array"
        assert out.shape == (2, 3), "the returned table must keep the input's shape"
        assert out.dtype == np.float64, "the returned table must be float64, even for integer input"


@settings(report_multiple_bugs=False, max_examples=60, deadline=None)
@given(transitions())
def test_both_updates_return_fresh_tables(case):
    Q, s, a, r, s_next, a_next, alpha, gamma, epsilon = case
    before = np.asarray(Q, dtype=np.float64).copy()
    first = sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma)
    second = expected_sarsa_update(Q, s, a, r, s_next, alpha, gamma, epsilon)
    assert first is not second and not np.shares_memory(first, np.asarray(Q)), (
        "return a new table, not a view or the input array"
    )
    _equal(np.asarray(Q, dtype=np.float64), before, err_msg="input modified")


def test_complex_valued_table_is_rejected():
    complex_table = [[0.0 + 0.0j, 0.0j, 0.5 + 0.0j], [1.0 + 0.0j, 3.0 + 0.0j, 3.0 + 0.0j]]
    _expect_value_error(
        lambda: sarsa_update(complex_table, 0, 1, 0.5, 1, 0, 0.5, 0.5),
        "a complex-valued table must raise ValueError: Q must hold real numbers",
    )


def test_nonfinite_table_entry_is_rejected():
    nan_table = [[0.0, float("nan"), 0.5], [1.0, 3.0, 3.0]]
    inf_table = [[0.0, float("inf"), 0.5], [1.0, 3.0, 3.0]]
    _expect_value_error(
        lambda: sarsa_update(nan_table, 0, 2, 0.5, 1, 0, 0.5, 0.5),
        "a NaN entry anywhere in Q must raise ValueError: Q entries must be finite",
    )
    _expect_value_error(
        lambda: expected_sarsa_update(inf_table, 0, 2, 0.5, 1, 0.5, 0.5, 0.5),
        "an infinite entry anywhere in Q must raise ValueError: Q entries must be finite",
    )


def test_non_integer_index_is_rejected():
    _expect_value_error(
        lambda: sarsa_update(TIE_TABLE, 0.5, 1, 0.5, 1, 0, 0.5, 0.5),
        "a non-integer state index must raise ValueError: indices must be integers",
    )
    _expect_value_error(
        lambda: expected_sarsa_update(TIE_TABLE, 0, 1.5, 0.5, 1, 0.5, 0.5, 0.5),
        "a non-integer action index must raise ValueError: indices must be integers",
    )


def test_empty_one_dimensional_table_is_rejected():
    _expect_value_error(
        lambda: sarsa_update(np.zeros(0), 0, 0, 0.0, None, None, 0.5, 0.5),
        "an empty one-dimensional Q must raise ValueError: Q must be a non-empty 2-D table",
    )


def test_numpy_integer_indices_are_accepted_as_integer_indices():
    # The Contract asks for integer indices. NumPy integers (as returned by np.argmax) are integers.
    # Table [[0, 0], [1, 3]], alpha = 1, so the new entry is the target:
    # target = r + gamma * Q[1, 0] = 1.0 + 0.5 * 1.0 = 1.5, and Q[0, 1] becomes 1.5.
    table = [[0.0, 0.0], [1.0, 3.0]]
    result = _run(
        lambda: sarsa_update(
            table,
            s=np.int64(0),
            a=np.int64(1),
            r=1.0,
            s_next=np.int64(1),
            a_next=np.int64(0),
            alpha=1.0,
            gamma=0.5,
        ),
        "a NumPy integer index must be accepted where an integer index is allowed",
    )
    _close(
        result,
        [[0.0, 1.5], [1.0, 3.0]],
        err_msg="a NumPy integer index must select the same entries as a Python int",
        atol=1e-9,
    )
