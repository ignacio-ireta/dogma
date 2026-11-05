from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import epsilon_greedy_action


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


def _expect(condition, message):
    if not condition:
        pytest.fail(message, pytrace=False)


# Frequency tolerance: N = 20000 draws, so the standard error is at most 0.0035; 0.02 is about
# 5.7 standard errors. The seeds are fixed, so a correct implementation always passes.
N_DRAWS = 20_000
FREQ_ATOL = 2e-2


def _call(q, epsilon, rng):
    try:
        return epsilon_greedy_action(q, epsilon, rng)
    except Exception as exc:  # a valid call must not fail, whatever the exception type
        pytest.fail(f"valid input raised {type(exc).__name__}: check indexing and the draws")


def _rejects(exc_type, *args):
    try:
        epsilon_greedy_action(*args)
    except exc_type:
        return
    except Exception:
        pytest.fail(f"invalid input must raise {exc_type.__name__}, not another exception type")
    pytest.fail(f"invalid input was accepted: check the {exc_type.__name__} validation")


def _reference_choice(q, epsilon, rng):
    """Independent oracle that encodes the draw order in the contract, step by step."""
    u = rng.random()
    if u < epsilon:
        return int(rng.integers(len(q)))
    best = max(q)
    return next(i for i, value in enumerate(q) if value == best)


def test_replayed_draw_order_matches_reference_sequence():
    patterns = [
        np.array([0.2, 0.9, 0.3]),
        np.array([1.0, 1.0, 0.5, 1.0]),
        np.array([0.4, 0.4, 0.4]),
        np.array([7.0]),
    ]
    for epsilon in (0.0, 0.1, 0.5, 1.0):
        for q in patterns:
            got_rng = np.random.default_rng(2024)
            want_rng = np.random.default_rng(2024)
            got = [_call(q, epsilon, got_rng) for _ in range(300)]
            want = [_reference_choice(q, epsilon, want_rng) for _ in range(300)]
            _expect(
                got == want,
                "action sequence differs from the draw order: check that u is drawn on every "
                "call and that exploration draws one integer",
            )


def test_zero_epsilon_consumes_exactly_one_uniform_per_call():
    seed = 99
    q = np.array([0.5, 0.1, 0.5])
    used = np.random.default_rng(seed)
    for _ in range(50):
        _call(q, 0.0, used)
    expected = np.random.default_rng(seed)
    for _ in range(50):
        expected.random()
    _expect(
        used.random() == expected.random(),
        "exploitation must draw exactly one uniform and nothing else: check for extra draws",
    )


def test_exploration_uses_one_integer_draw_after_the_uniform():
    seed = 7
    q = np.array([0.0, 1.0])
    used = np.random.default_rng(seed)
    expected = np.random.default_rng(seed)
    for _ in range(200):
        u = expected.random()
        if u < 1.0:
            expected.integers(2)
    for _ in range(200):
        _call(q, 1.0, used)
    _expect(
        used.random() == expected.random(),
        "exploration must draw one integer after the uniform: the draw count is wrong",
    )


def test_selection_frequencies_match_analytic_table():
    q = np.array([0.2, 0.9, 0.9, 0.4])
    epsilon = 0.2
    rng = np.random.default_rng(1)
    counts = np.bincount([_call(q, epsilon, rng) for _ in range(N_DRAWS)], minlength=4)
    freq = counts / N_DRAWS
    expected = np.array([0.05, 0.85, 0.05, 0.05])  # lowest greedy index gets the exploit share
    _close(
        freq,
        expected,
        atol=FREQ_ATOL,
        rtol=0.0,
        err_msg="selection frequencies differ from 1 - e + e/A (greedy) and e/A (others): "
        "check the exploration probability and the greedy set",
    )


def test_draw_equal_to_epsilon_exploits():
    # The rule is u < epsilon, strictly. With epsilon equal to the first uniform of the seed,
    # the first call must exploit, so no integer is drawn; a "<=" comparison would explore.
    q = np.array([0.0, 1.0, 0.0, 0.5])
    for seed in range(10):
        epsilon = float(np.random.default_rng(seed).random())
        got_rng = np.random.default_rng(seed)
        want_rng = np.random.default_rng(seed)
        got = [_call(q, epsilon, got_rng) for _ in range(20)]
        want = [_reference_choice(q, epsilon, want_rng) for _ in range(20)]
        _expect(
            got == want,
            "when u equals epsilon the rule must exploit (strict u < epsilon): "
            "check the comparison direction",
        )


def test_exploration_includes_the_greedy_action():
    q = np.array([0.0, 5.0, 0.0])  # unique greedy action is index 1
    rng = np.random.default_rng(3)
    counts = np.bincount([_call(q, 1.0, rng) for _ in range(N_DRAWS)], minlength=3)
    _close(
        counts / N_DRAWS,
        [1 / 3, 1 / 3, 1 / 3],
        atol=FREQ_ATOL,
        rtol=0.0,
        err_msg="full exploration must be uniform over all actions, including the greedy one",
    )


def test_ties_break_by_lowest_index_when_exploiting():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        _expect(
            _call(np.array([1.0, 3.0, 3.0, 3.0, 0.0]), 0.0, rng) == 1,
            "tie among greedy actions must return the lowest index",
        )
    rng = np.random.default_rng(0)
    _expect(
        _call(np.array([-5.0, -5.0]), 0.0, rng) == 0,
        "all-equal ties must return index 0",
    )


def test_tiny_gap_between_values_is_respected():
    q = np.array([1.0, 1.0 + 2.2e-16])  # one ulp apart: a unique maximum at index 1
    rng = np.random.default_rng(0)
    _expect(
        _call(q, 0.0, rng) == 1,
        "a one-ulp gap must still select the larger value",
    )


def test_huge_magnitudes_pick_the_correct_maximum():
    q = np.array([1e308, 5e307, 1e308])
    rng = np.random.default_rng(0)
    _expect(
        _call(q, 0.0, rng) == 0,
        "huge equal values must tie and return the lowest index",
    )
    q2 = np.array([-1e308, 0.0])
    _expect(
        _call(q2, 0.0, rng) == 1,
        "large negative values must not overflow the comparison",
    )


def test_integer_q_values_are_accepted():
    rng = np.random.default_rng(0)
    _expect(
        _call(np.array([0, 2, 1]), 0.0, rng) == 1,
        "check failed: _call(np.array([0, 2, 1]), 0.0, rng) == 1",
    )


def test_returns_a_python_int_in_range():
    rng = np.random.default_rng(4)
    for _ in range(100):
        choice = _call(np.array([0.3, 0.1, 0.2]), 0.5, rng)
        _expect(
            type(choice) is int,
            "return a Python int, not a NumPy integer",
        )
        _expect(
            0 <= choice < 3,
            "choice must be a valid action index",
        )


@settings(max_examples=60, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    st.lists(st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=8),
    st.floats(0.0, 1.0, allow_nan=False),
    st.integers(0, 10_000),
)
def test_random_inputs_match_reference_choice(values, epsilon, seed):
    q = np.array(values)
    got = _call(q, epsilon, np.random.default_rng(seed))
    want = _reference_choice(q, epsilon, np.random.default_rng(seed))
    _expect(
        got == want,
        "choice differs from the draw-order reference",
    )


def test_q_values_are_not_mutated():
    q = np.array([0.3, 0.7, 0.7])
    original = q.copy()
    _call(q, 0.5, np.random.default_rng(0))
    _equal(q, original, err_msg="q_values was modified")


def test_global_random_state_is_not_used():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    for _ in range(50):
        _call(np.array([0.1, 0.2]), 0.9, np.random.default_rng(0))
    after = np.random.get_state()[1]
    _equal(before, after, err_msg="the global numpy.random state was consumed: use the passed rng")


def test_invalid_epsilon_rejected():
    q = np.array([1.0, 2.0])
    rng = np.random.default_rng(0)
    for epsilon in (-0.01, 1.01, float("nan"), True):
        _rejects(ValueError, q, epsilon, rng)


def test_invalid_q_values_rejected():
    rng = np.random.default_rng(0)
    for bad in (
        np.array([]),
        np.ones((2, 2)),
        np.array([1.0, float("nan")]),
        np.array([1.0, float("inf")]),
        np.array([1 + 1j]),
        np.array([True, False]),
    ):
        _rejects(ValueError, bad, 0.1, rng)


def test_rng_must_be_a_numpy_generator():
    _rejects(TypeError, np.array([1.0, 2.0]), 0.1, np.random.RandomState(0))
    _rejects(TypeError, np.array([1.0, 2.0]), 0.1, 42)
