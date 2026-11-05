"""Hidden tests: a linear-scan oracle, the half-open rule, the top of the CDF, validation and edges.

Each failure message names the kind of mistake. No message contains a hidden input or an expected
answer.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from submission import sample_discrete

TOP = 1.0 - 2.0**-53  # the largest uniform that rng.random can return


class _Uniforms:
    """A stand-in generator that hands out the given uniforms in order.

    ``random(n)`` returns the next ``n`` values, and ``random()`` returns the next value as a
    scalar. Every call is recorded, so a test can check the call pattern.
    """

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)
        self.pos = 0
        self.calls = []

    def random(self, n=None):
        self.calls.append(n)
        if n is None:
            value = self.values[self.pos]
            self.pos += 1
            return float(value)
        out = self.values[self.pos : self.pos + n].copy()
        self.pos += n
        return out


def _run(call, mistake):
    """Call the learner on valid inputs. An exception becomes a failure that names the mistake."""
    try:
        return call()
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")


def _linear_scan(probs, uniforms):
    """Independent reference: walk the running sums and stop at the first one above each u."""
    cdf = list(itertools.accumulate(float(p) for p in probs))
    last = len(cdf) - 1
    chosen = []
    for u in uniforms:
        category = last
        for i, bound in enumerate(cdf):
            if u < bound:
                category = i
                break
        chosen.append(category)
    return np.asarray(chosen, dtype=np.intp)


def _raises_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@settings(max_examples=150, deadline=None, report_multiple_bugs=False)
@given(
    weights=st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=1, max_size=8).filter(
        lambda w: sum(w) > 0
    ),
    seed=st.integers(min_value=0, max_value=10**6),
    n=st.integers(min_value=0, max_value=60),
)
def test_matches_a_linear_scan_for_the_same_uniforms(weights, seed, n):
    probs = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    u = np.random.default_rng(seed).random(n)
    got = _run(
        lambda: sample_discrete(probs, n, _Uniforms(u)),
        "valid inputs must not raise; the categories come from rng.random(n) and the interval "
        "rule: sample_discrete raised",
    )
    np.testing.assert_array_equal(
        got,
        _linear_scan(probs, u),
        err_msg="categories differ from the linear scan of the running sums: check the half-open "
        "rule c[i-1] <= u < c[i] and the zero-probability categories",
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3], ids=["seed-a", "seed-b", "seed-c", "seed-d"])
def test_seeded_draws_match_the_linear_scan_of_the_same_uniforms(seed):
    probs = np.array([0.05, 0.0, 0.15, 0.3, 0.5])
    n = 500
    got = _run(
        lambda: sample_discrete(probs, n, np.random.default_rng(seed)),
        "valid inputs must not raise: sample_discrete raised",
    )
    np.testing.assert_array_equal(
        got,
        _linear_scan(probs, np.random.default_rng(seed).random(n)),
        err_msg="seeded draws differ from the protocol: use u = rng.random(n), "
        "then the interval rule",
    )


def test_chi_square_goodness_of_fit_at_a_fixed_seed():
    probs = np.array([0.05, 0.0, 0.15, 0.3, 0.5])
    draws = _run(
        lambda: sample_discrete(probs, 20_000, np.random.default_rng(11)),
        "valid inputs must not raise: sample_discrete raised",
    )
    counts = np.bincount(draws, minlength=len(probs))
    assert counts[1] == 0, "a category with probability zero was returned"
    nonzero = probs > 0
    p_value = stats.chisquare(counts[nonzero], f_exp=probs[nonzero] * draws.size).pvalue
    assert p_value > 0.001, (
        "category frequencies do not match the probabilities: check the interval of each category"
    )


def test_a_draw_exactly_at_a_cumulative_sum_goes_to_the_next_category():
    probs = [0.5, 0.25, 0.25]
    got = _run(
        lambda: sample_discrete(probs, 3, _Uniforms([0.5, 0.75, 0.4999999999999999])),
        "valid inputs must not raise: sample_discrete raised",
    )
    np.testing.assert_array_equal(
        got,
        [1, 2, 0],
        err_msg="a draw equal to a cumulative sum must go to the category that starts there",
    )


def test_zero_probability_category_is_never_returned():
    probs = [0.5, 0.0, 0.5]
    boundary = _run(
        lambda: sample_discrete(probs, 4, _Uniforms([0.5, 0.4999999999999999, 0.0, 0.9999])),
        "valid inputs must not raise: sample_discrete raised",
    )
    np.testing.assert_array_equal(
        boundary,
        [2, 0, 0, 2],
        err_msg="a category with probability zero owns an empty stretch: it must never be drawn",
    )
    random_draws = _run(
        lambda: sample_discrete(probs, 20_000, np.random.default_rng(5)),
        "valid inputs must not raise: sample_discrete raised",
    )
    assert not np.any(random_draws == 1), "a category with probability zero was drawn"


@pytest.mark.parametrize("probs", [[0.1] * 10, [0.6, 0.3, 0.1]], ids=["ten-equal", "three-way"])
def test_draw_above_the_last_cumulative_sum_maps_to_the_last_category(probs):
    got = _run(
        lambda: sample_discrete(probs, 1, _Uniforms([TOP])),
        "valid inputs must not raise: sample_discrete raised",
    )
    assert np.asarray(got).tolist() == [len(probs) - 1], (
        "a draw above the last cumulative sum must map to the last category: clip the index"
    )


def test_probabilities_off_by_a_small_real_amount_are_rejected():
    _raises_value_error(
        lambda: sample_discrete([0.5, 0.500000005], 3, np.random.default_rng(0)),
        "probabilities summing to slightly more than 1 are not a distribution: the tolerance "
        "is 1e-9, so raise ValueError",
    )


def test_probabilities_that_sum_to_one_only_up_to_rounding_are_accepted():
    draws = _run(
        lambda: sample_discrete([0.6, 0.3, 0.1], 50, np.random.default_rng(2)),
        "probabilities whose sum is 1 up to rounding must be accepted: sample_discrete raised. "
        "Compare the sum with a tolerance, not with ==",
    )
    assert draws.shape == (50,), "one category per requested draw"


@pytest.mark.parametrize(
    "bad",
    [[0.5, 0.6], [0.2, 0.3], [-0.1, 1.1], [float("nan"), 1.0], [], [[0.5, 0.5]]],
    ids=["sum-too-big", "sum-too-small", "negative-entry", "nan-entry", "empty", "2-d"],
)
def test_invalid_probabilities_raise_value_error(bad):
    _raises_value_error(
        lambda: sample_discrete(bad, 3, np.random.default_rng(0)),
        "invalid probabilities: raise ValueError for a sum other than 1, a negative or NaN entry, "
        "or an empty or 2-D vector",
    )


def test_negative_count_raises_value_error():
    # The stand-in does not reject a negative count itself, so the check must be the learner's.
    _raises_value_error(
        lambda: sample_discrete([0.5, 0.5], -1, _Uniforms([0.1, 0.2])),
        "a negative number of draws is invalid: raise ValueError",
    )


def test_draws_come_from_one_call_to_random_with_n():
    generator = _Uniforms([0.1, 0.4, 0.9])
    _run(
        lambda: sample_discrete([0.5, 0.5], 3, generator),
        "the draws must come from rng.random(n) alone: sample_discrete raised",
    )
    assert generator.calls == [3], (
        "the draws must come from one call rng.random(n), not from one call per draw"
    )


def test_output_is_an_integer_array_of_length_n_and_empty_for_zero_draws():
    draws = _run(
        lambda: sample_discrete([0.5, 0.5], 9, np.random.default_rng(4)),
        "valid inputs must not raise: sample_discrete raised",
    )
    assert draws.shape == (9,), "the output must have one category per requested draw"
    assert draws.dtype.kind == "i", "category indices must be integers"
    empty = _run(
        lambda: sample_discrete([0.5, 0.5], 0, np.random.default_rng(4)),
        "zero draws are valid input: sample_discrete raised",
    )
    assert empty.shape == (0,), "zero draws should give an empty array"


def test_probabilities_are_not_mutated():
    probs = np.array([0.2, 0.3, 0.5])
    before = probs.copy()
    _run(
        lambda: sample_discrete(probs, 10, np.random.default_rng(0)),
        "valid inputs must not raise: sample_discrete raised",
    )
    np.testing.assert_array_equal(probs, before, err_msg="probs was modified: work on a copy")


def test_sum_just_outside_the_tolerance_is_rejected():
    # The sum is off by about 1.5e-9, which is outside the 1e-9 tolerance the statement sets.
    _raises_value_error(
        lambda: sample_discrete([0.5, 0.5 + 1.5e-9], 3, np.random.default_rng(0)),
        "probabilities whose sum is more than the 1e-9 tolerance away from 1 must raise ValueError",
    )
