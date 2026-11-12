from __future__ import annotations

import math
import signal
from collections import Counter
from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from sklearn.tree import DecisionTreeClassifier
from submission import best_split

from dogma_data import rng, sample_rows

TIME_BUDGET_S = 10.0
TIE_TOLERANCE = 1e-12


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


class _OverBudget(BaseException):
    """Raised inside the learner's call when the time budget runs out. Not an Exception, so
    _call does not turn it into a crash message."""


def _stop_at_budget(signum, frame):
    raise _OverBudget


def _call(x, y, **kwargs):
    """Call best_split and check the shape of the answer. Messages never show the inputs."""
    try:
        result = best_split(x, y, **kwargs)
    except Exception as exc:
        pytest.fail(
            f"valid input was rejected or crashed ({type(exc).__name__}): "
            "check the input rules in the contract"
        )
    if not isinstance(result, tuple) or len(result) != 2:
        pytest.fail("best_split must return a (threshold, impurity_decrease) pair")
    threshold, decrease = result
    if threshold is not None:
        try:
            threshold = float(threshold)
            decrease = float(decrease)
        except (TypeError, ValueError):
            pytest.fail("the threshold and the decrease must be numbers, or None for the threshold")
    return threshold, decrease


def _timed_call(x, y, **kwargs):
    """Run the call under an interval timer: a too-slow scan fails instead of hanging."""
    previous = signal.signal(signal.SIGALRM, _stop_at_budget)
    signal.setitimer(signal.ITIMER_REAL, TIME_BUDGET_S)
    try:
        return _call(x, y, **kwargs)
    except _OverBudget:
        pytest.fail(
            "the call did not finish within the time budget: sort once and scan prefix counts, "
            "instead of rescanning the rows for every candidate threshold"
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _expect_value_error(x, y, mistake, **kwargs):
    try:
        best_split(x, y, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__} instead")
    pytest.fail(f"{mistake}; the function returned a value instead")


def _impurity(labels, criterion):
    """Independent impurity from class counts: Gini, or entropy in bits."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels).values()
    if criterion == "gini":
        return 1.0 - sum((c / n) ** 2 for c in counts)
    return -sum((c / n) * math.log2(c / n) for c in counts)


def _brute_force(x, y, criterion):
    """Test every midpoint of consecutive distinct values, in increasing order, with the same
    tie rule as the contract. No sorting tricks, no prefix sums: the reference is the definition."""
    values = sorted(set(x))
    if len(values) < 2:
        return None, 0.0
    parent = _impurity(list(y), criterion)
    n = len(y)
    best_t, best_d = None, None
    for a, b in pairwise(values):
        t = (a + b) / 2.0
        left = [yy for xx, yy in zip(x, y, strict=True) if xx <= t]
        right = [yy for xx, yy in zip(x, y, strict=True) if xx > t]
        d = parent - (len(left) / n) * _impurity(left, criterion)
        d -= (len(right) / n) * _impurity(right, criterion)
        if best_t is None or d > best_d + TIE_TOLERANCE:
            best_t, best_d = t, d
    return best_t, best_d


def test_matches_brute_force_search_under_both_criteria():
    for seed in range(10):
        # One real breast_cancer feature and the real diagnosis; measurements repeat.
        gen = rng(f"best-split-p01:brute-force:{seed}")
        n = int(gen.integers(5, 40))
        feature, target = sample_rows("breast_cancer", gen, n, 1)
        x = feature[:, 0]
        y = target.astype(int).tolist()
        for criterion in ("gini", "entropy"):
            want_t, want_d = _brute_force(x.tolist(), y, criterion)
            got_t, got_d = _call(x, y, criterion=criterion)
            assert got_t is not None and abs(got_t - want_t) <= TIE_TOLERANCE, (
                "threshold differs from the brute-force optimum: thresholds are midpoints of "
                "consecutive sorted values, and x <= t goes left"
            )
            np.testing.assert_allclose(
                got_d,
                want_d,
                rtol=1e-9,
                atol=1e-12,
                err_msg="impurity decrease is wrong: weight each child by its share of the rows",
            )


@pytest.mark.parametrize("criterion", ["gini", "entropy"], ids=["gini", "entropy"])
def test_threshold_agrees_with_a_depth_one_scikit_learn_tree(criterion):
    rng = np.random.default_rng(21)
    x = _real_normal(rng, 60)  # continuous values: no ties, so the optimum is unique
    y = ((x - 0.3 + _real_normal(rng, 60, scale=0.5)) > 0).astype(int)
    stump = DecisionTreeClassifier(max_depth=1, criterion=criterion, random_state=0)
    stump.fit(x.reshape(-1, 1), y)
    want = float(stump.tree_.threshold[0])
    got_t, _ = _call(x, y, criterion=criterion)
    assert got_t is not None and abs(got_t - want) <= 1e-6, (
        "threshold disagrees with the depth-one tree: check the midpoints "
        "and which side x <= t goes"
    )


def test_exact_tie_returns_the_smallest_threshold_under_both_criteria():
    for criterion in ("gini", "entropy"):
        threshold, _ = _call([0.0, 1.0, 2.0, 3.0], [0, 1, 1, 0], criterion=criterion)
        assert threshold == pytest.approx(0.5, abs=1e-12), (
            "tied candidates must return the smallest threshold, not the largest"
        )


def test_exact_tie_hidden_by_float_noise_still_goes_to_the_smallest_threshold():
    # Candidates after the 2nd and the 6th sorted value tie exactly at 1/24 (Gini). In floating
    # point the later one comes out a few ulps larger; the tie tolerance must still keep the
    # earlier threshold, -0.578, not the later one near 0.2965.
    x = [-0.696, -0.627, -0.529, -0.181, 0.102, 0.262, 0.331, 0.539]
    y = [0, 0, 1, 0, 0, 0, 1, 0]
    threshold, decrease = _call(x, y, criterion="gini")
    assert threshold is not None and abs(threshold - (-0.578)) <= 1e-9, (
        "an exact tie must go to the smallest threshold even when rounding favours a later one: "
        "replace the incumbent only when the gain is larger by more than the tolerance"
    )
    assert abs(decrease - 1.0 / 24.0) <= 1e-12, (
        "the reported Gini decrease must equal the weighted impurity drop at th"
        "e selected tied threshold"
    )


def test_row_order_does_not_change_the_answer():
    gen = rng("best-split-p01:row-order")
    feature, target = sample_rows("iris", gen, 25, 1)  # three species; lengths repeat
    x = feature[:, 0]
    y = target.astype(int)
    perm = gen.permutation(25)
    t1, d1 = _call(x, y, criterion="entropy")
    t2, d2 = _call(x[perm], y[perm], criterion="entropy")
    assert t1 == pytest.approx(t2, abs=1e-12), "the result depends on the order of the rows"
    assert d1 == pytest.approx(d2, abs=1e-12), "the decrease depends on the order of the rows"


def test_duplicate_values_never_create_a_threshold_between_equal_values():
    x = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    y = [0, 0, 1, 1, 0, 0]
    threshold, decrease = _call(x, y)
    assert threshold is not None and min(abs(threshold - 1.5), abs(threshold - 2.5)) <= 1e-12, (
        "a threshold must lie between two distinct values; equal values have no midpoint split"
    )
    assert decrease == pytest.approx(1.0 / 9.0, abs=1e-12), (
        "the decrease must use child sizes and class proportions correctly when"
        " feature values repeat"
    )


def test_single_row_has_no_valid_split():
    assert _call([2.0], [1]) == (None, 0.0), "one row cannot be split: return (None, 0.0)"


def test_pure_labels_return_the_smallest_candidate_with_zero_decrease():
    threshold, decrease = _call([3.0, 1.0, 2.0], ["a", "a", "a"])
    assert threshold == pytest.approx(1.5, abs=1e-12), (
        "pure labels still give a valid split: the smallest midpoint, with decrease 0"
    )
    assert abs(decrease) <= 1e-12, "a split of pure labels decreases impurity by zero"


def test_large_input_finishes_within_the_time_budget_and_matches_a_stump():
    rng = np.random.default_rng(30)
    n = 30_000
    x = _real_normal(rng, n)
    y = ((x + _real_normal(rng, n, scale=1.0)) > 0).astype(int)
    stump = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=0)
    stump.fit(x.reshape(-1, 1), y)
    got_t, _ = _timed_call(x, y, criterion="gini")
    assert got_t is not None and abs(got_t - float(stump.tree_.threshold[0])) <= 1e-5, (
        "large-input threshold disagrees with the depth-one tree: the scan must be exact on "
        "every candidate, not on a subsample"
    )


def test_unknown_criterion_raises_value_error():
    _expect_value_error(
        [0.0, 1.0], [0, 1], "an unknown criterion must raise ValueError", criterion="log-loss"
    )


def test_empty_x_raises_value_error():
    _expect_value_error([], [], "an empty feature must raise ValueError")


def test_nan_in_x_raises_value_error():
    _expect_value_error([0.0, float("nan"), 2.0], [0, 1, 1], "a NaN feature must raise ValueError")


def test_two_dimensional_x_raises_value_error():
    _expect_value_error(
        np.zeros((3, 2)), [0, 1, 1], "a two-dimensional feature must raise ValueError"
    )


def test_mismatched_labels_raise_value_error():
    _expect_value_error(
        [0.0, 1.0, 2.0], [0, 1], "labels of a different length must raise ValueError"
    )


def test_inputs_are_not_mutated():
    x = np.array([2.0, -1.0, 0.5, 3.0])
    y = np.array([1, 0, 0, 1])
    x0, y0 = x.copy(), y.copy()
    _call(x, y, criterion="entropy")
    np.testing.assert_array_equal(x, x0, err_msg="x was modified in place: sort a copy")
    np.testing.assert_array_equal(y, y0, err_msg="y was modified in place")


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=25),
    st.integers(min_value=0, max_value=10_000),
    st.sampled_from(["gini", "entropy"]),
)
def test_threshold_is_a_midpoint_and_the_decrease_is_not_negative(xs, seed, criterion):
    x = [float(v) for v in xs]
    rng = np.random.default_rng(seed)
    y = _real_ints(rng, 0, 3, len(x)).tolist()
    threshold, decrease = _call(x, y, criterion=criterion)
    distinct = sorted(set(x))
    if len(distinct) < 2:
        assert threshold is None, "a single distinct value has no threshold"
        return
    midpoints = {(a + b) / 2.0 for a, b in pairwise(distinct)}
    assert threshold is not None and any(abs(threshold - m) <= 1e-12 for m in midpoints), (
        "the threshold must be a midpoint of two consecutive distinct values"
    )
    assert decrease >= -1e-12, "the impurity decrease can never be negative"


def test_large_finite_features_have_a_finite_midpoint():
    threshold, decrease = _call([1e308, 1.4e308], [0, 1])
    assert threshold is not None and math.isfinite(threshold), (
        "midpoint overflow: finite feature endpoints need a finite split threshold"
    )
    assert math.isclose(threshold, 1.2e308, rel_tol=1e-12), (
        "the threshold must be the midpoint of the selected distinct feature values"
    )
    assert math.isclose(decrease, 0.5, abs_tol=1e-12), (
        "the reported decrease must agree with the returned threshold's partition"
    )


def test_distinct_scalar_label_types_are_not_coerced_to_one_class():
    threshold, decrease = _call([0.0, 1.0, 2.0, 3.0], [1, "1", 1, "1"])
    assert threshold == pytest.approx(0.5, abs=1e-12), (
        "class-label equality must preserve distinct scalar types when choosing the threshold"
    )
    assert decrease == pytest.approx(1.0 / 6.0, abs=1e-12), (
        "class-label coercion must not merge unequal numeric and string labels"
    )
