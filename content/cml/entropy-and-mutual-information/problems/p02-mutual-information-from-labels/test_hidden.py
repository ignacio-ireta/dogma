from __future__ import annotations

import functools
import math
import signal
from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from sklearn.metrics import mutual_info_score
from submission import mutual_information

from dogma_data import load_frame, rng, sample_rows

TIME_BUDGET_S = 10.0


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


@functools.cache
def _adult():
    """The adult census table, read once: real categorical columns with very uneven shares."""
    return load_frame("adult")


CATEGORICAL = ("workclass", "education", "marital.status", "occupation", "relationship", "race")


def _real_labels(key, n):
    """Two different categorical columns of n people drawn from adult, as pandas Series."""
    gen = rng(key)
    table = _adult()
    first, second = gen.choice(len(CATEGORICAL), size=2, replace=False)
    rows = gen.choice(len(table), size=n, replace=False)
    return table[CATEGORICAL[first]].iloc[rows], table[CATEGORICAL[second]].iloc[rows]


def _call(x, y):
    """Call the function and check the value is a float. Messages never show the inputs."""
    try:
        value = mutual_information(x, y)
    except Exception as exc:
        pytest.fail(
            f"valid labels were rejected or crashed ({type(exc).__name__}): "
            "check the input rules in the contract"
        )
    if not isinstance(value, float):
        pytest.fail("the mutual information must be returned as a Python float")
    if not math.isfinite(value):
        pytest.fail("the mutual information is not finite: check the joint count table")
    return value


class _OverBudget(BaseException):
    """Raised inside the learner's call when the time budget runs out. Not an Exception, so
    _call does not turn it into a crash message."""


def _stop_at_budget(signum, frame):
    raise _OverBudget


def _timed_call(x, y):
    """Run the call under an interval timer: a too-slow scan fails instead of hanging."""
    previous = signal.signal(signal.SIGALRM, _stop_at_budget)
    signal.setitimer(signal.ITIMER_REAL, TIME_BUDGET_S)
    try:
        return _call(x, y)
    except _OverBudget:
        pytest.fail(
            "the call did not finish within the time budget: count the joint table in one "
            "pass, not by comparing every pair of rows"
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _expect_value_error(x, y, mistake):
    try:
        mutual_information(x, y)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__} instead")
    pytest.fail(f"{mistake}; the function returned a value instead")


def _entropy_nats(labels):
    counts = Counter(labels)
    n = len(labels)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def test_matches_sklearn_mutual_info_on_random_labels():
    for seed in range(12):
        n = int(np.random.default_rng(seed).integers(20, 150))
        x, y = _real_labels(f"mutual-information-p02:sklearn:{seed}", n)
        if seed % 2:  # the category names as they are in the table
            x_labels = x.tolist()
            y_labels = y.tolist()
        else:  # the same labels as integer codes
            x_labels = x.astype("category").cat.codes.tolist()
            y_labels = y.astype("category").cat.codes.tolist()
        np.testing.assert_allclose(
            _call(x_labels, y_labels),
            mutual_info_score(x_labels, y_labels),
            rtol=1e-9,
            atol=1e-12,
            err_msg="mutual information disagrees with the reference: use the joint counts and "
            "the natural logarithm, not bits or a correlation",
        )


def test_is_symmetric_in_its_two_arguments():
    x, y = (labels.tolist() for labels in _real_labels("mutual-information-p02:symmetry", 60))
    np.testing.assert_allclose(
        _call(x, y),
        _call(y, x),
        rtol=1e-12,
        atol=1e-12,
        err_msg="mutual information must be symmetric: I(X;Y) equals I(Y;X)",
    )


def test_is_invariant_to_renaming_the_labels():
    rng = np.random.default_rng(4)
    x = _real_ints(rng, 0, 3, 50).tolist()
    y = _real_ints(rng, 0, 4, 50).tolist()
    rename_x = {0: 7, 1: 3, 2: 9}
    rename_y = {0: "d", 1: "a", 2: "c", 3: "b"}
    np.testing.assert_allclose(
        _call([rename_x[v] for v in x], [rename_y[v] for v in y]),
        _call(x, y),
        rtol=1e-12,
        atol=1e-12,
        err_msg="renaming labels must not change the mutual information: compare labels, "
        "not their numeric values",
    )


@pytest.mark.parametrize("k", [3, 4], ids=["case-a", "case-b"])
def test_identical_balanced_labels_give_the_entropy_of_the_label(k):
    labels = [i % k for i in range(12 * k)]
    np.testing.assert_allclose(
        _call(labels, labels),
        math.log(k),
        rtol=1e-12,
        err_msg="identical labels must give the entropy of the label, which is log k nats",
    )


def test_independent_balanced_table_gives_zero():
    x = [0, 0, 1, 1, 0, 0, 1, 1]
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    assert abs(_call(x, y)) <= 1e-12, (
        "a balanced independent table must give zero mutual information"
    )


def test_constant_labels_carry_no_information():
    x = _real_labels("mutual-information-p02:constant", 40)[0].tolist()
    assert abs(_call(["only"] * 40, x)) <= 1e-12, (
        "a constant label carries no information about the other label"
    )


def test_detects_a_non_linear_dependence_that_correlation_misses():
    x = [-1, 0, 1] * 100
    y = [v * v for v in x]
    assert abs(np.corrcoef(x, y)[0, 1]) < 1e-9, "test data must have zero linear correlation"
    expected = -(1.0 / 3.0) * math.log(1.0 / 3.0) - (2.0 / 3.0) * math.log(2.0 / 3.0)
    np.testing.assert_allclose(
        _call(x, y),
        expected,
        rtol=1e-9,
        err_msg="a deterministic non-linear dependence must give the entropy of the output: "
        "the joint table sees it even when the linear correlation is zero",
    )


def test_empty_input_raises_value_error():
    _expect_value_error([], [], "empty labels must raise ValueError: there is no table to count")
    _expect_value_error([0, 1], [], "an empty second sequence must raise ValueError")


def test_unequal_lengths_raise_value_error():
    _expect_value_error(
        [0, 1, 1], [0, 1], "sequences of unequal length must raise ValueError, not be truncated"
    )


def test_two_dimensional_input_raises_value_error():
    _expect_value_error(
        np.zeros((2, 2)), [0, 1], "a two-dimensional label array must raise ValueError"
    )


def test_inputs_are_not_mutated():
    x = [2, 0, 1, 1, 0]
    y = ["b", "a", "a", "b", "a"]
    x_before, y_before = list(x), list(y)
    _call(x, y)
    assert x == x_before and y == y_before, "the inputs were modified: work on a copy of the labels"


def test_large_input_is_counted_in_one_pass_within_the_time_budget():
    rng = np.random.default_rng(11)
    n = 200_000
    x = _real_ints(rng, 0, 6, n)
    y = (x + _real_ints(rng, 0, 4, n)) % 6
    np.testing.assert_allclose(
        _timed_call(x.tolist(), y.tolist()),
        mutual_info_score(x, y),
        rtol=1e-9,
        atol=1e-12,
        err_msg="large input disagrees with the reference: check the joint counts",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=40),
    st.integers(min_value=0, max_value=10_000),
)
def test_lies_between_zero_and_the_smaller_entropy(x, seed):
    rng = np.random.default_rng(seed)
    y = _real_ints(rng, 0, 3, len(x)).tolist()
    value = _call(x, y)
    bound = min(_entropy_nats(x), _entropy_nats(y))
    assert -1e-12 <= value <= bound + 1e-12, (
        "mutual information must be non-negative and at most the smaller of the two entropies"
    )


def test_one_dimensional_array_of_labels_is_accepted():
    value = _call(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    assert value == pytest.approx(math.log(2.0), abs=1e-12), (
        "a one-dimensional array is a valid label sequence, the same as a list of the same labels"
    )


def test_nested_sequences_of_labels_raise_value_error():
    _expect_value_error(
        [[0, 0], [1, 1]],
        [0, 1],
        "labels must be scalars: a sequence of sequences must raise ValueError",
    )


def test_mixed_scalars_and_nested_sequences_raise_value_error():
    _expect_value_error(
        [0, [1], 1],
        [0, 1, 0],
        "labels must be scalars: a sequence that mixes scalars and sequences must raise ValueError",
    )


def test_shorter_first_sequence_raises_value_error_instead_of_truncating():
    # x has 2 labels and y has 3: the lengths differ, so the call must raise, not pair up 2 rows.
    _expect_value_error(
        [0, 1], [0, 1, 1], "a shorter first sequence must raise ValueError, not be truncated"
    )


def test_zero_dimensional_array_raises_value_error():
    _expect_value_error(
        np.array(1),
        [0, 1],
        "a zero-dimensional array is not one-dimensional: it must raise ValueError",
    )
