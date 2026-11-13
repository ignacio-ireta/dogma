import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from submission import kmeans_plus_plus

from dogma_data import rng as data_rng
from dogma_data import sample_rows

RATIO_COLUMNS = (2, 3, 5)  # rooms, bedrooms and occupants per household


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


def _points(gen, n, d):
    """n real block groups of california_housing with d per-household averages, standardised.

    The averages are ratios of counts, so no two distances tie exactly, and they are
    heavy-tailed: a few block groups sit far from everything else, as real outliers do.
    """
    X, _ = sample_rows("california_housing", gen, n, standardize=True)
    return X[:, RATIO_COLUMNS[:d]]


def _call(X, k, seed):
    """Run with NumPy warnings as errors; a crash on valid input is a contract failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = kmeans_plus_plus(X, k, seed=seed)
    except RuntimeWarning:
        pytest.fail("the D^2 arithmetic emitted a NumPy warning: check overflow and zero sums")
    except Exception as exc:
        pytest.fail(f"valid input must be accepted; the call raised {type(exc).__name__}")
    out = np.asarray(out)
    if out.shape != (k, np.asarray(X).shape[1]):
        pytest.fail("shape mismatch: return a (k, d) array of centres")
    return out


def _expect_value_error(mistake, X, k, seed):
    try:
        kmeans_plus_plus(X, k, seed=seed)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the call raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{mistake}; the call returned centres instead of raising ValueError")


def _reference(X, k, seed):
    """The protocol, written again: fresh distances to every chosen centre each step."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(X)
    chosen = [int(rng.integers(n))]
    for _ in range(1, k):
        d2 = np.min(
            np.stack([np.sum((X - X[c]) ** 2, axis=1) for c in chosen]),
            axis=0,
        )
        chosen.append(int(rng.choice(n, p=d2 / d2.sum())))
    return X[chosen]


def test_matches_the_protocol_exactly_on_random_data():
    for seed in range(15):
        rng = data_rng(f"kmeans-plus-plus-p02:protocol:{seed}")
        n = int(rng.integers(4, 30))
        d = int(rng.integers(1, 4))
        X = _points(rng, n, d)
        k = int(rng.integers(1, min(6, n) + 1))
        np.testing.assert_array_equal(
            _call(X, k, seed),
            _reference(X, k, seed),
            err_msg="centres differ from the k-means++ protocol: seeded integers, then choice",
        )


def test_second_centre_frequencies_follow_squared_distance_weights():
    X = np.array([[0.0], [1.0], [3.0], [7.0]])
    n = len(X)
    # The first centre is uniform over rows, so the second centre has this mixture law.
    expected = np.zeros(n)
    for first in range(n):
        d2 = (X[:, 0] - X[first, 0]) ** 2
        expected += (d2 / d2.sum()) / n
    counts = np.zeros(n)
    trials = 3000
    for seed in range(trials):
        centre = _call(X, 2, seed)[1, 0]
        counts[int(np.flatnonzero(X[:, 0] == centre)[0])] += 1
    freq = counts / trials
    assert np.max(np.abs(freq - expected)) <= 0.04, (
        "second-centre frequencies do not follow the squared-distance weights: sample with "
        "probability proportional to D^2, not D and not uniform"
    )


def _sorted_rows(a):
    """Rows of a sorted lexicographically, so two row sets compare regardless of order."""
    return a[np.lexsort(a.T[::-1])]


def test_k_equal_to_n_selects_every_row_exactly_once():
    rng = data_rng("kmeans-plus-plus-p02:910")
    X = _points(rng, 7, 2)
    out = _call(X, 7, seed=5)
    np.testing.assert_array_equal(
        _sorted_rows(out),
        _sorted_rows(X),
        err_msg="with k = n every distinct row must be chosen exactly once",
    )


def test_centres_are_copies_of_distinct_rows():
    rng = data_rng("kmeans-plus-plus-p02:920")
    X = _points(rng, 25, 3)
    for seed in range(10):
        out = _call(X, 6, seed)
        for centre in out:
            matches = np.all(np.equal(X, centre), axis=1)
            assert matches.any(), "every centre must be a row of X, copied exactly"
        assert len({tuple(c) for c in out}) == 6, (
            "centres must be distinct when X has at least k distinct rows"
        )


def test_same_seed_reproduces_and_different_seeds_can_differ():
    rng = data_rng("kmeans-plus-plus-p02:930")
    X = _points(rng, 30, 2)
    a = _call(X, 4, seed=11)
    b = _call(X, 4, seed=11)
    np.testing.assert_array_equal(a, b, err_msg="the same seed must reproduce the same centres")
    outputs = {tuple(map(tuple, _call(X, 4, seed=s))) for s in range(12)}
    assert len(outputs) > 1, "different seeds should not all give the same centres"


def test_power_of_two_rescaling_gives_the_same_choices():
    rng = data_rng("kmeans-plus-plus-p02:940")
    X = _points(rng, 20, 2)
    base = _call(X, 4, seed=2)
    scaled = _call(X * 2.0**-30, 4, seed=2)
    np.testing.assert_array_equal(
        scaled,
        base * 2.0**-30,
        err_msg="rescaling by a power of two must give exactly scaled centres: compute D^2 from "
        "the data itself, without a fixed threshold",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000), st.integers(min_value=1, max_value=5))
def test_protocol_matches_reference_on_small_random_sets(seed, k):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 12))
    X = _real_ints(rng, -4, 5, (n, 2)).astype(float)
    if len(np.unique(X, axis=0)) < k:
        return
    np.testing.assert_array_equal(
        _call(X, k, seed),
        _reference(X, k, seed),
        err_msg="centres differ from the k-means++ protocol on a small integer set",
    )


INVALID = [
    ("k-zero", [[0.0], [1.0]], 0, 0),
    ("k-above-n", [[0.0], [1.0]], 3, 0),
    ("k-bool", [[0.0], [1.0]], True, 0),
    ("seed-negative", [[0.0], [1.0]], 1, -1),
    ("seed-float", [[0.0], [1.0]], 1, 2.0),
    ("seed-bool", [[0.0], [1.0]], 1, True),
    ("too-few-distinct-rows", [[1.0], [1.0], [2.0]], 3, 0),
    ("one-dimensional-x", [0.0, 1.0], 1, 0),
    ("nan-in-x", [[0.0], [np.nan]], 1, 0),
]


@pytest.mark.parametrize(
    ("X", "k", "seed"),
    [case[1:] for case in INVALID],
    ids=[case[0] for case in INVALID],
)
def test_invalid_inputs_raise_value_error(X, k, seed):
    _expect_value_error("k, seed, X shape or distinct-row rules must be enforced", X, k, seed)


def test_inputs_are_not_mutated():
    rng = data_rng("kmeans-plus-plus-p02:950")
    X = _points(rng, 12, 2)
    X0 = X.copy()
    _call(X, 3, seed=4)
    np.testing.assert_array_equal(X, X0, err_msg="X was modified in place")


def test_extreme_power_of_two_rescaling_preserves_seed_choices():
    X = np.array([[0.0], [1.0], [3.0], [7.0]])
    baseline = _call(X, 3, seed=0)
    for exponent in (-600, 600):
        scale = 2.0**exponent
        out = _call(X * scale, 3, seed=0)
        np.testing.assert_array_equal(
            out,
            baseline * scale,
            err_msg="finite power-of-two scaling must not destroy the squared-distance "
            "sampling probabilities",
        )


def test_empty_one_dimensional_input_raises_value_error():
    _expect_value_error("X must be a 2-D array with at least one row and one column", [], 1, 0)


def test_single_row_is_a_valid_input_for_one_centre():
    out = _call([[2.0, 3.0]], 1, seed=0)
    np.testing.assert_array_equal(
        out,
        [[2.0, 3.0]],
        err_msg="a single row with k = 1 is valid and must be returned as the one centre",
    )


def test_large_negative_values_do_not_overflow_the_squared_distances():
    # The same data scaled by an exact power of two must give exactly scaled centres, even when
    # the largest value is small and the largest magnitude is a large negative number.
    small = np.array([[-1.0], [0.0], [2.0**-600]])
    big = small * 2.0**600
    expected = _call(small, 2, seed=0) * 2.0**600
    np.testing.assert_array_equal(
        _call(big, 2, seed=0),
        expected,
        err_msg="large negative values made the squared distances overflow: scale by the largest "
        "magnitude, not the largest value",
    )


def test_integer_samples_give_float64_centres():
    # The samples are integers, but the Contract promises a float64 array of centres.
    out = _call([[0, 1], [3, 7], [2, 2]], 2, 0)
    assert out.dtype == np.float64, "the centres must be a float64 array whatever the dtype of X"


def test_three_dimensional_input_raises_value_error():
    _expect_value_error(
        "a sample array with more than two dimensions must be rejected", np.ones((3, 2, 2)), 1, 0
    )


def test_centres_come_back_as_a_numpy_array():
    out = kmeans_plus_plus([[0.0], [1.0], [3.0], [7.0]], 2, seed=0)
    assert isinstance(out, np.ndarray), (
        "the result must be a NumPy array: a list of the same numbers is another type"
    )
