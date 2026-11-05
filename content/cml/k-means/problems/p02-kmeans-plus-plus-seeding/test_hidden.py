import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import kmeans_plus_plus


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
        rng = np.random.default_rng(900 + seed)
        n = int(rng.integers(4, 30))
        d = int(rng.integers(1, 4))
        X = rng.normal(size=(n, d))
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
    rng = np.random.default_rng(910)
    X = rng.normal(size=(7, 2))
    out = _call(X, 7, seed=5)
    np.testing.assert_array_equal(
        _sorted_rows(out),
        _sorted_rows(X),
        err_msg="with k = n every distinct row must be chosen exactly once",
    )


def test_centres_are_copies_of_distinct_rows():
    rng = np.random.default_rng(920)
    X = rng.normal(size=(25, 3))
    for seed in range(10):
        out = _call(X, 6, seed)
        for centre in out:
            matches = np.all(np.equal(X, centre), axis=1)
            assert matches.any(), "every centre must be a row of X, copied exactly"
        assert len({tuple(c) for c in out}) == 6, (
            "centres must be distinct when X has at least k distinct rows"
        )


def test_same_seed_reproduces_and_different_seeds_can_differ():
    rng = np.random.default_rng(930)
    X = rng.normal(size=(30, 2))
    a = _call(X, 4, seed=11)
    b = _call(X, 4, seed=11)
    np.testing.assert_array_equal(a, b, err_msg="the same seed must reproduce the same centres")
    outputs = {tuple(map(tuple, _call(X, 4, seed=s))) for s in range(12)}
    assert len(outputs) > 1, "different seeds should not all give the same centres"


def test_power_of_two_rescaling_gives_the_same_choices():
    rng = np.random.default_rng(940)
    X = rng.normal(size=(20, 2))
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
    X = rng.integers(-4, 5, size=(n, 2)).astype(float)
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
    rng = np.random.default_rng(950)
    X = rng.normal(size=(12, 2))
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
