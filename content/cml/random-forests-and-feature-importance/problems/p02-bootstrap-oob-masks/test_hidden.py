from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import bootstrap_oob

from dogma_data import load


def _call(n, n_trees, seed):
    """Call the function and check the answer's types and shapes. Messages never show inputs."""
    try:
        result = bootstrap_oob(n, n_trees, seed)
    except ValueError:
        pytest.fail("valid input was rejected: check the input rules in the contract")
    except Exception as exc:
        pytest.fail(f"valid input crashed the call ({type(exc).__name__}): check the contract")
    if not isinstance(result, tuple) or len(result) != 2:
        pytest.fail("bootstrap_oob must return an (in_bag, oob) pair")
    in_bag, oob = (np.asarray(part) for part in result)
    if in_bag.shape != (n_trees, n) or oob.shape != (n_trees, n):
        pytest.fail("shape mismatch: both outputs need shape (n_trees, n)")
    return in_bag, oob


def _reference(n, n_trees, seed):
    """Independent counter: plain Python lists, one draw at a time, same generator protocol."""
    rng = np.random.default_rng(seed)
    counts_per_tree = []
    for _ in range(n_trees):
        drawn = [int(i) for i in rng.integers(0, n, size=n)]
        counts = [0] * n
        for index in drawn:
            counts[index] += 1
        counts_per_tree.append(counts)
    return np.array(counts_per_tree, dtype=np.int64)


def _expect_value_error(mistake, n, n_trees, seed):
    try:
        bootstrap_oob(n, n_trees, seed)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the function raised {type(exc).__name__} instead")
    pytest.fail(f"{mistake}; the function returned samples instead")


REAL_CASES = [("iris", 5, 3), ("wine", 4, 4)]


def test_matches_an_independent_counter_that_follows_the_protocol():
    # The last two sizes are the row counts of real datasets: iris (150) and wine (178).
    real = [(int(load(name).X.shape[0]), trees, seed) for name, trees, seed in REAL_CASES]
    for n, n_trees, seed in [(5, 3, 0), (17, 4, 1), (40, 6, 2), *real]:
        in_bag, _ = _call(n, n_trees, seed)
        np.testing.assert_array_equal(
            in_bag,
            _reference(n, n_trees, seed),
            err_msg="in-bag counts disagree with the protocol: one generator, trees in order, "
            "each tree draws n indices with replacement",
        )


def test_every_tree_sums_to_n_and_oob_is_the_zero_count_mask():
    for seed in range(5):
        in_bag, oob = _call(23, 4, seed)
        np.testing.assert_array_equal(
            in_bag.sum(axis=1), np.full(4, 23), err_msg="a tree's counts must sum to n"
        )
        np.testing.assert_array_equal(
            oob, in_bag == 0, err_msg="the OOB mask must be True exactly for zero counts"
        )


def test_out_of_bag_fraction_is_close_to_one_over_e_on_a_large_sample():
    # One bootstrap sample per tree of the 1,797 handwritten digits.
    n = int(load("digits").X.shape[0])
    _, oob = _call(n, 20, 11)
    fraction = float(oob.mean())
    assert abs(fraction - 1.0 / math.e) <= 0.01, (
        "the out-of-bag fraction should be near 1/e: draw with replacement, not without"
    )


def test_different_trees_get_different_bootstrap_samples():
    in_bag, _ = _call(50, 3, 4)
    assert not np.array_equal(in_bag[0], in_bag[1]) and not np.array_equal(in_bag[1], in_bag[2]), (
        "each tree must draw its own sample from the shared generator"
    )


def test_one_row_is_always_in_bag_with_a_count_of_one():
    in_bag, oob = _call(1, 3, 0)
    np.testing.assert_array_equal(
        in_bag,
        np.ones((3, 1), dtype=np.int64),
        err_msg="a one-row bootstrap must select its sole row on every draw",
    )
    assert not oob.any(), "with one row nothing is out of bag"


def test_same_seed_reproduces_and_another_seed_changes_the_samples():
    a, _ = _call(30, 3, 9)
    b, _ = _call(30, 3, 9)
    c, _ = _call(30, 3, 10)
    np.testing.assert_array_equal(a, b, err_msg="the same seed must reproduce the samples")
    assert not np.array_equal(a, c), "a different seed must change the samples"


def test_dtypes_are_integer_counts_and_boolean_mask():
    in_bag, oob = _call(7, 2, 0)
    assert np.issubdtype(in_bag.dtype, np.integer), "in-bag counts must be integers"
    assert oob.dtype == bool, "the out-of-bag mask must be boolean"


@pytest.mark.parametrize(
    ("n", "n_trees", "seed", "mistake"),
    [
        (0, 2, 0, "zero rows must raise ValueError"),
        (3, 0, 0, "zero trees must raise ValueError"),
        (3, 2, -1, "a negative seed must raise ValueError"),
        (2.0, 2, 0, "a float number of rows must raise ValueError"),
        (True, 2, 0, "a boolean row count must raise ValueError"),
    ],
    ids=["no-rows", "no-trees", "negative-seed", "float-rows", "bool-rows"],
)
def test_invalid_arguments_raise_value_error(n, n_trees, seed, mistake):
    _expect_value_error(mistake, n, n_trees, seed)


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.integers(min_value=1, max_value=30),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=0, max_value=10_000),
)
def test_counts_are_non_negative_and_consistent_with_the_mask(n, n_trees, seed):
    in_bag, oob = _call(n, n_trees, seed)
    assert (in_bag >= 0).all(), "a count cannot be negative"
    assert (in_bag.sum(axis=1) == n).all(), "each tree's counts must sum to n"
    assert (oob == (in_bag == 0)).all(), "the mask must agree with the zero counts"


def test_fractional_seed_raises_value_error():
    _expect_value_error(
        "a fractional seed must raise ValueError: seed must be a non-negative integer",
        3,
        2,
        2.5,
    )
