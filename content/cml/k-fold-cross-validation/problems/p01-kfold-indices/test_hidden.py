import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.model_selection import KFold
from submission import kfold_indices

UNSHUFFLED_CASES = [(10, 3), (7, 2), (12, 4), (9, 9), (25, 6)]
UNSHUFFLED_IDS = ["remainder-first", "two-folds-odd", "even-split", "leave-one-out", "many-folds"]
SHUFFLED_CASES = [(10, 3, 0), (6, 3, 0), (17, 5, 1), (8, 2, 7), (30, 4, 123)]
SHUFFLED_IDS = ["remainder", "equal-blocks", "uneven-blocks", "two-folds", "wide-folds"]


def _folds(n, k, **kwargs):
    """Call the learner's function and check the basic output shape, naming the mistake."""
    try:
        folds = kfold_indices(n, k, **kwargs)
    except ValueError:
        pytest.fail("a valid split raised ValueError: check the fold count and seed rules")
    except Exception as exc:
        pytest.fail(f"a valid split raised {type(exc).__name__}: check the index arithmetic")
    if len(folds) != k:
        pytest.fail("the function must return exactly one (train_idx, test_idx) pair per fold")
    for train, test in folds:
        if not (isinstance(train, np.ndarray) and isinstance(test, np.ndarray)):
            pytest.fail("train_idx and test_idx must be numpy arrays, not lists")
        if train.ndim != 1 or test.ndim != 1:
            pytest.fail("train_idx and test_idx must be 1-D arrays of positions")
    return folds


def _require(condition, mistake):
    if not condition:
        pytest.fail(mistake)


def _reference_shuffled(n, k, seed):
    """Independent reference of the documented protocol, written out in full."""
    perm = np.random.default_rng(seed).permutation(n)
    sizes = [n // k + (1 if j < n % k else 0) for j in range(k)]
    folds = []
    start = 0
    for size in sizes:
        test = np.sort(perm[start : start + size])
        mask = np.ones(n, dtype=bool)
        mask[test] = False
        folds.append((np.flatnonzero(mask), test))
        start += size
    return folds


@pytest.mark.parametrize(("n", "k"), UNSHUFFLED_CASES, ids=UNSHUFFLED_IDS)
def test_unshuffled_folds_match_sklearn_kfold(n, k):
    expected = list(KFold(n_splits=k).split(np.arange(n)))
    folds = _folds(n, k)
    for (train, test), (exp_train, exp_test) in zip(folds, expected, strict=True):
        _require(
            np.array_equal(test, exp_test),
            "unshuffled test block differs from KFold: use consecutive blocks in position order",
        )
        _require(
            np.array_equal(train, exp_train),
            "unshuffled training indices differ from KFold: train is the sorted complement of test",
        )


@pytest.mark.parametrize(("n", "k", "seed"), SHUFFLED_CASES, ids=SHUFFLED_IDS)
def test_shuffled_folds_follow_the_seeded_permutation_protocol(n, k, seed):
    expected = _reference_shuffled(n, k, seed)
    folds = _folds(n, k, shuffle=True, seed=seed)
    for train, test in folds:
        _require(
            bool(np.all(np.diff(train) > 0)) and bool(np.all(np.diff(test) > 0)),
            "shuffled index arrays must be sorted ascending: sort each fold's test and train",
        )
    for (train, test), (exp_train, exp_test) in zip(folds, expected, strict=True):
        _require(
            np.array_equal(test, exp_test),
            "shuffled test block differs from the protocol: chunk default_rng(seed).permutation(n)",
        )
        _require(
            np.array_equal(train, exp_train),
            "shuffled training indices must be the sorted complement of the test block",
        )


@pytest.mark.parametrize(
    ("n", "k"),
    [(11, 4), (10, 4), (7, 3), (13, 5)],
    ids=["one-extra", "two-extra", "two-of-three", "three-of-five"],
)
def test_fold_sizes_give_the_extra_positions_to_the_first_folds(n, k):
    sizes = [test.size for _, test in _folds(n, k)]
    q, r = divmod(n, k)
    expected = [q + 1] * r + [q] * (k - r)
    _require(
        sizes == expected,
        "fold sizes must put the n % k extra positions in the first folds, not the last",
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=2, max_value=60), st.data())
def test_every_position_is_tested_once_and_training_is_its_complement(n, data):
    k = data.draw(st.integers(min_value=2, max_value=n), label="k")
    shuffle = data.draw(st.booleans(), label="shuffle")
    seed = data.draw(st.integers(min_value=0, max_value=10_000), label="seed")
    folds = _folds(n, k, shuffle=shuffle, seed=seed)
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested.tolist()) == list(range(n)), (
        "test blocks must partition range(n): a position is tested twice or never"
    )
    for train, test in folds:
        assert np.intersect1d(train, test).size == 0, (
            "a training set contains positions of its own test set"
        )
        assert train.size + test.size == n, "train and test must together cover every position"


def test_leave_one_out_tests_exactly_one_position_per_fold():
    folds = _folds(5, 5)
    _require(
        all(test.size == 1 for _, test in folds) and len(folds) == 5,
        "with k = n each fold must test exactly one position (leave-one-out)",
    )


@pytest.mark.parametrize("k", [1, 0, -3], ids=["one", "zero", "negative"])
def test_fewer_than_two_folds_raises_value_error(k):
    try:
        kfold_indices(10, k)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"k below 2 raised {type(exc).__name__} instead of ValueError")
    pytest.fail("k below 2 must raise ValueError: a single fold has no training split")


@pytest.mark.parametrize("n", [3, 7, 40], ids=["small", "medium", "large"])
def test_more_folds_than_samples_raises_value_error(n):
    try:
        kfold_indices(n, n + 1)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"k above n raised {type(exc).__name__} instead of ValueError")
    pytest.fail("k above n must raise ValueError: a fold would have no test position")


def test_shuffle_without_a_seed_raises_value_error():
    try:
        kfold_indices(10, 3, shuffle=True)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a shuffle without seed raised {type(exc).__name__} instead of ValueError")
    pytest.fail("a shuffled split without a seed must raise ValueError: it cannot be reproduced")


def test_same_seed_reproduces_the_same_folds():
    first = _folds(23, 4, shuffle=True, seed=42)
    second = _folds(23, 4, shuffle=True, seed=42)
    for (tr1, te1), (tr2, te2) in zip(first, second, strict=True):
        _require(
            np.array_equal(tr1, tr2) and np.array_equal(te1, te2),
            "the same seed must reproduce the folds: draw from default_rng(seed), not a global",
        )


def test_seed_is_ignored_when_not_shuffling():
    plain = _folds(9, 3)
    seeded = _folds(9, 3, seed=5)
    for (tr1, te1), (tr2, te2) in zip(plain, seeded, strict=True):
        _require(
            np.array_equal(tr1, tr2) and np.array_equal(te1, te2),
            "without shuffle the folds must be identical whatever seed is passed",
        )


def test_index_arrays_are_sorted_and_integer_typed():
    for shuffle in (False, True):
        for train, test in _folds(14, 4, shuffle=shuffle, seed=3):
            _require(
                np.issubdtype(train.dtype, np.integer) and np.issubdtype(test.dtype, np.integer),
                "index arrays must hold integers, not floats or booleans",
            )
            _require(
                bool(np.all(np.diff(train) > 0)) and bool(np.all(np.diff(test) > 0)),
                "index arrays must be sorted in ascending order within each fold",
            )
