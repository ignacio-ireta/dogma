"""Hidden tests: an independent loop-by-loop statement of the protocol, Hypothesis properties,
scikit-learn's ceiling rule, and edge cases. The reference draws the same permutations from the
same generator but builds every count with plain Python loops. Each failure names the likely
mistake first."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sklearn.model_selection import train_test_split
from submission import train_test_split_indices

from dogma_data import rng, sample_rows

SETTINGS = settings(max_examples=60, deadline=None, report_multiple_bugs=False)


def _reference(n, test_size, seed, labels=None):
    """The protocol as explicit loops. Returns (train, test), or None where it must raise."""
    n_test = math.ceil(test_size * n)
    rng = np.random.default_rng(seed)
    if labels is None:
        if n_test >= n:
            return None
        perm = rng.permutation(n)
        return sorted(int(i) for i in perm[n_test:]), sorted(int(i) for i in perm[:n_test])
    classes = sorted(set(labels))
    counts = [labels.count(label) for label in classes]
    ideal = [test_size * count for count in counts]
    base = [math.floor(value) for value in ideal]
    leftover = n_test - sum(base)
    ranked = sorted(range(len(classes)), key=lambda j: (-(ideal[j] - base[j]), j))
    test_counts = list(base)
    for j in ranked[:leftover]:
        test_counts[j] += 1
    if any(t >= c for t, c in zip(test_counts, counts, strict=True)):
        return None
    train: list[int] = []
    test: list[int] = []
    for j, label in enumerate(classes):
        rows = [i for i, value in enumerate(labels) if value == label]
        shuffled = [int(i) for i in rng.permutation(np.array(rows, dtype=np.int64))]
        test.extend(shuffled[: test_counts[j]])
        train.extend(shuffled[test_counts[j] :])
    return sorted(train), sorted(test)


def _safe_split(n, test_size, seed, labels=None):
    """Return (train, test) from the learner, None on ValueError. Other crashes fail with a name."""
    try:
        train, test = train_test_split_indices(n, test_size, seed=seed, stratify=labels)
    except ValueError:
        return None
    except Exception as exc:
        pytest.fail(f"your function raised {type(exc).__name__} on a valid split")
    return train, test


def _same_rows(got, want, mistake):
    if not np.array_equal(np.asarray(got), np.asarray(want)):
        pytest.fail(mistake)


def _compare(n, test_size, seed, labels=None):
    want = _reference(n, test_size, seed, labels)
    got = _safe_split(n, test_size, seed, labels)
    if want is None:
        if got is not None:
            pytest.fail("an impossible split was accepted: a training side with no rows must raise")
        return
    if got is None:
        pytest.fail("a valid split was rejected: check the empty-training and class-count rules")
    if len(got[1]) != len(want[1]):
        pytest.fail("test side has the wrong number of rows: per-class shares must sum to n_test")
    _same_rows(
        got[1],
        want[1],
        "test rows differ from the protocol: check the draws and the per-class shares",
    )
    _same_rows(
        got[0],
        want[0],
        "train rows differ from the protocol: one generator from seed, one draw per class",
    )


def _real_labels(seed, n, classes):
    """n real class labels: the digits on n drawn images, among the digits below ``classes``.

    The labels come in the order of the draw, so the classes are interleaved, their counts are
    uneven, and in a small draw a class can have one member or none.
    """
    gen = rng(f"stratified-split-p01:labels:{seed}:{n}:{classes}")
    _, digit = sample_rows("digits", gen, 400, 1)
    return [int(label) for label in digit[digit < classes][:n]]


@SETTINGS
@given(n=st.integers(2, 60), tenths=st.integers(1, 99), seed=st.integers(0, 10_000))
def test_unstratified_split_matches_loop_reference(n, tenths, seed):
    _compare(n, tenths / 100, seed)


@SETTINGS
@given(data=st.data(), n=st.integers(2, 40), seed=st.integers(0, 10_000))
def test_stratified_split_matches_loop_reference(data, n, seed):
    labels = _real_labels(seed, n, 5)
    percent = data.draw(st.integers(1, 99))
    _compare(n, percent / 100, seed, labels)


@SETTINGS
@given(data=st.data(), n=st.integers(2, 40), seed=st.integers(0, 10_000))
def test_splits_partition_every_row_exactly_once(data, n, seed):
    stratified = data.draw(st.booleans())
    labels = _real_labels(seed, n, 4) if stratified else None
    got = _safe_split(n, data.draw(st.integers(1, 99)) / 100, seed, labels)
    assume(got is not None)
    train, test = got
    if len(set(train.tolist()) & set(test.tolist())) > 0:
        pytest.fail("train and test share a row: each row belongs to exactly one side")
    if not np.array_equal(np.sort(np.concatenate([train, test])), np.arange(n)):
        pytest.fail("rows are missing or repeated: train and test must cover 0..n-1")
    if not (np.all(np.diff(train) > 0) and np.all(np.diff(test) > 0)):
        pytest.fail("index arrays must be sorted in ascending order with no repeats")


@SETTINGS
@given(data=st.data(), n=st.integers(4, 60), seed=st.integers(0, 10_000))
def test_each_class_test_count_is_within_one_row_of_its_ideal(data, n, seed):
    labels = _real_labels(seed, n, 5)
    test_size = data.draw(st.integers(1, 99)) / 100
    ideals = {label: test_size * labels.count(label) for label in set(labels)}
    got = _safe_split(n, test_size, seed, labels)
    assume(got is not None)
    _, test = got
    test_labels = np.asarray(labels)[test]
    for label in set(labels):
        ideal = ideals[label]
        achieved = int(np.sum(test_labels == label))
        if not abs(achieved - ideal) < 1:
            pytest.fail("a class is off its share by a row or more: use largest remainders")


@SETTINGS
@given(n=st.integers(2, 200), tenths=st.integers(1, 99))
def test_test_count_matches_scikit_learn_ceiling_rule(n, tenths):
    test_size = tenths / 100
    try:
        _, sk_test = train_test_split(np.arange(n), test_size=test_size, random_state=0)
    except ValueError:
        assume(False)
    got = _safe_split(n, test_size, 0)
    assume(got is not None)
    if len(got[1]) != len(sk_test):
        pytest.fail(
            "test count differs from ceil(test_size * n): use math.ceil, not round or floor"
        )


@pytest.mark.parametrize("case", range(5), ids=["first", "second", "third", "fourth", "fifth"])
def test_test_count_is_ceiling_where_round_to_even_differs(case):
    test_size = [0.15, 0.25, 0.35, 0.45, 0.55][case]
    expected = [2, 3, 4, 5, 6][case]
    _, test = train_test_split_indices(10, test_size, seed=1)
    if len(test) != expected:
        pytest.fail("test count is not ceil(test_size * n) for this size: check the rounding")


@pytest.mark.parametrize(
    ("n", "test_size", "expected"),
    [(100, 0.07, 8), (25, 0.28, 8), (100, 0.55, 56)],
    ids=["just-above-seven", "just-above-seven-small", "just-above-fifty-five"],
)
def test_test_count_uses_the_float64_product_as_written(n, test_size, expected):
    got = _safe_split(n, test_size, 0)
    if got is None:
        pytest.fail("a valid split was rejected: check the test count and the empty-training rule")
    if len(got[1]) != expected:
        pytest.fail(
            "test count differs from the float64 protocol: take ceil of the float product, "
            "not of the exact decimal value"
        )


def test_stratified_shares_use_the_float64_products_as_written():
    labels = [0] * 37 + [1] * 26 + [2] * 12 + [3] * 40
    got = _safe_split(len(labels), 0.12, 0, labels)
    if got is None:
        pytest.fail("a valid stratified split was rejected: check the class-count rule")
    per_class = np.bincount(np.asarray(labels)[got[1]], minlength=4).tolist()
    if per_class != [4, 3, 2, 5]:
        pytest.fail(
            "per-class test counts differ from the float64 protocol: each ideal share is the float "
            "product test_size * n_j, then the largest remainders"
        )


def test_class_order_is_ascending_not_first_seen():
    labels = ["z", "a", "z", "a", "m", "a", "z", "m"]
    _compare(len(labels), 0.5, 4, labels)


def test_singleton_class_stays_in_training_when_its_share_is_zero():
    labels = [0] * 9 + [1]
    _compare(10, 0.1, 3, labels)
    train, _test = train_test_split_indices(10, 0.1, seed=3, stratify=labels)
    if 9 not in train.tolist():
        pytest.fail("a singleton class with a zero test share must stay in the training side")


def test_same_seed_reproduces_the_split_and_another_seed_changes_it():
    first = train_test_split_indices(30, 0.3, seed=7)
    again = train_test_split_indices(30, 0.3, seed=7)
    for a, b in zip(first, again, strict=True):
        if not np.array_equal(a, b):
            pytest.fail("the same seed must reproduce the split exactly")
    others = [train_test_split_indices(30, 0.3, seed=s)[1] for s in range(1, 6)]
    if all(np.array_equal(first[1], other) for other in others):
        pytest.fail("the seed must control the draws: every seed gave the same test rows")


@pytest.mark.parametrize(
    ("n", "test_size", "labels"),
    [
        (1, 0.5, None),
        (0, 0.5, None),
        (10, 0.0, None),
        (10, 1.0, None),
        (10, -0.2, None),
        (10, 1.5, None),
        (10, 0.5, [0, 1, 0]),
        (10, 0.5, [[0, 1]] * 5),
        (10.5, 0.5, None),
    ],
    ids=[
        "one-row",
        "no-rows",
        "zero-size",
        "full-size",
        "negative-size",
        "above-one",
        "short-labels",
        "two-dimensional-labels",
        "fractional-rows",
    ],
)
def test_invalid_arguments_raise_value_error(n, test_size, labels):
    _expect_value_error(
        lambda: train_test_split_indices(n, test_size, seed=0, stratify=labels),
        "invalid split arguments were accepted: check n, test_size and the shape of stratify",
    )


def test_singleton_class_with_a_test_share_raises_value_error():
    _expect_value_error(
        lambda: train_test_split_indices(4, 0.5, seed=0, stratify=["a", "b", "b", "b"]),
        "a class with no training row was accepted: raise ValueError when every class must train",
    )


def test_returned_indices_are_integer_arrays():
    train, test = train_test_split_indices(12, 0.25, seed=2, stratify=[0, 1] * 6)
    for name, arr in (("train", train), ("test", test)):
        if arr.dtype.kind != "i":
            pytest.fail(f"{name} indices must be integers, not {arr.dtype}")


def test_labels_and_stratify_input_are_not_mutated():
    labels = np.array([3, 1, 3, 1, 2, 2, 1, 3])
    before = labels.copy()
    train_test_split_indices(8, 0.5, seed=5, stratify=labels)
    if not np.array_equal(labels, before):
        pytest.fail("stratify was modified in place: sort or relabel a copy, not the input")
    as_list = [3, 1, 3, 1, 2, 2, 1, 3]
    train_test_split_indices(8, 0.5, seed=5, stratify=as_list)
    if as_list != [3, 1, 3, 1, 2, 2, 1, 3]:
        pytest.fail("a list passed as stratify was modified in place")


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_returned_index_arrays_are_int64():
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    for stratify in (None, labels):
        train_idx, test_idx = train_test_split_indices(10, 0.5, seed=0, stratify=stratify)
        for name, arr in (("train", train_idx), ("test", test_idx)):
            assert arr.dtype == np.int64, (
                f"the {name} indices must be int64 arrays, as the contract states"
            )


def test_stratify_must_be_one_dimensional_with_one_label_per_row():
    _expect_value_error(
        lambda: train_test_split_indices(10, 0.5, seed=0, stratify=[0, 1, 0, 1, 0, 1, 0, 1, 0]),
        "stratify with the wrong number of labels was accepted: it must have one label per row",
    )
    _expect_value_error(
        lambda: train_test_split_indices(
            10, 0.5, seed=0, stratify=np.array([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]])
        ),
        "two-dimensional stratify was accepted: stratify must be one-dimensional",
    )


def test_seed_is_keyword_only_in_the_signature():
    import inspect  # noqa: PLC0415

    # Contract: `seed`: integer, keyword-only. A positional seed must not be accepted.
    kind = inspect.signature(train_test_split_indices).parameters["seed"].kind
    if kind is not inspect.Parameter.KEYWORD_ONLY:
        pytest.fail("seed must be keyword-only: a positional seed was accepted")
