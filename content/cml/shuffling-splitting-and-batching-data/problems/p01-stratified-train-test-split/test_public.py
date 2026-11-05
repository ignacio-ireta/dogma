import numpy as np
import pytest
from submission import train_test_split_indices


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_unstratified_split_follows_the_permutation_protocol():
    train, test = train_test_split_indices(10, 0.25, seed=0)
    if not np.array_equal(test, [2, 4, 6]):
        pytest.fail(
            "test rows differ from the protocol: one rng.permutation(n), then the first n_test"
        )
    if not np.array_equal(train, [0, 1, 3, 5, 7, 8, 9]):
        pytest.fail("train rows differ from the protocol: the rest of the permutation, sorted")


def test_stratified_split_takes_exact_per_class_counts():
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    train, test = train_test_split_indices(10, 0.5, seed=0, stratify=labels)
    if not np.array_equal(train, [1, 3, 4, 5, 8]) or not np.array_equal(test, [0, 2, 6, 7, 9]):
        pytest.fail(
            "stratified rows differ from the protocol: per-class draws in ascending class order"
        )


def test_test_count_is_the_ceiling_not_the_rounded_value():
    _, test = train_test_split_indices(10, 0.25, seed=0)
    if len(test) != 3:
        pytest.fail(
            "test count rounds the wrong way: n_test is ceil(test_size * n), so 2.5 gives 3"
        )


def test_split_that_leaves_no_training_rows_raises_value_error():
    _expect_value_error(
        lambda: train_test_split_indices(2, 0.9, seed=0),
        "an empty training side was accepted: raise ValueError when n_test >= n",
    )
