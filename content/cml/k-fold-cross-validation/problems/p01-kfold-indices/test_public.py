import pytest
from submission import kfold_indices


def _tests_of(folds):
    return [test.tolist() for _, test in folds]


def test_ten_positions_in_three_folds_give_consecutive_blocks_with_the_remainder_first():
    folds = kfold_indices(10, 3)
    assert _tests_of(folds) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]], (
        "unshuffled folds must be consecutive blocks, with the extra position in the first fold"
    )


def test_shuffled_folds_follow_the_seeded_permutation_protocol():
    folds = kfold_indices(6, 3, shuffle=True, seed=0)
    assert _tests_of(folds) == [[2, 3], [4, 5], [0, 1]], (
        "shuffled folds must be chunks of default_rng(seed).permutation(n), sorted within each fold"
    )


def test_more_folds_than_samples_raises_value_error():
    try:
        kfold_indices(3, 4)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"k above n raised {type(exc).__name__} instead of ValueError")
    pytest.fail("k above n must raise ValueError: a fold would have no test position")
