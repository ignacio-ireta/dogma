import numpy as np
import pytest
from submission import minibatch_indices


def _as_lists(schedule):
    if not isinstance(schedule, list) or not all(isinstance(e, list) for e in schedule):
        pytest.fail("schedule must be a list with one list of batches per epoch")
    if not all(isinstance(b, np.ndarray) and b.ndim == 1 for e in schedule for b in e):
        pytest.fail("each batch must be a 1-D index array")
    return [[list(map(int, batch)) for batch in epoch] for epoch in schedule]


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_two_epochs_follow_one_generator_and_keep_the_short_batch():
    schedule = _as_lists(minibatch_indices(5, 2, 2, seed=0))
    if schedule != [[[2, 4], [3, 0], [1]], [[4, 1], [2, 0], [3]]]:
        pytest.fail(
            "epoch batches differ from the protocol: one generator, one permutation per epoch"
        )


def test_drop_last_removes_only_the_short_final_batch():
    schedule = _as_lists(minibatch_indices(5, 2, 1, seed=0, drop_last=True))
    if schedule != [[[2, 4], [3, 0]]]:
        pytest.fail("drop_last must remove the short final batch and nothing else")


def test_zero_epochs_gives_an_empty_schedule():
    if minibatch_indices(5, 2, 0, seed=0) != []:
        pytest.fail("zero epochs must give an empty list, with no draws")


def test_batch_size_zero_raises_value_error():
    _expect_value_error(
        lambda: minibatch_indices(5, 0, 1, seed=0),
        "batch_size 0 was accepted: batch_size must be at least 1",
    )
