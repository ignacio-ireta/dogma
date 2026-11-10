"""Hidden tests: an independent loop-by-loop statement of the protocol, Hypothesis properties and
edge cases. The reference builds each permutation with arange and rng.shuffle; for a Generator,
rng.permutation(n) makes the same draws, so the two routes must agree. Each failure names the
likely mistake first."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import minibatch_indices

SETTINGS = settings(max_examples=60, deadline=None, report_multiple_bugs=False)


def _reference(n, batch_size, epochs, seed, drop_last):
    rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(epochs):
        order = np.arange(n)
        rng.shuffle(order)
        batches = []
        start = 0
        while start < n:
            batch = order[start : start + batch_size]
            start += batch_size
            if drop_last and len(batch) < batch_size:
                continue
            batches.append([int(i) for i in batch])
        schedule.append(batches)
    return schedule


def _structure_ok(schedule, epochs):
    """Return a mistake message when the nesting is wrong, else None. Never raises."""
    if not isinstance(schedule, list) or len(schedule) != epochs:
        return "schedule must be a list with one entry per epoch"
    for epoch in schedule:
        if not isinstance(epoch, list):
            return "each epoch must be a list of batches, not a single array"
        for batch in epoch:
            if not isinstance(batch, np.ndarray) or batch.ndim != 1:
                return "each batch must be a 1-D index array"
    return None


def _schedule(n, batch_size, epochs, seed, drop_last=False):
    """Call the learner on valid hidden inputs. A rejection names the rule, never the values."""
    try:
        return minibatch_indices(n, batch_size, epochs, seed=seed, drop_last=drop_last)
    except Exception as exc:
        pytest.fail(
            f"valid schedule arguments were rejected with {type(exc).__name__}: "
            "check the batch size, epoch count and drop_last rules"
        )


def _as_lists(schedule):
    return [[[int(i) for i in batch] for batch in epoch] for epoch in schedule]


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@SETTINGS
@given(
    n=st.integers(1, 40),
    data=st.data(),
    epochs=st.integers(0, 4),
    seed=st.integers(0, 10_000),
    drop_last=st.booleans(),
)
def test_schedule_matches_loop_reference(n, data, epochs, seed, drop_last):
    batch_size = data.draw(st.integers(1, n))
    want = _reference(n, batch_size, epochs, seed, drop_last)
    got = _schedule(n, batch_size, epochs, seed, drop_last)
    problem = _structure_ok(got, epochs)
    if problem is not None:
        pytest.fail(problem)
    if _as_lists(got) != want:
        pytest.fail(
            "batches differ from the protocol: check the generator, the draws and the slices"
        )


@SETTINGS
@given(n=st.integers(1, 40), data=st.data(), seed=st.integers(0, 10_000))
def test_every_row_appears_once_per_epoch_when_the_last_batch_is_kept(n, data, seed):
    batch_size = data.draw(st.integers(1, n))
    schedule = _schedule(n, batch_size, 2, seed)
    problem = _structure_ok(schedule, 2)
    if problem is not None:
        pytest.fail(problem)
    for epoch in schedule:
        flat = np.concatenate(epoch) if epoch else np.array([], dtype=np.int64)
        if not np.array_equal(np.sort(flat), np.arange(n)):
            pytest.fail("each epoch must visit every row exactly once when drop_last is False")


@SETTINGS
@given(n=st.integers(1, 40), data=st.data(), seed=st.integers(0, 10_000))
def test_drop_last_only_removes_a_short_final_batch(n, data, seed):
    batch_size = data.draw(st.integers(1, n))
    full = _schedule(n, batch_size, 1, seed)[0]
    dropped = _schedule(n, batch_size, 1, seed, drop_last=True)[0]
    expected_count = n // batch_size
    if len(dropped) != expected_count:
        pytest.fail("drop_last must keep exactly the full batches: floor(n / batch_size) of them")
    if not all(len(b) == batch_size for b in dropped):
        pytest.fail("drop_last left a short batch in the schedule")
    for kept, original in zip(dropped, full[: len(dropped)], strict=True):
        if not np.array_equal(kept, original):
            pytest.fail("drop_last changed the order of the batches it kept")


def test_epochs_are_fresh_permutations_from_one_generator():
    schedule = minibatch_indices(12, 4, 3, seed=5)
    orders = [tuple(np.concatenate(epoch).tolist()) for epoch in schedule]
    if len(set(orders)) == 1:
        pytest.fail("every epoch repeated the same order: draw one permutation per epoch")


def test_batch_size_equal_to_n_gives_one_full_batch_per_epoch():
    schedule = minibatch_indices(7, 7, 3, seed=1)
    for epoch in schedule:
        if len(epoch) != 1 or sorted(epoch[0].tolist()) != list(range(7)):
            pytest.fail("with batch_size = n each epoch must be a single batch of all rows")


def test_same_seed_reproduces_and_another_seed_changes_the_schedule():
    first = _as_lists(minibatch_indices(20, 6, 2, seed=9))
    again = _as_lists(minibatch_indices(20, 6, 2, seed=9))
    if first != again:
        pytest.fail("the same seed must reproduce the schedule exactly")
    others = [_as_lists(minibatch_indices(20, 6, 2, seed=s)) for s in range(1, 6)]
    if all(other == first for other in others):
        pytest.fail("the seed must control the draws: every seed gave the same schedule")


def test_batches_keep_permutation_order_not_sorted_order():
    schedule = _as_lists(minibatch_indices(9, 9, 1, seed=0))
    if schedule[0][0] == sorted(schedule[0][0]):
        pytest.fail("batches were sorted: keep the order of the permutation inside each batch")


def test_returned_indices_are_integer_arrays():
    schedule = minibatch_indices(6, 4, 1, seed=3)
    for batch in schedule[0]:
        if np.asarray(batch).dtype.kind != "i":
            pytest.fail("batch indices must be integers")


@pytest.mark.parametrize(
    ("n", "batch_size", "epochs", "drop_last"),
    [
        (0, 1, 1, False),
        (5, 0, 1, False),
        (5, 6, 1, False),
        (5, 2, -1, False),
        (5, 2.5, 1, False),
        (5.0, 2, 1, False),
        (5, 2, 1.5, False),
    ],
    ids=[
        "no-rows",
        "zero-batch",
        "batch-above-n",
        "negative-epochs",
        "fractional-batch",
        "float-rows",
        "fractional-epochs",
    ],
)
def test_invalid_arguments_raise_value_error(n, batch_size, epochs, drop_last):
    _expect_value_error(
        lambda: minibatch_indices(n, batch_size, epochs, seed=0, drop_last=drop_last),
        "invalid schedule arguments were accepted: check n, batch_size and epochs",
    )


def test_drop_last_defaults_to_false():
    schedule = minibatch_indices(5, 2, 1, seed=0)
    if len(schedule[0]) != 3:
        pytest.fail("drop_last must default to False and keep the short final batch")


def test_zero_batch_size_raises_value_error_even_without_epochs():
    _expect_value_error(
        lambda: minibatch_indices(5, 0, 0, seed=0),
        "a batch_size of zero was accepted: batch_size must be an integer from 1 to n, "
        "raise ValueError",
    )


def test_batch_indices_are_int64():
    schedule = minibatch_indices(5, 2, 2, seed=0)
    for epoch in schedule:
        for batch in epoch:
            assert batch.dtype == np.int64, (
                "each batch must be an int64 index array, as the contract states"
            )


def test_seed_is_keyword_only_in_the_signature():
    import inspect  # noqa: PLC0415

    # Contract: `seed`: integer, keyword-only. A positional seed must not be accepted.
    kind = inspect.signature(minibatch_indices).parameters["seed"].kind
    if kind is not inspect.Parameter.KEYWORD_ONLY:
        pytest.fail("seed must be keyword-only: a positional seed was accepted")
