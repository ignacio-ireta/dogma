import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import early_stopping


def _call(losses, *, patience, min_delta=0.0):
    """Call the learner's function and check the result type, naming the mistake if it is off."""
    try:
        result = early_stopping(losses, patience=patience, min_delta=min_delta)
    except ValueError:
        pytest.fail("a valid sequence raised ValueError: check the patience and improvement rules")
    except Exception as exc:
        pytest.fail(f"a valid sequence raised {type(exc).__name__}: check the bookkeeping")
    if not (isinstance(result, tuple) and len(result) == 2):
        pytest.fail("the result must be a (best_epoch, stop_epoch) pair")
    best, stop = result
    if not (isinstance(best, int) and isinstance(stop, int)):
        pytest.fail("best_epoch and stop_epoch must be Python ints, not arrays or floats")
    return best, stop


def _require(condition, mistake):
    if not condition:
        pytest.fail(mistake)


def _reference(losses, patience, min_delta):
    """Brute force, written differently: list the improving epochs, then find the first stall."""
    improving = []
    best = float("inf")
    for epoch, loss in enumerate(losses):
        if loss < best - min_delta:
            improving.append(epoch)
            best = loss
    for epoch in range(len(losses)):
        last = (
            max(e for e in improving if e <= epoch) if any(e <= epoch for e in improving) else None
        )
        if last is not None and epoch - last >= patience:
            return last, epoch
    return improving[-1], len(losses) - 1


# Values are multiples of 0.25 and min_delta is a multiple of 0.25 too, so every comparison at the
# improvement boundary is exact in floating point.
@pytest.mark.parametrize(
    ("patience", "min_delta", "seed"),
    [(1, 0.0, 0), (2, 0.0, 1), (3, 0.25, 2), (4, 0.5, 3), (2, 0.25, 4), (5, 0.0, 5)],
    ids=[
        "patience-one",
        "patience-two",
        "small-delta",
        "large-delta",
        "mid-delta",
        "patience-five",
    ],
)
def test_matches_the_brute_force_reference_on_seeded_sequences(patience, min_delta, seed):
    rng = np.random.default_rng(seed)
    for _ in range(40):
        length = int(rng.integers(1, 25))
        losses = (np.round(rng.normal(size=length) * 4) / 4).tolist()
        exp_best, exp_stop = _reference(losses, patience, min_delta)
        got_best, got_stop = _call(losses, patience=patience, min_delta=min_delta)
        _require(
            0 <= got_stop <= length - 1,
            "the stop epoch must lie inside the sequence: without a stop it is the last epoch",
        )
        _require(
            got_stop == exp_stop,
            "the stop must be the first epoch whose wait, counted from the last improvement,"
            " reaches patience: a tie with the best is not an improvement and does not reset it",
        )
        _require(
            got_best == exp_best,
            "the best checkpoint must be the last epoch that beat the best loss by more than"
            " min_delta: ties and exact min_delta decreases do not count",
        )


def test_a_decrease_of_exactly_min_delta_is_not_an_improvement():
    _require(
        _call([1.0, 0.75, 0.6], patience=5, min_delta=0.25) == (2, 2),
        "a decrease of exactly min_delta is not an improvement: use a strict comparison",
    )
    _require(
        _call([1.0, 0.5, 0.25], patience=1, min_delta=0.5) == (0, 1),
        "a decrease of exactly min_delta must not reset the wait counter",
    )


def test_ties_with_the_best_loss_are_not_improvements():
    _require(
        _call([0.5, 0.5, 0.5, 0.5], patience=2) == (0, 2),
        "ties with the best loss are not improvements: the best checkpoint is the first one",
    )


def test_wait_counter_resets_after_each_improvement():
    losses = [1.0, 1.25, 1.25, 0.75, 1.0, 1.0, 1.0]
    _require(
        _call(losses, patience=3) == (3, 6),
        "the wait counter must restart from zero after every improvement",
    )


def test_patience_one_stops_at_the_first_epoch_without_improvement():
    _require(
        _call([1.0, 2.0, 0.5, 0.75], patience=1) == (0, 1),
        "with patience 1 the first non-improving epoch must stop training",
    )


def test_single_epoch_is_its_own_best_checkpoint_and_stop():
    _require(_call([4.25], patience=1) == (0, 0), "a single epoch must give (0, 0)")


def test_stop_comes_exactly_patience_epochs_after_the_best_checkpoint():
    rng = np.random.default_rng(11)
    for _ in range(60):
        losses = (np.round(rng.normal(size=30) * 4) / 4).tolist()
        patience = int(rng.integers(1, 6))
        best, stop = _call(losses, patience=patience)
        if stop < len(losses) - 1:
            _require(
                stop - best == patience,
                "a stop must come exactly patience epochs after the best checkpoint",
            )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.lists(st.integers(min_value=-8, max_value=8), min_size=1, max_size=30),
    st.integers(min_value=1, max_value=6),
)
def test_with_zero_min_delta_the_best_checkpoint_is_the_first_running_minimum(raw, patience):
    losses = [v / 4.0 for v in raw]
    best, stop = _call(losses, patience=patience, min_delta=0.0)
    _require(0 <= best <= stop <= len(losses) - 1, "epochs must satisfy 0 <= best <= stop <= last")
    prefix = losses[: stop + 1]
    _require(
        losses[best] == min(prefix) and best == prefix.index(min(prefix)),
        "the best checkpoint must be the first epoch holding the lowest loss seen before the stop",
    )


INVALID = [
    ("empty", [], 2, 0.0, "an empty loss sequence has no epochs: raise ValueError"),
    ("two_dimensional", [[1.0, 0.5], [0.4, 0.3]], 2, 0.0, "losses must be 1-D: raise ValueError"),
    ("nan_loss", [1.0, float("nan"), 0.5], 2, 0.0, "a NaN loss must raise ValueError"),
    ("zero_patience", [1.0, 0.5], 0, 0.0, "patience below 1 must raise ValueError"),
    ("negative_min_delta", [1.0, 0.5], 2, -0.1, "a negative min_delta must raise ValueError"),
]


@pytest.mark.parametrize(
    ("case", "losses", "patience", "min_delta", "mistake"),
    INVALID,
    ids=[case[0] for case in INVALID],
)
def test_invalid_arguments_raise_value_error(case, losses, patience, min_delta, mistake):
    try:
        early_stopping(losses, patience=patience, min_delta=min_delta)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_input_array_is_not_mutated():
    losses = np.array([0.9, 0.4, 0.6, 0.7, 0.8])
    before = losses.copy()
    early_stopping(losses, patience=2)
    _require(bool(np.array_equal(losses, before)), "the loss array was modified in place")


def test_tiny_improvement_below_single_precision_resolution_is_an_improvement():
    _require(
        early_stopping([1.0, 1.0 - 1e-9], patience=1) == (1, 1),
        "a loss lower than the best by 1e-9 is an improvement: compare the losses at full "
        "float64 precision, not after rounding them to a coarser format",
    )
