import pytest
from submission import early_stopping


def _call(losses, **kwargs):
    try:
        return early_stopping(losses, **kwargs)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(f"a valid example raised {type(exc).__name__}: check the bookkeeping")


def test_best_checkpoint_and_stop_after_three_epochs_without_improvement():
    assert _call([1.0, 0.8, 0.7, 0.72, 0.71, 0.75, 0.74], patience=3) == (2, 5), (
        "best epoch 2; epochs 3, 4 and 5 are the three without improvement, so stop at 5"
    )


def test_a_decrease_of_exactly_min_delta_is_not_an_improvement():
    assert _call([1.0, 0.75, 0.6], patience=5, min_delta=0.25) == (2, 2), (
        "0.75 is exactly min_delta below 1.0, which must not count as an improvement"
    )


def test_a_sequence_that_never_triggers_the_stop_ends_on_the_last_epoch():
    assert _call([3.0, 2.0, 1.0], patience=2) == (2, 2), (
        "with steady improvement training is never stopped, so the stop is the last epoch"
    )
