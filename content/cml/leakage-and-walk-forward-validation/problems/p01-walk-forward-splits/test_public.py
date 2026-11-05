import pytest
from submission import walk_forward_splits


def _pairs(splits):
    return [(train.tolist(), test.tolist()) for train, test in splits]


def _raises_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_expanding_windows_grow_from_row_zero_and_advance_by_step():
    splits = walk_forward_splits(10, initial_train=4, horizon=2, step=2)
    assert _pairs(splits) == [
        ([0, 1, 2, 3], [4, 5]),
        ([0, 1, 2, 3, 4, 5], [6, 7]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]),
    ], "expanding splits: train from row 0 up to the cutoff, test the next horizon rows"


def test_gap_skips_rows_between_training_and_testing():
    splits = walk_forward_splits(10, initial_train=4, horizon=2, step=2, gap=1)
    assert _pairs(splits) == [([0, 1, 2, 3], [5, 6]), ([0, 1, 2, 3, 4, 5], [7, 8])], (
        "with gap=1 the test window must start one row after the training window ends"
    )


def test_rolling_window_keeps_its_length():
    splits = walk_forward_splits(10, initial_train=4, horizon=2, step=2, expanding=False)
    assert _pairs(splits) == [
        ([0, 1, 2, 3], [4, 5]),
        ([2, 3, 4, 5], [6, 7]),
        ([4, 5, 6, 7], [8, 9]),
    ], "rolling splits: every training window has exactly initial_train rows and slides by step"


def test_series_too_short_for_one_split_raises_value_error():
    _raises_value_error(
        lambda: walk_forward_splits(3, initial_train=4, horizon=1, step=1),
        "a series shorter than initial_train + gap + horizon must raise ValueError",
    )
