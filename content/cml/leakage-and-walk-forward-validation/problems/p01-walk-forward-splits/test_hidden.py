from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.model_selection import TimeSeriesSplit
from submission import walk_forward_splits

from dogma_data import load_series

# (n, initial_train, horizon, step, gap, expanding), covering both modes, gaps and boundary cases.
CONFIGS = [
    (10, 4, 2, 2, 0, True),
    (10, 4, 2, 2, 1, True),
    (10, 4, 2, 2, 0, False),
    (17, 5, 3, 1, 2, True),
    (17, 5, 3, 1, 2, False),
    (31, 6, 4, 3, 0, False),
    (12, 3, 1, 4, 3, True),
    (9, 2, 7, 1, 0, True),
]
CONFIG_IDS = [
    "expanding-plain",
    "expanding-gap",
    "rolling-plain",
    "expanding-unit-step",
    "rolling-gap",
    "rolling-wide-step",
    "long-gap",
    "single-split-edge",
]


def _reference(*, n, initial_train, horizon, step, gap, expanding):
    """Brute force: try every cutoff, and keep those on the step grid whose test window fits."""
    splits = []
    for cut in range(initial_train, n + 1):
        if (cut - initial_train) % step != 0:
            continue
        if cut + gap + horizon > n:
            continue
        lo = 0 if expanding else cut - initial_train
        splits.append((list(range(lo, cut)), list(range(cut + gap, cut + gap + horizon))))
    return splits


def _splits(*, n, initial_train, horizon, step, gap, expanding):
    """Call the learner's function and check the output shape, naming the mistake."""
    try:
        splits = walk_forward_splits(
            n, initial_train=initial_train, horizon=horizon, step=step, gap=gap, expanding=expanding
        )
    except ValueError:
        pytest.fail(
            "a valid configuration raised ValueError: check the window arithmetic and bounds"
        )
    except Exception as exc:
        pytest.fail(f"a valid configuration raised {type(exc).__name__}: check the index ranges")
    for train, test in splits:
        if not (isinstance(train, np.ndarray) and isinstance(test, np.ndarray)):
            pytest.fail("train_idx and test_idx must be numpy arrays, not lists")
    return splits


def _require(condition, mistake):
    if not condition:
        pytest.fail(mistake)


def _as_lists(splits):
    return [(train.tolist(), test.tolist()) for train, test in splits]


@pytest.mark.parametrize(
    ("n", "initial_train", "horizon", "step", "gap", "expanding"), CONFIGS, ids=CONFIG_IDS
)
def test_splits_match_the_brute_force_reference(*, n, initial_train, horizon, step, gap, expanding):
    expected = _reference(
        n=n, initial_train=initial_train, horizon=horizon, step=step, gap=gap, expanding=expanding
    )
    got = _as_lists(
        _splits(
            n=n,
            initial_train=initial_train,
            horizon=horizon,
            step=step,
            gap=gap,
            expanding=expanding,
        )
    )
    for (train, test), (exp_train, exp_test) in zip(got, expected, strict=False):
        _require(len(test) == horizon, "every test window must contain exactly horizon rows")
        _require(
            test == exp_test,
            "a test window starts in the wrong place: the cutoff is initial_train + j * step,"
            " and the test window begins gap rows after it",
        )
        _require(
            train == exp_train,
            "a training window differs from the reference: expanding windows start at row 0,"
            " rolling windows keep initial_train rows",
        )
    _require(
        len(got) == len(expected),
        "the number of splits is wrong: a split exists for every cutoff on the step grid whose"
        " test window ends by n",
    )


@pytest.mark.parametrize(
    ("n", "initial_train", "horizon", "step", "gap", "expanding"), CONFIGS, ids=CONFIG_IDS
)
def test_every_training_row_precedes_every_test_row_by_more_than_the_gap(
    *, n, initial_train, horizon, step, gap, expanding
):
    for train, test in _splits(
        n=n, initial_train=initial_train, horizon=horizon, step=step, gap=gap, expanding=expanding
    ):
        _require(
            train.max() < test.min() - gap,
            "training rows must end before the test window starts, with the gap skipped between",
        )
        _require(
            np.intersect1d(train, test).size == 0,
            "a training window overlaps its test window: move the test window after the cutoff",
        )


@pytest.mark.parametrize(
    ("n", "initial_train", "horizon", "step", "gap", "expanding"), CONFIGS, ids=CONFIG_IDS
)
def test_test_windows_stay_inside_the_series_and_no_further_split_fits(
    *, n, initial_train, horizon, step, gap, expanding
):
    splits = _splits(
        n=n, initial_train=initial_train, horizon=horizon, step=step, gap=gap, expanding=expanding
    )
    for _, test in splits:
        _require(
            test.size == horizon and test.max() < n,
            "every test window must have exactly horizon rows and stay inside the series",
        )
    last_cut = initial_train + (len(splits) - 1) * step
    _require(
        last_cut + step + gap + horizon > n,
        "a further split still fits: keep generating while the test window ends by n",
    )


@pytest.mark.parametrize(
    ("n", "initial_train", "horizon", "step", "gap", "expanding"), CONFIGS, ids=CONFIG_IDS
)
def test_windows_advance_by_exactly_step_and_rolling_windows_keep_their_length(
    *, n, initial_train, horizon, step, gap, expanding
):
    splits = _splits(
        n=n, initial_train=initial_train, horizon=horizon, step=step, gap=gap, expanding=expanding
    )
    ends = [int(train.max()) + 1 for train, _ in splits]
    _require(
        all(b - a == step for a, b in pairwise(ends)),
        "consecutive cutoffs must differ by exactly step",
    )
    if expanding:
        _require(
            all(train[0] == 0 for train, _ in splits),
            "an expanding training window must start at row 0 in every split",
        )
    else:
        _require(
            all(train.size == initial_train for train, _ in splits),
            "a rolling training window must have exactly initial_train rows in every split",
        )


def test_matches_sklearn_time_series_split_for_a_matching_configuration():
    # A real series: the 100 annual flows of the Nile, tested ten years at a time.
    n, horizon, splits_count, gap = int(load_series("nile").shape[0]), 10, 3, 2
    initial = n - splits_count * horizon - gap
    expected = list(
        TimeSeriesSplit(n_splits=splits_count, test_size=horizon, gap=gap).split(np.arange(n))
    )
    got = _splits(
        n=n, initial_train=initial, horizon=horizon, step=horizon, gap=gap, expanding=True
    )
    _require(len(got) == len(expected), "expanding split count differs from TimeSeriesSplit")
    for (tr, te), (exp_tr, exp_te) in zip(got, expected, strict=True):
        _require(
            np.array_equal(tr, exp_tr) and np.array_equal(te, exp_te),
            "expanding splits differ from TimeSeriesSplit for the same n, test size and gap",
        )
    expected_rolling = list(
        TimeSeriesSplit(
            n_splits=splits_count, test_size=horizon, gap=gap, max_train_size=initial
        ).split(np.arange(n))
    )
    got_rolling = _splits(
        n=n, initial_train=initial, horizon=horizon, step=horizon, gap=gap, expanding=False
    )
    for (tr, te), (exp_tr, exp_te) in zip(got_rolling, expected_rolling, strict=True):
        _require(
            np.array_equal(tr, exp_tr) and np.array_equal(te, exp_te),
            "rolling splits differ from TimeSeriesSplit with max_train_size equal to the window",
        )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    initial=st.integers(min_value=1, max_value=40),
    horizon=st.integers(min_value=1, max_value=8),
    step=st.integers(min_value=1, max_value=6),
    gap=st.integers(min_value=0, max_value=4),
    expanding=st.booleans(),
    extra=st.integers(min_value=0, max_value=40),
)
def test_random_configurations_match_the_reference(
    *, initial, horizon, step, gap, expanding, extra
):
    n = initial + gap + horizon + extra
    expected = _reference(
        n=n, initial_train=initial, horizon=horizon, step=step, gap=gap, expanding=expanding
    )
    got = _as_lists(
        _splits(
            n=n, initial_train=initial, horizon=horizon, step=step, gap=gap, expanding=expanding
        )
    )
    _require(got == expected, "splits differ from the reference for a random configuration")


INVALID = [
    ("zero_initial_train", {"initial_train": 0, "horizon": 2, "step": 1}),
    ("zero_horizon", {"initial_train": 3, "horizon": 0, "step": 1}),
    ("zero_step", {"initial_train": 3, "horizon": 2, "step": 0}),
    ("negative_gap", {"initial_train": 3, "horizon": 2, "step": 1, "gap": -1}),
    ("series_too_short", {"initial_train": 8, "horizon": 3, "step": 1, "gap": 1}),
]


@pytest.mark.parametrize(("case", "kwargs"), INVALID, ids=[c for c, _ in INVALID])
def test_invalid_parameters_raise_value_error(case, kwargs):
    try:
        walk_forward_splits(10, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid parameters raised {type(exc).__name__} instead of ValueError")
    pytest.fail("invalid parameters must raise ValueError with a message naming the rule")


def test_index_arrays_are_integer_typed_and_ascending():
    for train, test in _splits(n=15, initial_train=4, horizon=3, step=2, gap=1, expanding=True):
        _require(
            np.issubdtype(train.dtype, np.integer) and np.issubdtype(test.dtype, np.integer),
            "index arrays must hold integers, not floats or booleans",
        )
        _require(
            bool(np.all(np.diff(train) == 1)) and bool(np.all(np.diff(test) == 1)),
            "index arrays must be consecutive and ascending rows",
        )
